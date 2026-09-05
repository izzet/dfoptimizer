import dataclasses as dc
from typing import Any, Dict, List, Literal, Optional


@dc.dataclass
class KnobResponse:
    """How a knob responds to a specific opportunity_tag."""

    direction: Literal["increase", "decrease", "set"]
    step: Any = None  # for increase/decrease
    step_mode: Literal["add", "multiply", "evidence"] = "add"  # "add": old+step, "multiply": old*step, "evidence": computed from finding metrics
    set_to: Any = None  # for direction="set"
    min_severity: float = 0.5
    min_persistence: int = 2
    cooldown_windows: int = 3
    apply_when: str = "epoch_boundary"
    effectiveness_threshold: float = 0.10  # min severity improvement to allow next action


@dc.dataclass
class ActionDef:
    """An actuator that may COST something, may be IRREVERSIBLE, and may have a PRECONDITION.

    `KnobDef` is the degenerate case of this: cost ~= the loader rebuild (measured 0.21 s on the TF
    path), reversible, precondition = a range check, and the "side effect" is assigning a function
    argument. Every actuator in the DL-input-tuner literature (AUTOTUNE, Cachew, Pecan, cedar,
    Plumber) is degenerate in exactly that way, which is why none of them carries a cost model.

    Non-degenerate actuators break the assumption those systems are built on:

        read/decode threads, prefetch   0.21 s      reversible
        staging to node-local storage   465-1365 s  abortable during, not undoable after
        restripe (`lfs migrate`)        4.4 h       not reversible

    Costs are functions, not constants, because they are NOT stable: stage-in was measured at
    465, 474, 477, 679, 1181 and 1365 s for byte-identical work (2.9x). With R1's 378 s/epoch gain
    that is the difference between a break-even of 1.23 and 3.61 epochs, so a precomputed threshold
    is wrong at one end or the other.

    `cost_fn`, `precondition_fn` and `effect_fn` take (observation, value); `reconcile_fn` takes the
    desired value and performs the side effect on the rank that owns it.
    """

    id: str  # e.g. "dlio.data_source"
    values: List[Any]  # desired states, e.g. ["pfs", "local"]
    target_domain: str = "storage"  # storage | decode | comm | compute | transfer
    reversible: bool = True
    abortable: bool = False  # can be given up part-way, paying only the work already done
    scope: str = "node"

    # MUST every rank apply this in the SAME window, or none?
    #
    # This is NOT a precondition. A precondition asks "may I?" and is evaluated per node; its failure
    # is a declined action. Unanimity asks "must everyone, simultaneously?" and its failure is a
    # HANG -- no error, no timeout, nothing in the log.
    #
    # Raised by the Tuolumne branch from a root cause they already paid for: `halo_width` sets how
    # many boundary planes are exchanged, so it changes the P2P message size, and their `_tune_*_live`
    # enable flags were set only on optimizer-context ranks. Collective actuators therefore ran on
    # 4 of 16 ranks and the job hung. `halo3d_train.py:447` still carries the scar comment.
    #
    # Our staging action independently needed the same mechanism (allreduce(min) veto,
    # allreduce(max) bail) but special-cased it. Declaring it makes `halo_width` and `staging` the
    # same kind of object, which is the point of the generalisation.
    #
    # NOTE the axes are independent: `comm_overlap` is free, reversible AND unanimity-free (it
    # reschedules the same ops, so disagreeing ranks skew rather than deadlock), while `halo_width`
    # is free and reversible but REQUIRES unanimity. Cost does not predict it.
    requires_unanimity: bool = False

    # For actuators that touch the math. `comm_overlap` was verified numerically identical to the
    # blocking path at 4.3e-07 relative max difference, making it a scheduling change rather than a
    # model change. `amp` (bf16) genuinely changes numerics and cannot claim this. An actuator that
    # silently changes the computation is not an actuator, so anything touching the math carries
    # either a verified bound here or an explicit note that it changes results.
    equivalence_verified: Optional[float] = None
    cost_fn: Optional[Any] = None
    precondition_fn: Optional[Any] = None
    effect_fn: Optional[Any] = None
    reconcile_fn: Optional[Any] = None
    target_function: str = ""

    def cost(self, obs, value) -> float:
        return 0.0 if self.cost_fn is None else float(self.cost_fn(obs, value))

    def precondition(self, obs, value):
        return (True, "") if self.precondition_fn is None else self.precondition_fn(obs, value)

    @staticmethod
    def from_knob(kdef: "KnobDef", rebuild_seconds: float = 0.21) -> "ActionDef":
        """Wrap an existing knob so old declarations keep working and gain honest cost fields."""
        return ActionDef(
            id=kdef.id,
            values=list(kdef.values) if kdef.values else [],
            target_domain="storage",
            reversible=True,
            abortable=True,
            scope=kdef.scope,
            requires_unanimity=kdef.requires_unanimity,
            cost_fn=lambda _o, _v, c=rebuild_seconds: c,
            precondition_fn=lambda _o, v, k=kdef: (k.clamp(v) == v,
                                                  "" if k.clamp(v) == v else "outside knob range"),
            target_function=kdef.target_function,
        )


@dc.dataclass
class KnobDef:
    """Definition of a tunable knob."""

    id: str  # e.g. "dlio.prefetch_size"
    default: Any
    type: type  # int, float, str
    range: Optional[tuple] = None  # (min, max) for numeric
    values: Optional[List[Any]] = None  # for enum-like knobs
    scope: str = "job"  # "job", "rank_local", "node"

    # SECONDS TO APPLY THIS KNOB. Default 0.0 -- every knob declared before this existed is free and
    # reversible, which is exactly the assumption every DL input-pipeline tuner makes, and why none
    # of them needs a cost model. Measured here: a loader rebuild is 0.21 s, staging a shard is
    # 465-1365 s, `lfs migrate` is 4.4 h. When actions differ by seven orders of magnitude, firing
    # every knob that matches a tag is not a policy.
    #
    # The planner uses this to rank candidates cheapest-first. With all costs at the default the
    # ranking is a no-op and behaviour is unchanged, so this cannot regress existing knobs.
    # float, OR a dict keyed by VALUE. Cost is a property of what you are asking for, not of the
    # knob: data_source="pfs" costs nothing, "local" costs 465-1365 s, "restriped" costs ~4.4 h.
    # Collapsing those into one number per knob makes the expensive value look cheap or the cheap
    # value look expensive, and either way the ranking is wrong.
    cost_seconds: Any = 0.0

    # True when every participant must apply the value at the same boundary. This is plan
    # metadata, not permission to insert a collective into an application from the optimizer.
    requires_unanimity: bool = False

    # True when moving this knob changes the REGIME the objective is measured in, so samples taken
    # before the move and samples taken after it are not observations of the same quantity.
    # Measured (T3-0 section 4p): staging the corpus off Lustre took a cosmoflow analysis window
    # from ~37 s to ~5.6 s, a 6.7x shift, and every other knob's sample deque pooled both sides of
    # it. Within-arm variance then swamps any between-arm difference and Welch cannot separate two
    # prefetch levels however well they are sampled. This is declared by the knob rather than
    # inferred from a cost or a name: only the knob knows that its action relocates the bytes.
    changes_regime: bool = False

    # True when moving this knob RELOCATES the bytes to a different device, so its gain is bounded
    # by what that device can deliver and its cost scales with how much there is to move. Measured
    # (T3-0 section 4s): the same price cleared staging on cosmoflow, where the node read Lustre at
    # 0.71 GB/s and moved to an SSD delivering about 2.05, and on unet3d, where the node was already
    # reading at 2.89 to 3.36 GB/s and moved to a slower device and lost. Nothing the rule looked at
    # separated them, because it asked only how bad the SOURCE was.
    relocates_data: bool = False

    def cost_for(self, value=None) -> float:
        """Cost of moving this knob TO `value`.

        With no value the MAXIMUM is returned. That is deliberate: when the planner cannot yet tell
        which value a response will produce, the safe assumption for an expensive knob is the worst
        case, so it defers rather than fires on an optimistic guess.
        """
        c = self.cost_seconds
        if isinstance(c, dict):
            if value is None:
                return float(max(c.values())) if c else 0.0
            return float(c.get(value, max(c.values()) if c else 0.0))
        return float(c or 0.0)
    responds_to: Dict[str, KnobResponse] = dc.field(default_factory=dict)
    target_function: str = ""  # which @tunable function owns this knob

    def clamp(self, value: Any) -> Any:
        if self.values is not None:
            return value if value in self.values else self.default
        if self.range is not None:
            lo, hi = self.range
            return self.type(max(lo, min(hi, value)))
        return self.type(value)


@dc.dataclass
class ActionPlan:
    """A concrete action the optimizer wants to apply."""

    plan_id: str
    knob_id: str
    target_function: str  # e.g. "make_dataloader"
    old_value: Any
    new_value: Any
    apply_when: str
    rationale: str  # human-readable reason
    finding_type: str
    severity: float
    opportunity_tag: str
    window_index: int
    target_nodes: List[str] = dc.field(default_factory=list)  # empty = all nodes
    # A tag consumed by collective-aware runtimes. The planner does not implement the collective.
    requires_unanimity: bool = False


@dc.dataclass
class ActionAck:
    """Acknowledgement from the app that an action was applied (or rejected)."""

    plan_id: str
    knob_id: str
    status: Literal["applied", "rejected", "deferred"]
    old_value: Any = None
    new_value: Any = None
    reason: str = ""


@dc.dataclass
class DiagnosisFindingMsg:
    """Wire format of a DiagnosisFinding received from Mofka."""

    finding_type: str
    motif: str
    severity: str  # human-readable label for logging
    severity_score: float  # continuous [0.0, 1.0] for gating/scaling
    confidence: float
    prevalence: float
    persistence: int
    trend_direction: str
    contributing_facts: List[tuple]
    recommendation_bundle: str
    opportunity_tags: List[str]
    summary: str
    scope: str = ""
    layer: Optional[str] = None
    support_windows: int = 0
    last_seen_window: int = 0
    window_index: int = 0
    publish_mode: str = "control"
    suppresses_tags: List[str] = dc.field(default_factory=list)
    key_metrics: Dict[str, float] = dc.field(default_factory=dict)


@dc.dataclass
class KnobRegistration:
    """Wire format: an app registering its knobs with the optimizer."""

    namespace: str
    function_name: str
    knobs: Dict[str, dict]  # param_name -> serialized KnobDef + responds_to
