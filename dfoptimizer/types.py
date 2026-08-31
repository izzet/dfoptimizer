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
