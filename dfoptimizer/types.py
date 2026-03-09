import dataclasses as dc
from typing import Any, Dict, List, Literal, Optional


@dc.dataclass
class KnobResponse:
    """How a knob responds to a specific opportunity_tag."""

    direction: Literal["increase", "decrease", "set"]
    step: Any = None  # for increase/decrease
    set_to: Any = None  # for direction="set"
    min_severity: float = 0.5
    min_persistence: int = 2
    cooldown_windows: int = 3
    apply_when: str = "epoch_boundary"
    skip_motifs: List[str] = dc.field(default_factory=lambda: ["warmup_transient"])


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
    severity: str
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


@dc.dataclass
class KnobRegistration:
    """Wire format: an app registering its knobs with the optimizer."""

    namespace: str
    function_name: str
    knobs: Dict[str, dict]  # param_name -> serialized KnobDef + responds_to
