from typing import Any, Dict, List, Optional

from ..types import KnobDef, KnobResponse


def knob(
    default: Any,
    type: type = int,
    range: Optional[tuple] = None,
    values: Optional[List[Any]] = None,
    scope: str = "job",
    responds_to: Optional[Dict[str, dict]] = None,
) -> dict:
    """Helper to declare a knob in @tunable.

    Usage::

        knob(default=2, range=(1, 16), type=int,
             responds_to={
                 "dataloader_prefetch": {"direction": "increase", "step": 2},
             })

    ``responds_to`` maps opportunity_tags to adjustment specs.
    Unspecified fields get sensible defaults from KnobResponse.
    """
    return {
        "default": default,
        "type": type,
        "range": range,
        "values": values,
        "scope": scope,
        "responds_to": responds_to or {},
    }


def knob_def_from_dict(knob_id: str, d: dict, target_function: str = "") -> KnobDef:
    """Build a KnobDef from the dict returned by knob()."""
    responds_to = {}
    for tag, spec in d.get("responds_to", {}).items():
        responds_to[tag] = KnobResponse(
            direction=spec.get("direction", "increase"),
            step=spec.get("step"),
            set_to=spec.get("set_to"),
            min_severity=spec.get("min_severity", 0.5),
            min_persistence=spec.get("min_persistence", 2),
            cooldown_windows=spec.get("cooldown_windows", 3),
            apply_when=spec.get("apply_when", "epoch_boundary"),
            skip_motifs=spec.get("skip_motifs", ["warmup_transient"]),
        )

    return KnobDef(
        id=knob_id,
        default=d["default"],
        type=d.get("type", type(d["default"])),
        range=d.get("range"),
        values=d.get("values"),
        scope=d.get("scope", "job"),
        responds_to=responds_to,
        target_function=target_function,
    )


def knob_def_to_wire(kdef: KnobDef) -> dict:
    """Serialize a KnobDef for publishing to the optimizer registry topic."""
    responds_wire = {}
    for tag, resp in kdef.responds_to.items():
        responds_wire[tag] = {
            "direction": resp.direction,
            "step": resp.step,
            "set_to": resp.set_to,
            "min_severity": resp.min_severity,
            "min_persistence": resp.min_persistence,
            "cooldown_windows": resp.cooldown_windows,
            "apply_when": resp.apply_when,
            "skip_motifs": resp.skip_motifs,
        }

    return {
        "id": kdef.id,
        "default": kdef.default,
        "type": kdef.type.__name__,
        "range": list(kdef.range) if kdef.range else None,
        "values": kdef.values,
        "scope": kdef.scope,
        "responds_to": responds_wire,
        "target_function": kdef.target_function,
    }


def knob_def_from_wire(d: dict) -> KnobDef:
    """Deserialize a KnobDef from the wire format."""
    type_map = {"int": int, "float": float, "str": str, "bool": bool}
    ktype = type_map.get(d.get("type", "int"), int)

    responds_to = {}
    for tag, spec in d.get("responds_to", {}).items():
        responds_to[tag] = KnobResponse(
            direction=spec.get("direction", "increase"),
            step=spec.get("step"),
            set_to=spec.get("set_to"),
            min_severity=spec.get("min_severity", 0.5),
            min_persistence=spec.get("min_persistence", 2),
            cooldown_windows=spec.get("cooldown_windows", 3),
            apply_when=spec.get("apply_when", "epoch_boundary"),
            skip_motifs=spec.get("skip_motifs", ["warmup_transient"]),
        )

    return KnobDef(
        id=d["id"],
        default=d["default"],
        type=ktype,
        range=tuple(d["range"]) if d.get("range") else None,
        values=d.get("values"),
        scope=d.get("scope", "job"),
        responds_to=responds_to,
        target_function=d.get("target_function", ""),
    )
