from typing import Any, Dict, List, Optional

from ..types import ActionDef, KnobDef, KnobResponse


def _coerce_cost(c):
    """cost_seconds may be a scalar OR a per-value mapping; preserve whichever was declared.

    Coercing unconditionally with float() raised TypeError on the dict form, which is the form that
    matters: cost belongs to the VALUE asked for, not to the knob.
    """
    if isinstance(c, dict):
        return {k: float(v) for k, v in c.items()}
    return float(c or 0.0)


def knob(
    default: Any,
    type: type = int,
    range: Optional[tuple] = None,
    values: Optional[List[Any]] = None,
    scope: str = "job",
    responds_to: Optional[Dict[str, dict]] = None,
    cost_seconds: float = 0.0,
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
        "cost_seconds": cost_seconds,
    }


def knob_def_from_dict(knob_id: str, d: dict, target_function: str = "") -> KnobDef:
    """Build a KnobDef from the dict returned by knob()."""
    responds_to = {}
    for tag, spec in d.get("responds_to", {}).items():
        responds_to[tag] = KnobResponse(
            direction=spec.get("direction", "increase"),
            step=spec.get("step"),
            step_mode=spec.get("step_mode", "add"),
            set_to=spec.get("set_to"),
            min_severity=spec.get("min_severity", 0.5),
            min_persistence=spec.get("min_persistence", 2),
            cooldown_windows=spec.get("cooldown_windows", 3),
            apply_when=spec.get("apply_when", "epoch_boundary"),
            effectiveness_threshold=spec.get("effectiveness_threshold", 0.10),
        )

    return KnobDef(
        id=knob_id,
        default=d["default"],
        type=d.get("type", type(d["default"])),
        range=d.get("range"),
        values=d.get("values"),
        scope=d.get("scope", "job"),
        cost_seconds=_coerce_cost(d.get("cost_seconds", 0.0)),
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
            "step_mode": resp.step_mode,
            "set_to": resp.set_to,
            "min_severity": resp.min_severity,
            "min_persistence": resp.min_persistence,
            "cooldown_windows": resp.cooldown_windows,
            "apply_when": resp.apply_when,
            "effectiveness_threshold": resp.effectiveness_threshold,
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
        "cost_seconds": kdef.cost_seconds,
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
            step_mode=spec.get("step_mode", "add"),
            set_to=spec.get("set_to"),
            min_severity=spec.get("min_severity", 0.5),
            min_persistence=spec.get("min_persistence", 2),
            cooldown_windows=spec.get("cooldown_windows", 3),
            apply_when=spec.get("apply_when", "epoch_boundary"),
            effectiveness_threshold=spec.get("effectiveness_threshold", 0.10),
        )

    return KnobDef(
        id=d["id"],
        default=d["default"],
        type=ktype,
        range=tuple(d["range"]) if d.get("range") else None,
        values=d.get("values"),
        scope=d.get("scope", "job"),
        cost_seconds=_coerce_cost(d.get("cost_seconds", 0.0)),
        responds_to=responds_to,
        target_function=d.get("target_function", ""),
    )


def action(
    values: List[Any],
    target_domain: str = "storage",
    reversible: bool = True,
    abortable: bool = False,
    scope: str = "node",
    cost_fn=None,
    precondition_fn=None,
    effect_fn=None,
    reconcile_fn=None,
) -> dict:
    """Declare a COSTED actuator, the non-degenerate sibling of knob().

    knob() describes something free and reversible -- set an argument, change your mind next window.
    Every actuator in the DL input-tuner literature is of that kind, which is why none of those
    systems carries a cost model. action() describes the rest:

        staging   465-1365 s   abortable during, not undoable after
        restripe  4.4 h        neither

    `cost_fn` is a FUNCTION because the cost is not stable -- stage-in varied 2.9x for byte-identical
    work, which is the difference between a 1.23- and a 3.61-epoch break-even.
    """
    return {
        "kind": "action",
        "values": values,
        "target_domain": target_domain,
        "reversible": reversible,
        "abortable": abortable,
        "scope": scope,
        "cost_fn": cost_fn,
        "precondition_fn": precondition_fn,
        "effect_fn": effect_fn,
        "reconcile_fn": reconcile_fn,
    }


def action_def_from_dict(action_id: str, d: dict, target_function: str = "") -> ActionDef:
    return ActionDef(
        id=action_id,
        values=d.get("values", []),
        target_domain=d.get("target_domain", "storage"),
        reversible=d.get("reversible", True),
        abortable=d.get("abortable", False),
        scope=d.get("scope", "node"),
        cost_fn=d.get("cost_fn"),
        precondition_fn=d.get("precondition_fn"),
        effect_fn=d.get("effect_fn"),
        reconcile_fn=d.get("reconcile_fn"),
        target_function=target_function,
    )
