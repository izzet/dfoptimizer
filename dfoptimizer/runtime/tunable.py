import functools
from typing import Dict

import structlog

from .knob import knob_def_from_dict
from . import context as _ctx_module

logger = structlog.get_logger()


def tunable(knobs: Dict[str, dict]):
    """Decorator that marks a function as tunable by DFOptimizer.

    Usage::

        @tunable(knobs={
            "prefetch_size": knob(default=2, range=(1, 16), type=int,
                                  responds_to={
                                      "dataloader_prefetch": {"direction": "increase", "step": 2},
                                  }),
        })
        def make_dataloader(prefetch_size, **kwargs):
            ...

    When the decorated function is called, pending optimization actions
    are applied by overriding matching keyword arguments.

    The knob definitions (including responds_to) are published to the
    optimizer service so it can build action rules dynamically.
    """

    # We defer building KnobDefs until context + namespace are known.
    _knob_specs = knobs  # raw dicts from knob()
    _registered = False

    def decorator(func):
        func_name = func.__qualname__

        def _build_knob_defs(wrapper, namespace: str):
            if wrapper._tunable_knobs and wrapper._tunable_namespace == namespace:
                return wrapper._tunable_knobs

            knob_defs = {}
            for param_name, spec in _knob_specs.items():
                full_id = f"{namespace}.{param_name}"
                knob_defs[param_name] = knob_def_from_dict(
                    full_id, spec, target_function=func_name
                )

            wrapper._tunable_knobs = knob_defs
            wrapper._tunable_namespace = namespace
            return knob_defs

        def _register_knobs(wrapper, ctx, current_values=None):
            nonlocal _registered

            if _registered:
                return

            knob_defs = _build_knob_defs(wrapper, ctx.namespace)
            ctx.register_knobs(
                func_name,
                knob_defs,
                current_values=current_values or {},
            )
            _registered = True

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Pop _apply_at early — before the ctx check — so non-rank-0
            # processes (where ctx is None) don't pass it to the real function.
            apply_at = kwargs.pop("_apply_at", "epoch_boundary")

            ctx = _ctx_module._global_context
            if ctx is None or ctx._noop:
                return func(*args, **kwargs)

            if not _registered:
                live_values = {
                    param_name: kwargs[param_name]
                    for param_name in _knob_specs
                    if param_name in kwargs
                }
                _register_knobs(wrapper, ctx, current_values=live_values)

            # Drain pending actions for this function
            _reconciling = getattr(ctx, "_reconcile", False)
            namespace = ctx.namespace
            # Reconcile under a broadcast transport, drain under Mofka. See
            # OptimizerContext.desired_actions_for for why draining diverges across ranks.
            actions = ctx.actions_for(func_name, at=apply_at)

            applied = []
            for action in actions:
                # Strip namespace prefix: "dlio.prefetch_size" -> "prefetch_size"
                param_name = action.knob_id
                if param_name.startswith(namespace + "."):
                    param_name = param_name[len(namespace) + 1:]

                if param_name not in wrapper._tunable_knobs:
                    logger.warning(
                        "optimizer.tunable.knob_not_declared",
                        knob_id=action.knob_id,
                        function=func_name,
                    )
                    ctx.ack_action(action, None, None, status="rejected")
                    continue

                if param_name not in kwargs:
                    logger.warning(
                        "optimizer.tunable.knob_not_in_kwargs",
                        knob=param_name,
                        function=func_name,
                    )
                    ctx.ack_action(action, None, None, status="rejected")
                    continue

                kdef = wrapper._tunable_knobs[param_name]
                old_val = kwargs[param_name]
                new_val = kdef.clamp(action.new_value)
                kwargs[param_name] = new_val

                # RE-ASSERT ALWAYS, ACK ONLY ON CHANGE.
                #
                # Under reconciliation the same desired value is seen at every boundary, so the
                # override above has to run every time -- that is what makes a rank that joined
                # late, or restarted, converge on the current value. But acking every time would
                # be wrong twice over: it floods the ack path, and the cooldown/effectiveness
                # gates key off acks (sec 83.5), so a stream of no-op acks would look like
                # repeated successful actions and corrupt the very signal the gates need.
                #
                # An ack means "this knob moved", not "I looked at this knob".
                if old_val != new_val:
                    applied.append((action, old_val, new_val))
                elif _reconciling:
                    continue

                logger.info(
                    "optimizer.tunable.override",
                    function=func_name,
                    knob=param_name,
                    old_value=old_val,
                    new_value=new_val,
                    plan_id=action.plan_id,
                    rationale=action.rationale,
                )

            result = func(*args, **kwargs)

            # Send ACKs, carrying what the actuator actually DID where it can say (4s.33).
            #
            # The ack goes out after `func` returns, so by now the target knows whether a
            # relocation copied, was already in place, or was declined, and when it finished. An
            # object that does not implement the hook is unaffected.
            _target = args[0] if args else None
            _hook = getattr(_target, "_liveflow_ack_reason", None)
            for action, old_val, new_val in applied:
                _reason, _completed = "", None
                if callable(_hook):
                    try:
                        _hooked = _hook(action.knob_id.split(".")[-1])
                        _reason, _completed = (_hooked or (None, None))[:2]
                    except Exception:                      # noqa: BLE001
                        _reason, _completed = "", None
                ctx.ack_action(action, old_val, new_val, status="applied",
                               reason=_reason or "", completed_at=_completed)
                # A CHANGE THAT TAKES EFFECT LATER GETS TO SAY SO LATER (T3-0 4s.37).
                #
                # A relocation's copy ends inside this call; the loader that reads the moved bytes
                # is not built until the next epoch boundary, and every window in between measures
                # the OLD source. One ack cannot carry both instants, so the target is handed the
                # plan and sends its own second ack, with `effective_at`, from wherever the change
                # actually starts being true. A target without the hook is unaffected.
                _defer = getattr(_target, "_liveflow_ack_pending", None)
                if callable(_defer):
                    try:
                        _defer(action, old_val, new_val, ctx)
                    except Exception:                      # noqa: BLE001
                        logger.warning("optimizer.tunable.defer_failed",
                                       knob=action.knob_id, plan_id=action.plan_id)

            return result

        # Attach metadata for introspection (populated on first call)
        wrapper._tunable_knobs = {}
        wrapper._tunable_namespace = None
        wrapper._tunable_func_name = func_name
        wrapper._tunable_register = lambda ctx, current_values=None, _wrapper=wrapper: _register_knobs(
            _wrapper, ctx, current_values=current_values
        )
        return wrapper

    return decorator
