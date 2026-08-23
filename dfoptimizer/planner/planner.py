import dataclasses as dc
import uuid
from typing import Any, Dict, List, Optional, Tuple

import structlog

from ..types import ActionPlan, DiagnosisFindingMsg, KnobDef, KnobResponse
from .cooldown import CooldownTracker

logger = structlog.get_logger()


@dc.dataclass
class SuppressionRecord:
    """Active suppression of a specific opportunity tag by a global finding."""

    window_index: int
    severity: float
    finding_type: str


class Planner:
    """Decision engine: consumes DiagnosisFindings, produces ActionPlans.

    Dynamically populated from knob registrations sent by instrumented apps.
    Each KnobDef carries ``responds_to`` — a mapping from opportunity_tags
    to adjustment specs (KnobResponse).  The planner builds its rule index
    from these registrations, so no hardcoded per-app rules are needed.

    4-gate decision pipeline per finding:
    1. Trend filter — skip if severity is improving (previous action is working)
    2. Severity gate — require minimum severity score
    3. Persistence gate — require evidence across N consecutive windows
    4. Cooldown — time-based + effectiveness check since last action

    Global findings that carry ``suppresses_tags`` activate data-driven
    suppression after passing the same 4-gate pipeline.  Per-node findings
    whose tags are suppressed are blocked, unless the node's local severity
    exceeds the global severity (severity-weighted agreement).
    """

    def __init__(self):
        self.knobs: Dict[str, KnobDef] = {}
        self.cooldown = CooldownTracker()

        # Current knob values (global baseline, start at defaults)
        self.current_values: Dict[str, any] = {}

        # Per-node value overrides: (knob_id, node) -> value
        self.node_values: Dict[Tuple[str, str], Any] = {}

        # Index: opportunity_tag -> list of (knob_id, KnobResponse)
        self._responses_by_tag: Dict[str, List[tuple]] = {}

        # Pending plans: (knob_id, node) -> (plan_id, severity_score, plan_window_index)
        # node="" means global (all-node) plan. The window index is carried because acks
        # report window_index=-1, and the cooldown record is gated on a valid window.
        self._pending_plans: Dict[Tuple[str, str], Tuple[str, float, int]] = {}

        # Data-driven suppression: tag -> SuppressionRecord
        # Populated when global findings with suppresses_tags pass 4 gates.
        self._active_suppressions: Dict[str, SuppressionRecord] = {}

        # Latest per-node severity per tag — used for severity-weighted
        # agreement check (local vs global).
        self._latest_node_severity: Dict[Tuple[str, str], float] = {}  # (tag, node) -> severity

    def register_knobs(
        self,
        knob_defs: Dict[str, KnobDef],
        current_values: Optional[Dict[str, object]] = None,
    ):
        """Add knobs from an app registration. Rebuilds the tag index."""
        current_values = current_values or {}
        for knob_id, kdef in knob_defs.items():
            self.knobs[knob_id] = kdef
            if knob_id in current_values:
                self.current_values[knob_id] = current_values[knob_id]
            else:
                self.current_values.setdefault(knob_id, kdef.default)

            for tag, response in kdef.responds_to.items():
                self._responses_by_tag.setdefault(tag, []).append(
                    (knob_id, response)
                )

        logger.info(
            "optimizer.planner.updated",
            knob_count=len(self.knobs),
            tag_count=len(self._responses_by_tag),
            current_values=dict(self.current_values),
        )

    # ── Global plan acceptance ──

    def apply_global_plan(self, plan: ActionPlan) -> Optional[ActionPlan]:
        """Accept a plan from the global optimizer.

        Validates the knob is registered and the value is within range.
        Global plans have higher priority than local plans — they update
        current_values immediately and start cooldown to prevent the
        local planner from overriding them.

        Returns the plan (possibly with updated old_value) if accepted,
        or None if rejected.
        """
        knob_id = plan.knob_id
        kdef = self.knobs.get(knob_id)
        if kdef is None:
            logger.warning(
                "planner.global_plan.rejected",
                knob_id=knob_id,
                reason="knob_not_registered",
            )
            return None

        # Validate value is within knob range
        new_value = kdef.clamp(plan.new_value)

        old_value = self.current_values.get(knob_id, kdef.default)
        if new_value == old_value:
            logger.debug(
                "planner.global_plan.no_change",
                knob_id=knob_id,
                value=old_value,
            )
            return None

        # Apply: update current value
        self.current_values[knob_id] = new_value

        # Start cooldown so local planner doesn't immediately override
        self.cooldown.record(
            knob_id=knob_id,
            window_index=plan.window_index,
            severity_score=plan.severity,
        )

        accepted_plan = dc.replace(
            plan,
            old_value=old_value,
            new_value=new_value,
        )

        logger.info(
            "planner.global_plan.accepted",
            knob_id=knob_id,
            old_value=old_value,
            new_value=new_value,
            target_nodes=plan.target_nodes,
            window_index=plan.window_index,
            rationale=plan.rationale,
        )
        return accepted_plan

    # ── Suppression management ──

    def _update_suppressions(
        self, finding: DiagnosisFindingMsg, suppresses_tags: List[str],
    ):
        """Activate or refresh suppressions from a global finding."""
        for tag in suppresses_tags:
            prev = self._active_suppressions.get(tag)
            self._active_suppressions[tag] = SuppressionRecord(
                window_index=finding.window_index,
                severity=finding.severity_score,
                finding_type=finding.finding_type,
            )
            if prev is None:
                logger.info(
                    "optimizer.suppression.activated",
                    tag=tag,
                    finding_type=finding.finding_type,
                    severity=round(finding.severity_score, 3),
                    window=finding.window_index,
                )

    def _clear_suppressions_for(self, finding_type: str):
        """Remove suppressions that were set by *finding_type*.

        Called when a global finding without suppresses_tags arrives,
        indicating the condition that triggered suppression has ended.
        """
        cleared = []
        for tag, rec in list(self._active_suppressions.items()):
            if rec.finding_type == finding_type:
                del self._active_suppressions[tag]
                cleared.append(tag)
        if cleared:
            logger.info(
                "optimizer.suppression.cleared",
                finding_type=finding_type,
                cleared_tags=cleared,
            )

    def process_finding(self, finding: DiagnosisFindingMsg) -> List[ActionPlan]:
        """Evaluate a finding through the 4-gate pipeline.

        Returns zero or more ActionPlans for findings that pass all gates.
        """
        plans = []

        logger.debug(
            "planner.finding.detail",
            finding_type=finding.finding_type,
            scope=finding.scope,
            layer=finding.layer,
            motif=finding.motif,
            severity=finding.severity,
            severity_score=round(finding.severity_score, 3),
            persistence=finding.persistence,
            support_windows=finding.support_windows,
            trend_direction=finding.trend_direction,
            opportunity_tags=finding.opportunity_tags,
            window_index=finding.window_index,
            key_metrics=finding.key_metrics if finding.key_metrics else None,
            publish_mode=getattr(finding, "publish_mode", None),
            contributing_facts=getattr(finding, "contributing_facts", None),
        )

        logger.info(
            "optimizer.finding.received",
            finding_type=finding.finding_type,
            scope=finding.scope,
            layer=finding.layer,
            motif=finding.motif,
            severity_score=round(finding.severity_score, 3),
            persistence=finding.persistence,
            support_windows=finding.support_windows,
            trend_direction=finding.trend_direction,
            opportunity_tags=finding.opportunity_tags,
            window_index=finding.window_index,
        )

        if not self.knobs:
            logger.debug("optimizer.finding.skipped", reason="no_knobs_registered")
            return plans

        is_global_scope = finding.scope in ("global", "global:global")

        # ── Gate 1: Trend filter ──
        # If severity is improving, the previous action is working — don't stack.
        if finding.trend_direction == "improving":
            logger.info(
                "optimizer.gate.rejected",
                finding_type=finding.finding_type,
                gate="trend",
                trend=finding.trend_direction,
            )
            return plans

        # Use continuous severity score from the analyzer (no label re-quantization)
        severity_score = finding.severity_score

        # Extract node scope from finding (e.g. "node:tuolumne1022:...")
        node = ""
        if finding.scope and finding.scope.startswith("node:"):
            parts = finding.scope.split(":", 2)
            if len(parts) >= 2:
                node = parts[1]

        # Track latest per-node severity per tag (for agreement check).
        if node:
            for tag in finding.opportunity_tags:
                self._latest_node_severity[(tag, node)] = severity_score

        # ── Global suppression activation ──
        # Global findings with suppresses_tags activate suppression AFTER
        # passing the trend gate above (so improving global findings don't
        # spuriously suppress).  Suppression is data-driven from rule YAML.
        suppresses_tags = getattr(finding, "suppresses_tags", None) or []
        if is_global_scope and suppresses_tags:
            self._update_suppressions(finding, suppresses_tags)

        # Clear suppressions when a global finding's suppresses_tags disappear
        # (the global condition that caused suppression is no longer firing).
        if is_global_scope and not suppresses_tags:
            self._clear_suppressions_for(finding.finding_type)

        for tag in finding.opportunity_tags:
            # Data-driven suppression: check if this tag is actively
            # suppressed by a global finding.
            if tag in self._active_suppressions:
                suppression = self._active_suppressions[tag]
                if not is_global_scope and node:
                    # Hierarchical mode: severity-weighted agreement —
                    # if the node's own severity for this tag exceeds
                    # the global severity, exempt it from suppression.
                    local_sev = self._latest_node_severity.get((tag, node))
                    if local_sev is not None and local_sev > suppression.severity:
                        logger.info(
                            "optimizer.suppression.node_exempted",
                            node=node, tag=tag,
                            local_severity=round(local_sev, 3),
                            global_severity=round(suppression.severity, 3),
                        )
                    else:
                        logger.info(
                            "optimizer.suppression.active",
                            tag=tag,
                            finding_type=finding.finding_type,
                            scope=finding.scope,
                            suppressed_by=suppression.finding_type,
                            suppressed_since_window=suppression.window_index,
                        )
                        continue
                elif finding.severity_score > suppression.severity:
                    # Local mode / global scope: severity-weighted agreement, the SAME rule the
                    # hierarchical branch above applies. The previous behaviour enforced
                    # suppression unconditionally here, justified as "the same-scope finding that
                    # set suppression has a stronger signal". That assumption is false and it was
                    # costing whole runs: in the sec 47 campaign a `comm_dominance` finding with
                    # severity 0.187 suppressed `fetch_pressure` findings with severity 1.000, so
                    # the input domain was locked out from window 0 for the entire run. Those runs
                    # sat ~7x worse than the ones that escaped the race (LIVEFLOW-STATUS.md 48).
                    #
                    # A weaker signal must not veto a stronger one. When the suppressed finding is
                    # itself more severe than the finding that suppressed it, the suppressor is
                    # stale evidence and the tag is exempt.
                    logger.info(
                        "optimizer.suppression.exempted",
                        tag=tag,
                        finding_type=finding.finding_type,
                        scope=finding.scope,
                        severity=round(finding.severity_score, 3),
                        suppressed_by=suppression.finding_type,
                        suppressor_severity=round(suppression.severity, 3),
                    )
                else:
                    logger.info(
                        "optimizer.suppression.active",
                        tag=tag,
                        finding_type=finding.finding_type,
                        scope=finding.scope,
                        suppressed_by=suppression.finding_type,
                        suppressed_since_window=suppression.window_index,
                    )
                    continue

            responses = self._responses_by_tag.get(tag, [])
            logger.debug(
                "planner.tag.resolve",
                tag=tag,
                matched_knobs=[knob_id for knob_id, _ in responses],
            )
            for knob_id, response in responses:
                # Pending plan safety check (not a gate — prevents duplicates)
                pending_key = (knob_id, node)
                if pending_key in self._pending_plans:
                    # Index rather than unpack. The sec 83.5 cooldown fix widened this entry to
                    # (plan_id, severity_score, window_index) at the write site (line ~497) but
                    # left this read unpacking two values, so the FIRST time a second finding
                    # arrived for a knob that already had a pending plan, process_finding raised
                    # `ValueError: too many values to unpack (expected 2)` and the optimizer
                    # dropped the event. That is precisely the duplicate-suppression path the
                    # cooldown and effectiveness gates sit behind, so sec 83.5's fix could never
                    # take effect past the first plan per knob.
                    pending_plan_id = self._pending_plans[pending_key][0]
                    logger.debug(
                        "optimizer.gate.rejected",
                        knob_id=knob_id, tag=tag, gate="pending_plan",
                        pending_plan_id=pending_plan_id,
                        node=node or "global",
                    )
                    continue

                # ── Gate 2: Severity threshold ──
                if severity_score < response.min_severity:
                    logger.debug(
                        "optimizer.gate.rejected",
                        knob_id=knob_id, tag=tag, gate="severity",
                        value=round(severity_score, 3),
                        threshold=response.min_severity,
                    )
                    continue

                # ── Gate 3: Persistence threshold ──
                if finding.persistence < response.min_persistence:
                    logger.debug(
                        "optimizer.gate.rejected",
                        knob_id=knob_id, tag=tag, gate="persistence",
                        value=finding.persistence,
                        threshold=response.min_persistence,
                    )
                    continue

                # ── Gate 4: Cooldown (time-based + effectiveness) ──
                cooldown_result = self.cooldown.check(
                    knob_id,
                    finding.window_index,
                    response.cooldown_windows,
                    current_severity=severity_score,
                    effectiveness_threshold=response.effectiveness_threshold,
                    node=node,
                )
                if cooldown_result.blocked:
                    logger.debug(
                        "optimizer.gate.rejected",
                        knob_id=knob_id, tag=tag, gate="cooldown",
                        reason=cooldown_result.reason,
                        window_index=finding.window_index,
                    )
                    continue

                # Compute new value
                plan = self._make_plan(knob_id, response, finding, severity_score, tag, node=node)
                if plan is not None:
                    plans.append(plan)

        return plans

    def _make_plan(
        self, knob_id: str, response: KnobResponse,
        finding: DiagnosisFindingMsg, severity_score: float, tag: str,
        node: str = "",
    ) -> Optional[ActionPlan]:
        kdef = self.knobs.get(knob_id)
        if kdef is None:
            return None

        # Per-node value lookup: check node-specific override first,
        # then fall back to global current value.
        if node:
            old_value = self.node_values.get(
                (knob_id, node),
                self.current_values.get(knob_id, kdef.default),
            )
        else:
            old_value = self.current_values.get(knob_id, kdef.default)

        step_mode = getattr(response, "step_mode", "add")

        if response.direction == "set":
            new_value = response.set_to
        elif step_mode == "evidence":
            new_value = self._compute_from_evidence(
                old_value, kdef, finding, response.direction
            )
            if new_value is None:
                # Fallback to doubling if evidence metrics unavailable
                new_value = max(old_value * 2, old_value + 1)
        elif response.direction == "increase":
            if step_mode == "multiply":
                factor = response.step  # e.g., 2 means double
                new_value = max(old_value * factor, old_value + 1)
            else:
                delta = self._scaled_delta(kdef, response.step, severity_score, finding)
                new_value = old_value + delta
        elif response.direction == "decrease":
            if step_mode == "multiply":
                factor = response.step
                new_value = old_value // factor
            else:
                delta = self._scaled_delta(kdef, response.step, severity_score, finding)
                new_value = old_value - delta
        else:
            return None

        new_value = kdef.clamp(new_value)

        # Don't emit a plan if value wouldn't change
        if new_value == old_value:
            return None

        logger.debug(
            "planner.action.generate",
            knob_id=knob_id,
            old_value=old_value,
            new_value=new_value,
            direction=response.direction,
            reason=f"{finding.finding_type}: {finding.motif} -> {tag}",
        )

        target_nodes = [node] if node else []

        plan = ActionPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            knob_id=knob_id,
            target_function=kdef.target_function,
            old_value=old_value,
            new_value=new_value,
            apply_when=response.apply_when,
            rationale=(
                f"{finding.finding_type}: {finding.motif} "
                f"(severity={severity_score:.3f}, persistence={finding.persistence}, "
                f"trend={finding.trend_direction}, scope={finding.scope}) -> {tag}"
            ),
            finding_type=finding.finding_type,
            severity=severity_score,
            opportunity_tag=tag,
            window_index=finding.window_index,
            target_nodes=target_nodes,
        )

        # Update state: mark as pending until app acks
        if node:
            self.node_values[(knob_id, node)] = new_value
        else:
            self.current_values[knob_id] = new_value
        # Carry the PLAN-TIME window index too. The ack is supposed to supply the window the
        # plan was applied in, but every observed ack reports window_index=-1, and the record
        # below is gated on `window_index >= 0` -- so cooldown.record() was never called,
        # `_last_action` stayed empty, and BOTH the cooldown timer and the effectiveness veto
        # silently no-opped for the entire life of the planner (LIVEFLOW-STATUS sec 83).
        self._pending_plans[(knob_id, node)] = (plan.plan_id, severity_score,
                                                finding.window_index)

        logger.info(
            "optimizer.plan.created",
            plan_id=plan.plan_id,
            knob_id=plan.knob_id,
            old_value=old_value,
            new_value=new_value,
            rationale=plan.rationale,
        )
        return plan

    @staticmethod
    def _scaled_delta(
        kdef: KnobDef,
        base_step,
        severity_score: float,
        finding: DiagnosisFindingMsg,
    ):
        scale = 0.5 + 0.5 * severity_score  # [0.5, 1.0]
        if finding.trend_direction == "stable":
            scale *= 0.5

        scaled_step = base_step * scale
        if kdef.type is int:
            if scaled_step <= 0:
                return 0
            if scaled_step < 1:
                return 1
            return int(scaled_step)
        return kdef.type(scaled_step)

    @staticmethod
    def _compute_from_evidence(
        old_value,
        kdef: KnobDef,
        finding: DiagnosisFindingMsg,
        direction: str,
    ):
        """Compute a target value from the finding's evidence metrics.

        Uses the I/O-to-compute ratio (Amdahl's Law) to estimate how much
        parallelism is needed to fully overlap I/O with compute.  Returns
        None if the required metrics are not available.
        """
        km = getattr(finding, "key_metrics", None) or {}

        # Try rule-derived names first (fetch_frac / compute_frac from
        # fetch_pressure rules), then fall back to flat-view cross-layer
        # metric names.  Use `is None` to avoid treating 0.0 as missing.
        fetch_frac = km.get("fetch_frac")
        if fetch_frac is None:
            fetch_frac = km.get("fetch_iter_time_frac_parent")
        compute_frac = km.get("compute_frac")
        if compute_frac is None:
            compute_frac = km.get("compute_time_frac_parent")

        if fetch_frac is None:
            return None

        fetch_frac = float(fetch_frac)
        # compute_frac may be None if the column was NaN in the flat_view.
        # Derive it from fetch_frac: fetch + compute ≈ 1 (ignoring overhead).
        if compute_frac is not None:
            compute_frac = float(compute_frac)
        else:
            compute_frac = 1.0 - fetch_frac

        if fetch_frac <= 0.05:
            # Already compute-bound — no increase needed
            return old_value

        compute_frac = max(compute_frac, 0.01)  # avoid division by zero
        # Amdahl's Law: to fully overlap I/O with compute, need
        # fetch_time / compute_time workers.  Since fetch_frac and
        # compute_frac are fractions of the same total, the ratio
        # gives the absolute target parallelism — no multiplication
        # by old_value (which compounds when the ratio is stale due
        # to pipeline latency).
        amdahl_target = max(int(fetch_frac / compute_frac), 1)

        if direction == "increase":
            if amdahl_target <= old_value:
                # Amdahl formula saturated, but if fetch_frac is still
                # significant, the model underestimates the real need
                # (e.g. with gradient accumulation, fast no-sync steps
                # can't be prefetched in time).  Probe by stepping +1.
                if fetch_frac > 0.20:
                    target = old_value + 1
                    logger.info(
                        "planner.evidence_target.probe",
                        old_value=old_value,
                        amdahl_target=amdahl_target,
                        probe_target=target,
                        fetch_frac=round(fetch_frac, 3),
                        compute_frac=round(compute_frac, 3),
                    )
                else:
                    logger.info(
                        "planner.evidence_target.saturated",
                        old_value=old_value,
                        amdahl_target=amdahl_target,
                        fetch_frac=round(fetch_frac, 3),
                        compute_frac=round(compute_frac, 3),
                    )
                    return old_value
            else:
                target = amdahl_target
        elif direction == "decrease":
            if compute_frac > fetch_frac:
                target = max(amdahl_target, 1)
            else:
                return old_value  # still I/O bound, don't decrease
        else:
            return None

        logger.info(
            "planner.evidence_target",
            old_value=old_value,
            target=target,
            fetch_frac=round(fetch_frac, 3),
            compute_frac=round(compute_frac, 3),
        )
        return target

    def apply_ack(self, plan_id: str, knob_id: str, status: str,
                  old_value=None, new_value=None, window_index: int = -1,
                  target_nodes: Optional[List[str]] = None):
        """Process an ACK from the app.

        Clears the pending-plan gate for this knob and starts cooldown
        from the APPLICATION window (not the publish window).  The severity
        recorded at plan-creation time is used for effectiveness tracking.
        """
        # Determine node scope from the plan's target_nodes.
        node = target_nodes[0] if target_nodes else ""

        logger.info(
            "optimizer.ack.received",
            plan_id=plan_id,
            knob_id=knob_id,
            status=status,
            old_value=old_value,
            new_value=new_value,
            window_index=window_index,
            node=node or "global",
        )

        # Clear pending state and recover severity recorded at plan time
        pending_key = (knob_id, node)
        pending_entry = self._pending_plans.pop(pending_key, None)
        if pending_entry is not None:
            pending_plan_id, plan_severity, plan_window = pending_entry
            if pending_plan_id != plan_id:
                logger.warning(
                    "optimizer.ack.plan_id_mismatch",
                    expected=pending_plan_id,
                    received=plan_id,
                    knob_id=knob_id,
                )
        else:
            # No pending entry. This is normally a DUPLICATE ack: under CTX_SCOPE=all
            # every rank acks the same plan, so only the first ack finds the entry and
            # the rest land here. It can also be an ack for a plan this instance did not
            # create.
            #
            # There is no severity to recover, and fabricating 0.0 would be worse than
            # useless: cooldown.record() stores it as severity_at_action, which
            # CooldownTracker.check() later compares against current severity to decide
            # whether the last action helped. Writing 0.0 over the real plan-time
            # severity makes every subsequent action look ineffective. Leave it unknown
            # and skip the re-record below; the first ack already recorded the truth.
            #
            # (Until 2026-08-21 this line read `plan_severity = severity_score`, an
            # undefined name. It raised NameError on every duplicate ack, which was
            # invisible only because the sec 61 consumer bug meant plans were never
            # created in the first place.)
            plan_severity = None
            plan_window = -1

        if status == "applied":
            # Update current value to what was actually applied
            if new_value is not None:
                if node:
                    self.node_values[(knob_id, node)] = new_value
                else:
                    self.current_values[knob_id] = new_value
            # Start cooldown from APPLICATION point, not publish point.
            # Record the severity at plan time for effectiveness comparison.
            # Prefer the window the app reports it applied in; fall back to the window the
            # plan was CREATED in when the ack does not carry one (it currently never does).
            # Without the fallback this record never happens and the gates are inert.
            _eff_window = window_index if window_index >= 0 else plan_window
            if _eff_window >= 0 and plan_severity is not None:
                self.cooldown.record(
                    knob_id, _eff_window,
                    severity_score=plan_severity,
                    node=node,
                )
                logger.info(
                    "optimizer.cooldown.recorded",
                    knob_id=knob_id, node=node or "global",
                    window_index=_eff_window,
                    severity_at_action=round(float(plan_severity), 3),
                    from_ack=bool(window_index >= 0),
                )
        elif status == "rejected":
            # Revert current value — plan was not applied
            if node:
                self.node_values.pop((knob_id, node), None)
            elif old_value is not None:
                self.current_values[knob_id] = old_value
            else:
                kdef = self.knobs.get(knob_id)
                if kdef:
                    self.current_values[knob_id] = kdef.default
            logger.warning(
                "optimizer.ack.rejected",
                plan_id=plan_id,
                knob_id=knob_id,
                node=node or "global",
            )
