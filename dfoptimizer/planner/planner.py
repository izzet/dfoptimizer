import uuid
from typing import Dict, List, Optional

import structlog

from ..types import ActionPlan, DiagnosisFindingMsg, KnobDef, KnobResponse
from .cooldown import CooldownTracker

logger = structlog.get_logger()


# Severity label -> numeric score for gating
SEVERITY_SCORE_MAP = {
    "critical": 1.0,
    "high": 0.8,
    "medium": 0.5,
    "low": 0.3,
    "none": 0.0,
    "unknown": 0.0,
}


class Planner:
    """Decision engine: consumes DiagnosisFindings, produces ActionPlans.

    Dynamically populated from knob registrations sent by instrumented apps.
    Each KnobDef carries ``responds_to`` — a mapping from opportunity_tags
    to adjustment specs (KnobResponse).  The planner builds its rule index
    from these registrations, so no hardcoded per-app rules are needed.

    Logic per finding:
    1. Skip warmup_transient motif
    2. Skip improving trend
    3. For each opportunity_tag in the finding, find registered knob responses
    4. Gate on severity, persistence, motif exclusion, cooldown
    5. Compute new knob value (scaled by severity)
    6. Emit ActionPlan
    """

    def __init__(self):
        self.knobs: Dict[str, KnobDef] = {}
        self.cooldown = CooldownTracker()

        # Current knob values (start at defaults, updated on plan emission)
        self.current_values: Dict[str, any] = {}

        # Index: opportunity_tag -> list of (knob_id, KnobResponse)
        self._responses_by_tag: Dict[str, List[tuple]] = {}

        # Pending plans: knob_id -> plan_id (published but not yet applied)
        self._pending_plans: Dict[str, str] = {}

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

    def process_finding(self, finding: DiagnosisFindingMsg) -> List[ActionPlan]:
        """Evaluate a finding and return zero or more ActionPlans."""
        plans = []

        logger.info(
            "optimizer.finding.received",
            finding_type=finding.finding_type,
            scope=finding.scope,
            layer=finding.layer,
            motif=finding.motif,
            severity=finding.severity,
            persistence=finding.persistence,
            support_windows=finding.support_windows,
            trend_direction=finding.trend_direction,
            opportunity_tags=finding.opportunity_tags,
            window_index=finding.window_index,
        )

        if not self.knobs:
            logger.debug("optimizer.finding.skipped", reason="no_knobs_registered")
            return plans

        # Gate 1: motif-based skip
        if finding.motif == "warmup_transient":
            logger.info("optimizer.finding.skipped", finding_type=finding.finding_type, reason="warmup_transient")
            return plans

        if finding.publish_mode != "control":
            logger.info(
                "optimizer.finding.skipped",
                finding_type=finding.finding_type,
                scope=finding.scope,
                reason="non_control_publish_mode",
                publish_mode=finding.publish_mode,
            )
            return plans

        # Gate 2: improving trend — don't fix what's getting better
        if finding.trend_direction == "improving":
            logger.info("optimizer.finding.skipped", finding_type=finding.finding_type, reason="improving_trend")
            return plans

        # Numeric severity from label
        severity_score = SEVERITY_SCORE_MAP.get(finding.severity.lower(), 0)

        for tag in finding.opportunity_tags:
            responses = self._responses_by_tag.get(tag, [])
            for knob_id, response in responses:
                # Gate 3: pending plan — don't stack plans before previous is applied
                if knob_id in self._pending_plans:
                    logger.debug(
                        "optimizer.gate.rejected",
                        knob_id=knob_id, tag=tag, gate="pending_plan",
                        pending_plan_id=self._pending_plans[knob_id],
                    )
                    continue

                # Gate 4: severity threshold (was gate 3)
                if severity_score < response.min_severity:
                    logger.debug(
                        "optimizer.gate.rejected",
                        knob_id=knob_id, tag=tag, gate="severity",
                        value=round(severity_score, 2), threshold=response.min_severity,
                    )
                    continue

                # Gate 4: persistence threshold
                if finding.persistence < response.min_persistence:
                    logger.debug(
                        "optimizer.gate.rejected",
                        knob_id=knob_id, tag=tag, gate="persistence",
                        value=finding.persistence, threshold=response.min_persistence,
                    )
                    continue

                # Gate 5: motif exclusion
                if finding.motif in response.skip_motifs:
                    logger.debug(
                        "optimizer.gate.rejected",
                        knob_id=knob_id, tag=tag, gate="motif_excluded",
                        motif=finding.motif,
                    )
                    continue

                # Gate 6: cooldown
                if self.cooldown.in_cooldown(
                    knob_id, finding.window_index, response.cooldown_windows
                ):
                    logger.debug(
                        "optimizer.gate.rejected",
                        knob_id=knob_id, tag=tag, gate="cooldown",
                        window_index=finding.window_index,
                    )
                    continue

                # Compute new value
                plan = self._make_plan(knob_id, response, finding, severity_score, tag)
                if plan is not None:
                    plans.append(plan)

        return plans

    def _make_plan(
        self, knob_id: str, response: KnobResponse,
        finding: DiagnosisFindingMsg, severity_score: float, tag: str,
    ) -> Optional[ActionPlan]:
        kdef = self.knobs.get(knob_id)
        if kdef is None:
            return None

        old_value = self.current_values.get(knob_id, kdef.default)

        if response.direction == "set":
            new_value = response.set_to
        elif response.direction == "increase":
            delta = self._scaled_delta(kdef, response.step, severity_score, finding)
            new_value = old_value + delta
        elif response.direction == "decrease":
            delta = self._scaled_delta(kdef, response.step, severity_score, finding)
            new_value = old_value - delta
        else:
            return None

        new_value = kdef.clamp(new_value)

        # Don't emit a plan if value wouldn't change
        if new_value == old_value:
            return None

        plan = ActionPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            knob_id=knob_id,
            target_function=kdef.target_function,
            old_value=old_value,
            new_value=new_value,
            apply_when=response.apply_when,
            rationale=(
                f"{finding.finding_type}: {finding.motif} "
                f"(severity={finding.severity}, persistence={finding.persistence}, "
                f"trend={finding.trend_direction}, scope={finding.scope}) -> {tag}"
            ),
            finding_type=finding.finding_type,
            severity=severity_score,
            opportunity_tag=tag,
            window_index=finding.window_index,
        )

        # Update state: mark as pending until app acks
        self.current_values[knob_id] = new_value
        self._pending_plans[knob_id] = plan.plan_id

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

    def apply_ack(self, plan_id: str, knob_id: str, status: str,
                  old_value=None, new_value=None, window_index: int = -1):
        """Process an ACK from the app.

        Clears the pending-plan gate for this knob and starts cooldown
        from the APPLICATION window (not the publish window).
        """
        logger.info(
            "optimizer.ack.received",
            plan_id=plan_id,
            knob_id=knob_id,
            status=status,
            old_value=old_value,
            new_value=new_value,
            window_index=window_index,
        )

        # Clear pending state regardless of status
        pending_plan = self._pending_plans.pop(knob_id, None)
        if pending_plan and pending_plan != plan_id:
            logger.warning(
                "optimizer.ack.plan_id_mismatch",
                expected=pending_plan,
                received=plan_id,
                knob_id=knob_id,
            )

        if status == "applied":
            # Update current value to what was actually applied
            if new_value is not None:
                self.current_values[knob_id] = new_value
            # Start cooldown from APPLICATION point, not publish point
            if window_index >= 0:
                self.cooldown.record(knob_id, window_index)
        elif status == "rejected":
            # Revert current value — plan was not applied
            if old_value is not None:
                self.current_values[knob_id] = old_value
            else:
                kdef = self.knobs.get(knob_id)
                if kdef:
                    self.current_values[knob_id] = kdef.default
            logger.warning(
                "optimizer.ack.rejected",
                plan_id=plan_id,
                knob_id=knob_id,
            )
