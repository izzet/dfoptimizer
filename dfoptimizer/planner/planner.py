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

    def register_knobs(self, knob_defs: Dict[str, KnobDef]):
        """Add knobs from an app registration. Rebuilds the tag index."""
        for knob_id, kdef in knob_defs.items():
            self.knobs[knob_id] = kdef
            self.current_values.setdefault(knob_id, kdef.default)

            for tag, response in kdef.responds_to.items():
                self._responses_by_tag.setdefault(tag, []).append(
                    (knob_id, response)
                )

        logger.info(
            "optimizer.planner.updated",
            knob_count=len(self.knobs),
            tag_count=len(self._responses_by_tag),
        )

    def process_finding(self, finding: DiagnosisFindingMsg) -> List[ActionPlan]:
        """Evaluate a finding and return zero or more ActionPlans."""
        plans = []

        logger.info(
            "optimizer.finding.received",
            finding_type=finding.finding_type,
            motif=finding.motif,
            severity=finding.severity,
            persistence=finding.persistence,
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

        # Gate 2: improving trend — don't fix what's getting better
        if finding.trend_direction == "improving":
            logger.info("optimizer.finding.skipped", finding_type=finding.finding_type, reason="improving_trend")
            return plans

        # Numeric severity from label
        severity_score = SEVERITY_SCORE_MAP.get(finding.severity.lower(), 0)

        for tag in finding.opportunity_tags:
            responses = self._responses_by_tag.get(tag, [])
            for knob_id, response in responses:
                # Gate 3: severity threshold
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
            scale = 0.5 + 0.5 * severity_score  # [0.5, 1.0]
            if finding.trend_direction == "stable":
                scale *= 0.5
            step = response.step * scale
            new_value = old_value + kdef.type(step)
        elif response.direction == "decrease":
            scale = 0.5 + 0.5 * severity_score
            if finding.trend_direction == "stable":
                scale *= 0.5
            step = response.step * scale
            new_value = old_value - kdef.type(step)
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
                f"trend={finding.trend_direction}) -> {tag}"
            ),
            finding_type=finding.finding_type,
            severity=severity_score,
            opportunity_tag=tag,
            window_index=finding.window_index,
        )

        # Update state
        self.current_values[knob_id] = new_value
        self.cooldown.record(knob_id, finding.window_index)

        logger.info(
            "optimizer.plan.created",
            plan_id=plan.plan_id,
            knob_id=plan.knob_id,
            old_value=old_value,
            new_value=new_value,
            rationale=plan.rationale,
        )
        return plan

    def apply_ack(self, plan_id: str, knob_id: str, status: str,
                  old_value=None, new_value=None):
        """Process an ACK from the app."""
        logger.info(
            "optimizer.ack.received",
            plan_id=plan_id,
            knob_id=knob_id,
            status=status,
            old_value=old_value,
            new_value=new_value,
        )

        if status == "rejected":
            kdef = self.knobs.get(knob_id)
            if kdef:
                self.current_values[knob_id] = kdef.default
            logger.warning(
                "optimizer.ack.rejected",
                plan_id=plan_id,
                knob_id=knob_id,
            )
