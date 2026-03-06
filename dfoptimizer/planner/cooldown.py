from typing import Dict


class CooldownTracker:
    """Tracks when each knob was last changed to enforce cooldown periods."""

    def __init__(self):
        # knob_id -> window_index when last action was applied
        self._last_action: Dict[str, int] = {}

    def record(self, knob_id: str, window_index: int):
        self._last_action[knob_id] = window_index

    def in_cooldown(self, knob_id: str, current_window: int, cooldown_windows: int) -> bool:
        last = self._last_action.get(knob_id)
        if last is None:
            return False
        return (current_window - last) < cooldown_windows
