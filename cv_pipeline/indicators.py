"""Temporal state machines that convert per-frame signals into observable
events. A condition must persist past its duration threshold before it is
reported as an event (proposal: "A condition must persist for a configured
duration before it becomes an event.").
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import PipelineConfig

# Event type keys (must match apps.monitoring.models.EventType values).
LOOKING_AWAY = "looking_away"
HEAD_DOWN = "head_down"
HIGH_MOVEMENT = "high_movement"
FACE_NOT_VISIBLE = "face_not_visible"
POSSIBLE_DROWSINESS = "possible_drowsiness"

LOOKING_FORWARD = "looking_forward"       # display-only, never persisted


@dataclass
class FrameSignals:
    face_visible: bool
    yaw: float = 0.0
    pitch: float = 0.0
    blink: float = 0.0
    movement_norm: float = 0.0
    quality: float = 0.0


@dataclass
class Transition:
    kind: str          # "open" or "close"
    condition: str
    quality: float = 0.0


@dataclass
class _CondState:
    active_since: float | None = None
    event_open: bool = False
    cooldown_until: float = 0.0


@dataclass
class TrackIndicatorState:
    current_label: str = LOOKING_FORWARD
    current_is_event: bool = False
    conditions: dict[str, _CondState] = field(default_factory=dict)


class IndicatorEngine:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._tracks: dict[int, TrackIndicatorState] = {}
        self._rules = {
            FACE_NOT_VISIBLE: (self._is_face_not_visible, cfg.face_not_visible_min_seconds),
            POSSIBLE_DROWSINESS: (self._is_drowsy, cfg.drowsiness_min_seconds),
            HEAD_DOWN: (self._is_head_down, cfg.head_down_min_seconds),
            LOOKING_AWAY: (self._is_looking_away, cfg.looking_away_min_seconds),
            HIGH_MOVEMENT: (self._is_high_movement, cfg.high_movement_min_seconds),
        }

    # --- instantaneous condition tests ---------------------------------
    def _is_face_not_visible(self, s: FrameSignals) -> bool:
        return not s.face_visible

    def _is_drowsy(self, s: FrameSignals) -> bool:
        return s.face_visible and s.blink >= self.cfg.eye_closed_blink_score

    def _is_head_down(self, s: FrameSignals) -> bool:
        return s.face_visible and s.pitch >= self.cfg.head_down_pitch_deg

    def _is_looking_away(self, s: FrameSignals) -> bool:
        return s.face_visible and abs(s.yaw) >= self.cfg.looking_away_yaw_deg

    def _is_high_movement(self, s: FrameSignals) -> bool:
        return s.movement_norm >= self.cfg.high_movement_norm

    # --- per-frame update --------------------------------------------
    def update(self, track_id: int, now: float, signals: FrameSignals):
        state = self._tracks.setdefault(track_id, TrackIndicatorState())
        transitions: list[Transition] = []
        active_now: list[str] = []

        for cond, (test, min_seconds) in self._rules.items():
            cs = state.conditions.setdefault(cond, _CondState())
            holds = test(signals)
            if holds:
                active_now.append(cond)
                if cs.active_since is None:
                    cs.active_since = now
                duration = now - cs.active_since
                if (
                    duration >= min_seconds
                    and not cs.event_open
                    and now >= cs.cooldown_until
                ):
                    cs.event_open = True
                    transitions.append(Transition("open", cond, signals.quality))
            else:
                if cs.event_open:
                    cs.event_open = False
                    cs.cooldown_until = now + self.cfg.event_cooldown_seconds
                    transitions.append(Transition("close", cond))
                cs.active_since = None

        # Current display label: highest-priority active condition.
        label, is_event = LOOKING_FORWARD, False
        for cond in self.cfg.indicator_priority:
            if cond in active_now:
                label = cond
                is_event = state.conditions[cond].event_open
                break
        state.current_label = label
        state.current_is_event = is_event
        return state, transitions

    def drop_track(self, track_id: int) -> list[Transition]:
        """Close any open events when a track disappears for good."""
        state = self._tracks.pop(track_id, None)
        if not state:
            return []
        return [
            Transition("close", cond)
            for cond, cs in state.conditions.items()
            if cs.event_open
        ]
