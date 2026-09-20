"""Temporal state machines that convert per-frame signals into observable
events. A condition must persist past its duration threshold before it is
reported as an event (proposal: "A condition must persist for a configured
duration before it becomes an event.").
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from .config import PipelineConfig

# Event type keys (must match apps.monitoring.models.EventType values).
LOOKING_AWAY = "looking_away"
HEAD_DOWN = "head_down"
HIGH_MOVEMENT = "high_movement"
FACE_NOT_VISIBLE = "face_not_visible"
POSSIBLE_DROWSINESS = "possible_drowsiness"

LOOKING_FORWARD = "looking_forward"       # display-only, never persisted
UNCERTAIN = "uncertain"
EYES_CLOSED = "eyes_closed"               # short closure, not yet a prolonged event


@dataclass
class FrameSignals:
    face_visible: bool
    yaw: float = 0.0
    pitch: float = 0.0
    blink: float = 0.0
    movement_norm: float = 0.0
    quality: float = 0.0
    pose_valid: bool = False
    eye_left: float | None = None
    eye_right: float | None = None


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
    last_holds: bool = False
    observed_seconds: float = 0.0
    release_since: float | None = None


@dataclass
class TrackIndicatorState:
    current_label: str = UNCERTAIN
    current_is_event: bool = False
    conditions: dict[str, _CondState] = field(default_factory=dict)
    last_update: float | None = None


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
    def _is_face_not_visible(self, s: FrameSignals, active=False) -> bool:
        return not s.face_visible

    def _eyes_valid(self, s):
        return (s.face_visible and s.quality >= self.cfg.indicator_min_quality
                and all(v is not None and math.isfinite(v) and 0 <= v <= 1
                        for v in (s.eye_left, s.eye_right)))

    def _pose_valid(self, s):
        return (s.face_visible and s.pose_valid and s.quality >= self.cfg.indicator_min_quality
                and math.isfinite(s.yaw) and math.isfinite(s.pitch))

    def _is_drowsy(self, s: FrameSignals, active=False) -> bool | None:
        if not self._eyes_valid(s):
            return None
        threshold = self.cfg.eye_closed_exit_score if active else self.cfg.eye_closed_blink_score
        return min(s.eye_left, s.eye_right) >= threshold

    def _is_head_down(self, s: FrameSignals, active=False) -> bool | None:
        if not self._pose_valid(s):
            return None
        threshold = self.cfg.head_down_exit_deg if active else self.cfg.head_down_pitch_deg
        return self.cfg.pitch_direction * (s.pitch - self.cfg.neutral_pitch_deg) >= threshold

    def _is_looking_away(self, s: FrameSignals, active=False) -> bool | None:
        if not self._pose_valid(s):
            return None
        threshold = self.cfg.looking_away_exit_deg if active else self.cfg.looking_away_yaw_deg
        return abs(s.yaw - self.cfg.neutral_yaw_deg) >= threshold

    def _is_high_movement(self, s: FrameSignals, active=False) -> bool | None:
        if not s.face_visible or not math.isfinite(s.movement_norm) or s.quality < self.cfg.indicator_min_quality:
            return None
        return s.movement_norm >= self.cfg.high_movement_norm

    # --- per-frame update --------------------------------------------
    def update(self, track_id: int, now: float, signals: FrameSignals):
        state = self._tracks.setdefault(track_id, TrackIndicatorState())
        transitions: list[Transition] = []
        active_now: list[str] = []
        dt = 0.0 if state.last_update is None else now - state.last_update
        interrupted = dt < 0 or dt > self.cfg.indicator_max_gap_seconds
        state.last_update = now

        for cond, (test, min_seconds) in self._rules.items():
            cs = state.conditions.setdefault(cond, _CondState())
            if interrupted:
                self._reset(cs, cond, now, transitions)
            # Expire a grace period even when the next observation is positive.
            if cs.release_since is not None and now - cs.release_since >= self.cfg.condition_release_seconds:
                self._reset(cs, cond, now, transitions)
            holds = test(signals, cs.active_since is not None)
            if holds is True:
                active_now.append(cond)
                if cs.active_since is None:
                    cs.active_since = now
                if cs.last_holds and not interrupted:
                    cs.observed_seconds += max(0.0, dt)
                cs.release_since = None
                if (
                    cs.observed_seconds >= min_seconds
                    and not cs.event_open
                    and now >= cs.cooldown_until
                ):
                    cs.event_open = True
                    transitions.append(Transition("open", cond, signals.quality))
            else:
                if cs.active_since is not None:
                    if cs.release_since is None:
                        cs.release_since = now
                    if now - cs.release_since >= self.cfg.condition_release_seconds:
                        self._reset(cs, cond, now, transitions)
                    elif holds is False:
                        active_now.append(cond)
                # Missing measurements never count as evidence or a display label.
            cs.last_holds = holds is True

        # Current display label: highest-priority active condition.
        forward = (self._pose_valid(signals) and self._eyes_valid(signals)
                   and abs(signals.yaw - self.cfg.neutral_yaw_deg) <= self.cfg.forward_yaw_deg
                   and abs(signals.pitch - self.cfg.neutral_pitch_deg) <= self.cfg.forward_pitch_deg
                   and max(signals.eye_left, signals.eye_right) < self.cfg.eye_closed_exit_score)
        label, is_event = (LOOKING_FORWARD if forward else UNCERTAIN), False
        for cond in self.cfg.indicator_priority:
            if cond in active_now:
                label = cond
                is_event = state.conditions[cond].event_open
                if cond == POSSIBLE_DROWSINESS and not is_event:
                    label = EYES_CLOSED
                break
        state.current_label = label
        state.current_is_event = is_event
        return state, transitions

    def _reset(self, cs, condition, now, transitions):
        if cs.event_open:
            transitions.append(Transition("close", condition))
            cs.cooldown_until = now + self.cfg.event_cooldown_seconds
        cs.event_open = False
        cs.active_since = None
        cs.observed_seconds = 0.0
        cs.release_since = None
        cs.last_holds = False

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
