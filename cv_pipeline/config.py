from dataclasses import dataclass, field
import json
import math
from pathlib import Path


@dataclass
class PipelineConfig:
    """All tunable thresholds live here. Calibrate during Months 3-4
    (proposal: "threshold calibration").
    """

    # --- Device / detection -------------------------------------------------
    device: str = "cpu"                       # "cpu" or "cuda"
    det_size: tuple[int, int] = (480, 480)   # smaller = faster; raise for far/small faces
    min_det_score: float = 0.45              # discard weak face detections
    process_every_n_frames: int = 2          # run the pipeline on 1 of every N frames
    capture_width: int = 640                  # downscale incoming frames to this width
    burn_in_overlay: bool = True              # draw boxes onto the frame (False = the web UI draws them)
    landmark_every_n: int = 3                 # run MediaPipe on 1 of every N processed frames

    # --- Recognition ------------------------------------------------------
    recognition_threshold: float = 0.35     # cosine similarity to accept identity
    vote_window: int = 20                    # frames of identity history per track
    min_votes: int = 6                       # agreeing votes before an identity sticks

    # --- Tracking --------------------------------------------------------
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    track_match_thresh: float = 0.8
    track_buffer: int = 30                    # frames to keep a lost track alive

    # --- Observable indicators (degrees / seconds / normalised units) ----
    looking_away_yaw_deg: float = 28.0
    looking_away_min_seconds: float = 3.0

    head_down_pitch_deg: float = 20.0
    head_down_min_seconds: float = 3.0

    eye_closed_blink_score: float = 0.55     # MediaPipe eyeBlink blendshape
    drowsiness_min_seconds: float = 2.5

    high_movement_norm: float = 0.06         # centroid displacement / frame diag
    high_movement_min_seconds: float = 2.0

    face_not_visible_min_seconds: float = 3.0

    event_cooldown_seconds: float = 4.0      # gap before the same event re-opens

    # Calibrate against consented recordings; these are initial defaults.
    neutral_yaw_deg: float = 0.0
    neutral_pitch_deg: float = 0.0
    pitch_direction: float = 1.0            # set -1 if a verified downward nod is negative
    looking_away_exit_deg: float = 23.0
    head_down_exit_deg: float = 15.0
    eye_closed_exit_score: float = 0.40
    forward_yaw_deg: float = 15.0
    forward_pitch_deg: float = 12.0
    indicator_min_quality: float = 0.60     # detector score, not state confidence
    condition_release_seconds: float = 0.35
    landmark_max_age_seconds: float = 0.50
    indicator_max_gap_seconds: float = 1.0

    # --- Assumed capture rate for duration maths when timestamps absent --
    assumed_fps: float = 15.0

    model_dir: str = "cv_pipeline/models"
    # buffalo_s (SCRFD-500M + MobileFaceNet) is ~5x faster on CPU than buffalo_l.
    # Switch to "buffalo_l" for best accuracy once you have a GPU.
    insightface_name: str = "buffalo_s"
    # Only load the models we use — skips 3D/2D landmarks and age/gender.
    insightface_modules: tuple[str, ...] = ("detection", "recognition")

    indicator_priority: list[str] = field(
        default_factory=lambda: [
            "face_not_visible",
            "possible_drowsiness",
            "head_down",
            "looking_away",
            "high_movement",
        ]
    )

    def __post_init__(self):
        for name in CALIBRATION_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if name not in {"neutral_yaw_deg", "neutral_pitch_deg", "pitch_direction"} and value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.pitch_direction not in (-1, 1):
            raise ValueError("pitch_direction must be 1 or -1")
        for low, high in (("looking_away_exit_deg", "looking_away_yaw_deg"),
                          ("head_down_exit_deg", "head_down_pitch_deg"),
                          ("eye_closed_exit_score", "eye_closed_blink_score")):
            if getattr(self, low) >= getattr(self, high):
                raise ValueError(f"{low} must be below {high}")
        for name in ("eye_closed_blink_score", "eye_closed_exit_score", "indicator_min_quality"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.forward_yaw_deg >= self.looking_away_yaw_deg or self.forward_pitch_deg >= self.head_down_pitch_deg:
            raise ValueError("Forward limits must be below away/down entry thresholds")
        if self.landmark_max_age_seconds <= 0 or self.indicator_max_gap_seconds <= 0:
            raise ValueError("Freshness and observation gap limits must be positive")


CALIBRATION_FIELDS = {
    "neutral_yaw_deg", "neutral_pitch_deg", "pitch_direction",
    "looking_away_yaw_deg", "looking_away_exit_deg", "looking_away_min_seconds",
    "head_down_pitch_deg", "head_down_exit_deg", "head_down_min_seconds",
    "eye_closed_blink_score", "eye_closed_exit_score", "drowsiness_min_seconds",
    "forward_yaw_deg", "forward_pitch_deg", "indicator_min_quality",
    "condition_release_seconds", "landmark_max_age_seconds", "indicator_max_gap_seconds",
    "face_not_visible_min_seconds", "high_movement_norm", "high_movement_min_seconds",
    "event_cooldown_seconds",
}


def load_calibration(path: str) -> dict:
    """Read numeric indicator overrides; reject typos instead of silently ignoring them."""
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Calibration must be a JSON object")
    unknown = values.keys() - CALIBRATION_FIELDS
    if unknown:
        raise ValueError(f"Unknown calibration fields: {', '.join(sorted(unknown))}")
    PipelineConfig(**values)
    return values
