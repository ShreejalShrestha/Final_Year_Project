from dataclasses import dataclass, field


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
