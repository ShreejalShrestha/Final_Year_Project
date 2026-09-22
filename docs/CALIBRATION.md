# Calibrating observable classroom indicators

The pipeline uses pretrained MediaPipe measurements and configurable temporal
rules. These changes do not train a classifier or establish accuracy on your
classroom. Validate with manually labeled recordings before relying on events.

## Updated behavior

- `looking_forward` requires a valid head pose near the calibrated neutral
  angles and low closure scores for both eyes. It describes head orientation,
  not gaze or attention. Intermediate angles, upward poses, missing landmarks,
  and low-quality measurements produce `uncertain` when no other reliable
  condition applies.
- A single closed eye cannot trigger an eye-closure event. Short closures show
  `eyes_closed`; sustained bilateral closure shows **prolonged eye closure**.
  The existing database/API key `possible_drowsiness` and setting
  `drowsiness_min_seconds` remain for compatibility. Existing events are not
  reclassified; historical records were generated with the old rules.
- Separate entry/exit thresholds reduce flicker near a boundary. A short release
  grace period tolerates glitches. Missing intervals pause accumulated evidence;
  they never open events. Sustained missing data or a long processing gap resets
  evidence and closes an open event. Missing measurements show uncertainty
  immediately even while the internal event is waiting for the grace period.
- Cached landmarks expire and are cleared on analysis failure. Current face
  detection loss cannot be hidden by a cached landmark. Detector score is a
  quality gate, not a calibrated probability of the displayed state.
- `head_up` mirrors `head_down`: the same signed pitch past a threshold in the
  opposite direction (chin raised). It has its own entry/exit thresholds
  (`head_up_pitch_deg`, `head_up_exit_deg`) and duration (`head_up_min_seconds`).
- Movement is split into two mutually exclusive conditions: **high movement**
  (`high_movement_norm` or more, sustained) and **restless movement**
  (`restless_movement_norm` up to, but below, `high_movement_norm`, recurring
  at least `restless_min_bursts` times inside `restless_window_seconds`). A
  single big sustained shift is never also counted as restlessness.
- **Prolonged absence** (`left_seat`) uses the same "no face detected" signal
  as `face_not_visible` but held for much longer (`left_seat_min_seconds`,
  default 20s vs. 3s). It only takes over the displayed label once it has
  actually become an event, so a brief look-away still reads as
  `face_not_visible` first.
- Local video files use media timestamps for rule durations; live streams use a
  monotonic clock. Database event start/end timestamps still record processing
  wall time. Use CSV media timestamps for offline duration evaluation.

## Configure a camera

Copy `data/calibration.example.json` to `data/calibration.json`. In `.env`, set:

```dotenv
CV_CALIBRATION_FILE=data/calibration.json
```

Both the dashboard and `run_pipeline` load this configuration. Restart the camera
worker after changing it. Unspecified fields keep their defaults. Invalid names,
nonfinite values and inconsistent entry/exit thresholds are rejected.

Collect a short, explicitly instructed neutral-head sequence with participants
facing the intended reference (for example, the board). Use the median valid yaw
and pitch as `neutral_yaw_deg` and `neutral_pitch_deg`. Verify a downward nod:
if pitch decreases relative to neutral, set `pitch_direction` to `-1`.
Do not automatically treat the first arbitrary pose as neutral.

Offsets currently apply to the whole camera. If seats have substantially different
reference angles, a single baseline may be insufficient; collect evidence by seat
before adding seat-specific calibration. Do not use offsets learned on test clips.

Tune thresholds and durations on a validation set covering normal blinking,
winks, glasses, writing, head turns, occlusion, camera distance and illumination.
Lower thresholds are not automatically more accurate. Check the sampling cadence:
if valid landmark samples are more than 0.5 seconds apart, reduce
`landmark_every_n` in `PipelineConfig` or measure an appropriate freshness limit.
Increasing cache lifetime also increases how long stale observations can persist.

## Export measurements for labeling

For a consented recording and an existing session:

```powershell
python manage.py run_pipeline --session 1 --source data/clips/lecture.mp4 --signals-csv data/lecture-signals.csv
```

The output must be a new file; existing files are not overwritten. This command
also performs the normal session attendance/event writes, so use a dedicated
evaluation session. Export is opt-in and includes no images or face embeddings.

CSV rows contain session and track IDs, measurement timestamps, elapsed time,
video time (blank for live input), raw yaw/pitch, separate eye scores, detector
quality, validity flags, movement, whether landmarks were freshly measured, and
the predicted display state. `label` is deliberately blank. Fill labels from
manual video review, not from `indicator`. `blink` is a compatibility field
containing the minimum of the two eye scores; missing scores must be interpreted
using the separate nullable eye fields. `elapsed_seconds` starts at the first
exported track; use `video_seconds` to align local video annotations.

For supervised learning, summarize short windows of these measurements and train
a small classifier on manually labeled data. Keep recordings and overlapping
windows together in a split; hold out students and sessions for final evaluation.
Track IDs can change after occlusion and are not person identifiers. Maintain a
separate participant/session grouping manifest when making the split. Compare
per-state precision/recall, false events per hour, detection delay and processing
speed against the calibrated rule baseline. A trained classifier and a validated
threshold profile still require your labeled recordings; neither is bundled here.

## Apply and verify

```powershell
python manage.py migrate
python manage.py test cv_pipeline apps.attendance
```

The migration updates the event choice label and preserves existing event keys.
Tests cover signal validity, bilateral eye closure, hysteresis, brief dropouts,
stale caches, camera offsets, event lifecycle and attendance regressions without
loading vision weights or requiring a camera.
