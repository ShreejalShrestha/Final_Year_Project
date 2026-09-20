# Evaluation plan

Mirrors the proposal's "Testing and Evaluation" and Appendix C.

## 1. Face identification
Metrics: accuracy, precision, recall, F1, FAR, FRR.
Method: held-out probe images per enrolled student + impostor set (non-enrolled
faces). Sweep `CV_RECOGNITION_THRESHOLD`; pick the operating point that keeps
FAR low while accuracy ≥ 90% in normal light.

## 2. Attendance correctness
Checklist against observed ground truth per session:
- session must be `active` before any row is written;
- first valid recognition → exactly one `present` row (see
  `apps/attendance/tests.py`);
- timestamp recorded; repeated detections create no duplicates;
- unknown faces create nothing;
- manual correction sets `corrected`, `corrected_by`, `audit_note`.
Target: ≥ 98% correct duplicate-free recording once a student is recognised.

## 3. Tracking
Metrics: track continuity, ID switches during walking / partial occlusion.
Method: annotate short clips with per-person ID; compare to tracker output.

## 4. Observable indicators
Metrics: precision, recall, F1 vs manually labelled reference clips for
looking away / head down / high movement / face not visible / possible
drowsiness. Target F1 ≥ 0.80 for the main indicators.
Vary: lighting (normal / low / backlit), head rotation, distance, crowd size.
Calibrate thresholds in `cv_pipeline/config.py`.

## 5. Performance
≥ 8 FPS end-to-end on the target laptop (`fps` shown on the dashboard and by
`run_pipeline`); profile panel opens < 1.5 s (`student_profile_json`).

## 6. Security & usability
Role checks, session auth, password hashing; a small teacher user test on
whether the dashboard is understood *without* over-reading the indicators.
