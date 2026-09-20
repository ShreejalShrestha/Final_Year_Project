# Privacy & ethics implementation notes

Maps proposal section "Legal, Social and Ethical Issues" to the code.

| Commitment | Where it lives |
|-----------|----------------|
| Informed consent before enrollment | `Student.consent_given` / `consent_recorded_at`; set when photos are uploaded (`students.services.enroll_student_images`). Recognition gallery (`dashboard.gallery.build_gallery`) only includes `consent_given=True`. |
| Purpose limitation | `Student` stores only id, name, program, semester/section, one profile photo, and face templates. No address/family/demographic fields exist. |
| No continuous raw video retention | The worker holds only the latest annotated JPEG in memory; nothing writes frames to disk. `run_pipeline` likewise. |
| Restricted access | `accounts.permissions` — teachers vs admins; enrollment/correction gated. Passwords hashed (PBKDF2). |
| Uncertainty disclosed | Unknown/low-confidence tracks stay `Unknown` (`recognition.TrackIdentity`); attendance rows carry `recognition_confidence`; indicators carry duration + `quality_score`. |
| No automated discipline | System only records observations. No status is derived beyond present/absent; corrections are always manual and audited (`Attendance.corrected`, `corrected_by`, `audit_note`). |
| No inferred attributes | No age/ethnicity/emotion/personality/ability model is called or stored. `student_profile_json` returns an explicit `excluded_by_default` list shown in the UI. |
| Withdrawal / deletion | Delete the `Student`; `FaceEmbedding` and history cascade. (A soft `is_active=False` also removes them from recognition.) |
| Opt-out path | Manual attendance remains available for every session (`attendance.services.set_attendance_manually`, "Not marked — add manually" table). |

## Before any real classroom trial

* Written consent form per volunteer; log the version/date.
* Retention window agreed with the supervisor; add a scheduled purge.
* Review third-party licences: InsightFace models (non-commercial research),
  MediaPipe (Apache-2.0), ByteTrack paper/algorithm (MIT reference impl).
