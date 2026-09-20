# Weekly Meeting Minutes — Week 1

**Project:** AI-Based Smart Classroom Monitoring Using Facial Recognition and Sentiment Analysis
**Programme:** BSc. IT — Final Year Project, Presidential Graduate School
**Supervisor:** Bibek Khanal
**Group members:** Uttam Chaudhary, Shreejal Shrestha, Udhav Kharel, Amit Mukhiya

| | |
|---|---|
| **Meeting no.** | 1 |
| **Date** | «2026-08-26» |
| **Time** | «2:00 PM – 2:40 PM» |
| **Venue / mode** | «College — Supervisor's office / Online» |
| **Present** | Bibek Khanal (Supervisor); Uttam Chaudhary, Shreejal Shrestha, Udhav Kharel, Amit Mukhiya |
| **Absent** | «None» |
| **Minute taker** | «Shreejal Shrestha» |

---

## 1. Agenda

1. Kick-off after proposal approval; confirm project scope and boundaries.
2. Agree the Week 1–2 focus (literature review, requirements, consent/privacy preparation).
3. Set up the development environment and code repository.
4. Plan the consent-based student enrollment and controlled reference-clip collection.
5. Assign responsibilities and fix the weekly meeting slot.

## 2. Discussion / points noted

- **Scope confirmed.** The system records attendance from the first confident
  face recognition (once per student per session), tracks recognised students in
  the live feed, and reports **observable indicators only** (looking away,
  head-down, high movement, face not visible, possible drowsiness) with
  duration/confidence. The supervisor stressed that the report and UI must not
  claim to detect emotion, engagement, boredom or ability, and that a human must
  review every result.
- **Prototype boundary.** One classroom, one camera, ~20–30 consenting student
  volunteers. Processing runs locally; continuous raw video is not stored.
- **Literature review.** Core references identified and being read: ArcFace
  (Deng et al., 2019), MTCNN (Zhang et al., 2016), ByteTrack (Zhang et al., 2022),
  MediaPipe Face/Pose Landmarker, Barrett et al. (2019) on the limits of
  inferring emotion from facial movement, and NIST FRVT demographic effects
  (Grother et al., 2019).
- **Requirements.** Draft functional list started from Appendix A of the
  proposal (user management, enrollment, attendance session, live monitoring,
  indicators, profile interaction, privacy controls). To be finalised next week.
- **Consent & privacy.** Nepal's Privacy Act, 2075 applies (facial images and
  templates are personal data). A written consent form, retention window and a
  withdrawal/deletion procedure must be prepared before any photos are collected.
  The supervisor asked to see the consent form draft at the next meeting.
- **Development environment.** Repository initialised with Git. Django 5 project
  scaffolded with apps for accounts, students, classes, attendance, monitoring
  and dashboard, plus a `cv_pipeline` package (detection → tracking → recognition
  → landmarks → indicators). SQLite for development, MySQL for later deployment.
- **Hardware / acceleration.** InsightFace (ArcFace / SCRFD, `buffalo_l`) and
  MediaPipe installed. NVIDIA GPU (RTX 4060) acceleration configured and
  verified working via ONNX Runtime CUDA.
- **Datasets.** Only a consent-based primary dataset will be used for enrollment
  and short controlled reference clips (looking forward, looking away, head down,
  normal/high movement, temporary face loss, prolonged eye closure). Public
  benchmark data only where licensing permits, and only for background tests.

## 3. Decisions

| # | Decision |
|---|---|
| D1 | Weekly supervisor meeting fixed for «Tuesdays, 2:00 PM». |
| D2 | Weeks 1–2 focus: literature review, requirements finalisation, consent/privacy documents, and preparation for enrollment. |
| D3 | Consent form + participant information sheet to be drafted and reviewed by the supervisor before any data collection. |
| D4 | Code, datasets, annotations and experiment notes kept in Git / documented separately; no raw classroom video committed. |
| D5 | Terminology rule: UI and report use descriptive indicator names only — never "bored", "distracted", "disengaged". |

## 4. Action items

| # | Task | Owner | Due |
|---|---|---|---|
| A1 | Finalise the functional and non-functional requirements document | Uttam Chaudhary | Next meeting |
| A2 | Draft the consent form + participant information sheet (purpose, data stored, retention, withdrawal) | Shreejal Shrestha | Next meeting |
| A3 | Complete the literature review write-up and citation list | Udhav Kharel | Next meeting |
| A4 | Prepare the enrollment plan: photo capture protocol (pose/lighting variation), folder structure, target 20–30 volunteers | Amit Mukhiya | Next meeting |
| A5 | Stand up the base Django app + data model (Student, Class/Session, Attendance, MonitoringEvent) and confirm it runs | Shreejal Shrestha | Next meeting |
| A6 | Confirm classroom, camera and demo machine availability | Uttam Chaudhary | Next meeting |

## 5. Next meeting

- **Date/time:** «2026-09-02, 2:00 PM»
- **Expected review:** requirements document, consent form draft, literature
  review draft, enrollment plan, base app demo.

---

_Prepared by «Shreejal Shrestha». Reviewed by Supervisor: ____________________  Date: ___________
