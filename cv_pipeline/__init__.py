"""Computer-vision pipeline for the Smart Classroom Monitoring system.

Modules:
    detection   - InsightFace (SCRFD detector + ArcFace embeddings, buffalo_l)
    tracking    - ByteTrack-style multi-object tracker (Kalman + Hungarian)
    recognition - match track embeddings against the enrolled gallery
    landmarks   - MediaPipe Face Landmarker (head pose + blendshapes)
    indicators  - temporal state machines that turn signals into events
    engine      - per-frame orchestration used by the dashboard runtime
    annotate    - draw boxes / labels onto frames

The heavy libraries are imported lazily so that Django management commands and
migrations work without the CV stack installed.
"""
