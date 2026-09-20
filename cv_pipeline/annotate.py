"""Draw tracker boxes, identity labels and the current observable indicator."""
from __future__ import annotations

_LABELS = {
    "looking_forward": ("forward", (90, 200, 90)),
    "looking_away": ("looking away", (60, 170, 240)),
    "head_down": ("head down", (60, 170, 240)),
    "high_movement": ("high movement", (60, 200, 240)),
    "face_not_visible": ("face not visible", (150, 150, 150)),
    "possible_drowsiness": ("prolonged eye closure", (70, 90, 240)),
    "eyes_closed": ("eyes closed", (150, 150, 150)),
    "uncertain": ("uncertain", (150, 150, 150)),
}


def draw_overlay(frame, track_views):
    import cv2

    for tv in track_views:
        x1, y1, x2, y2 = (int(v) for v in tv["bbox"])
        indicator = tv.get("indicator", "uncertain")
        text, colour = _LABELS.get(indicator, ("", (200, 200, 200)))
        if tv.get("student_name"):
            name = tv["student_name"]
            if tv.get("attendance_status") == "present":
                name += "  ✓"
        else:
            name = "Unknown"
        box_colour = (0, 200, 0) if tv.get("student_name") else (0, 165, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 2)
        cv2.rectangle(frame, (x1, y1 - 22), (x2, y1), box_colour, -1)
        cv2.putText(frame, f"#{tv['track_id']} {name}", (x1 + 3, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        if text:
            marker = "! " if tv.get("indicator_is_event") else ""
            cv2.putText(frame, marker + text, (x1 + 3, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return frame
