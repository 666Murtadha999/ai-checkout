# AI_Checkout/engine/detection_engine.py

import cv2
from collections import deque
from typing import List, Dict, Tuple

from ultralytics import YOLO

DEFAULT_MODEL_PATH = "yolov8x6.pt"
DEFAULT_CONF = 0.3

# People & line heuristics
CHILD_HEIGHT_RATIO = 0.50        # <50% of frame height => child
LINE_Y_MIN_RATIO = 0.35          # center Y between these => "in line"
LINE_Y_MAX_RATIO = 0.95

# Physical estimate
FEET_PER_ADULT = 2.5             # ~2.5 ft per adult
LINE_TOO_LONG_FEET = 20.0        # alert if longer than this

# Smoothing window
WINDOW_SIZE = 3


def _median(values):
    if not values:
        return 0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid - 2])


class DetectionEngine:
    """
    Core detection logic for the AI checkout system.
    This version only focuses on lane health (no theft / under-basket logic).
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, conf_threshold: float = DEFAULT_CONF):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names

        self._customers_window = deque(maxlen=WINDOW_SIZE)
        self._items_window = deque(maxlen=WINDOW_SIZE)

    def process_frame(self, frame) -> Tuple:
        """
        Process a single frame and return:
            annotated_frame, detections, summary

        summary contains:
            - adults_in_line
            - items
            - estimated_wait_seconds
            - line_length_feet
            - alerts (list of strings)
        """
        h, w = frame.shape[:2]

        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)

        detections: List[Dict] = []
        annotated = frame.copy()

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                raw_name = self.class_names.get(cls_id, str(cls_id)).lower()

                bbox_h = max(1.0, y2 - y1)
                height_ratio = bbox_h / float(h)
                cy = (y1 + y2) / 2.0
                cy_ratio = cy / float(h)

                # Is the detection in the vertical "line band"?
                in_line = LINE_Y_MIN_RATIO <= cy_ratio <= LINE_Y_MAX_RATIO

                # Decide role (ADULT / CHILD / ITEM)
                if raw_name == "person":
                    is_child = height_ratio < CHILD_HEIGHT_RATIO
                    if is_child:
                        role = "CHILD"
                        color = (147, 51, 234)   # purple
                    else:
                        role = "ADULT"
                        color = (0, 200, 0)      # green
                else:
                    role = "ITEM"
                    is_child = False
                    in_line = False
                    color = (0, 170, 255)       # orange/blue

                det = {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": raw_name,
                    "role": role,
                    "in_line": in_line,
                    "is_child": is_child,
                }
                detections.append(det)

                # Draw bounding box
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                cv2.rectangle(annotated, pt1, pt2, color, 2)

                # Label text
                label_parts = [role, f"{conf:.2f}"]
                if role in ("ADULT", "CHILD") and in_line:
                    label_parts.append("[LINE]")
                text = " ".join(label_parts)

                cv2.putText(
                    annotated,
                    text,
                    (int(x1), max(0, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # Summarize lane stats (no theft logic)
        adults_in_line, items_count, est_wait_sec, line_feet, base_alerts = self._summarize(
            detections
        )

        summary = {
            "adults_in_line": adults_in_line,
            "items": items_count,
            "estimated_wait_seconds": est_wait_sec,
            "line_length_feet": line_feet,
            "alerts": base_alerts,
        }

        # Overlay summary text
        info_text = (
            f"Adults in line: {adults_in_line} | "
            f"Items: {items_count} | "
            f"Line ≈ {line_feet:.1f} ft | "
            f"ETA ≈ {int(est_wait_sec)} s"
        )
        cv2.putText(
            annotated,
            info_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Visualize line band
        y_min = int(LINE_Y_MIN_RATIO * h)
        y_max = int(LINE_Y_MAX_RATIO * h)
        cv2.rectangle(annotated, (0, y_min), (w, y_max), (30, 64, 128), 1)

        return annotated, detections, summary

    def _summarize(self, detections: List[Dict]):
        """
        Take raw detections and compute:
          - smoothed adults_in_line
          - smoothed items
          - estimated wait (sec)
          - line length (feet)
          - alerts
        """
        adult_in_line = 0
        items = 0

        for d in detections:
            role = d.get("role")
            in_line = d.get("in_line", False)

            if role == "ITEM":
                items += 1
            elif role == "ADULT" and in_line:
                adult_in_line += 1

        # Clamp to reasonable values
        adult_in_line = max(0, min(adult_in_line, 8))
        items = max(0, min(items, 100))

        # Smoothing windows
        self._customers_window.append(adult_in_line)
        self._items_window.append(items)

        sm_customers = int(round(_median(self._customers_window)))
        sm_items = int(round(_median(self._items_window)))

        line_feet = sm_customers * FEET_PER_ADULT
        est_wait = sm_customers * 25 + sm_items * 2

        alerts = []
        if sm_customers >= 3:
            alerts.append("Lane crowded (3+ adult customers in line)")
        if est_wait >= 120:
            alerts.append("Long wait time (2+ minutes)")
        if sm_items >= 20:
            alerts.append("Many items on belt")
        if sm_customers == 0 and sm_items > 0:
            alerts.append("Items on belt with no adult in line")

        if line_feet >= LINE_TOO_LONG_FEET:
            alerts.append(
                f"Line too long (~{line_feet:.1f} ft). Consider opening another lane."
            )

        return sm_customers, sm_items, est_wait, line_feet, alerts

    def run_on_video(
            self,
            source,
            show_window: bool = True,
            on_frame=None,
            window_title: str = "AI Checkout – Detection",
    ):
        """
        Convenience method to test the engine on a camera or video file.

        - source: camera index (0,1,...) or video path
        - show_window: whether to show the annotated video
        - on_frame: optional callback(detections, summary) per frame
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video source: {source}")
            return

        print("[INFO] Starting detection. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video or failed to read frame.")
                break

            annotated, detections, summary = self.process_frame(frame)

            if on_frame is not None:
                try:
                    on_frame(detections, summary)
                except Exception as e:
                    print(f"[WARN] on_frame callback error: {e}")

            print(
                f"Adults in line={summary['adults_in_line']}, "
                f"items={summary['items']}, "
                f"line≈{summary['line_length_feet']:.1f} ft, "
                f"eta≈{summary['estimated_wait_seconds']:.1f}s"
            )

            if show_window:
                cv2.imshow(window_title, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if show_window:
            cv2.destroyAllWindows()
