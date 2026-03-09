# tools/calibrate_queue.py

from __future__ import annotations

import os
from typing import Tuple

import cv2
import yaml


def find_project_root() -> Tuple[str, str]:
    """
    Walk upward from this file until we find config.yaml.
    Returns (project_root, config_path).
    """
    here = os.path.abspath(os.path.dirname(__file__))
    cur = here
    while True:
        candidate = os.path.join(cur, "config.yaml")
        if os.path.exists(candidate):
            return cur, candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise RuntimeError("config.yaml not found while walking up from %s" % here)


def load_config(cfg_path: str) -> dict:
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(cfg_path: str, cfg: dict) -> None:
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[INFO] Saved queue_geometry to {cfg_path}")


def main():
    project_root, cfg_path = find_project_root()
    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Using config file: {cfg_path}")

    cfg = load_config(cfg_path)
    qg = cfg.get("queue_geometry", {})

    point_near = qg.get("point_near")
    point_far = qg.get("point_far")
    corridor_width_px = int(qg.get("corridor_width_px", 90))
    line_length_feet = float(qg.get("line_length_feet", 40.0))

    source = 0  # laptop cam; change to 1 / 2 / RTSP if needed

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Could not open camera source:", source)
        return

    print("\n=== Slanted Queue Calibration ===")
    print("• Left click = set NEAR point (by cashier)")
    print("• Right click = set FAR point (end of line)")
    print("• Press '+' or '-' to adjust corridor width")
    print("• Press 's' to SAVE to config.yaml")
    print("• Press 'q' to quit without saving\n")

    window_name = "Calibrate Queue Line"
    cv2.namedWindow(window_name)

    clicks = {"near": point_near, "far": point_far}

    def on_mouse(event, x, y, flags, userdata):
        nonlocal clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks["near"] = [float(x), float(y)]
            print(f"[INFO] NEAR point set to {clicks['near']}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            clicks["far"] = [float(x), float(y)]
            print(f"[INFO] FAR point set to {clicks['far']}")

    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from camera.")
            break

        h, w = frame.shape[:2]

        pn = clicks.get("near")
        pf = clicks.get("far")

        # draw slanted line + corridor if both points exist
        if pn is not None and pf is not None:
            ax, ay = int(pn[0]), int(pn[1])
            bx, by = int(pf[0]), int(pf[1])

            cv2.line(frame, (ax, ay), (bx, by), (64, 196, 255), 2)
            cv2.circle(frame, (ax, ay), 6, (64, 196, 255), -1)
            cv2.circle(frame, (bx, by), 6, (64, 196, 255), -1)

            # visualize corridor width with parallel lines
            import math

            vx = bx - ax
            vy = by - ay
            length = math.hypot(vx, vy) or 1.0
            nx = -vy / length
            ny = vx / length
            offset = corridor_width_px

            ax1 = int(ax + nx * offset)
            ay1 = int(ay + ny * offset)
            bx1 = int(bx + nx * offset)
            by1 = int(by + ny * offset)

            ax2 = int(ax - nx * offset)
            ay2 = int(ay - ny * offset)
            bx2 = int(bx - nx * offset)
            by2 = int(by - ny * offset)

            cv2.line(frame, (ax1, ay1), (bx1, by1), (80, 80, 80), 1)
            cv2.line(frame, (ax2, ay2), (bx2, by2), (80, 80, 80), 1)

        info = f"width={corridor_width_px}px | line_length={line_length_feet:.1f} ft"
        cv2.putText(
            frame,
            info,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            print("[INFO] Quit without saving.")
            break
        elif key == ord("s"):
            if clicks.get("near") is None or clicks.get("far") is None:
                print("[WARN] You must click NEAR and FAR points before saving.")
                continue

            if "queue_geometry" not in cfg:
                cfg["queue_geometry"] = {}

            cfg["queue_geometry"]["point_near"] = [
                float(clicks["near"][0]),
                float(clicks["near"][1]),
            ]
            cfg["queue_geometry"]["point_far"] = [
                float(clicks["far"][0]),
                float(clicks["far"][1]),
            ]
            cfg["queue_geometry"]["corridor_width_px"] = int(corridor_width_px)
            cfg["queue_geometry"]["line_length_feet"] = float(line_length_feet)
            cfg["queue_geometry"]["use_slanted_line"] = True

            save_config(cfg_path, cfg)
            print("[INFO] Saved. You can now run run_camera_bt.py in slanted mode.")
        elif key == ord("+") or key == ord("="):
            corridor_width_px += 5
            print(f"[INFO] corridor_width_px = {corridor_width_px}")
        elif key == ord("-") or key == ord("_"):
            corridor_width_px = max(10, corridor_width_px - 5)
            print(f"[INFO] corridor_width_px = {corridor_width_px}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
