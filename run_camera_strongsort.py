# AI_Checkout/run_camera_strongsort.py
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
import yaml
from ultralytics import YOLO
from boxmot import StrongSort

# ------------------------------
# CONSTANTS & GLOBALS
# ------------------------------

DEFAULT_MODEL_PATH = "yolov8x6.pt"
BACKEND_URL = "http://127.0.0.1:5000/update_lane"
LANE_DECIDER_URL = "http://127.0.0.1:3000/decide"

LINE_Y_MIN_RATIO = 0.25
LINE_Y_MAX_RATIO = 0.95
CHILD_HEIGHT_RATIO = 0.38

_cfg: Optional[Dict[str, Any]] = None
_track_states: Dict[int, Dict[str, Any]] = {}
_alias_map: Dict[int, int] = {}
_next_alias = 1


# ------------------------------
# CONFIG LOADING
# ------------------------------

def load_config() -> Dict[str, Any]:
    global _cfg
    if _cfg is not None:
        return _cfg

    here = os.path.abspath(os.path.dirname(__file__))
    cfg_path = os.path.join(here, "config.yaml")

    if not os.path.exists(cfg_path):
        print(f"[WARN] config.yaml not found at {cfg_path}, using defaults.")
        _cfg = {
            "model": {
                "path": DEFAULT_MODEL_PATH,
                "imgsz": 1280,
                "conf": 0.35,
                "iou": 0.45,
                # you can override in config.yaml
                "precision": "fp16",   # fp16 / fp32 / int8 (stub)
            },
            "smoothing": {
                "window_size": 2,
                "update_every_n_frames": 3,
            },
            "modes": {
                "default": "calibration",
                "calibration": {
                    "draw_boxes": True,
                    "draw_labels": True,
                    "draw_line_band": True,
                    "draw_slanted_line": True,
                },
                "production": {
                    "draw_boxes": False,
                    "draw_labels": False,
                    "draw_line_band": True,
                    "draw_slanted_line": True,
                },
            },
            "screenshots": {
                "enabled": True,
                "out_dir": "logs/screenshots",
            },
            "fps_monitor": {
                "enabled": True,
            },
            "queue_geometry": {
                "use_slanted_line": True,
                "line_length_feet": 40.0,
                "corridor_width_px": 80.0,
                "point_near": [640, 700],
                "point_far": [640, 120],
            },
            "tracking": {
                "min_track_age_frames": 3,
            },
            "offline": False,
        }
        return _cfg

    with open(cfg_path, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f) or {}
    return _cfg


# ------------------------------
# HELPERS
# ------------------------------

def ema(prev: Optional[float], value: float, alpha: float) -> float:
    if prev is None:
        return value
    return (1.0 - alpha) * prev + alpha * value


def project_point_onto_segment(
        px: float,
        py: float,
        ax: float,
        ay: float,
        bx: float,
        by: float,
) -> Tuple[float, float, float, float]:
    abx = bx - ax
    aby = by - ay
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-6:
        return ax, ay, 0.0, (px - ax) ** 2 + (py - ay) ** 2
    apx = px - ax
    apy = py - ay
    t = (apx * abx + apy * aby) / ab2
    t_clamped = max(0.0, min(1.0, t))
    proj_x = ax + t_clamped * abx
    proj_y = ay + t_clamped * aby
    dist2 = (proj_x - px) ** 2 + (proj_y - py) ** 2
    return proj_x, proj_y, t_clamped, dist2


def update_track_state(
        track_id: int,
        cx: float,
        cy: float,
        pos_norm: float,
        ts: float,
        alpha_center: float = 0.4,
        alpha_speed: float = 0.4,
) -> Dict[str, Any]:
    s = _track_states.get(track_id)
    if s is None:
        s = {
            "smooth_cx": cx,
            "smooth_cy": cy,
            "smooth_pos_norm": pos_norm,
            "last_pos_norm": pos_norm,
            "last_ts": ts,
            "flow_speed": 0.0,
        }
        _track_states[track_id] = s
        return s

    dt = max(ts - s["last_ts"], 1e-3)
    s["smooth_cx"] = ema(s["smooth_cx"], cx, alpha_center)
    s["smooth_cy"] = ema(s["smooth_cy"], cy, alpha_center)
    s["smooth_pos_norm"] = ema(s["smooth_pos_norm"], pos_norm, alpha_center)

    vel = (pos_norm - s["last_pos_norm"]) / dt
    s["flow_speed"] = ema(s["flow_speed"], vel, alpha_speed)

    s["last_pos_norm"] = pos_norm
    s["last_ts"] = ts
    return s


def get_alias_for_track(tid: int) -> int:
    global _next_alias
    if tid not in _alias_map:
        _alias_map[tid] = _next_alias
        _next_alias += 1
    return _alias_map[tid]


def summarize_tracks(
        detections: List[Dict[str, Any]],
        window_size: int,
        min_track_age_frames: int,
        queue_cfg: Dict[str, Any],
        frame_shape: Tuple[int, int],
) -> Tuple[int, int, float, float, List[str]]:
    h, w = frame_shape
    line_len_feet_cfg = float(queue_cfg.get("line_length_feet", 40.0))
    max_slots = 6.0

    customers = 0
    items = 0

    # Mark in_line and count
    for d in detections:
        role = d["role"]
        pos = float(d.get("queue_pos_norm", 0.0))
        in_line = 0.0 <= pos <= 1.0
        d["in_line"] = in_line

        if role == "ADULT" and in_line:
            customers += 1
        elif role == "ITEM":
            items += 1

    line_feet = (customers / max_slots) * line_len_feet_cfg
    line_feet = max(0.0, line_feet)

    wait_sec = customers * 20.0

    alerts: List[str] = []
    if customers >= 5 or wait_sec >= 240:
        alerts.append("LONG_LINE")
    if customers == 0:
        alerts.append("NO_LINE")

    return customers, items, wait_sec, line_feet, alerts


def send_to_backend(
        lane_id: int,
        customers: int,
        items: int,
        wait_sec: float,
        line_feet: float,
        alerts: List[str],
        lane_health_score: Optional[float] = None,
        lane_recommendation: Optional[str] = None,
        avg_observed_wait_sec: Optional[float] = None,
        throughput_cph: Optional[float] = None,
        customers_served_total: Optional[int] = None,
) -> None:
    payload = {
        "lane_id": lane_id,
        "customers_in_line": int(customers),
        "items_on_belt": int(items),
        "estimated_wait_seconds": float(wait_sec),
        "line_length_feet": float(line_feet),
        "alerts": alerts,
    }

    if lane_health_score is not None:
        payload["lane_health_score"] = float(lane_health_score)
    if lane_recommendation is not None:
        payload["lane_recommendation"] = str(lane_recommendation)
    if avg_observed_wait_sec is not None:
        payload["avg_observed_wait_sec"] = float(avg_observed_wait_sec)
    if throughput_cph is not None:
        payload["throughput_cph"] = float(throughput_cph)
    if customers_served_total is not None:
        payload["customers_served_total"] = int(customers_served_total)

    try:
        r = requests.post(BACKEND_URL, json=payload, timeout=1.0)
        print(f"[DEBUG] Sent to backend {BACKEND_URL}, status={r.status_code}")
        if r.status_code != 200:
            print(f"[WARN] Backend responded {r.status_code}: {r.text}")
    except Exception as e:
        print(f"[WARN] Could not reach backend: {e}")


def save_screenshot_if_needed(
        frame: np.ndarray,
        lane_id: int,
        alerts: List[str],
        cfg: Dict[str, Any],
) -> None:
    ss_cfg = cfg.get("screenshots", {})
    if not ss_cfg.get("enabled", True):
        return
    if not alerts:
        return

    out_dir = ss_cfg.get("out_dir", "logs/screenshots")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"lane{lane_id}_{ts}_{'_'.join(alerts)}.jpg"
    cv2.imwrite(os.path.join(out_dir, name), frame)


# ------------------------------
# RUST LANE-DECIDER CLIENT
# ------------------------------

def call_lane_decider(
        lane_id: int,
        customers: int,
        children: int,
        items: int,
        wait_sec: float,
        line_feet: float,
        throughput_cph: float,
        avg_observed_wait_sec: float,
        avg_flow_speed: float,
        alerts: List[str],
) -> Optional[Dict[str, Any]]:
    now_utc = datetime.now(timezone.utc).isoformat()

    snapshot = {
        "schema_version": "1.0",
        "lane_id": int(lane_id),
        "timestamp_utc": now_utc,
        "metrics": {
            "customers_in_line": int(max(customers, 0)),
            "children_in_line": int(max(children, 0)),
            "items_on_belt": int(max(items, 0)),
            "estimated_wait_seconds": float(max(wait_sec, 0.0)),
            "physical_line_feet": float(max(line_feet, 0.0)),
            "throughput_cph": float(max(throughput_cph, 0.0)),
            "avg_observed_wait_seconds": float(max(avg_observed_wait_sec, 0.0)),
            "avg_flow_speed": float(avg_flow_speed),
        },
        "config_profile": {
            "profile_name": "default",
            "seconds_per_customer": 20.0,
            "seconds_per_item": 1.5,
            "feet_per_customer": 6.0,
            "count_children_as_customers": True,
        },
        "flags": {
            "is_access_friendly_lane": False,
            "camera_ok": True,
            "tracking_ok": True,
        },
        "alerts": alerts,
    }

    try:
        r = requests.post(LANE_DECIDER_URL, json=snapshot, timeout=0.25)
        if r.status_code != 200:
            print(f"[WARN] lane_decider HTTP {r.status_code}: {r.text}")
            return None
        return r.json()
    except Exception as e:
        print(f"[WARN] lane_decider not reachable at {LANE_DECIDER_URL}: {e}")
        return None


# ------------------------------
# MAIN
# ------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AI Checkout – StrongSORT camera with Rust lane_decider")
    parser.add_argument("--lane-id", type=int, default=1)
    parser.add_argument(
        "--source",
        default="1",  # camera index or video path
        help="Camera index or video path",
    )
    parser.add_argument("--mode", default="auto", help="calibration / production / auto")
    args = parser.parse_args()

    cfg = load_config()

    model_cfg = cfg.get("model", {})
    model_path = model_cfg.get("path", DEFAULT_MODEL_PATH)
    imgsz = int(model_cfg.get("imgsz", 1280))
    conf = float(model_cfg.get("conf", 0.35))
    iou = float(model_cfg.get("iou", 0.45))
    precision = str(model_cfg.get("precision", "fp16")).lower()

    smoothing_cfg = cfg.get("smoothing", {})
    window_size = int(smoothing_cfg.get("window_size", 1))
    update_every_n_frames = int(smoothing_cfg.get("update_every_n_frames", 1))

    modes_cfg = cfg.get("modes", {})
    default_mode = modes_cfg.get("default", "production")
    effective_mode = args.mode if args.mode != "auto" else default_mode

    mode_cfg = modes_cfg.get(effective_mode, {})
    draw_boxes = bool(mode_cfg.get("draw_boxes", True))
    draw_labels = bool(mode_cfg.get("draw_labels", True))
    draw_line_band = bool(mode_cfg.get("draw_line_band", True))
    draw_slanted_line = bool(mode_cfg.get("draw_slanted_line", True))

    fps_cfg = cfg.get("fps_monitor", {})
    fps_enabled = bool(fps_cfg.get("enabled", True))

    queue_cfg = cfg.get("queue_geometry", {})
    track_cfg = cfg.get("tracking", {})
    min_track_age = int(track_cfg.get("min_track_age_frames", 3))
    offline = bool(cfg.get("offline", False))

    print(f"[INFO] Using mode: {effective_mode}")
    print(f"[INFO] Loading model {model_path} (this may download once).")

    model = YOLO(model_path)

    # -------- DEVICE + PRECISION ----------
    use_half = False

    if torch.cuda.is_available():
        model.to("cuda")
        device = torch.device("cuda:0")
        print("[INFO] Using GPU:", torch.cuda.get_device_name(0))

        if precision == "fp16":
            use_half = True
            print("[INFO] Precision set to FP16 (half) on CUDA.")
        elif precision == "fp32":
            use_half = False
            print("[INFO] Precision set to FP32 on CUDA.")
        elif precision == "int8":
            print("[WARN] INT8 precision requested but not implemented yet. Falling back to FP16.")
            use_half = True
            precision = "fp16"
        else:
            print(f"[WARN] Unknown precision '{precision}', defaulting to FP16.")
            use_half = True
            precision = "fp16"

        if use_half:
            try:
                model.model.half()
            except Exception as e:
                print(f"[WARN] Could not switch model to half precision: {e}")
                use_half = False
                precision = "fp32"
    else:
        device = torch.device("cpu")
        print("[WARN] CUDA not available. Using CPU.")
        if precision != "fp32":
            print("[WARN] Forcing FP32 on CPU (half/INT8 not used).")
        use_half = False
        precision = "fp32"

    print(f"[INFO] Effective precision in use: {precision.upper()} (half={use_half})")

    # -------- REID WEIGHTS ----------
    base_dir = os.path.abspath(os.path.dirname(__file__))
    reid_path = Path(os.path.join(base_dir, "weights", "osnet_x0_25_msmt17.pt"))
    if not reid_path.exists():
        print(f"[ERROR] ReID weights not found at {reid_path}")
        raise SystemExit(1)

    lane_start_time = time.time()
    customers_served_total = 0

    window_title = f"AI Checkout – Lane {args.lane_id}"

    print(
        f"[INFO] Starting YOLO + StrongSORT for Lane {args.lane_id}, "
        f"source={args.source}. Press 'q' to quit."
    )

    last_fps_time = time.time()
    frames_in_second = 0
    current_fps = 0.0

    lane_smooth: Dict[str, Optional[float]] = {
        "customers": None,
        "items": None,
        "wait_sec": None,
        "line_feet": None,
        "avg_flow": None,
        "throughput_cph": None,
    }

    track_ages: Dict[int, int] = {}
    line_status: Dict[int, Dict[str, Any]] = {}
    line_entry_times: Dict[int, float] = {}
    completed_waits: List[float] = []

    state = {"frame_index": 0}

    src = args.source
    if isinstance(src, str) and src.isdigit():
        video_source = int(src)
    else:
        video_source = src

    while True:
        user_quit = False
        cap = None
        try:
            print("[INFO] (Re)starting tracking loop...")

            tracker = StrongSort(
                reid_weights=reid_path,
                device=str(device),
                half=False,
            )

            cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video source {video_source}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Failed to grab frame from camera.")
                    break

                frame = frame.copy()
                h, w = frame.shape[:2]

                detections: List[Dict[str, Any]] = []
                current_ids = set()

                use_slanted = bool(queue_cfg.get("use_slanted_line", False))
                line_len_feet_cfg = float(queue_cfg.get("line_length_feet", 40.0))
                corridor_width_px = float(queue_cfg.get("corridor_width_px", 80.0))
                pt_near = queue_cfg.get("point_near") or [w * 0.5, h * 0.7]
                pt_far = queue_cfg.get("point_far") or [w * 0.5, h * 0.1]
                ax, ay = float(pt_near[0]), float(pt_near[1])
                bx, by = float(pt_far[0]), float(pt_far[1])

                now_ts = time.time()

                # YOLO detection
                results = model(
                    frame,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    verbose=False,
                )[0]
                boxes = results.boxes

                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    conf_arr = boxes.conf.cpu().numpy().reshape(-1, 1)
                    cls_arr = boxes.cls.cpu().numpy().reshape(-1, 1)
                else:
                    xyxy = np.empty((0, 4), dtype=np.float32)
                    conf_arr = np.empty((0, 1), dtype=np.float32)
                    cls_arr = np.empty((0, 1), dtype=np.float32)

                PERSON_CLASS_ID = 0

                # PERSON dets → StrongSORT
                if xyxy.shape[0] > 0:
                    person_mask = (cls_arr.flatten() == PERSON_CLASS_ID)
                    if person_mask.any():
                        dets_person = np.concatenate(
                            [xyxy[person_mask], conf_arr[person_mask], cls_arr[person_mask]],
                            axis=1,
                        ).astype(np.float32)
                    else:
                        dets_person = np.empty((0, 6), dtype=np.float32)
                else:
                    person_mask = np.array([], dtype=bool)
                    dets_person = np.empty((0, 6), dtype=np.float32)

                tracks = tracker.update(dets_person, frame)

                # tracked persons
                for t in tracks:
                    x1, y1, x2, y2, track_id, t_conf, t_cls, _ind = t
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    tid = int(track_id)
                    cls_id = int(t_cls)
                    box_conf = float(t_conf)

                    raw_name = model.names.get(cls_id, str(cls_id)).lower()

                    bbox_h = max(1.0, y2 - y1)
                    height_ratio = bbox_h / float(h)
                    cy = (y1 + y2) / 2.0
                    cx = (x1 + x2) / 2.0
                    cy_ratio = cy / float(h)

                    if raw_name == "person":
                        is_child = height_ratio < CHILD_HEIGHT_RATIO
                        if is_child:
                            role = "CHILD"
                            color = (147, 51, 234)
                        else:
                            role = "ADULT"
                            color = (0, 200, 0)
                    else:
                        role = "ITEM"
                        is_child = False
                        color = (0, 170, 255)

                    current_ids.add(tid)
                    track_ages[tid] = track_ages.get(tid, 0) + 1

                    # position along queue
                    if use_slanted:
                        _, _, t_norm, _ = project_point_onto_segment(cx, cy, ax, ay, bx, by)
                        pos_norm_raw = float(t_norm)
                    else:
                        denom = max(LINE_Y_MAX_RATIO - LINE_Y_MIN_RATIO, 1e-6)
                        pos_norm_raw = (cy_ratio - LINE_Y_MIN_RATIO) / denom
                        pos_norm_raw = max(0.0, min(1.0, float(pos_norm_raw)))

                    smooth_cx = cx
                    smooth_cy = cy
                    flow_speed = 0.0
                    pos_norm = pos_norm_raw

                    if role in ("ADULT", "CHILD"):
                        st = update_track_state(
                            tid, cx, cy, pos_norm_raw, now_ts, 0.4, 0.4
                        )
                        smooth_cx = st["smooth_cx"]
                        smooth_cy = st["smooth_cy"]
                        flow_speed = st["flow_speed"]
                        pos_norm = st["smooth_pos_norm"]

                    det = {
                        "track_id": tid,
                        "role": role,
                        "in_line": False,
                        "is_child": is_child,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": box_conf,
                        "class_id": cls_id,
                        "class_name": raw_name,
                        "smooth_cx": smooth_cx,
                        "smooth_cy": smooth_cy,
                        "queue_pos_norm": pos_norm,
                        "flow_speed": flow_speed,
                    }
                    detections.append(det)

                    if draw_boxes:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # non-person items (no tracking)
                if xyxy.shape[0] > 0:
                    if person_mask.size == 0:
                        item_inds = range(xyxy.shape[0])
                    else:
                        item_inds = np.where(~person_mask)[0]

                    for idx in item_inds:
                        x1, y1, x2, y2 = xyxy[idx].tolist()
                        box_conf = float(conf_arr[idx][0])
                        cls_id = int(cls_arr[idx][0])
                        raw_name = model.names.get(cls_id, str(cls_id)).lower()

                        cy = (y1 + y2) / 2.0
                        cx = (x1 + x2) / 2.0
                        cy_ratio = cy / float(h)

                        role = "ITEM"
                        is_child = False
                        color = (0, 170, 255)

                        if use_slanted:
                            _, _, t_norm, _ = project_point_onto_segment(cx, cy, ax, ay, bx, by)
                            pos_norm_raw = float(t_norm)
                        else:
                            denom = max(LINE_Y_MAX_RATIO - LINE_Y_MIN_RATIO, 1e-6)
                            pos_norm_raw = (cy_ratio - LINE_Y_MIN_RATIO) / denom
                            pos_norm_raw = max(0.0, min(1.0, float(pos_norm_raw)))

                        det = {
                            "track_id": None,
                            "role": role,
                            "in_line": False,
                            "is_child": is_child,
                            "bbox": [x1, y1, x2, y2],
                            "confidence": box_conf,
                            "class_id": cls_id,
                            "class_name": raw_name,
                            "smooth_cx": cx,
                            "smooth_cy": cy,
                            "queue_pos_norm": pos_norm_raw,
                            "flow_speed": 0.0,
                        }
                        detections.append(det)

                        if draw_boxes:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # cleanup old tracks
                for tid in list(track_ages.keys()):
                    if tid not in current_ids:
                        del track_ages[tid]

                # aggregate analytics
                customers, items, wait_sec, line_feet, alerts = summarize_tracks(
                    detections,
                    window_size,
                    min_track_age,
                    queue_cfg,
                    (h, w),
                )

                # children count (for Rust)
                children_in_line = sum(
                    1 for d in detections
                    if d["role"] == "CHILD" and d.get("in_line", False)
                )

                # per-person wait logic
                for d in detections:
                    tid = d.get("track_id")
                    role = d.get("role")
                    in_line = d.get("in_line", False)

                    if tid is None or role != "ADULT":
                        continue

                    prev = line_status.get(tid, {"in_line_prev": False})
                    prev_in = prev["in_line_prev"]

                    if not prev_in and in_line:
                        line_entry_times[tid] = now_ts
                    elif prev_in and not in_line:
                        start = line_entry_times.pop(tid, None)
                        if start is not None:
                            dur = now_ts - start
                            if 1.0 < dur < 900.0:
                                completed_waits.append(dur)
                                customers_served_total += 1

                    line_status[tid] = {"in_line_prev": in_line}

                # exited tracks that were in line
                for tid in list(line_status.keys()):
                    if tid not in current_ids and line_status[tid].get("in_line_prev"):
                        start = line_entry_times.pop(tid, None)
                        if start is not None:
                            dur = now_ts - start
                            if 1.0 < dur < 900.0:
                                completed_waits.append(dur)
                                customers_served_total += 1
                        line_status[tid]["in_line_prev"] = False

                # wait stats
                if completed_waits:
                    avg_observed_wait_sec = sum(completed_waits) / len(completed_waits)
                    avg_wait_source = "obs"
                else:
                    avg_observed_wait_sec = float(wait_sec)
                    avg_wait_source = "est"

                # throughput
                if lane_start_time is not None:
                    elapsed_hours = max((now_ts - lane_start_time) / 3600.0, 1e-6)
                    throughput_cph_raw = customers_served_total / elapsed_hours
                else:
                    throughput_cph_raw = 0.0

                lane_smooth["throughput_cph"] = ema(
                    lane_smooth.get("throughput_cph"), throughput_cph_raw, alpha=0.3
                )
                throughput_cph = (
                    throughput_cph_raw
                    if lane_smooth["throughput_cph"] is None
                    else lane_smooth["throughput_cph"]
                )

                lane_smooth["customers"] = ema(lane_smooth["customers"], customers, alpha=0.4)
                lane_smooth["items"] = ema(lane_smooth["items"], items, alpha=0.4)
                lane_smooth["wait_sec"] = ema(lane_smooth["wait_sec"], wait_sec, alpha=0.3)
                lane_smooth["line_feet"] = ema(lane_smooth["line_feet"], line_feet, alpha=0.3)

                customers_s = customers if lane_smooth["customers"] is None else int(
                    round(lane_smooth["customers"])
                )
                items_s = items if lane_smooth["items"] is None else int(
                    round(lane_smooth["items"])
                )
                wait_sec_s = float(wait_sec) if lane_smooth["wait_sec"] is None else float(
                    lane_smooth["wait_sec"]
                )
                line_feet_s = float(line_feet) if lane_smooth["line_feet"] is None else float(
                    lane_smooth["line_feet"]
                )

                flow_values = [
                    d["flow_speed"]
                    for d in detections
                    if d["role"] in ("ADULT", "CHILD")
                       and d.get("in_line", False)
                       and d.get("track_id") is not None
                ]
                if flow_values:
                    avg_flow_raw = sum(flow_values) / len(flow_values)
                else:
                    avg_flow_raw = 0.0

                lane_smooth["avg_flow"] = ema(
                    lane_smooth["avg_flow"], avg_flow_raw, alpha=0.3
                )
                avg_flow_s = (
                    avg_flow_raw if lane_smooth["avg_flow"] is None else lane_smooth["avg_flow"]
                )

                if customers_s == 0:
                    speed_state = "No line"
                else:
                    if abs(avg_flow_s) < 0.002:
                        speed_state = "Stopped"
                    elif avg_flow_s < -0.015:
                        speed_state = "Fast"
                    elif avg_flow_s < -0.005:
                        speed_state = "Normal"
                    else:
                        speed_state = "Slow"

                # local Python health scoring (fallback)
                score = 100.0
                if customers_s >= 5:
                    score -= 40
                elif customers_s >= 3:
                    score -= 25
                elif customers_s == 2:
                    score -= 10

                if wait_sec_s >= 240:
                    score -= 40
                elif wait_sec_s >= 180:
                    score -= 30
                elif wait_sec_s >= 120:
                    score -= 20

                if line_feet_s >= 30:
                    score -= 25
                elif line_feet_s >= 20:
                    score -= 15

                if speed_state == "Stopped":
                    score -= 20
                elif speed_state == "Slow":
                    score -= 10

                score = max(0.0, min(100.0, score))

                if score <= 40:
                    lane_recommendation = "OPEN_NEW_LANE"
                elif score <= 65:
                    lane_recommendation = "WATCH_LANE"
                else:
                    lane_recommendation = "OK"

                lane_health_score = score

                # ---- ask Rust lane_decider ----
                decider_resp = call_lane_decider(
                    lane_id=args.lane_id,
                    customers=customers_s,
                    children=children_in_line,
                    items=items_s,
                    wait_sec=wait_sec_s,
                    line_feet=line_feet_s,
                    throughput_cph=throughput_cph,
                    avg_observed_wait_sec=avg_observed_wait_sec,
                    avg_flow_speed=avg_flow_s,
                    alerts=alerts,
                )

                if decider_resp is not None:
                    try:
                        lane_health_score = float(decider_resp.get("health_score", lane_health_score))
                        lane_recommendation = decider_resp.get("recommendation", lane_recommendation)
                        reason_codes = decider_resp.get("reason_codes", [])
                    except Exception:
                        reason_codes = []
                else:
                    reason_codes = []

                # labels
                if draw_labels:
                    for d in detections:
                        x1, y1, x2, y2 = d["bbox"]
                        role = d["role"]
                        tid = d["track_id"]
                        conf_val = d["confidence"]
                        in_line = d.get("in_line", False)

                        if role == "ADULT":
                            color = (0, 200, 0)
                        elif role == "CHILD":
                            color = (147, 51, 234)
                        else:
                            color = (0, 170, 255)

                        if draw_boxes:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                        text_parts = [role]
                        if tid is not None:
                            alias = get_alias_for_track(tid)
                            text_parts.append(f"#{alias}")
                        text_parts.append(f"{conf_val:.2f}")
                        if role in ("ADULT", "CHILD") and in_line:
                            text_parts.append("[LINE]")

                        text = " ".join(text_parts)
                        cv2.putText(
                            frame,
                            text,
                            (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

                # ----------------- LOGGING / BACKEND / OVERLAYS -----------------
                state["frame_index"] += 1

                if state["frame_index"] % update_every_n_frames == 0:
                    print(
                        f"[Lane {args.lane_id}] adults={customers_s}, items={items_s}, "
                        f"line={line_feet_s:.1f}ft, wait={wait_sec_s:.1f}s, "
                        f"avg_wait={avg_observed_wait_sec:.1f}s({avg_wait_source}), "
                        f"thr={throughput_cph:.1f}cust/hr, "
                        f"speed={speed_state}, health={lane_health_score:.0f}, "
                        f"rec={lane_recommendation}, alerts={alerts}, "
                        f"rust_reasons={reason_codes}"
                    )

                    if not offline:
                        send_to_backend(
                            args.lane_id,
                            customers_s,
                            items_s,
                            wait_sec_s,
                            line_feet_s,
                            alerts,
                            lane_health_score,
                            lane_recommendation,
                            avg_observed_wait_sec,
                            throughput_cph,
                            customers_served_total,
                        )

                    save_screenshot_if_needed(frame, args.lane_id, alerts, cfg)

                # FPS + overlays
                if fps_enabled:
                    frames_in_second += 1
                    now = time.time()
                    if now - last_fps_time >= 1.0:
                        current_fps = frames_in_second / (now - last_fps_time)
                        frames_in_second = 0
                        last_fps_time = now

                summary_text = (
                    f"Lane {args.lane_id} | Adults: {customers_s} | Items: {items_s} | "
                    f"Line ~ {line_feet_s:.1f} ft | ETA ~ {int(wait_sec_s)} s"
                )
                speed_text = f"Speed: {speed_state}"
                fps_text = f"FPS: {current_fps:.1f}" if fps_enabled else ""
                health_text = f"Health: {lane_health_score:.0f} ({lane_recommendation})"
                if avg_wait_source == "obs":
                    avg_wait_text = f"Avg wait (past): {avg_observed_wait_sec:.0f} s"
                else:
                    avg_wait_text = f"Avg wait (est): {avg_observed_wait_sec:.0f} s"
                throughput_text = f"Throughput: {throughput_cph:.0f} cust/hr"

                cv2.putText(frame, summary_text, (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, speed_text, (10, 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 2)
                if fps_enabled:
                    cv2.putText(frame, fps_text, (10, 72),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 2)
                cv2.putText(frame, health_text, (10, 96),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 150), 2)
                cv2.putText(frame, avg_wait_text, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 2)
                cv2.putText(frame, throughput_text, (10, 144),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2)

                if draw_line_band and not queue_cfg.get("use_slanted_line", False):
                    y_min = int(LINE_Y_MIN_RATIO * h)
                    y_max = int(LINE_Y_MAX_RATIO * h)
                    cv2.rectangle(frame, (0, y_min), (w, y_max), (30, 64, 128), 1)

                if draw_slanted_line and queue_cfg.get("use_slanted_line", False):
                    pt_near_draw = queue_cfg.get("point_near") or [w * 0.5, h * 0.7]
                    pt_far_draw = queue_cfg.get("point_far") or [w * 0.5, h * 0.1]
                    ax_draw, ay_draw = int(pt_near_draw[0]), int(pt_near_draw[1])
                    bx_draw, by_draw = int(pt_far_draw[0]), int(pt_far_draw[1])
                    cv2.line(frame, (ax_draw, ay_draw), (bx_draw, by_draw), (64, 196, 255), 2)
                    cv2.circle(frame, (ax_draw, ay_draw), 6, (64, 196, 255), -1)
                    cv2.circle(frame, (bx_draw, by_draw), 6, (64, 196, 255), -1)

                cv2.imshow(window_title, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    user_quit = True
                    break

        except Exception as e:
            print(f"[ERROR] Tracking loop crashed: {e}")
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            if user_quit:
                break
            print("[INFO] Restarting tracking in 2 seconds...")
            time.sleep(2)
            continue

        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if user_quit:
            print("[INFO] User requested quit. Exiting.")
            break
        else:
            print("[WARN] Tracking ended unexpectedly. Restarting in 2 seconds...")
            time.sleep(2)
            continue


if __name__ == "__main__":
    main()

