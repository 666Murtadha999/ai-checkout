# backend/app.py (or whatever name you use)
from __future__ import annotations

import csv
import os
from datetime import datetime, date, timezone
from typing import Dict, Any, List

import yaml
from flask import Flask, jsonify, render_template, request

from engine.lane_state import (
    update_lane_from_payload,
    get_lane_status,
    get_lanes_with_hint,
    get_lane_history,
)

# 👇 AI decision engine
from engine.ai_decision import (
    AiDecisionState,
    compute_ai_decision,
    handle_action,
)

# 👇 NEW: forecast module (lane + store predictions)
from engine.forecast import (
    build_store_heatmap,
    predict_lane_load,
)

app = Flask(__name__, template_folder="templates")

# ---------- paths & logging config ----------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "lane_events.csv")

LOG_FIELDS = [
    "timestamp",
    "lane_id",
    "customers_in_line",
    "items_on_belt",
    "estimated_wait_seconds",
    "line_length_feet",
    "alerts",
]

CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

# 👇 global AI decision state (keeps snooze / block info in memory)
_ai_state = AiDecisionState()


# ---------- helpers: CSV log ----------

def append_log(row: dict) -> None:
    """Append a single event row to logs/lane_events.csv."""
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------- helpers: config.yaml ----------

def load_config_file() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def save_config_file(cfg: Dict[str, Any]) -> None:
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    except Exception as e:
        print(f"[WARN] Could not save config.yaml: {e}")


# ---------- helpers: heatmap from lane_events.csv ----------

def compute_daily_heatmap() -> Dict[str, Any]:
    """
    Build a 'today-only' shift heatmap from lane_events.csv.

    - hours: [0..23]
    - lanes: [1,2,...]
    - matrix_customers[h][lane_index] = avg customers_in_line at that hour (today)
    - matrix_wait[h][lane_index]      = avg estimated_wait_seconds at that hour (today)
    """
    today = datetime.now().date()

    if not os.path.exists(LOG_FILE):
        return {
            "hours": list(range(24)),
            "lanes": [],
            "matrix_customers": [],
            "matrix_wait": [],
            "max_customers": 0,
            "max_wait": 0,
        }

    # (lane_id, hour) -> aggregations for TODAY ONLY
    agg: Dict[tuple, Dict[str, float]] = {}
    lanes_set = set()
    max_customers_val = 0.0
    max_wait_val = 0.0

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("timestamp") or ""
            try:
                dt = datetime.fromisoformat(ts_str)
            except Exception:
                continue

            if dt.date() != today:
                continue

            try:
                hour = dt.hour
                lane_id = int(row.get("lane_id", 1))
                c = float(row.get("customers_in_line", 0) or 0)
                eta = float(row.get("estimated_wait_seconds", 0) or 0)
            except Exception:
                continue

            lanes_set.add(lane_id)
            key = (lane_id, hour)
            if key not in agg:
                agg[key] = {"sum_c": 0.0, "sum_eta": 0.0, "n": 0.0}
            agg[key]["sum_c"] += c
            agg[key]["sum_eta"] += eta
            agg[key]["n"] += 1.0

    if not lanes_set:
        return {
            "hours": list(range(24)),
            "lanes": [],
            "matrix_customers": [],
            "matrix_wait": [],
            "max_customers": 0,
            "max_wait": 0,
        }

    hours = list(range(24))
    lanes = sorted(lanes_set)

    matrix_customers: List[List[float]] = []
    matrix_wait: List[List[float]] = []

    for h in hours:
        row_c: List[float] = []
        row_eta: List[float] = []
        for lane_id in lanes:
            cell = agg.get((lane_id, h))
            if cell and cell["n"] > 0:
                avg_c = cell["sum_c"] / cell["n"]
                avg_eta = cell["sum_eta"] / cell["n"]
            else:
                avg_c = 0.0
                avg_eta = 0.0

            row_c.append(avg_c)
            row_eta.append(avg_eta)

            if avg_c > max_customers_val:
                max_customers_val = avg_c
            if avg_eta > max_wait_val:
                max_wait_val = avg_eta

        matrix_customers.append(row_c)
        matrix_wait.append(row_eta)

    return {
        "hours": hours,
        "lanes": lanes,
        "matrix_customers": matrix_customers,
        "matrix_wait": matrix_wait,
        "max_customers": max_customers_val,
        "max_wait": max_wait_val,
    }


# ---------- NEW: daily summary for QueVision-style report ----------

def compute_daily_summary(target_date: date | None = None) -> Dict[str, Any]:
    """
    Build a one-day summary from lane_events.csv for QueVision-style reporting.
    """
    if target_date is None:
        target_date = datetime.now().date()

    if not os.path.exists(LOG_FILE):
        return {
            "date": target_date.isoformat(),
            "total_snapshots": 0,
            "avg_customers_in_line": 0.0,
            "avg_wait_seconds": 0.0,
            "pct_within_120s": 0.0,
            "snapshots_over_120s": 0,
            "lanes": [],
            "by_hour": [],
        }

    total_c = 0.0
    total_wait = 0.0
    total_n = 0
    over_120 = 0
    lanes_set = set()

    # per-hour stats for a simple table
    per_hour: Dict[int, Dict[str, float]] = {}

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("timestamp") or ""
            try:
                dt = datetime.fromisoformat(ts_str)
            except Exception:
                continue

            if dt.date() != target_date:
                continue

            try:
                lane_id = int(row.get("lane_id", 1))
                c = float(row.get("customers_in_line", 0) or 0)
                eta = float(row.get("estimated_wait_seconds", 0) or 0)
            except Exception:
                continue

            hour = dt.hour
            lanes_set.add(lane_id)

            total_c += c
            total_wait += eta
            total_n += 1

            if eta > 120:
                over_120 += 1

            if hour not in per_hour:
                per_hour[hour] = {"sum_c": 0.0, "sum_eta": 0.0, "n": 0.0, "over120": 0.0}
            per_hour[hour]["sum_c"] += c
            per_hour[hour]["sum_eta"] += eta
            per_hour[hour]["n"] += 1.0
            if eta > 120:
                per_hour[hour]["over120"] += 1.0

    if total_n == 0:
        return {
            "date": target_date.isoformat(),
            "total_snapshots": 0,
            "avg_customers_in_line": 0.0,
            "avg_wait_seconds": 0.0,
            "pct_within_120s": 0.0,
            "snapshots_over_120s": 0,
            "lanes": sorted(lanes_set),
            "by_hour": [],
        }

    avg_customers = total_c / total_n
    avg_wait = total_wait / total_n
    pct_within = 100.0 * (1.0 - over_120 / total_n)

    by_hour_list: List[Dict[str, Any]] = []
    for h in range(24):
        cell = per_hour.get(h)
        if cell and cell["n"] > 0:
            avg_c_h = cell["sum_c"] / cell["n"]
            avg_eta_h = cell["sum_eta"] / cell["n"]
            pct_ok_h = 100.0 * (1.0 - cell["over120"] / cell["n"])
        else:
            avg_c_h = 0.0
            avg_eta_h = 0.0
            pct_ok_h = 100.0
        by_hour_list.append(
            {
                "hour": h,
                "avg_customers": avg_c_h,
                "avg_wait": avg_eta_h,
                "pct_within_120s": pct_ok_h,
            }
        )

    return {
        "date": target_date.isoformat(),
        "total_snapshots": total_n,
        "avg_customers_in_line": avg_customers,
        "avg_wait_seconds": avg_wait,
        "pct_within_120s": pct_within,
        "snapshots_over_120s": over_120,
        "lanes": sorted(lanes_set),
        "by_hour": by_hour_list,
    }


# ---------- basic pages ----------

@app.get("/health")
def health_check():
    return jsonify(status="ok", message="AI Checkout backend is running")


@app.get("/")
def dashboard():
    return render_template("dashboard.html")


@app.get("/customer")
def customer_screen():
    return render_template("customer.html")


@app.get("/help")
def help_page():
    """
    Simple help / settings page for managers.
    """
    return render_template("help.html")


@app.get("/calibrate")
def calibrate_page():
    """
    Simple click-to-calibrate UI for the slanted queue line.
    """
    return render_template("calibrate.html")


# ---------- NEW: Daily report page (printable / PDF) ----------

@app.get("/report/daily")
def daily_report_page():
    """
    Render a simple QueVision-style daily report as HTML.
    You can print this page or 'Save as PDF' in the browser.
    Optional: /report/daily?date=2025-12-09
    """
    date_str = request.args.get("date")
    target_date: date | None = None
    if date_str:
        try:
            target_date = datetime.fromisoformat(date_str).date()
        except Exception:
            target_date = None

    summary = compute_daily_summary(target_date)
    heatmap = compute_daily_heatmap()  # reserved for future charts

    return render_template("daily_report.html", summary=summary, heatmap=heatmap)


# ---------- lane data APIs ----------

@app.post("/update_lane")
def update_lane():
    """
    Called by run_camera / detection code.

    Expects JSON like:
    {
      "lane_id": 1,
      "customers_in_line": int,
      "items_on_belt": int,
      "estimated_wait_seconds": float,
      "line_length_feet": float,
      "alerts": [...]
    }
    """
    payload = request.get_json(force=True) or {}
    lane_id = int(payload.get("lane_id", 1))

    snapshot = update_lane_from_payload(lane_id, payload)

    # CSV log
    append_log(
        {
            "timestamp": snapshot["timestamp"],
            "lane_id": snapshot["lane_id"],
            "customers_in_line": snapshot["customers_in_line"],
            "items_on_belt": snapshot["items_on_belt"],
            "estimated_wait_seconds": snapshot["estimated_wait_seconds"],
            "line_length_feet": snapshot.get("line_length_feet", 0.0),
            "alerts": " | ".join(snapshot.get("alerts", [])),
        }
    )

    return jsonify(status="ok")


@app.get("/lane_status")
def lane_status():
    """
    Used by dashboard + customer screen.
    Always returns an idle lane if we have no data yet.
    """
    lane_id = int(request.args.get("lane_id", "1"))
    data = get_lane_status(lane_id) or {}

    if "lane_id" not in data:
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        data = {
            "lane_id": lane_id,
            "customers_in_line": 0,
            "items_on_belt": 0,
            "estimated_wait_seconds": 0.0,
            "line_length_feet": 0.0,
            "alerts": [],
            "timestamp": now,
            "last_update_iso": now,
            "rebalance_hint": None,
        }

    return jsonify(data)


@app.get("/lanes")
def lanes():
    """
    Returns all lanes + a global rebalance hint.
    """
    return jsonify(get_lanes_with_hint())


@app.get("/history")
def history():
    """
    Chart data for one lane or all lanes.
    /history?lane_id=1
    /history?lane_id=all
    """
    lane_id = request.args.get("lane_id", "1")
    return jsonify(get_lane_history(lane_id))


# ---------- AI Lane Decision APIs ----------

@app.get("/ai_decision")
def ai_decision_api():
    """
    Returns current AI lane recommendation for the manager.
    """
    lanes_info = get_lanes_with_hint()
    lanes_raw = lanes_info.get("lanes", [])
    rebalance_hint = lanes_info.get("rebalance_hint")

    now = datetime.now(timezone.utc)
    result = compute_ai_decision(lanes_raw=lanes_raw, now=now, state=_ai_state)

    if rebalance_hint:
        result.setdefault("extra", {})["rebalance_hint"] = rebalance_hint

    return jsonify(result)


@app.post("/ai_decision_action")
def ai_decision_action_api():
    """
    Called when manager presses:
    - 'done'
    - 'snooze'
    - 'cannot'
    """
    data = request.get_json(force=True, silent=True) or {}
    decision_id = int(data.get("decision_id") or 0)
    action = str(data.get("action") or "").strip().lower()

    if action not in ("done", "snooze", "cannot"):
        return jsonify(error="Invalid action"), 400

    now = datetime.now(timezone.utc)
    result = handle_action(
        state=_ai_state,
        decision_id=decision_id,
        action=action,
        now=now,
    )
    return jsonify(result)


# ---------- TODAY HEATMAP API (per-hour, per-lane) ----------

@app.get("/heatmap")
def heatmap_api():
    """
    Returns full-day shift heatmap built from lane_events.csv (today only).
    """
    data = compute_daily_heatmap()
    return jsonify(data)


# ---------- STORE-WIDE FORECAST HEATMAP + LANE FORECAST ----------

@app.get("/store_heatmap")
def store_heatmap_api():
    """
    Typical store-wide heatmap by weekday (0=Mon..6=Sun).

    /store_heatmap?weekday=2  -> Wednesday
    """
    weekday = int(request.args.get("weekday", datetime.now().weekday()))
    data = build_store_heatmap(LOG_FILE, weekday=weekday)
    return jsonify(data)


@app.get("/predict_lane")
def predict_lane_api():
    """
    Predict the future load for one lane using historical bins.

    /predict_lane?lane_id=1&minutes_ahead=15
    """
    lane_id = int(request.args.get("lane_id", 1))
    minutes_ahead = int(request.args.get("minutes_ahead", 15))

    now = datetime.now(timezone.utc)
    weekday_now = now.weekday()
    seconds_since_midnight = now.hour * 3600 + now.minute * 60 + now.second

    # current bin
    bin_now = seconds_since_midnight // (15 * 60)
    if bin_now < 0:
        bin_now = 0
    if bin_now > 95:
        bin_now = 95

    # target time
    future_seconds = seconds_since_midnight + minutes_ahead * 60
    if future_seconds < 0:
        future_seconds = 0

    # handle wrap across midnight (roughly)
    future_days_offset = future_seconds // (24 * 3600)
    future_seconds_in_day = future_seconds % (24 * 3600)

    weekday_future = (weekday_now + future_days_offset) % 7
    bin_future = future_seconds_in_day // (15 * 60)
    if bin_future < 0:
        bin_future = 0
    if bin_future > 95:
        bin_future = 95

    pred = predict_lane_load(
        log_path=LOG_FILE,
        lane_id=lane_id,
        weekday=weekday_future,
        bin_index=bin_future,
    )

    pred["minutes_ahead"] = int(minutes_ahead)
    return jsonify(pred)


@app.get("/forecast_required_lanes")
def forecast_required_lanes_api():
    """
    Rough store-wide forecast of how many lanes should be open
    during the NEXT 30 minutes, based on historical store-wide averages.

    We look at the next two 15-min bins (excluding "right now") and
    average their expected customers.

    Returns:
      {
        "minutes_window": 30,
        "required_lanes": int,
        "avg_customers_window": float,
        "weekday": int,
        "bin_index_now": int,
        "bin_index_start": int,
        "bin_index_end": int
      }
    """
    now = datetime.now(timezone.utc)
    weekday_now = now.weekday()

    seconds_since_midnight = now.hour * 3600 + now.minute * 60 + now.second
    bin_now = seconds_since_midnight // (15 * 60)
    if bin_now < 0:
        bin_now = 0
    if bin_now > 95:
        bin_now = 95

    # NEXT 30 minutes = next two 15-min bins (bin_now+1 and bin_now+2)
    bin_start = bin_now + 1
    bin_end = bin_now + 2

    if bin_start > 95:
        bin_start = 95
    if bin_end > 95:
        bin_end = 95

    heat = build_store_heatmap(LOG_FILE, weekday=weekday_now)
    avg_list = heat.get("avg_customers") or []

    total_c = 0.0
    n_bins = 0
    for b in range(bin_start, bin_end + 1):
        if 0 <= b < len(avg_list):
            try:
                total_c += float(avg_list[b] or 0.0)
                n_bins += 1
            except Exception:
                continue

    if n_bins > 0:
        avg_customers_window = total_c / n_bins
    else:
        avg_customers_window = 0.0

    # lanes needed, assume ~4 customers per lane
    required_lanes = 0
    if avg_customers_window > 0:
        required_lanes = max(1, int(-(-avg_customers_window // 4)))  # ceiling

    return jsonify({
        "minutes_window": 30,
        "required_lanes": int(required_lanes),
        "avg_customers_window": float(avg_customers_window),
        "weekday": int(weekday_now),
        "bin_index_now": int(bin_now),
        "bin_index_start": int(bin_start),
        "bin_index_end": int(bin_end),
    })


# ---------- calibration APIs ----------

@app.get("/api/queue_geometry")
def get_queue_geometry_api():
    """
    Returns the current queue_geometry section from config.yaml.
    """
    cfg = load_config_file()
    geom = cfg.get("queue_geometry", {})
    return jsonify(geom)


@app.post("/api/save_queue_geometry")
def save_queue_geometry_api():
    """
    Save slanted queue line config from calibration UI into config.yaml.
    """
    data = request.get_json(force=True) or {}

    pn = data.get("point_near")
    pf = data.get("point_far")
    corridor = data.get("corridor_width_px", 90)
    line_feet = data.get("line_length_feet", 40.0)

    if not (isinstance(pn, list) and len(pn) == 2 and isinstance(pf, list) and len(pf) == 2):
        return jsonify(error="point_near and point_far must be [x, y] lists"), 400

    try:
        corridor = int(corridor)
    except Exception:
        corridor = 90

    try:
        line_feet = float(line_feet)
    except Exception:
        line_feet = 40.0

    cfg = load_config_file()
    if "queue_geometry" not in cfg:
        cfg["queue_geometry"] = {}

    qg = cfg["queue_geometry"]
    qg["point_near"] = [float(pn[0]), float(pn[1])]
    qg["point_far"] = [float(pf[0]), float(pf[1])]
    qg["corridor_width_px"] = corridor
    qg["line_length_feet"] = line_feet
    qg["use_slanted_line"] = True

    save_config_file(cfg)

    return jsonify(status="ok", queue_geometry=qg)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
