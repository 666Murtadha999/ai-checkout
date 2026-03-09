# AI_Checkout/engine/forecast.py
from __future__ import annotations

import csv
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional


def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """
    Parse ISO timestamp like '2025-11-25T03:14:15+00:00'.
    Returns None on failure.
    """
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str)
    except Exception:
        # try stripping Z if present
        try:
            if ts_str.endswith("Z"):
                return datetime.fromisoformat(ts_str[:-1])
        except Exception:
            return None
    return None


def _weekday_and_bin(dt: datetime) -> Tuple[int, int]:
    """
    Convert a datetime into:
      - weekday: Monday=0..Sunday=6
      - bin_index: 0..95 for 15-minute bins
    """
    weekday = dt.weekday()
    seconds_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
    bin_index = seconds_since_midnight // (15 * 60)
    if bin_index < 0:
        bin_index = 0
    if bin_index > 95:
        bin_index = 95
    return weekday, int(bin_index)


def _load_stats(log_path: str) -> Tuple[
    Dict[Tuple[int, int], Dict[str, float]],
    Dict[Tuple[int, int, int], Dict[str, float]],
]:
    """
    Scan lane_events.csv and build aggregates:

    store_stats[(weekday, bin)] = { sum_c, sum_eta, n }
    lane_stats[(lane_id, weekday, bin)] = { sum_c, sum_eta, n }
    """
    store_stats: Dict[Tuple[int, int], Dict[str, float]] = {}
    lane_stats: Dict[Tuple[int, int, int], Dict[str, float]] = {}

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_str = row.get("timestamp") or ""
                dt = _parse_timestamp(ts_str)
                if dt is None:
                    continue

                try:
                    lane_id = int(row.get("lane_id", 1))
                    c = float(row.get("customers_in_line", 0) or 0.0)
                    eta = float(row.get("estimated_wait_seconds", 0) or 0.0)
                except Exception:
                    continue

                weekday, bin_index = _weekday_and_bin(dt)

                # store-wide
                key_store = (weekday, bin_index)
                s = store_stats.setdefault(key_store, {"sum_c": 0.0, "sum_eta": 0.0, "n": 0.0})
                s["sum_c"] += c
                s["sum_eta"] += eta
                s["n"] += 1.0

                # per-lane
                key_lane = (lane_id, weekday, bin_index)
                l = lane_stats.setdefault(key_lane, {"sum_c": 0.0, "sum_eta": 0.0, "n": 0.0})
                l["sum_c"] += c
                l["sum_eta"] += eta
                l["n"] += 1.0
    except FileNotFoundError:
        # no logs yet
        pass
    except Exception as e:
        print(f"[WARN] forecast._load_stats error: {e}")

    return store_stats, lane_stats


def build_store_heatmap(log_path: str, weekday: int) -> Dict[str, Any]:
    """
    Build typical store-wide heatmap for a given weekday (0=Mon..6=Sun).

    Returns:
      {
        "weekday": int,
        "bins": [0..95],
        "avg_customers": [96 floats],
        "avg_wait": [96 floats],
        "samples": [96 ints],
        "total_samples": int
      }
    """
    store_stats, _ = _load_stats(log_path)

    bins = list(range(96))
    avg_customers: List[float] = []
    avg_wait: List[float] = []
    samples: List[int] = []
    total_samples = 0

    for b in bins:
        agg = store_stats.get((weekday, b))
        if agg and agg["n"] > 0:
            n = int(agg["n"])
            c_avg = agg["sum_c"] / agg["n"]
            eta_avg = agg["sum_eta"] / agg["n"]
        else:
            n = 0
            c_avg = 0.0
            eta_avg = 0.0
        avg_customers.append(float(c_avg))
        avg_wait.append(float(eta_avg))
        samples.append(n)
        total_samples += n

    return {
        "weekday": int(weekday),
        "bins": bins,
        "avg_customers": avg_customers,
        "avg_wait": avg_wait,
        "samples": samples,
        "total_samples": int(total_samples),
    }


def predict_lane_load(
        log_path: str,
        lane_id: int,
        weekday: int,
        bin_index: int,
) -> Dict[str, Any]:
    """
    Predict lane load for given lane_id, weekday, and 15-min bin.

    Combines:
      - lane-specific history (Lane X on that weekday+time)
      - store-wide history (ALL lanes at that weekday+time)

    Weighted by how many samples we have.
    """
    store_stats, lane_stats = _load_stats(log_path)

    lane_key = (lane_id, weekday, bin_index)
    store_key = (weekday, bin_index)

    lane_agg = lane_stats.get(lane_key)
    store_agg = store_stats.get(store_key)

    lane_n = int(lane_agg["n"]) if lane_agg else 0
    store_n = int(store_agg["n"]) if store_agg else 0

    lane_c_avg = (lane_agg["sum_c"] / lane_agg["n"]) if (lane_agg and lane_agg["n"] > 0) else 0.0
    lane_eta_avg = (lane_agg["sum_eta"] / lane_agg["n"]) if (lane_agg and lane_agg["n"] > 0) else 0.0

    store_c_avg = (store_agg["sum_c"] / store_agg["n"]) if (store_agg and store_agg["n"] > 0) else 0.0
    store_eta_avg = (store_agg["sum_eta"] / store_agg["n"]) if (store_agg and store_agg["n"] > 0) else 0.0

    # weights depending on how much lane data we have
    if lane_n >= 40:
        w_lane = 0.7
        w_store = 0.3
    elif lane_n >= 10:
        w_lane = 0.5
        w_store = 0.5
    elif lane_n >= 3:
        w_lane = 0.3
        w_store = 0.7
    else:
        # lane is new, mostly trust store
        w_lane = 0.1
        w_store = 0.9

    if store_n == 0 and lane_n > 0:
        # only lane history available
        w_lane = 1.0
        w_store = 0.0
    elif store_n == 0 and lane_n == 0:
        # no data at all -> neutral prediction
        pred_c = 0.0
        pred_eta = 0.0
        return {
            "lane_id": int(lane_id),
            "weekday": int(weekday),
            "bin_index": int(bin_index),
            "predicted_customers": float(pred_c),
            "predicted_wait_seconds": float(pred_eta),
            "lane_samples": int(lane_n),
            "store_samples": int(store_n),
        }

    pred_customers = w_lane * lane_c_avg + w_store * store_c_avg
    pred_wait = w_lane * lane_eta_avg + w_store * store_eta_avg

    # safety clamp
    if pred_customers < 0:
        pred_customers = 0.0
    if pred_wait < 0:
        pred_wait = 0.0

    return {
        "lane_id": int(lane_id),
        "weekday": int(weekday),
        "bin_index": int(bin_index),
        "predicted_customers": float(pred_customers),
        "predicted_wait_seconds": float(pred_wait),
        "lane_samples": int(lane_n),
        "store_samples": int(store_n),
    }

