# AI_Checkout/engine/lane_state.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# How many snapshots per-lane we keep in memory (for charts / history)
MAX_HISTORY = 600  # ~10 minutes if you update about once per second


@dataclass
class LaneSnapshot:
    timestamp: datetime
    lane_id: int
    customers_in_line: int
    items_on_belt: int
    estimated_wait_seconds: float
    line_length_feet: float
    alerts: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        iso = self.timestamp.isoformat(timespec="seconds")
        d["timestamp"] = iso
        d["last_update_iso"] = iso
        return d


class LaneStateStore:
    def __init__(self) -> None:
        self._lanes: Dict[int, LaneSnapshot] = {}
        self._history: Dict[int, deque[LaneSnapshot]] = {}

    # ---------- core update ----------

    def update_lane(self, lane_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply smarter alerts + store last snapshot + history.
        `payload` is JSON coming from detection/tracking.
        """
        lane_id = int(lane_id)

        base_alerts = payload.get("alerts") or []
        alerts: List[str] = list(base_alerts)

        c = int(payload.get("customers_in_line", 0))
        i = int(payload.get("items_on_belt", 0))
        eta = float(payload.get("estimated_wait_seconds", 0.0))
        line_feet = float(payload.get("line_length_feet", 0.0))

        # --- smarter alerts (no theft, just lane health) ---

        # Busy / overloaded
        if c >= 5:
            alerts.append("CRITICAL: 5+ adult customers in line")
        elif c >= 3:
            alerts.append("Busy lane: 3+ adults in line")

        # Wait time
        if eta >= 180:
            alerts.append("Very long wait (3+ minutes)")
        elif eta >= 120:
            alerts.append("Long wait (2+ minutes)")

        # Items on belt
        if i >= 30:
            alerts.append("Many items on belt (30+ items)")

        # Idle cashier (items but no adults)
        if c == 0 and i > 0:
            alerts.append("Items on belt but no adult in line (possible idle cashier)")

        # Deduplicate & keep stable order (sort for consistency)
        alerts = sorted(set(alerts))

        ts = datetime.now(timezone.utc)
        snap = LaneSnapshot(
            timestamp=ts,
            lane_id=lane_id,
            customers_in_line=c,
            items_on_belt=i,
            estimated_wait_seconds=eta,
            line_length_feet=line_feet,
            alerts=alerts,
        )

        self._lanes[lane_id] = snap
        hist = self._history.setdefault(lane_id, deque(maxlen=MAX_HISTORY))
        hist.append(snap)

        return snap.to_dict()

    # ---------- getters ----------

    def get_lane(self, lane_id: int) -> Optional[Dict[str, Any]]:
        snap = self._lanes.get(int(lane_id))
        return snap.to_dict() if snap else None

    def get_all_lanes(self) -> List[Dict[str, Any]]:
        snaps = sorted(self._lanes.values(), key=lambda s: s.lane_id)
        return [s.to_dict() for s in snaps]

    def get_history(self, lane_id: str | int) -> List[Dict[str, Any]]:
        if lane_id == "all":
            combined: List[LaneSnapshot] = []
            for d in self._history.values():
                combined.extend(d)
            combined.sort(key=lambda s: s.timestamp)
            return [s.to_dict() for s in combined]

        lane_id = int(lane_id)
        hist = self._history.get(lane_id)
        if not hist:
            return []
        return [s.to_dict() for s in hist]

    def compute_rebalance_hint(self) -> Optional[str]:
        """
        If one lane is very crowded and another is empty/light,
        suggest sending customers to the lighter lane.
        """
        snaps = list(self._lanes.values())
        if len(snaps) < 2:
            return None

        snaps.sort(key=lambda s: s.customers_in_line)
        least = snaps[0]
        most = snaps[-1]

        if most.customers_in_line - least.customers_in_line >= 3 and most.customers_in_line >= 4:
            return (
                f"Consider directing customers from Lane {most.lane_id} "
                f"to Lane {least.lane_id} (large difference in line length)."
            )
        return None

    def get_busiest_lane_id(self) -> Optional[int]:
        """
        Return lane_id of busiest lane (by customers, then wait).
        """
        if not self._lanes:
            return None
        snaps = list(self._lanes.values())
        snaps.sort(key=lambda s: (s.customers_in_line, s.estimated_wait_seconds), reverse=True)
        return snaps[0].lane_id


# ---------- module-level helpers used by server.py ----------

_store = LaneStateStore()


def update_lane_from_payload(lane_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    return _store.update_lane(lane_id, payload)


def get_lane_status(lane_id: int) -> Optional[Dict[str, Any]]:
    lane = _store.get_lane(lane_id)
    if not lane:
        return None
    lane["rebalance_hint"] = _store.compute_rebalance_hint()
    return lane


def get_lanes_with_hint() -> Dict[str, Any]:
    lanes = _store.get_all_lanes()
    rebalance_hint = _store.compute_rebalance_hint()
    busiest_lane_id = _store.get_busiest_lane_id()

    overload_predictions = []
    for s in lanes:
        c = s.get("customers_in_line", 0)
        eta = s.get("estimated_wait_seconds", 0.0)
        if c >= 3 and eta >= 90:
            overload_predictions.append(
                {
                    "lane_id": s["lane_id"],
                    "reason": "High load and rising wait time (3+ customers, ETA >= 90s).",
                }
            )

    return {
        "lanes": lanes,
        "rebalance_hint": rebalance_hint,
        "busiest_lane_id": busiest_lane_id,
        "overload_predictions": overload_predictions,
    }


def get_lane_history(lane_id: str) -> Dict[str, Any]:
    return {"history": _store.get_history(lane_id)}


