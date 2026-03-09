# AI_Checkout/engine/ai_decision.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from math import ceil
from typing import Any, Dict, List, Optional


@dataclass
class LaneSnapshotLite:
    lane_id: int
    customers_in_line: int
    estimated_wait_seconds: float
    is_open: bool


@dataclass
class AiDecisionState:
    last_decision_id: int = 0
    snoozed_until: Optional[datetime] = None
    blocked_until: Optional[datetime] = None
    last_payload: Optional[Dict[str, Any]] = None


def _classify_store_status(total_customers: int,
                           max_eta: float,
                           required_lanes: int,
                           open_now: int) -> str:
    """
    Rough global classification for the front-end pill:
    'ok' | 'busy' | 'critical'
    """
    if total_customers == 0 and open_now == 0:
        return "ok"

    if max_eta >= 240 or required_lanes > open_now + 1:
        return "critical"

    if max_eta >= 120 or required_lanes > open_now:
        return "busy"

    return "ok"


def _pick_lane_to_open(lanes: List[LaneSnapshotLite]) -> Optional[int]:
    """
    Choose a lane that looks 'closed' or idle as best candidate to open.
    """
    if not lanes:
        return None

    # Prefer lanes with 0 customers and 0 eta (idle)
    idle = [l for l in lanes if l.customers_in_line == 0 and l.estimated_wait_seconds == 0]
    if idle:
        # smallest lane id for stability
        return sorted(idle, key=lambda x: x.lane_id)[0].lane_id

    # Fall back to the lane with the fewest customers
    return sorted(lanes, key=lambda x: (x.customers_in_line, x.lane_id))[0].lane_id


def _pick_lane_to_close(lanes: List[LaneSnapshotLite]) -> Optional[int]:
    """
    If we are overserved (more lanes than needed), pick a lane that is
    least loaded as suggestion to close.
    """
    if not lanes:
        return None

    # Prefer lanes that are nearly idle
    idle = [l for l in lanes if l.customers_in_line <= 1 and l.estimated_wait_seconds <= 30]
    if idle:
        return sorted(idle, key=lambda x: (x.customers_in_line, x.lane_id))[0].lane_id

    # Otherwise the smallest load
    return sorted(lanes, key=lambda x: (l.customers_in_line, l.estimated_wait_seconds, l.lane_id))[0].lane_id


def compute_ai_decision(
        lanes_raw: List[Dict[str, Any]],
        now: Optional[datetime],
        state: AiDecisionState,
) -> Dict[str, Any]:
    """
    lanes_raw: list of dicts like lane_status returned by your backend
               (lane_id, customers_in_line, estimated_wait_seconds, etc.)
    now: current datetime (UTC)
    state: mutable AiDecisionState for snooze / block
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Convert incoming data to lite objects
    lanes: List[LaneSnapshotLite] = []
    for ld in lanes_raw:
        lane_id = int(ld.get("lane_id", 0) or 0)
        c = int(ld.get("customers_in_line") or 0)
        eta = float(ld.get("estimated_wait_seconds") or 0.0)
        is_open = bool(c > 0 or eta > 0)
        lanes.append(LaneSnapshotLite(
            lane_id=lane_id,
            customers_in_line=c,
            estimated_wait_seconds=eta,
            is_open=is_open,
        ))

    total_customers = sum(l.customers_in_line for l in lanes)
    max_eta = max((l.estimated_wait_seconds for l in lanes), default=0.0)
    open_now = sum(1 for l in lanes if l.is_open)

    if total_customers == 0:
        required_lanes = 0
    else:
        # Rough rule: 1 lane per 4 people
        required_lanes = max(1, ceil(total_customers / 4))

    store_status = _classify_store_status(
        total_customers=total_customers,
        max_eta=max_eta,
        required_lanes=required_lanes,
        open_now=open_now,
    )

    # Base payload even if we decide "no action"
    payload: Dict[str, Any] = {
        "status": store_status,              # 'ok' | 'busy' | 'critical'
        "action": "none",                    # 'open_lane' | 'close_lane' | 'none'
        "lane_id": None,                     # recommended lane
        "seconds_until_deadline": None,      # countdown seconds
        "reason": "",
        "total_customers": total_customers,
        "open_now": open_now,
        "required_lanes": required_lanes,
        "max_eta": max_eta,
        "decision_id": state.last_decision_id,
        "snoozed_until": state.snoozed_until.isoformat() if state.snoozed_until else None,
        "blocked_until": state.blocked_until.isoformat() if state.blocked_until else None,
        "ts": now.isoformat(),
    }

    # If no lanes or no customers, just report status
    if not lanes or total_customers == 0:
        payload["reason"] = "Traffic is low. No action required."
        state.last_payload = payload
        return payload

    # Respect snooze / block windows
    if state.snoozed_until and now < state.snoozed_until:
        payload["reason"] = "Manager snoozed lane recommendation recently."
        state.last_payload = payload
        return payload

    if state.blocked_until and now < state.blocked_until:
        payload["reason"] = "Manager marked recommendation as handled."
        state.last_payload = payload
        return payload

    # --------------------------
    # ACTION LOGIC
    # --------------------------

    # Case 1: not enough lanes -> suggest opening
    if required_lanes > open_now:
        lane_to_open = _pick_lane_to_open(lanes)
        if lane_to_open is not None:
            state.last_decision_id += 1
            deadline = 240 if store_status == "busy" else 120  # seconds
            payload.update({
                "action": "open_lane",
                "lane_id": lane_to_open,
                "seconds_until_deadline": deadline,
                "reason": (
                    f"Traffic rising: {total_customers} customer(s), "
                    f"{open_now} lane(s) open, need about {required_lanes}."
                ),
                "decision_id": state.last_decision_id,
            })
            state.last_payload = payload
            return payload

    # Case 2: we might be overserved (too many lanes open with low load)
    if required_lanes + 1 < open_now and total_customers <= 4 and max_eta < 60:
        lane_to_close = _pick_lane_to_close(lanes)
        if lane_to_close is not None:
            state.last_decision_id += 1
            payload.update({
                "action": "close_lane",
                "lane_id": lane_to_close,
                "seconds_until_deadline": 300,
                "reason": (
                    f"Traffic is light: {total_customers} customer(s), "
                    f"{open_now} lane(s) open, need about {required_lanes}. "
                    "You can safely close one lane."
                ),
                "decision_id": state.last_decision_id,
            })
            state.last_payload = payload
            return payload

    # Default: no specific action, just status
    payload["reason"] = "Traffic is balanced for current open lanes."
    state.last_payload = payload
    return payload


def handle_action(
        state: AiDecisionState,
        decision_id: int,
        action: str,
        now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Called when manager presses a button: 'done' | 'snooze' | 'cannot'.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # If decision ids don't match, still accept but don't blindly trust
    if action == "done":
        # Assume manager handled it -> block new suggestions for a short period
        state.blocked_until = now + timedelta(minutes=8)
        state.snoozed_until = None
    elif action == "snooze":
        state.snoozed_until = now + timedelta(minutes=10)
        state.blocked_until = None
    elif action == "cannot":
        # Manager can't follow recommendation (short-staffed)
        state.blocked_until = now + timedelta(minutes=15)
        state.snoozed_until = None

    return {
        "ok": True,
        "action_applied": action,
        "decision_id": decision_id,
        "snoozed_until": state.snoozed_until.isoformat() if state.snoozed_until else None,
        "blocked_until": state.blocked_until.isoformat() if state.blocked_until else None,
    }
