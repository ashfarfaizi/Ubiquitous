"""
query_engine.py

The last two boxes of the pipeline: "question/query engine" and "retrieve
sensor evidence -> structured answer". Takes a plain-English question about
the day's activity timeline and answers it, showing which timeline segments
it used as evidence (so the answer is checkable, not just asserted).

This is intentionally rule-based rather than a wrapped LLM call: the whole
point of the project is that the sensor pipeline itself is the source of
truth, and a query engine that can't point at the exact evidence it used
would defeat that.
"""

import re
from datetime import datetime, timezone

ACTIVITY_WORDS = {
    "LYING_DOWN": ["lying down", "lying", "in bed", "asleep", "sleeping"],
    "SITTING": ["sitting", "sat"],
    "STANDING_IN_PLACE": ["standing still", "standing in place"],
    "STANDING_AND_MOVING": ["standing and moving", "standing around"],
    "WALKING": ["walking", "walk"],
    "RUNNING": ["running", "run", "jogging"],
    "BICYCLING": ["biking", "bicycling", "cycling", "bike"],
}


def _to_ms_of_day(hh: int, mm: int) -> int:
    return (hh * 3600 + mm * 60) * 1000


def _find_activity_in_text(text: str):
    text = text.lower()
    for activity, words in ACTIVITY_WORDS.items():
        for w in words:
            if w in text:
                return activity
    return None


def _find_time_in_text(text: str):
    m = re.search(r"(\d{1,2}):(\d{2})\s*(am|pm)?", text.lower())
    if not m:
        return None
    hh, mm, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
    if ampm == "pm" and hh != 12:
        hh += 12
    if ampm == "am" and hh == 12:
        hh = 0
    return _to_ms_of_day(hh, mm)


def answer(question: str, timeline: list) -> dict:
    q = question.strip()
    ql = q.lower()

    time_ms = _find_time_in_text(ql)
    activity = _find_activity_in_text(ql)

    # --- "what was I doing at 14:32?" ---
    if time_ms is not None and ("what" in ql or "doing" in ql):
        match = next((seg for seg in timeline if seg["start_ms"] <= time_ms <= seg["end_ms"]), None)
        if match:
            return {
                "question": q,
                "answer": f"You were {match['activity'].replace('_', ' ').lower()} "
                          f"from {match['start_time']} to {match['end_time']} "
                          f"(confidence {match['avg_confidence']:.0%}).",
                "evidence": [match],
            }
        return {"question": q, "answer": "No classified activity covers that time.", "evidence": []}

    # --- "how long did I spend walking?" / "how much time was I sitting?" ---
    if activity and ("how long" in ql or "how much time" in ql or "total time" in ql):
        matches = [seg for seg in timeline if seg["activity"] == activity]
        total_s = sum(seg["duration_s"] for seg in matches)
        mins = round(total_s / 60, 1)
        label = activity.replace("_", " ").lower()
        return {
            "question": q,
            "answer": f"You spent about {mins} minutes {label} today, across {len(matches)} segment(s).",
            "evidence": matches,
        }

    # --- "when did I go running?" / "when was I walking?" ---
    if activity and ("when" in ql):
        matches = [seg for seg in timeline if seg["activity"] == activity]
        if not matches:
            label = activity.replace("_", " ").lower()
            return {"question": q, "answer": f"No {label} segments were found today.", "evidence": []}
        spans = ", ".join(f"{seg['start_time']}\u2013{seg['end_time']}" for seg in matches)
        label = activity.replace("_", " ").lower()
        return {
            "question": q,
            "answer": f"You were {label} during: {spans}.",
            "evidence": matches,
        }

    # --- "how many times did I sit today?" ---
    if activity and ("how many times" in ql or "how many segments" in ql):
        matches = [seg for seg in timeline if seg["activity"] == activity]
        label = activity.replace("_", " ").lower()
        return {
            "question": q,
            "answer": f"{len(matches)} separate segment(s) of {label} were detected today.",
            "evidence": matches,
        }

    # --- "what did I do most today?" ---
    if "most" in ql and ("do" in ql or "activity" in ql):
        totals = {}
        for seg in timeline:
            totals[seg["activity"]] = totals.get(seg["activity"], 0) + seg["duration_s"]
        if not totals:
            return {"question": q, "answer": "No activity was recorded.", "evidence": []}
        top = max(totals, key=totals.get)
        matches = [seg for seg in timeline if seg["activity"] == top]
        label = top.replace("_", " ").lower()
        mins = round(totals[top] / 60, 1)
        return {
            "question": q,
            "answer": f"Your most common activity was {label} ({mins} minutes total).",
            "evidence": matches,
        }

    # --- fallback: just describe the whole timeline briefly ---
    if not timeline:
        return {"question": q, "answer": "No timeline is available yet - run the pipeline first.", "evidence": []}

    return {
        "question": q,
        "answer": "I couldn't parse that as a specific question. Try things like "
                  "\"what was I doing at 3:15pm?\", \"how long did I spend walking?\", "
                  "\"when did I go running?\", or \"what did I do most today?\".",
        "evidence": timeline[:3],
    }