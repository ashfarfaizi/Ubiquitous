"""
Question answering over a classified activity timeline.

Every answer uses the CS60055 Challenge 1 fields, in this order:

    Answer
    Activity/Event
    Evidence:
        Timestamp(s)
        Sensor Modality
        Sensor Channel(s)
        Explanation

Timestamps are seconds from the start of the recording.
"""

import re

ACTIVITY_WORDS = {
    "LYING_DOWN": ["lying down", "lie down", "lied down", "lying", "in bed", "asleep", "sleeping", "resting"],
    "SITTING": ["sitting", "sat", "sit"],
    "STANDING_AND_MOVING": ["standing and moving", "standing around"],
    "STANDING_IN_PLACE": ["standing still", "standing in place", "standing"],
    "WALKING": ["walking", "walked", "walk"],
    "RUNNING": ["running", "jogged", "jogging", "ran", "run"],
    "BICYCLING": ["bicycling", "biking", "cycling", "bicycle", "bike", "pedal"],
}

DISPLAY = {
    "LYING_DOWN": "Lying down",
    "SITTING": "Sitting",
    "STANDING_IN_PLACE": "Standing in place",
    "STANDING_AND_MOVING": "Standing and moving",
    "WALKING": "Walking",
    "RUNNING": "Running",
    "BICYCLING": "Bicycling",
}

# short, signal-level reasons (not class slogans)
SIGNAL_WHY = {
    "LYING_DOWN": (
        "accel variance is almost flat and the gyro barely moves, which is what a long lie looks like, not a short pause"
    ),
    "SITTING": (
        "gravity sits on one axis, accel wobble is small, and gyro motion stays low — typical seated posture"
    ),
    "STANDING_IN_PLACE": (
        "there is a bit of upright sway around gravity, but no step cadence in the accel magnitude"
    ),
    "STANDING_AND_MOVING": (
        "accel energy is higher than quiet standing, yet there is no clean heel-strike rhythm"
    ),
    "WALKING": (
        "accel magnitude has a walking-like step beat and the gyro swings with each stride"
    ),
    "RUNNING": (
        "accel peaks get larger and the step rate jumps, with bigger gyro swings than walking"
    ),
    "BICYCLING": (
        "the accel is smooth and cyclic, without the sharp heel-strike spikes of gait, and the gyro keeps a steady pedal-like oscillation"
    ),
}


def _pretty(activity):
    return DISPLAY.get(activity, str(activity).replace("_", " ").title())


def _find_activities(text):
    found = []
    seen = set()
    lower = text.lower()
    items = []
    for activity, words in ACTIVITY_WORDS.items():
        for w in words:
            items.append((len(w), w, activity))
    for _, word, activity in sorted(items, reverse=True):
        if activity in seen:
            continue
        if re.search(r"\b" + re.escape(word) + r"\b", lower):
            found.append(activity)
            seen.add(activity)
    return found


def _seconds_in_text(text):
    m = re.search(r"(?:at\s+)?(\d+(?:\.\d+)?)\s*(?:seconds?|s)\b", text.lower())
    if m:
        return float(m.group(1))
    m = re.search(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?", text.lower())
    if not m:
        return None
    hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    ampm = m.group(4)
    if ampm == "pm" and hh != 12:
        hh += 12
    if ampm == "am" and hh == 12:
        hh = 0
    return hh * 3600 + mm * 60 + ss


def _matches(timeline, activity):
    return [seg for seg in timeline if seg["activity"] == activity]


def _totals(timeline):
    totals = {}
    for seg in timeline:
        totals[seg["activity"]] = totals.get(seg["activity"], 0.0) + seg["duration_s"]
    return totals


def _dominant(timeline):
    totals = _totals(timeline)
    if not totals:
        return None, 0.0
    act = max(totals, key=totals.get)
    return act, totals[act]


def _fmt_spans(segs, limit=8):
    if not segs:
        return "N/A"
    parts = []
    for seg in segs[:limit]:
        a = int(round(seg["start_s"]))
        b = int(round(seg["end_s"]))
        parts.append("%d to %d" % (a, b))
    extra = "" if len(segs) <= limit else ", ... (%d intervals)" % len(segs)
    return ", ".join(parts) + extra + " (seconds from start)"


def _signal_stats(segs):
    if not segs:
        return ""
    mag = sum(s.get("accel_mag_std", 0.0) for s in segs) / len(segs)
    freq = sum(s.get("ax_dom_freq", 0.0) for s in segs) / len(segs)
    gyr = sum(s.get("gyro_energy", 0.0) for s in segs) / len(segs)
    return (
        " On the cited stretch, accel magnitude std was %.3f, the main Acc-X frequency "
        "was %.2f Hz, and mean gyro energy was %.3f."
        % (mag, freq, gyr)
    )


def _pack(question, answer, activity, timestamps, modality, channels, explanation):
    result = {
        "question": question,
        "Answer": answer,
        "Activity/Event": activity,
        "Timestamp(s)": timestamps,
        "Sensor Modality": modality,
        "Sensor Channel(s)": channels,
        "Explanation": explanation,
        "answer": answer,
        "activity_event": activity,
        "activity": activity,
        "timestamps": timestamps,
        "sensor_modality": modality,
        "sensor_channels": channels,
        "explanation": explanation,
        "evidence": {
            "timestamps": timestamps,
            "sensor_modality": modality,
            "sensor_channels": channels,
        },
        "time_base": "seconds from start of recording",
    }
    result["formatted"] = format_answer(result)
    return result


def format_answer(result):
    return (
        "Answer: %s\n"
        "Activity/Event: %s\n"
        "Evidence:\n"
        "Timestamp(s): %s\n"
        "Sensor Modality: %s\n"
        "Sensor Channel(s): %s\n"
        "Explanation: %s"
        % (
            result.get("Answer", "N/A"),
            result.get("Activity/Event", "N/A"),
            result.get("Timestamp(s)", "N/A"),
            result.get("Sensor Modality", "N/A"),
            result.get("Sensor Channel(s)", "N/A"),
            result.get("Explanation", "N/A"),
        )
    )


def _grounded(question, answer, activity_label, segs, explanation):
    return _pack(
        question,
        answer=answer,
        activity=activity_label,
        timestamps=_fmt_spans(segs),
        modality="Accelerometer, Gyroscope" if segs else "N/A",
        channels="All" if segs else "N/A",
        explanation=explanation,
    )


def _empty(question):
    return _pack(
        question,
        answer="N/A",
        activity="N/A",
        timestamps="N/A",
        modality="N/A",
        channels="N/A",
        explanation="No classified activity timeline is available yet.",
    )


def answer(question, timeline, stats=None):
    q = (question or "").strip()
    if not q:
        return _pack(q, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A")
    if not timeline:
        return _empty(q)

    ql = q.lower()
    activities = _find_activities(ql)
    activity = activities[0] if activities else None
    at_s = _seconds_in_text(ql)

    if at_s is not None and any(w in ql for w in ("what", "doing", "activity")):
        match = next((seg for seg in timeline if seg["start_s"] <= at_s <= seg["end_s"]), None)
        if not match:
            return _pack(
                q, "N/A", "N/A", "N/A", "N/A", "N/A",
                "Nothing classified covers %.1f seconds from start." % at_s,
            )
        label = _pretty(match["activity"])
        return _grounded(
            q, label, label, [match],
            "At %.0f s the windowed accel/gyro features look like %s: %s.%s"
            % (at_s, label.lower(), SIGNAL_WHY[match["activity"]], _signal_stats([match])),
        )

    # Task 4 first so "did ... lie down for a prolonged period" is not treated as yes/no class check
    if any(w in ql for w in ("prolonged", "long period", "long time", "extended")) and any(
        w in ql for w in ("lie", "lying", "rest", "sleep", "bed")
    ):
        segs = _matches(timeline, "LYING_DOWN")
        longest = max((s["duration_s"] for s in segs), default=0.0)
        rec_len = max((s["end_s"] for s in timeline), default=1.0)
        if longest >= max(60.0, 0.08 * rec_len):
            ans = "Likely yes"
        elif segs:
            ans = "Possibly, but only briefly"
        else:
            ans = "No"
        return _grounded(
            q, ans, "Prolonged lying down", segs,
            "%s. The longest quiet stretch labelled lying down lasts %.0f seconds. %s.%s"
            % (ans, longest, SIGNAL_WHY["LYING_DOWN"].rstrip("."), _signal_stats(segs[:1])),
        )

    if any(w in ql for w in ("wheel", "pedal", "cycl", "bike", "outdoor physical")):
        segs = _matches(timeline, "BICYCLING")
        if segs:
            return _grounded(
                q, "Yes",
                "Unknown outdoor physical activity, consistent with cycling",
                segs,
                "Yes. That stretch looks like a low-impact wheeled/pedal mode: %s.%s"
                % (SIGNAL_WHY["BICYCLING"], _signal_stats(segs)),
            )
        return _grounded(
            q, "No", "Unknown outdoor physical activity, consistent with cycling", [],
            "No. We did not see the smooth cyclic accel / gyro pattern that usually comes with pedaling.",
        )

    if activity and any(w in ql for w in ("how long", "how much time", "total time", "duration", "spend", "spent")) and "more" not in ql and " or " not in ql:
        segs = _matches(timeline, activity)
        total = int(round(sum(s["duration_s"] for s in segs)))
        label = _pretty(activity)
        if not segs:
            return _grounded(q, "0 seconds", label, [], "%s was not detected in this recording." % label)
        bits = " and ".join("%d seconds" % int(round(s["duration_s"])) for s in segs[:6])
        return _grounded(
            q, "%d seconds" % total, label, segs,
            "%s was detected in %d interval(s), of %s, which sum to %d seconds. %s.%s"
            % (label, len(segs), bits, total, SIGNAL_WHY[activity].rstrip("."), _signal_stats(segs)),
        )

    if "more" in ql or "longer" in ql or (" or " in ql and any(w in ql for w in ("walk", "run", "sit", "lying", "stand", "cycl"))):
        pair = activities[:2]
        if len(pair) < 2:
            totals = _totals(timeline)
            ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
            if len(ranked) >= 2:
                pair = [ranked[0][0], ranked[1][0]]
        if len(pair) >= 2:
            a, b = pair[0], pair[1]
            totals = _totals(timeline)
            ta, tb = int(round(totals.get(a, 0))), int(round(totals.get(b, 0)))
            winner = a if ta >= tb else b
            segs = _matches(timeline, a) + _matches(timeline, b)
            return _grounded(
                q, _pretty(winner), "%s, %s" % (_pretty(a), _pretty(b)), segs,
                "Total %s time was %d seconds and total %s time was %d seconds over the recording, so %s is larger."
                % (_pretty(a), ta, _pretty(b), tb, _pretty(winner)),
            )

    if activity and any(w in ql for w in ("how many times", "how many segments", "how often", "number of")):
        segs = _matches(timeline, activity)
        label = _pretty(activity)
        return _grounded(
            q, str(len(segs)), label, segs,
            "%d separate interval(s) of %s after merging neighbouring windows with the same label."
            % (len(segs), label),
        )

    if any(w in ql for w in ("begin", "began", "start", "started", "onset", "when did")):
        target = activity or "RUNNING"
        segs = _matches(timeline, target)
        label = _pretty(target)
        if not segs:
            return _grounded(
                q, "No", label, [],
                "%s was not detected, so there is no onset time." % label,
            )
        t0 = int(round(segs[0]["start_s"]))
        verb = "Yes, %s began at %d seconds" % (label.lower(), t0)
        return _grounded(
            q, verb, "Onset of %s" % label.lower(), segs,
            "%s. From that point the signal matches running/gait change: %s.%s"
            % (verb, SIGNAL_WHY[target], _signal_stats(segs[:1])),
        )

    if activity and "when" in ql:
        segs = _matches(timeline, activity)
        label = _pretty(activity)
        if not segs:
            return _grounded(q, "N/A", label, [], "No %s intervals were found." % label.lower())
        return _grounded(
            q, _fmt_spans(segs), label, segs,
            "%s shows up in the listed intervals (seconds from start)." % label,
        )

    if activity and (
        ql.startswith("is ")
        or ql.startswith("did ")
        or ql.startswith("was ")
        or " is the user" in ql
        or "was the user" in ql
    ) and not any(w in ql for w in ("prolonged", "wheeled", "pedal", "strenuous", "unsteady")):
        segs = _matches(timeline, activity)
        label = _pretty(activity)
        if segs:
            return _grounded(
                q, "Yes", label, segs,
                "Yes. %s turns up in the cited interval(s). %s.%s"
                % (label, SIGNAL_WHY[activity].rstrip("."), _signal_stats(segs[:3])),
            )
        return _grounded(
            q, "No", label, [],
            "No. None of the classified windows were labelled %s." % label.lower(),
        )

    if any(w in ql for w in ("what activity", "what is the user", "what was the user", "performing", "doing")):
        totals = _totals(timeline)
        rec_len = max((s["end_s"] for s in timeline), default=1.0)
        act, total = _dominant(timeline)
        share = total / rec_len if rec_len else 0.0
        if share < 0.45 and len(totals) > 1:
            order = []
            for seg in timeline:
                if seg["activity"] not in order:
                    order.append(seg["activity"])
            labels = ", ".join(_pretty(a) for a in order)
            return _grounded(
                q, labels, labels, timeline,
                "This clip is mixed, not one pose the whole time. In order: %s. "
                "Labels come from 2 s windows on 25 Hz accel + gyro."
                % labels,
            )
        segs = _matches(timeline, act)
        label = _pretty(act)
        extra = ""
        if len(totals) > 1:
            extra = " That is the longest total (%.0f s); other labels also appear." % total
        return _grounded(
            q, label, label, segs,
            "Most of the recording classifies as %s: %s.%s%s"
            % (label.lower(), SIGNAL_WHY[act], extra, _signal_stats(segs[:3])),
        )

    if "strenuous" in ql or "intense" in ql or "vigorous" in ql:
        strenuous = [s for s in timeline if s["activity"] in ("RUNNING", "BICYCLING")]
        if strenuous:
            labels = ", ".join(sorted({_pretty(s["activity"]) for s in strenuous}))
            return _grounded(
                q, "Yes", labels, strenuous,
                "Yes. %s shows up, with higher accel magnitude and gyro motion than the sedentary bits.%s"
                % (labels, _signal_stats(strenuous[:3])),
            )
        return _grounded(
            q, "No", "N/A", [],
            "No running or bicycling intervals, and the rest looks light or still.",
        )

    if "most" in ql:
        act, total = _dominant(timeline)
        segs = _matches(timeline, act)
        label = _pretty(act)
        return _grounded(
            q, label, label, segs,
            "Longest total duration is %s at %d seconds." % (label, int(round(total))),
        )

    act, total = _dominant(timeline)
    segs = _matches(timeline, act)
    label = _pretty(act)
    return _grounded(
        q, label, label, segs,
        "Took this as an identification question. Dominant label is %s (%d s total). "
        "Clearer examples: \"How long was the user walking?\" or "
        "\"Did the user begin running at any point, and if so, when?\"."
        % (label, int(round(total))),
    )
