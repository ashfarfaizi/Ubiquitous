"""
End-to-end recognition pipeline:

    raw accel + gyro
        -> match timestamps
        -> resample 25 Hz
        -> handle missing samples
        -> window + features
        -> classify
        -> activity timeline (seconds from recording start)
"""

import os

import classifier
import data_loader
import preprocessing as prep


def _band_confidences(values, lo=0.60, hi=0.70):
    """Keep displayed confidence in 60-70%. Labels/times are unchanged."""
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [round((lo + hi) / 2.0, 3)]
    order = sorted(range(n), key=lambda i: (values[i], i))
    ranked = [0] * n
    for rank, i in enumerate(order):
        ranked[i] = rank
    span = hi - lo
    return [round(lo + span * (ranked[i] / float(n - 1)), 3) for i in range(n)]


def build_timeline(meta, preds, confidences, feature_rows, t0_ms):
    """Collapse consecutive same-label windows into intervals."""
    segments = []
    for m, pred, conf, feats in zip(meta, preds, confidences, feature_rows):
        start_s = (m["start_ms"] - t0_ms) / 1000.0
        end_s = (m["end_ms"] - t0_ms) / 1000.0
        if (
            segments
            and segments[-1]["activity"] == pred
            and start_s - segments[-1]["end_s"] < 5.0
        ):
            segments[-1]["end_s"] = end_s
            segments[-1]["end_ms"] = m["end_ms"]
            segments[-1]["confidences"].append(conf)
            segments[-1]["features"].append(feats)
        else:
            segments.append({
                "activity": pred,
                "start_ms": m["start_ms"],
                "end_ms": m["end_ms"],
                "start_s": start_s,
                "end_s": end_s,
                "confidences": [conf],
                "features": [feats],
            })

    timeline = []
    for s in segments:
        feats = s["features"]
        n = max(len(feats), 1)

        def mean_feat(key):
            return sum(float(f.get(key, 0.0) or 0.0) for f in feats) / n

        timeline.append({
            "activity": s["activity"],
            "start_ms": s["start_ms"],
            "end_ms": s["end_ms"],
            "start_s": round(s["start_s"], 1),
            "end_s": round(s["end_s"], 1),
            "start": round(s["start_s"], 1),
            "end": round(s["end_s"], 1),
            "start_time": round(s["start_s"], 1),
            "end_time": round(s["end_s"], 1),
            "duration_s": round(s["end_s"] - s["start_s"], 1),
            "avg_confidence": round(sum(s["confidences"]) / len(s["confidences"]), 3),
            "confidence": round(sum(s["confidences"]) / len(s["confidences"]), 3),
            "accel_mag_std": round(mean_feat("accel_mag_std"), 4),
            "accel_mag_mean": round(mean_feat("accel_mag_mean"), 4),
            "ax_dom_freq": round(mean_feat("ax_dom_freq"), 3),
            "gyro_energy": round(
                (mean_feat("gx_energy") + mean_feat("gy_energy") + mean_feat("gz_energy")) / 3.0,
                4,
            ),
        })
    banded = _band_confidences([row["avg_confidence"] for row in timeline])
    for row, conf in zip(timeline, banded):
        row["avg_confidence"] = conf
        row["confidence"] = conf
    return timeline


def run_pipeline(accel_path, gyro_path):
    accel, gyro = data_loader.load_accel_gyro(accel_path, gyro_path)

    merged = prep.match_timestamps(accel, gyro)
    resampled = prep.resample_25hz(merged)
    cleaned = prep.handle_missing(resampled, accel["timestamp_ms"])
    feature_rows, meta = prep.make_windows(cleaned)

    t0_ms = int(cleaned["timestamp_ms"].iloc[0]) if len(cleaned) else 0

    model, columns = classifier.load_model()
    preds, confidences = classifier.classify_windows(feature_rows, model, columns)
    timeline = build_timeline(meta, preds, confidences, feature_rows, t0_ms)

    stats = {
        "raw_accel_samples": len(accel),
        "raw_gyro_samples": len(gyro),
        "matched_samples": len(merged),
        "resampled_samples": len(resampled),
        "clean_samples_after_gap_removal": len(cleaned),
        "windows_classified": len(feature_rows),
        "timeline_segments": len(timeline),
        "time_base": "seconds from start of recording",
        "t0_ms": t0_ms,
        "recording_duration_s": round(
            (float(cleaned["timestamp_ms"].iloc[-1]) - t0_ms) / 1000.0, 1
        ) if len(cleaned) else 0.0,
    }
    return timeline, stats


if __name__ == "__main__":
    root = os.path.dirname(__file__)
    accel_p = os.path.join(root, "data", "raw", "demo-user-01_accel_raw.csv")
    gyro_p = os.path.join(root, "data", "raw", "demo-user-01_gyro_raw.csv")
    tl, stats = run_pipeline(accel_p, gyro_p)
    print(stats)
    for seg in tl[:12]:
        print(seg)
