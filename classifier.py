"""
classifier.py

The 7-class activity classifier from the pipeline diagram. A RandomForest
over the windowed features from preprocessing.py - nothing fancier, because
for short accelerometer/gyroscope windows a forest of shallow trees is a
well-established, hard-to-beat baseline (this is basically what the
ExtraSensory paper's own per-sensor classifiers look like).

train() builds its own labeled training windows from generate_sample_data's
activity profiles (many short bursts per class, run through the same
preprocessing pipeline used at inference time) since we don't have the real
multi-gigabyte ExtraSensory raw download sitting on disk here. Swap this out
for real ExtraSensory raw + label data and nothing downstream changes -
the feature vector shape is identical either way.
"""

import os
import tempfile
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from generate_sample_data import ACTIVITIES, _make_burst, RNG
import preprocessing as prep

_ROOT = os.path.dirname(os.path.abspath(__file__))
_BUNDLED_MODEL = os.path.join(_ROOT, "data", "model.joblib")


def _model_candidates():
    """Bundled path first, then env, then /tmp — serverless roots are read-only."""
    paths = []
    env = os.environ.get("MODEL_PATH")
    if env:
        paths.append(env)
    paths.append(_BUNDLED_MODEL)
    paths.append(os.path.join(tempfile.gettempdir(), "ask_the_sensors_model.joblib"))
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _dir_writable(path):
    directory = os.path.dirname(path) or "."
    try:
        os.makedirs(directory, exist_ok=True)
        probe = os.path.join(directory, ".write_probe")
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
        return True
    except OSError:
        return False


def _save_path():
    for path in _model_candidates():
        if _dir_writable(path):
            return path
    return os.path.join(tempfile.gettempdir(), "ask_the_sensors_model.joblib")


MODEL_PATH = _BUNDLED_MODEL


def _build_training_windows(bursts_per_activity: int = 40):
    """Generates many short labeled bursts per activity and windows them."""
    rows, labels = [], []
    t = 0
    for activity in ACTIVITIES:
        for _ in range(bursts_per_activity):
            accel, gyro = _make_burst(activity, t, duration_s=4.0)
            t += 5000

            merged = prep.match_timestamps(accel, gyro)
            resampled = prep.resample_25hz(merged)
            cleaned = prep.handle_missing(resampled, accel["timestamp_ms"])
            feats, _ = prep.make_windows(cleaned, window_seconds=2.0, overlap=0.5)

            for f in feats:
                rows.append(f)
                labels.append(activity)

    X = pd.DataFrame(rows).fillna(0.0)
    y = np.array(labels)
    return X, y


def train(save: bool = True):
    X, y = _build_training_windows()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(f"holdout accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds, zero_division=0))

    if save:
        path = _save_path()
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            joblib.dump({"model": clf, "columns": list(X.columns)}, path, protocol=4)
            print("saved model -> %s" % path)
        except OSError as exc:
            print("could not persist model (%s); using in-memory classifier" % exc)

    return clf, list(X.columns)


def _try_load(path):
    bundle = joblib.load(path)
    model, columns = bundle["model"], bundle["columns"]
    probe = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
    model.predict(probe)
    return model, columns


def load_model():
    for path in _model_candidates():
        if not os.path.exists(path):
            continue
        try:
            return _try_load(path)
        except Exception as exc:
            print("could not use saved model at %s (%s)" % (path, exc))
    return train(save=True)


def classify_windows(feature_rows, model, columns):
    """feature_rows: list of dicts from preprocessing.make_windows"""
    if not feature_rows:
        return [], []
    X = pd.DataFrame(feature_rows).reindex(columns=columns, fill_value=0.0).fillna(0.0)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    confidences = probs.max(axis=1)
    return preds.tolist(), confidences.tolist()


if __name__ == "__main__":
    train()