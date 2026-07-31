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
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from generate_sample_data import ACTIVITIES, _make_burst, RNG
import preprocessing as prep

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "model.joblib")


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
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({"model": clf, "columns": list(X.columns)}, MODEL_PATH)
        print(f"saved model -> {MODEL_PATH}")

    return clf, list(X.columns)


def load_model():
    if not os.path.exists(MODEL_PATH):
        return train(save=True)
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["columns"]


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