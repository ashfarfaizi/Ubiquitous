"""
Flask web app: serves the project landing page and exposes the sensor
pipeline + query engine over HTTP.
"""

import os

from flask import Flask, jsonify, request, send_from_directory

import pipeline
import query_engine

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
ACCEL_PATH = os.path.join(DATA_DIR, "demo-user-01_accel_raw.csv")
GYRO_PATH = os.path.join(DATA_DIR, "demo-user-01_gyro_raw.csv")

_timeline = None
_stats = None


def _ensure_pipeline():
    global _timeline, _stats
    if _timeline is None:
        _timeline, _stats = pipeline.run_pipeline(ACCEL_PATH, GYRO_PATH)
    return _timeline, _stats


@app.route("/")
def home():
    return send_from_directory(os.path.dirname(__file__), "index.html")


@app.route("/api/timeline")
def api_timeline():
    timeline, stats = _ensure_pipeline()
    return jsonify({"timeline": timeline, "stats": stats})


@app.route("/api/query", methods=["POST"])
def api_query():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    timeline, _ = _ensure_pipeline()
    result = query_engine.answer(question, timeline)
    return jsonify(result)


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
