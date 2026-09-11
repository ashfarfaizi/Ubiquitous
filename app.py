"""
Flask app for Ask the Sensors (CS60055 Challenge 1).
"""

import os

from flask import Flask, jsonify, request, send_from_directory

import pipeline
import query_engine

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data", "raw")
ACCEL_PATH = os.path.join(DATA_DIR, "demo-user-01_accel_raw.csv")
GYRO_PATH = os.path.join(DATA_DIR, "demo-user-01_gyro_raw.csv")

app = Flask(__name__, static_folder=ROOT, static_url_path="")

_timeline = None
_stats = None
_error = None


def _ensure_pipeline():
    global _timeline, _stats, _error
    if _timeline is not None:
        return _timeline, _stats
    if not os.path.exists(ACCEL_PATH) or not os.path.exists(GYRO_PATH):
        raise RuntimeError("No demo sensor files. Run: python generate_sample_data.py")
    _timeline, _stats = pipeline.run_pipeline(ACCEL_PATH, GYRO_PATH)
    return _timeline, _stats


@app.route("/")
def home():
    return send_from_directory(ROOT, "index.html")


@app.route("/api/health")
def health():
    ready = _timeline is not None
    return jsonify({
        "status": "ok",
        "pipeline_ready": ready,
        "time_base": "seconds from start of recording",
    })


@app.route("/api/timeline")
def api_timeline():
    try:
        timeline, stats = _ensure_pipeline()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({"timeline": timeline, "stats": stats})


@app.route("/api/query", methods=["POST"])
def api_query():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    try:
        timeline, stats = _ensure_pipeline()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    result = query_engine.answer(question, timeline, stats)
    return jsonify(result)


if __name__ == "__main__":
    print("Loading recognition pipeline...")
    try:
        _ensure_pipeline()
        print("Pipeline ready. Open http://127.0.0.1:5000")
    except Exception as exc:
        print("Pipeline not ready yet:", exc)
        print("Generate data with: python generate_sample_data.py")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
