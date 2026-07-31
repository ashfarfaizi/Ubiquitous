# Ubiquitous Computing Project

Sensor-based activity recognition pipeline: raw accelerometer + gyroscope data is preprocessed, classified into seven activity types, and exposed through a query engine.

## Setup

```bash
pip install -r requirements.txt
python generate_sample_data.py
```

## Run

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### API

- `GET /api/timeline` — run the pipeline and return the activity timeline
- `POST /api/query` — ask a question, e.g. `{"question": "how long did I spend walking?"}`
- `GET /api/health` — health check

### CLI

```bash
python pipeline.py
python classifier.py
```

## Team

Ubiquitous Computing course project.
