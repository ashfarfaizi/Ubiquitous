# Ask the Sensors

CS60055 Ubiquitous Computing — Hackathon Challenge 1.

Grounded activity question answering from wearable accelerometer and gyroscope streams. Answers use the required challenge fields and timestamps in **seconds from the start of the recording**.

## Setup

```bash
pip install -r requirements.txt
python generate_sample_data.py
```

`generate_sample_data.py` writes a compact 7-activity demo clip (90 seconds of each ExtraSensory class) under `data/raw/`. Do not commit ExtraSensory raw files.

## Run the QA system (required output format)

```bash
python run_qa.py --questions questions.txt
```

Or a single question:

```bash
python run_qa.py --question "How long was the user walking?"
```

On a fresh recording:

```bash
python run_qa.py --accel path/to/accel.csv --gyro path/to/gyro.csv --questions questions.txt
```

CSV columns: `timestamp_ms,x,y,z` for both accelerometer and gyroscope. Streams are resampled to 25 Hz before classification.

## Web UI

```bash
python app.py
```

Open http://127.0.0.1:5000

## Output format

Every answer is:

```
Answer: <direct answer, or N/A>
Activity/Event: <activity or event, or N/A>
Evidence:
Timestamp(s): <ranges in seconds from start, or N/A>
Sensor Modality: Accelerometer, Gyroscope
Sensor Channel(s): All
Explanation: <reasoning grounded in the observed signal, or N/A>
```

The seven classes are: lying down, sitting, standing in place, standing and moving, walking, running, bicycling.

## Pipeline

1. Match accelerometer and gyroscope timestamps  
2. Resample to 25 Hz  
3. Drop long gaps instead of inventing samples  
4. 2-second overlapping windows and signal features  
5. Random forest over the seven ExtraSensory classes  
6. Collapse windows into intervals, then answer questions from those intervals  

The query layer only reports evidence the recognizer actually found. It does not invent activity from language alone.
