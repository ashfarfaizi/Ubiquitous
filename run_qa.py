"""
Evaluation entry point for CS60055 Challenge 1.

Given a recording (accelerometer CSV + gyroscope CSV) and one or more
natural-language questions, print answers in the required output format.

Examples:
    python run_qa.py --question "How long was the user walking?"
    python run_qa.py --questions questions.txt
    python run_qa.py --accel path/accel.csv --gyro path/gyro.csv --question "Is the user running?"
"""

from __future__ import print_function

import argparse
import os
import sys

import pipeline
import query_engine


def _read_questions(path):
    questions = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                questions.append(line)
    return questions


def main(argv=None):
    root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Ask the Sensors — CS60055 Challenge 1 QA")
    parser.add_argument("--accel", default=os.path.join(root, "data", "raw", "demo-user-01_accel_raw.csv"))
    parser.add_argument("--gyro", default=os.path.join(root, "data", "raw", "demo-user-01_gyro_raw.csv"))
    parser.add_argument("--question", action="append", default=[])
    parser.add_argument("--questions", help="Text file with one question per line")
    args = parser.parse_args(argv)

    questions = list(args.question)
    if args.questions:
        questions.extend(_read_questions(args.questions))
    if not questions:
        default_q = os.path.join(root, "questions.txt")
        if os.path.exists(default_q):
            questions = _read_questions(default_q)
    if not questions:
        print("No questions given. Use --question or --questions.", file=sys.stderr)
        return 2

    if not os.path.exists(args.accel) or not os.path.exists(args.gyro):
        print("Sensor files not found. Run: python generate_sample_data.py", file=sys.stderr)
        return 2

    timeline, stats = pipeline.run_pipeline(args.accel, args.gyro)
    print("TIME BASE: seconds from start of recording")
    print("RECORDING: %.1f s, %d windows, %d intervals"
          % (stats["recording_duration_s"], stats["windows_classified"], stats["timeline_segments"]))
    print("")

    for i, question in enumerate(questions, 1):
        result = query_engine.answer(question, timeline, stats)
        print("Query: \"%s\"" % question)
        print(result["formatted"])
        if i != len(questions):
            print("")
            print("---")
            print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
