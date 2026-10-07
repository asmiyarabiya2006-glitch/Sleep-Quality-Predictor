from flask import Flask, render_template, request
import joblib
import pandas as pd
import csv
import os
from datetime import datetime

app = Flask(__name__)
model = joblib.load("model/sleep_model.pkl")

HISTORY_FILE = "data/history.csv"


def save_prediction(sleep_duration, activity, stress, result):
    file_exists = os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "Sleep Duration", "Exercise", "Stress", "Result"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            sleep_duration, activity, stress, result
        ])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    sleep_duration = float(request.form["sleep_duration"])
    activity = float(request.form["activity"])
    stress = float(request.form["stress"])

    data = pd.DataFrame(
        [[sleep_duration, activity, stress]],
        columns=["Sleep Duration", "Physical Activity Level", "Stress Level"]
    )

    result = model.predict(data)[0]
    save_prediction(sleep_duration, activity, stress, result)

    tips = []
    if sleep_duration < 7:
        tips.append("Try to sleep at least 7 to 8 hours each night.")
    if activity < 30:
        tips.append("Increase your exercise to at least 30 minutes a day.")
    if stress >= 7:
        tips.append("Your stress is high. Try deep breathing or meditation before bed.")
    if not tips:
        tips.append("Great habits! Keep up your current routine.")

    return render_template("index.html", result=result, tips=tips)


@app.route("/history")
def history():
    rows = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, newline="") as f:
            rows = list(csv.DictReader(f))
        rows.reverse()
    return render_template("history.html", rows=rows)


if __name__ == "__main__":
    app.run(debug=True)