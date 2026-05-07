import sqlite3
import json
from datetime import datetime

conn = sqlite3.connect("predictions.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    features TEXT,
    probability REAL,
    prediction INTEGER
)
""")
conn.commit()


def save_prediction(features: dict, probability: float, prediction: int):
    cursor.execute(
        "INSERT INTO predictions (timestamp, features, probability, prediction) VALUES (?, ?, ?, ?)",
        (
            datetime.utcnow().isoformat(),
            json.dumps(features),
            probability,
            prediction
        )
    )
    conn.commit()


def get_recent_predictions(limit: int = 500):
    cursor.execute("""
        SELECT features, probability, prediction
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()

    features = []
    preds = []

    for f, p, pred in rows:
        features.append(json.loads(f))
        preds.append(pred)

    return features, preds