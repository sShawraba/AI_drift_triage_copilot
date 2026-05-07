# prediction_store.py - IMPROVED
import sqlite3
import json
import pandas as pd
from datetime import datetime

class PredictionStore:
    def __init__(self, db_path="predictions.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_table()
    
    def _init_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                features TEXT,
                probability REAL,
                prediction INTEGER
            )
        """)
        self.conn.commit()
    
    def save(self, features: dict, probability: float, prediction: int):
        self.conn.execute(
            "INSERT INTO predictions (timestamp, features, probability, prediction) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), json.dumps(features), probability, prediction)
        )
        self.conn.commit()
    
    def get_recent_as_dataframe(self, limit: int = 1000) -> pd.DataFrame:
        """Return DataFrame with all columns ready for drift calculation."""
        cursor = self.conn.execute("""
            SELECT features, probability, prediction
            FROM predictions ORDER BY id DESC LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        if not rows:
            return pd.DataFrame()
        
        records = []
        for features_json, prob, pred in rows:
            record = json.loads(features_json)
            record["probability"] = prob
            record["prediction"] = pred
            records.append(record)
        
        return pd.DataFrame(records)

    def get_count(self) -> int:
        """Get total number of predictions stored."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM predictions")
        return cursor.fetchone()[0]

prediction_store = PredictionStore()