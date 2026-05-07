import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd
from shared.schemas import DriftEvent
import uuid
from datetime import datetime
import requests
from backend.ml_system.services.prediction_store import get_recent_predictions

def calculate_psi(expected, actual, buckets=10):
    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.linspace(
        np.min(expected),
        np.max(expected),
        buckets + 1
    )

    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    psi = 0
    for e, a in zip(expected_percents, actual_percents):
        if e == 0 or a == 0:
            continue
        psi += (e - a) * np.log(e / a)

    return psi


def chi_square_test(expected, actual):
    table = pd.crosstab(expected, actual)
    chi2, p, _, _ = chi2_contingency(table)
    return chi2

def output_drift(expected_preds, actual_preds):
    return abs(np.mean(expected_preds) - np.mean(actual_preds))


def compute_drift(train_data, recent_data):

    numeric_features = ["age", "campaign", "pdays"]  # adjust if needed
    categorical_features = ["job", "marital", "education"]

    psi_scores = {}
    for col in numeric_features:
        psi_scores[col] = calculate_psi(
            train_data[col],
            recent_data[col]
        )

    chi2_scores = {}
    for col in categorical_features:
        chi2_scores[col] = chi_square_test(
            train_data[col],
            recent_data[col]
        )

    output = output_drift(
        train_data["prediction"],
        recent_data["prediction"]
    )

    return {
        "psi_numeric": max(psi_scores.values()),
        "chi2_categorical": max(chi2_scores.values()),
        "output_drift": output
    }

def get_severity(psi, chi2, output):

    score = max(psi, chi2, output)

    if score < 0.1:
        return "low"
    elif score < 0.25:
        return "medium"
    else:
        return "high"
    

def notify_agent(event: DriftEvent):

    requests.post(
        "http://agent:8001/webhook",
        json=event.dict()
    )   


def process_drift(train_data, recent_data, model_version):

    metrics = compute_drift(train_data, recent_data)

    severity = get_severity(
        metrics["psi_numeric"],
        metrics["chi2_categorical"],
        metrics["output_drift"]
    )

    event = DriftEvent(
        event_id=str(uuid.uuid4()),
        timestamp=datetime.utcnow(),
        model_version=model_version,
        psi_numeric=metrics["psi_numeric"],
        chi2_categorical=metrics["chi2_categorical"],
        output_drift=metrics["output_drift"],
        severity=severity
    )

    notify_agent(event)

def run_drift_check(train_data, model_version):

    # 1. get live data from DB
    recent_features, recent_preds = get_recent_predictions(limit=200)

    # safety check
    if len(recent_features) < 50:
        return None  # not enough data yet

    # 2. compute drift on real data
    metrics = compute_drift(train_data, recent_features)

    # 3. add output drift (important fix)
    metrics["output_drift"] = output_drift(
        [0] * len(recent_preds),   # baseline approximation
        recent_preds
    )

    # 4. trigger full pipeline
    process_drift(train_data, recent_features, model_version)

    return metrics