# ML System - Bank Marketing Classifier

ML service for UCI Bank Marketing dataset with drift detection and agent integration.

## Features

- Binary classifier with threshold tuning (recall ≥ 0.75)
- MLflow model registry integration
- REST API predictions with Pydantic validation
- Drift detection (PSI for numeric, Chi² for categorical)
- Webhook alerts to agent on drift severity change
- Auto-polling for production model changes (30s interval)
- Idempotent action endpoint for rollback/retrain/replay
- Persistent prediction storage (SQLite)
- Metrics endpoint for dashboard

## Prerequisites

- Python 3.12+
- MLflow server running on `http://localhost:5000`
- (Optional) Agent service for drift webhooks

## Installation

```bash
# Clone repository
git clone <your-repo>
cd AI-drift-triage-copilot

# Install dependencies
uv sync

# Set environment variables
cp .env.example .env
# Edit .env with your configuration