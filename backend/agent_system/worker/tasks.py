"""
Implementations of the slow actions that the worker executes.
In a real system these would call the model service API.
"""
import time

def retrain(investigation_id: str, model_version: str, **kwargs):
    """Simulate a retrain job."""
    print(f"⏳ Retraining model v{model_version} (investigation {investigation_id})...")
    time.sleep(2)   # simulate work
    print(f"✅ Retrain complete for model v{model_version}")

def rollback(investigation_id: str, model_version: str, **kwargs):
    """Simulate a rollback job."""
    print(f"⏳ Rolling back model to v{model_version} (investigation {investigation_id})...")
    time.sleep(2)
    print(f"✅ Rollback complete to v{model_version}")

def replay(investigation_id: str, model_version: str, **kwargs):
    """Simulate replaying the test set."""
    print(f"⏳ Replaying test set for model v{model_version} (investigation {investigation_id})...")
    time.sleep(2)
    print(f"✅ Replay complete for model v{model_version}")

TASK_MAP = {
    "retrain": retrain,
    "rollback": rollback,
    "replay": replay,
}