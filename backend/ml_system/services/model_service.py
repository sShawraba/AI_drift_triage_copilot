# backend/ml_system/services/model_service.py

import os
import time
import threading
from typing import Optional, Tuple

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


class ModelService:
    """
    Production model service.

    Responsibilities:
    - Load current Production model from MLflow
    - Cache model in memory for fast inference
    - Auto-refresh when Production model changes
    - Support manual force reload after rollback/promote
    """

    def __init__(
        self,
        model_name: str = "bank_marketing_model",
        poll_interval: int = 30
    ):
        self.model_name = model_name
        self.poll_interval = poll_interval

        # MLflow setup
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        )

        self.client = MlflowClient()

        # Cached state
        self.model = None
        self.threshold = 0.5

        self.current_production_version: Optional[str] = None
        self.current_run_id: Optional[str] = None

        # Thread safety
        self.lock = threading.Lock()

        # Initial load
        self._load_production_model(force=True)

        # Start background poller
        self._start_poller()

        print("✅ ModelService initialized")
        print(f"   Model name: {self.model_name}")
        print(f"   Poll interval: {self.poll_interval}s")
        print(f"   Current production version: {self.current_production_version}")

    # ---------------------------------------------------------
    # Get current Production model from MLflow
    # ---------------------------------------------------------
    def _get_production_model_version(self):
        """
        MLflow 2.9+ compatible production lookup.
        """

        versions = self.client.search_model_versions(
            f"name='{self.model_name}'"
        )

        production_versions = [
            v for v in versions
            if v.current_stage == "Production"
        ]

        if not production_versions:
            return None

        # Usually only one production model exists
        return production_versions[0]

    # ---------------------------------------------------------
    # Load production model
    # ---------------------------------------------------------
    def _load_production_model(self, force: bool = False) -> bool:
        """
        Load current Production model into memory.

        Returns:
            True  -> model changed/reloaded
            False -> no changes
        """

        with self.lock:

            try:
                prod_model = self._get_production_model_version()

                if prod_model is None:
                    print("❌ No Production model found")
                    return False

                version = str(prod_model.version)
                run_id = prod_model.run_id

                print("\n📦 Checking production model...")
                print(f"   MLflow Production version: {version}")
                print(f"   Currently loaded version: {self.current_production_version}")

                # Skip reload if already loaded
                if (
                    not force and
                    version == self.current_production_version
                ):
                    print("✅ No model change detected")
                    return False

                print(f"🔄 Loading Production model version {version}")

                # -----------------------------
                # Load sklearn model
                # -----------------------------
                model_uri = f"runs:/{run_id}/model"

                loaded_model = mlflow.sklearn.load_model(model_uri)

                # -----------------------------
                # Load threshold artifact
                # -----------------------------
                threshold_path = self.client.download_artifacts(
                    run_id,
                    "threshold.pkl"
                )

                loaded_threshold = joblib.load(threshold_path)

                # -----------------------------
                # Atomic swap
                # -----------------------------
                self.model = loaded_model
                self.threshold = loaded_threshold

                self.current_production_version = version
                self.current_run_id = run_id

                print("✅ Production model loaded successfully")
                print(f"   Version: {version}")
                print(f"   Run ID: {run_id}")
                print(f"   Threshold: {self.threshold:.4f}")

                return True

            except Exception as e:
                print(f"❌ Failed to load production model: {e}")
                return False

    # ---------------------------------------------------------
    # Background poller
    # ---------------------------------------------------------
    def _start_poller(self):
        """
        Background thread that keeps service synced with MLflow.
        """

        def poll_loop():

            print(
                f"📡 Poller started "
                f"(checking every {self.poll_interval}s)"
            )

            while True:

                time.sleep(self.poll_interval)

                try:
                    changed = self._load_production_model()

                    if changed:
                        print(
                            f"🔄 Auto-switched to model "
                            f"version {self.current_production_version}"
                        )

                except Exception as e:
                    print(f"❌ Poller error: {e}")

        poller_thread = threading.Thread(
            target=poll_loop,
            daemon=True
        )

        poller_thread.start()

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------
    def predict(self, df) -> Tuple[float, int]:
        """
        Predict using currently loaded production model.

        Returns:
            (probability, prediction)
        """

        with self.lock:

            if self.model is None:
                raise RuntimeError(
                    "No model loaded from MLflow."
                )

            probability = self.model.predict_proba(df)[0, 1]

            prediction = (
                1 if probability >= self.threshold else 0
            )

            return float(probability), int(prediction)

    # ---------------------------------------------------------
    # Metadata helpers
    # ---------------------------------------------------------
    def get_current_version(self) -> Optional[str]:
        return self.current_production_version

    def get_threshold(self) -> float:
        return self.threshold

    def get_current_run_id(self) -> Optional[str]:
        return self.current_run_id

    # ---------------------------------------------------------
    # Manual reload
    # ---------------------------------------------------------
    def force_reload(self) -> bool:
        """
        Force immediate reload from MLflow.

        Use after:
        - rollback
        - promote
        - retrain deployment
        """

        print("🔧 Force reload triggered")

        return self._load_production_model(force=True)


# ---------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------

model_service = ModelService()