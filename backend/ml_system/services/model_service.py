# backend/ml_system/services/model_service.py
import os
import mlflow
import mlflow.sklearn
import joblib
import threading
import time
from mlflow.tracking import MlflowClient
from typing import Optional, Tuple


class ModelService:
    """
    Model service that polls MLflow every 30 seconds for production model changes.
    When agent promotes/rollbacks a model, this service automatically switches within 30 seconds.
    """
    
    def __init__(self, model_name: str = "bank_marketing_model", poll_interval: int = 30):
        """
        Args:
            model_name: Name of registered model in MLflow
            poll_interval: How often to check for production changes (seconds)
        """
        self.model_name = model_name
        self.poll_interval = poll_interval
        
        # Set tracking URI from environment (fallback to localhost)
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        self.client = MlflowClient()
        
        # Current state (cached)
        self.model = None
        self.threshold = 0.5
        self.current_production_version: Optional[str] = None
        self.current_run_id: Optional[str] = None
        
        # Initial load
        self._load_production_model()
        
        # Start background poller
        self._start_poller()
        
        print(f"✅ ModelService initialized for '{model_name}'")
        print(f"   Polling every {poll_interval} seconds")
        print(f"   Initial production version: {self.current_production_version}")
    
    def _load_production_model(self) -> bool:
        """
        Load whatever model is currently in Production stage from MLflow.
        Returns True if model changed, False if same.
        """
        try:
            # Get the model version in Production stage
            prod_versions = self.client.get_latest_versions(self.model_name, stages=["Production"])
            
            if not prod_versions:
                # No production model yet, fall back to latest version
                print(f"⚠️ No Production model found, falling back to latest version")
                prod_versions = self.client.get_latest_versions(self.model_name, stages=["None"])
                
                if not prod_versions:
                    print(f"❌ No model found with name '{self.model_name}'")
                    return False
            
            version = prod_versions[0].version
            run_id = prod_versions[0].run_id
            
            # Check if this is already loaded
            if version == self.current_production_version:
                return False  # No change
            
            # Load the new model
            print(f"🔄 Loading production model: version {version} (was: {self.current_production_version})")
            
            model_uri = f"runs:/{run_id}/model"
            self.model = mlflow.sklearn.load_model(model_uri)
            
            # Load threshold from same run
            threshold_path = self.client.download_artifacts(run_id, "threshold.pkl")
            self.threshold = joblib.load(threshold_path)
            
            # Update state
            self.current_production_version = version
            self.current_run_id = run_id
            
            print(f"✅ Loaded model version {version} with threshold {self.threshold:.4f}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load production model: {e}")
            return False
    
    def _start_poller(self):
        """Start background thread that polls MLflow for production changes."""
        def poll_loop():
            print(f"📡 Poller thread started (checking every {self.poll_interval}s)")
            while True:
                time.sleep(self.poll_interval)
                try:
                    changed = self._load_production_model()
                    if changed:
                        print(f"🔄 Auto-switched to model version {self.current_production_version}")
                except Exception as e:
                    print(f"❌ Poller error: {e}")
        
        poller_thread = threading.Thread(target=poll_loop, daemon=True)
        poller_thread.start()
    
    def predict(self, df) -> Tuple[float, int]:
        """
        Make prediction using currently cached production model.
        Returns (probability, prediction)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Check MLflow connection.")
        
        probability = self.model.predict_proba(df)[0, 1]
        prediction = 1 if probability >= self.threshold else 0
        
        return probability, prediction
    
    def get_current_version(self) -> Optional[str]:
        """Return current production model version."""
        return self.current_production_version
    
    def get_threshold(self) -> float:
        """Return current threshold."""
        return self.threshold
    
    def force_reload(self):
        """Force an immediate reload (useful for testing)."""
        print("🔧 Force reload triggered")
        self._load_production_model()


# Single instance for the whole app
model_service = ModelService()