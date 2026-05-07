# backend/ml_system/services/registry_service.py
import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional

class RegistryService:
    def __init__(self):
        mlflow.set_tracking_uri('http://localhost:5000')
        self.client = MlflowClient()
        self.model_name = "bank_marketing_model"
    
    def model_exists(self, model_version: str) -> bool:
        """Check if model version still exists."""
        try:
            version = self.client.get_model_version(self.model_name, model_version)
            return version.status == "READY"
        except Exception:
            return False
    
    def promote_to_production(self, model_version: str, investigation_id: str) -> dict:
        """Promote a model version to Production stage."""
        # Archive current production model
        current_prod = self.client.get_latest_versions(self.model_name, stages=["Production"])
        for mv in current_prod:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=mv.version,
                stage="Archived"
            )
        
        # Promote new version to Production
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=model_version,
            stage="Production"
        )
        
        return {
            "promoted_version": model_version,
            "investigation_id": investigation_id
        }
    
    # Updated reading method
    def get_current_production_version(self) -> Optional[str]:
        """Get current Production model version (MLflow 2.9+ compatible)."""
        try:
            # Get all model versions
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            for v in versions:
                if v.current_stage == "Production":  # ← use .current_stage
                    return v.version
        except Exception:
            pass
        return None