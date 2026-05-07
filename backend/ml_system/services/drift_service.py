# backend/ml_system/services/drift_service.py
import numpy as np
import pandas as pd
import httpx
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from shared.schemas import DriftEvent


class DriftService:
    """Drift detection service that alerts agent via webhook when severity changes."""
    
    def __init__(
        self, 
        reference_data: pd.DataFrame, 
        agent_base_url: str,
        model_version: str
    ):
        """
        Args:
            reference_data: Validation set WITH predictions (from training time)
            agent_webhook_url: Where to send drift alerts (e.g., http://agent:8000/webhook/drift-event)
            model_version: Current model version string
        """
        # Ensure URL has the correct endpoint path
        self.agent_url = f"{agent_base_url.rstrip('/')}/webhook/drift-event"
        print(f"📡 Drift service will send webhooks to: {self.agent_url}")
        
        self.model_version = model_version
        self.last_severity: Optional[str] = None
        
        # Define columns for drift detection
        self.numeric_cols = ["age", "campaign", "euribor3m", "cons.price.idx", "emp.var.rate"]
        self.categorical_cols = ["job", "marital", "education", "poutcome", "housing"]
        
        # Pre-compute reference statistics
        self.ref_numeric_stats = {
            col: reference_data[col].values for col in self.numeric_cols 
            if col in reference_data.columns
        }
        self.ref_categorical_counts = {
            col: reference_data[col].value_counts().to_dict() 
            for col in self.categorical_cols 
            if col in reference_data.columns
        }
        self.ref_pos_rate = reference_data["prediction"].mean() if "prediction" in reference_data.columns else 0.11
        
        # Severity thresholds
        self.PSI_LOW = 0.08
        self.PSI_MEDIUM = 0.2
        self.CHI2_LOW = 0.5
        self.CHI2_MEDIUM = 2.0
        self.OUTPUT_LOW = 0.05
        self.OUTPUT_MEDIUM = 0.12
    
    def calculate_psi(self, expected_vals, actual_vals, buckets=10) -> float:
        """Population Stability Index for numeric features."""
        expected = np.array(expected_vals)
        actual = np.array(actual_vals)
        
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        
        # Use expected's distribution to define bins
        bins = np.linspace(expected.min(), expected.max(), buckets + 1)
        
        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)
        
        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)
        
        # Avoid log(0)
        expected_pct = np.maximum(expected_pct, 1e-5)
        actual_pct = np.maximum(actual_pct, 1e-5)
        
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)
    
    def calculate_chi2_distance(self, expected_counts: dict, actual_counts: dict) -> float:
        """Chi-square distance for categorical features."""
        all_categories = set(expected_counts.keys()) | set(actual_counts.keys())
        
        expected_arr = [expected_counts.get(cat, 0) for cat in all_categories]
        actual_arr = [actual_counts.get(cat, 0) for cat in all_categories]
        
        total_expected = sum(expected_arr)
        total_actual = sum(actual_arr)
        
        if total_expected == 0 or total_actual == 0:
            return 0.0
        
        expected_pct = np.array([e / total_expected for e in expected_arr])
        actual_pct = np.array([a / total_actual for a in actual_arr])
        
        # Chi-square distance (symmetric)
        chi2 = np.sum((expected_pct - actual_pct) ** 2 / (expected_pct + 1e-5))
        return float(chi2)
    
    def compute_drift_report(self, live_data: pd.DataFrame, min_samples: int = 50) -> Dict[str, Any]: # 50 is minimum samples needed to trigger alert
        """Compute drift metrics comparing live data to reference."""
        if len(live_data) < min_samples:
            return {"enough_data": False, "message": f"Only {len(live_data)} samples, need {min_samples}"}
        
        # PSI for numeric features
        psi_values = []
        for col in self.numeric_cols:
            if col in live_data.columns and col in self.ref_numeric_stats:
                psi = self.calculate_psi(
                    self.ref_numeric_stats[col],
                    live_data[col].values
                )
                psi_values.append(psi)
        
        avg_psi = np.mean(psi_values) if psi_values else 0.0
        
        # Chi2 for categorical features
        chi2_values = []
        for col in self.categorical_cols:
            if col in live_data.columns and col in self.ref_categorical_counts:
                actual_counts = live_data[col].value_counts().to_dict()
                chi2 = self.calculate_chi2_distance(
                    self.ref_categorical_counts[col],
                    actual_counts
                )
                chi2_values.append(chi2)
        
        avg_chi2 = np.mean(chi2_values) if chi2_values else 0.0
        
        # Output drift (prediction rate)
        live_pos_rate = live_data["prediction"].mean() if "prediction" in live_data.columns else 0.0
        output_drift = abs(self.ref_pos_rate - live_pos_rate)
        
        return {
            "enough_data": True,
            "psi_numeric": avg_psi,
            "chi2_categorical": avg_chi2,
            "output_drift": output_drift,
            "sample_size": len(live_data),
            "ref_pos_rate": self.ref_pos_rate,
            "live_pos_rate": live_pos_rate
        }
    
    def get_severity(self, psi: float, chi2: float, output_drift: float) -> str:
        """Determine severity based on thresholds."""
        # Normalize output drift to comparable scale
        normalized_output = output_drift * 2.0
        
        # Check each metric
        is_high = (psi > self.PSI_MEDIUM or chi2 > self.CHI2_MEDIUM or normalized_output > self.OUTPUT_MEDIUM)
        is_medium = (psi > self.PSI_LOW or chi2 > self.CHI2_LOW or normalized_output > self.OUTPUT_LOW)
        
        if is_high:
            return "high"
        elif is_medium:
            return "medium"
        else:
            return "low"
    
    async def _send_webhook(self, event: DriftEvent) -> bool:
        """Send drift event to agent's webhook endpoint."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.post(
                    self.agent_url,
                    json=event.model_dump()
                )
                response.raise_for_status()
                print(f"✅ Webhook sent to {self.agent_url}")
                print(f"   Event: {event.event_id} - Severity: {event.severity}")
                return True
            except httpx.HTTPStatusError as e:
                print(f"❌ Webhook HTTP error: {e.response.status_code} - {e.response.text}")
                return False
            except httpx.RequestError as e:
                print(f"❌ Webhook request failed: {e}")
                return False
            except Exception as e:
                print(f"❌ Webhook unexpected error: {e}")
                return False
    
    async def check_and_alert(self, live_data: pd.DataFrame) -> Optional[DriftEvent]:
        """Main entry point. Checks drift and sends webhook to agent if severity changed."""
        
        print(f"🔍 Drift check running with {len(live_data)} samples")  # ← ADD THIS

        report = self.compute_drift_report(live_data)
        
        if not report["enough_data"]:
            print(f"⚠️ {report.get('message')}") 
            return None
        
        severity = self.get_severity(
            report["psi_numeric"],
            report["chi2_categorical"],
            report["output_drift"]
        )
        
        print(f"Drift check - PSI: {report['psi_numeric']:.4f}, Chi2: {report['chi2_categorical']:.4f}, Output: {report['output_drift']:.4f}")
        print(f"Severity: {severity} (last: {self.last_severity})")
        
        # Only alert if severity changed AND not low (or changed from low to medium/high)
        should_alert = (
            severity != self.last_severity and 
            severity != "low"
        ) or (
            self.last_severity == "low" and severity != "low"
        )
        
        if not should_alert:
            print(f"No alert sent - severity unchanged or low")
            return None
        
        # Create drift event matching shared schema
        event = DriftEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            model_version=self.model_version,
            psi_numeric=report["psi_numeric"],
            chi2_categorical=report["chi2_categorical"],
            output_drift=report["output_drift"],
            severity=severity
        )
        
        # Send webhook to agent
        success = await self._send_webhook(event)
        
        if success:
            self.last_severity = severity
            return event
        else:
            # Don't update last_severity - will retry on next check
            print("Webhook failed, will retry on next drift check")
            return None
    
    def get_last_severity(self) -> Optional[str]:
        """For dashboard to show current drift status."""
        return self.last_severity

drift_service = None
