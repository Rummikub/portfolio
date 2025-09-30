"""
Data synchronization module for SyncFit-Storyblok integration
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import uuid
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.feature_builder import build_user_features
from src.guardian_alerts import generate_guardian_alerts
from src.llm_alerts import generate_alert_message
from storyblok_integration.client import storyblok_client
from storyblok_integration.models import (
    UserProfile, HealthMetrics, Alert, ChurnPrediction,
    Intervention, AlertSeverity, InterventionType
)
import joblib

logger = logging.getLogger(__name__)

class DataSyncManager:
    """Manages data synchronization between SyncFit and Storyblok"""
    
    def __init__(self):
        self.client = storyblok_client
        self.model_path = "models/syncfit_churn_model.pkl"
        self.churn_model = None
        self._load_churn_model()
        
    def _load_churn_model(self):
        """Load the trained churn model"""
        try:
            if Path(self.model_path).exists():
                self.churn_model = joblib.load(self.model_path)
                logger.info("Churn model loaded successfully")
            else:
                logger.warning("Churn model not found. Train the model first.")
        except Exception as e:
            logger.error(f"Failed to load churn model: {e}")
            
    def sync_wearable_data_to_storyblok(self, csv_path: str = "data/synthetic_wearable_logs.csv"):
        """Sync wearable data from CSV to Storyblok"""
        try:
            df = pd.read_csv(csv_path, parse_dates=['date'])
            
            # Group by user and date for daily metrics
            daily_metrics = df.groupby(['user_id', 'date']).agg({
                'steps': 'mean',
                'heart_rate': 'mean',
                'class_attended': 'max'
            }).reset_index()
            
            metrics_objects = []
            for _, row in daily_metrics.iterrows():
                metrics = HealthMetrics(
                    user_id=row['user_id'],
                    timestamp=pd.Timestamp(row['date']),
                    steps=int(row['steps']),
                    heart_rate=int(row['heart_rate']),
                    class_attended=bool(row['class_attended'])
                )
                metrics_objects.append(metrics)
                
            # Batch create metrics in Storyblok
            results = self.client.batch_create_metrics(metrics_objects)
            logger.info(f"Synced {len(results)} health metrics to Storyblok")
            return results
            
        except Exception as e:
            logger.error(f"Failed to sync wearable data: {e}")
            return []
            
    def create_user_profiles_from_data(self, csv_path: str = "data/synthetic_wearable_logs.csv"):
        """Create user profiles in Storyblok from wearable data"""
        try:
            df = pd.read_csv(csv_path)
            user_ids = df['user_id'].unique()
            
            profiles_created = []
            profiles_existing = []
            for user_id in user_ids:
                # Check if profile already exists
                try:
                    existing = self.client.get_story(f"users/{user_id}")
                    if existing:
                        logger.info(f"User profile {user_id} already exists")
                        profiles_existing.append(user_id)
                        continue
                except:
                    # Profile doesn't exist, we can create it
                    pass
                    
                # Create new profile
                profile = UserProfile(
                    user_id=user_id,
                    name=f"User {user_id.split('_')[1]}",
                    email=f"{user_id}@syncfit.com",
                    enrollment_date=datetime.now() - timedelta(days=30),
                    is_active=True,
                    device_id=f"device_{user_id}"
                )
                
                result = self.client.create_user_profile(profile)
                profiles_created.append(result)
                logger.info(f"Created profile for {user_id}")
                
            if profiles_existing:
                logger.info(f"Found {len(profiles_existing)} existing profiles (skipped)")
            logger.info(f"Created {len(profiles_created)} new user profiles in Storyblok")
            return profiles_created
            
        except Exception as e:
            logger.error(f"Failed to create user profiles: {e}")
            return []
            
    def sync_guardian_alerts(self, csv_path: str = "data/synthetic_wearable_logs.csv"):
        """Generate and sync guardian alerts to Storyblok"""
        try:
            # Generate alerts using existing logic
            alerts_df = generate_guardian_alerts(csv_path)
            
            alerts_created = []
            for _, row in alerts_df.iterrows():
                if row['alert'] == "Normal Activity - No alerts":
                    continue
                    
                # Determine severity based on alert message
                severity = AlertSeverity.LOW
                intervention_type = InterventionType.USER_NOTIFICATION
                
                if "HIGH : Call 911" in row['alert']:
                    severity = AlertSeverity.EMERGENCY
                    intervention_type = InterventionType.EMERGENCY_CALL
                elif "MIDDLE : Notify Doctor" in row['alert']:
                    severity = AlertSeverity.HIGH
                    intervention_type = InterventionType.DOCTOR_NOTIFICATION
                elif "LOW : Ping User" in row['alert']:
                    severity = AlertSeverity.MEDIUM
                    intervention_type = InterventionType.USER_NOTIFICATION
                    
                # Generate LLM summary if OpenAI is configured
                llm_summary = None
                try:
                    llm_summary = generate_alert_message(
                        row['user_id'],
                        row['deviation'],
                        row['inactive_hours']
                    )
                except:
                    llm_summary = "LLM summary unavailable"
                    
                # Create alert object
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    user_id=row['user_id'],
                    timestamp=datetime.now(),
                    severity=severity,
                    alert_type="inactivity_alert",
                    message=row['alert'],
                    metrics_snapshot={
                        "steps": row['steps'],
                        "steps_avg": row['steps_avg'],
                        "deviation": row['deviation']
                    },
                    inactive_hours=row['inactive_hours'],
                    activity_deviation=row['deviation'],
                    recommended_action=row['alert'],
                    llm_summary=llm_summary,
                    is_resolved=False
                )
                
                # Create alert in Storyblok
                alert_result = self.client.create_alert(alert)
                alerts_created.append(alert_result)
                
                # Create intervention if needed
                if severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                    intervention = Intervention(
                        intervention_id=str(uuid.uuid4()),
                        user_id=row['user_id'],
                        alert_id=alert.alert_id,
                        timestamp=datetime.now(),
                        intervention_type=intervention_type,
                        severity=severity,
                        description=f"Automated intervention for {row['alert']}",
                        automated=True,
                        recipient="emergency" if severity == AlertSeverity.EMERGENCY else "doctor",
                        contact_method="phone" if severity == AlertSeverity.EMERGENCY else "email",
                        message_sent=llm_summary or row['alert'],
                        follow_up_required=True,
                        follow_up_date=datetime.now() + timedelta(hours=24)
                    )
                    
                    self.client.create_intervention(intervention)
                    
            logger.info(f"Created {len(alerts_created)} alerts in Storyblok")
            return alerts_created
            
        except Exception as e:
            logger.error(f"Failed to sync guardian alerts: {e}")
            return []
            
    def sync_churn_predictions(self, csv_path: str = "data/synthetic_wearable_logs.csv"):
        """Generate and sync churn predictions to Storyblok"""
        if not self.churn_model:
            logger.error("Churn model not loaded. Cannot generate predictions.")
            return []
            
        try:
            # Build features
            X, y = build_user_features(csv_path)
            
            # Generate predictions
            predictions = self.churn_model.predict_proba(X)[:, 1]
            feature_importance = dict(zip(X.columns, self.churn_model.feature_importances_))
            
            predictions_created = []
            for idx, (_, row) in enumerate(X.iterrows()):
                user_id = row['user_id'] if 'user_id' in row else f"user_{idx}"
                churn_prob = predictions[idx]
                
                # Determine risk level
                risk_level = "low"
                if churn_prob > 0.7:
                    risk_level = "high"
                elif churn_prob > 0.4:
                    risk_level = "medium"
                    
                # Identify contributing factors
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                contributing_factors = [f[0] for f in top_features]
                
                # Generate recommendations
                recommendations = []
                if risk_level == "high":
                    recommendations = [
                        "Immediate engagement required",
                        "Personal trainer consultation",
                        "Offer incentive program"
                    ]
                elif risk_level == "medium":
                    recommendations = [
                        "Send motivational messages",
                        "Suggest group activities",
                        "Review fitness goals"
                    ]
                else:
                    recommendations = ["Continue monitoring", "Send weekly tips"]
                    
                prediction = ChurnPrediction(
                    prediction_id=str(uuid.uuid4()),
                    user_id=user_id,
                    timestamp=datetime.now(),
                    churn_probability=float(churn_prob),
                    churn_risk_level=risk_level,
                    feature_importance=feature_importance,
                    model_version="1.0",
                    prediction_horizon_days=30,
                    contributing_factors=contributing_factors,
                    recommended_interventions=recommendations,
                    confidence_score=float(max(predictions[idx], 1 - predictions[idx]))
                )
                
                result = self.client.create_churn_prediction(prediction)
                predictions_created.append(result)
                
            logger.info(f"Created {len(predictions_created)} churn predictions in Storyblok")
            return predictions_created
            
        except Exception as e:
            logger.error(f"Failed to sync churn predictions: {e}")
            return []
            
    def full_sync(self, csv_path: str = "data/synthetic_wearable_logs.csv"):
        """Perform full data synchronization"""
        logger.info("Starting full data synchronization to Storyblok...")
        
        results = {
            "user_profiles": self.create_user_profiles_from_data(csv_path),
            "health_metrics": self.sync_wearable_data_to_storyblok(csv_path),
            "alerts": self.sync_guardian_alerts(csv_path),
            "predictions": self.sync_churn_predictions(csv_path)
        }
        
        logger.info("Full synchronization completed")
        return results
        
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current synchronization status"""
        try:
            status = {
                "users": len(self.client.get_stories(starts_with="users/")),
                "metrics": len(self.client.get_stories(starts_with="metrics/")),
                "alerts": len(self.client.get_stories(starts_with="alerts/")),
                "predictions": len(self.client.get_stories(starts_with="predictions/")),
                "interventions": len(self.client.get_stories(starts_with="interventions/")),
                "active_alerts": len([a for a in self.client.get_stories(starts_with="alerts/") 
                                     if not a.get("content", {}).get("is_resolved", False)]),
                "active_interventions": len(self.client.get_active_interventions())
            }
            return status
        except Exception as e:
            logger.error(f"Failed to get sync status: {e}")
            return {}

# Create singleton instance
data_sync_manager = DataSyncManager()

if __name__ == "__main__":
    # Test synchronization
    manager = DataSyncManager()
    
    # Perform full sync
    results = manager.full_sync()
    print(f"Synchronization results: {results}")
    
    # Get status
    status = manager.get_sync_status()
    print(f"Current status: {status}")
