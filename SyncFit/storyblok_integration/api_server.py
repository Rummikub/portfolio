"""
FastAPI server for SyncFit-Storyblok integration
Provides REST API endpoints and webhook handlers
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import uuid
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from storyblok_integration.data_sync import data_sync_manager
from storyblok_integration.models import (
    UserProfile, HealthMetrics, Alert, ChurnPrediction,
    Intervention, WebhookPayload, AnalyticsReport,
    AlertSeverity, InterventionType
)
from storyblok_integration.client import storyblok_client
from src.llm_alerts import generate_alert_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SyncFit-Storyblok Integration API",
    description="API for managing wearable health data and Storyblok content",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class SyncRequest(BaseModel):
    """Request model for data synchronization"""
    csv_path: Optional[str] = "data/synthetic_wearable_logs.csv"
    sync_type: str = "full"  # full, metrics, alerts, predictions

class AlertRequest(BaseModel):
    """Request model for creating alerts"""
    user_id: str
    severity: str
    message: str
    inactive_hours: float
    activity_deviation: float

class InterventionRequest(BaseModel):
    """Request model for creating interventions"""
    user_id: str
    alert_id: Optional[str]
    intervention_type: str
    severity: str
    description: str
    recipient: str
    contact_method: str

class MetricsUpload(BaseModel):
    """Request model for uploading metrics"""
    user_id: str
    steps: int
    heart_rate: int
    class_attended: bool = False
    timestamp: Optional[datetime] = None

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "SyncFit-Storyblok Integration",
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "sync": "/api/sync",
            "users": "/api/users",
            "metrics": "/api/metrics",
            "alerts": "/api/alerts",
            "predictions": "/api/predictions",
            "interventions": "/api/interventions",
            "analytics": "/api/analytics",
            "webhooks": "/webhook/*"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        status = data_sync_manager.get_sync_status()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "sync_status": status
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

# Synchronization endpoints

@app.post("/api/sync")
async def sync_data(request: SyncRequest, background_tasks: BackgroundTasks):
    """Trigger data synchronization with Storyblok"""
    try:
        if request.sync_type == "full":
            background_tasks.add_task(
                data_sync_manager.full_sync,
                request.csv_path
            )
            return {"message": "Full synchronization started", "status": "processing"}
        
        elif request.sync_type == "metrics":
            results = data_sync_manager.sync_wearable_data_to_storyblok(request.csv_path)
            return {"message": "Metrics synchronized", "count": len(results)}
        
        elif request.sync_type == "alerts":
            results = data_sync_manager.sync_guardian_alerts(request.csv_path)
            return {"message": "Alerts synchronized", "count": len(results)}
        
        elif request.sync_type == "predictions":
            results = data_sync_manager.sync_churn_predictions(request.csv_path)
            return {"message": "Predictions synchronized", "count": len(results)}
        
        else:
            raise HTTPException(status_code=400, detail="Invalid sync type")
            
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sync/status")
async def get_sync_status():
    """Get current synchronization status"""
    try:
        status = data_sync_manager.get_sync_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get sync status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# User endpoints

@app.get("/api/users")
async def get_users():
    """Get all user profiles from Storyblok"""
    try:
        users = storyblok_client.get_stories(starts_with="users/")
        return {"users": users, "count": len(users)}
    except Exception as e:
        logger.error(f"Failed to get users: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}")
async def get_user(user_id: str):
    """Get specific user profile"""
    try:
        user = storyblok_client.get_story(f"users/{user_id}")
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except Exception as e:
        logger.error(f"Failed to get user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/users")
async def create_user(user: UserProfile):
    """Create a new user profile"""
    try:
        result = storyblok_client.create_user_profile(user)
        return {"message": "User created", "data": result}
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Metrics endpoints

@app.post("/api/metrics")
async def upload_metrics(metrics: MetricsUpload):
    """Upload health metrics for a user"""
    try:
        health_metrics = HealthMetrics(
            user_id=metrics.user_id,
            timestamp=metrics.timestamp or datetime.now(),
            steps=metrics.steps,
            heart_rate=metrics.heart_rate,
            class_attended=metrics.class_attended
        )
        result = storyblok_client.create_health_metrics(health_metrics)
        return {"message": "Metrics uploaded", "data": result}
    except Exception as e:
        logger.error(f"Failed to upload metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/{user_id}")
async def get_user_metrics(user_id: str, limit: int = 100):
    """Get health metrics for a specific user"""
    try:
        metrics = storyblok_client.get_user_metrics(user_id, limit)
        return {"metrics": metrics, "count": len(metrics)}
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Alert endpoints

@app.get("/api/alerts")
async def get_alerts(active_only: bool = True):
    """Get all alerts"""
    try:
        alerts = storyblok_client.get_stories(starts_with="alerts/")
        if active_only:
            alerts = [a for a in alerts if not a.get("content", {}).get("is_resolved", False)]
        return {"alerts": alerts, "count": len(alerts)}
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts/{user_id}")
async def get_user_alerts(user_id: str, active_only: bool = True):
    """Get alerts for a specific user"""
    try:
        alerts = storyblok_client.get_user_alerts(user_id, active_only)
        return {"alerts": alerts, "count": len(alerts)}
    except Exception as e:
        logger.error(f"Failed to get user alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/alerts")
async def create_alert(alert_request: AlertRequest):
    """Create a new alert"""
    try:
        # Generate LLM summary
        llm_summary = None
        try:
            llm_summary = generate_alert_message(
                alert_request.user_id,
                alert_request.activity_deviation,
                alert_request.inactive_hours
            )
        except:
            llm_summary = "LLM summary unavailable"
        
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            user_id=alert_request.user_id,
            timestamp=datetime.now(),
            severity=AlertSeverity(alert_request.severity),
            alert_type="manual_alert",
            message=alert_request.message,
            metrics_snapshot={},
            inactive_hours=alert_request.inactive_hours,
            activity_deviation=alert_request.activity_deviation,
            recommended_action="Review user status",
            llm_summary=llm_summary,
            is_resolved=False
        )
        
        result = storyblok_client.create_alert(alert)
        return {"message": "Alert created", "data": result}
    except Exception as e:
        logger.error(f"Failed to create alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, resolved_by: str = "system"):
    """Resolve an alert"""
    try:
        result = storyblok_client.update_alert_status(alert_id, resolved=True, resolved_by=resolved_by)
        if not result:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"message": "Alert resolved", "data": result}
    except Exception as e:
        logger.error(f"Failed to resolve alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Prediction endpoints

@app.get("/api/predictions")
async def get_predictions():
    """Get all churn predictions"""
    try:
        predictions = storyblok_client.get_stories(starts_with="predictions/")
        return {"predictions": predictions, "count": len(predictions)}
    except Exception as e:
        logger.error(f"Failed to get predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/{user_id}")
async def get_user_predictions(user_id: str):
    """Get churn predictions for a specific user"""
    try:
        predictions = storyblok_client.search_stories(user_id, content_type="churn_prediction")
        return {"predictions": predictions, "count": len(predictions)}
    except Exception as e:
        logger.error(f"Failed to get user predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Intervention endpoints

@app.get("/api/interventions")
async def get_interventions(active_only: bool = True):
    """Get all interventions"""
    try:
        if active_only:
            interventions = storyblok_client.get_active_interventions()
        else:
            interventions = storyblok_client.get_stories(starts_with="interventions/")
        return {"interventions": interventions, "count": len(interventions)}
    except Exception as e:
        logger.error(f"Failed to get interventions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/interventions")
async def create_intervention(intervention_request: InterventionRequest):
    """Create a new intervention"""
    try:
        intervention = Intervention(
            intervention_id=str(uuid.uuid4()),
            user_id=intervention_request.user_id,
            alert_id=intervention_request.alert_id,
            timestamp=datetime.now(),
            intervention_type=InterventionType(intervention_request.intervention_type),
            severity=AlertSeverity(intervention_request.severity),
            description=intervention_request.description,
            automated=False,
            recipient=intervention_request.recipient,
            contact_method=intervention_request.contact_method,
            message_sent=intervention_request.description,
            follow_up_required=True,
            follow_up_date=datetime.now() + timedelta(hours=24)
        )
        
        result = storyblok_client.create_intervention(intervention)
        return {"message": "Intervention created", "data": result}
    except Exception as e:
        logger.error(f"Failed to create intervention: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints

@app.get("/api/analytics/report")
async def get_analytics_report(days: int = 7):
    """Generate analytics report"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get all relevant data
        users = storyblok_client.get_stories(starts_with="users/")
        alerts = storyblok_client.get_stories(starts_with="alerts/")
        predictions = storyblok_client.get_stories(starts_with="predictions/")
        interventions = storyblok_client.get_stories(starts_with="interventions/")
        
        # Filter by date range
        alerts = [a for a in alerts if start_date <= datetime.fromisoformat(
            a.get("content", {}).get("timestamp", "2000-01-01")) <= end_date]
        
        # Calculate statistics
        alerts_by_severity = {}
        for alert in alerts:
            severity = alert.get("content", {}).get("severity", "unknown")
            alerts_by_severity[severity] = alerts_by_severity.get(severity, 0) + 1
        
        # Identify at-risk users
        at_risk_users = set()
        for pred in predictions:
            if pred.get("content", {}).get("churn_risk_level") in ["high", "medium"]:
                at_risk_users.add(pred.get("content", {}).get("user_id"))
        
        report = AnalyticsReport(
            report_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            period_start=start_date,
            period_end=end_date,
            total_users=len(users),
            active_users=len(users),  # Simplified for demo
            churned_users=0,  # Would need actual churn data
            at_risk_users=len(at_risk_users),
            total_alerts=len(alerts),
            alerts_by_severity=alerts_by_severity,
            interventions_triggered=len(interventions),
            intervention_success_rate=0.75,  # Placeholder
            average_response_time_minutes=30.0,  # Placeholder
            top_risk_factors=[
                {"factor": "Low activity", "impact": 0.45},
                {"factor": "Missed classes", "impact": 0.35},
                {"factor": "Irregular heart rate", "impact": 0.20}
            ],
            recommendations=[
                "Increase engagement for high-risk users",
                "Review alert thresholds",
                "Implement proactive wellness checks"
            ]
        )
        
        return report.dict()
    except Exception as e:
        logger.error(f"Failed to generate analytics report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Webhook endpoints

@app.post("/webhook/data-update")
async def webhook_data_update(request: Request):
    """Handle data update webhooks from Storyblok"""
    try:
        payload = await request.json()
        logger.info(f"Received data update webhook: {payload}")
        
        # Process the webhook based on content type
        story = payload.get("story", {})
        content_type = story.get("content", {}).get("component")
        
        if content_type == "health_metrics":
            # Trigger alert check for the user
            user_id = story.get("content", {}).get("user_id")
            # Add logic to check for alerts
            
        return {"status": "processed"}
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/alert-trigger")
async def webhook_alert_trigger(payload: WebhookPayload):
    """Handle alert trigger webhooks"""
    try:
        logger.info(f"Alert triggered for user {payload.user_id}")
        
        # Create intervention if needed based on alert severity
        if payload.data.get("severity") in ["high", "critical", "emergency"]:
            intervention = Intervention(
                intervention_id=str(uuid.uuid4()),
                user_id=payload.user_id,
                alert_id=payload.data.get("alert_id"),
                timestamp=datetime.now(),
                intervention_type=InterventionType.AUTOMATED_CHECKIN,
                severity=AlertSeverity(payload.data.get("severity")),
                description="Automated intervention triggered by alert",
                automated=True,
                recipient="user",
                contact_method="app_notification",
                message_sent="Please check in to confirm your status",
                follow_up_required=True,
                follow_up_date=datetime.now() + timedelta(hours=1)
            )
            
            storyblok_client.create_intervention(intervention)
            
        return {"status": "processed", "intervention_created": True}
    except Exception as e:
        logger.error(f"Alert webhook processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/intervention")
async def webhook_intervention(request: Request):
    """Handle intervention webhooks"""
    try:
        payload = await request.json()
        logger.info(f"Intervention webhook received: {payload}")
        
        # Process intervention response
        # Add logic to update intervention status
        
        return {"status": "processed"}
    except Exception as e:
        logger.error(f"Intervention webhook processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers

@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
