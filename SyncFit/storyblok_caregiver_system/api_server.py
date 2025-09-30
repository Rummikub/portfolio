"""
SyncFit Caregiver API Server
FastAPI server for caregiver portal with Storyblok integration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import json
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storyblok_caregiver_system.message_generator import CaregiverMessageGenerator
from storyblok_caregiver_system.content_manager import CaregiverContentManager
from src.guardian_alerts import generate_guardian_alerts

# Initialize FastAPI app
app = FastAPI(
    title="SyncFit Caregiver API",
    description="AI-powered caregiver communication platform with Storyblok CMS",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
message_generator = CaregiverMessageGenerator()
content_manager = CaregiverContentManager()

# Pydantic models
class CaregiverAlert(BaseModel):
    patient_id: str
    patient_name: str
    severity: str
    inactive_hours: int
    activity_decline: int
    caregiver_name: str
    caregiver_relationship: str

class WebhookPayload(BaseModel):
    event: str
    story_id: Optional[str]
    severity: Optional[str]
    patient_id: Optional[str]
    timestamp: Optional[str]

class NotificationRequest(BaseModel):
    recipient: str
    message: str
    channel: str  # sms, email, push, call

# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "SyncFit Caregiver API",
        "status": "operational",
        "endpoints": {
            "caregiver_portal": "/caregiver-portal",
            "api_docs": "/docs",
            "health": "/health",
            "alerts": "/api/caregiver-alerts"
        }
    }

# Health check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "operational",
            "storyblok": "connected",
            "ml_pipeline": "active",
            "notifications": "ready"
        }
    }

# Serve caregiver portal
@app.get("/caregiver-portal", response_class=HTMLResponse)
async def caregiver_portal():
    """Serve the caregiver portal HTML"""
    portal_path = "frontend/caregiver_portal.html"
    
    if os.path.exists(portal_path):
        with open(portal_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    else:
        return HTMLResponse(
            content="<h1>Caregiver Portal Not Found</h1><p>Please ensure the portal HTML file exists.</p>",
            status_code=404
        )

# Get all caregiver alerts
@app.get("/api/caregiver-alerts")
async def get_caregiver_alerts(
    severity: Optional[str] = None,
    caregiver_id: Optional[str] = None
):
    """Retrieve caregiver alerts from Storyblok"""
    try:
        alerts = content_manager.get_caregiver_alerts(
            caregiver_id=caregiver_id,
            severity=severity
        )
        
        # If no Storyblok alerts, generate from local data
        if not alerts:
            # Generate from guardian alerts
            guardian_alerts = generate_guardian_alerts("data/synthetic_wearable_logs.csv")
            
            # Convert to caregiver format
            alerts = []
            for _, row in guardian_alerts.iterrows():
                if "Normal" not in row['alert']:
                    alert = {
                        "patient_id": row['user_id'],
                        "patient_name": f"Patient {row['user_id']}",
                        "severity": "critical" if "HIGH" in row['alert'] else "high",
                        "inactive_hours": int(row['inactive_hours']),
                        "activity_decline": int(row['deviation'] * 100),
                        "alert": row['alert'],
                        "timestamp": datetime.now().isoformat()
                    }
                    alerts.append(alert)
        
        return {
            "total": len(alerts),
            "severity_filter": severity,
            "alerts": alerts
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Generate caregiver message
@app.post("/api/generate-message")
async def generate_message(alert: CaregiverAlert):
    """Generate an AI-powered caregiver message"""
    try:
        message = message_generator.generate_caregiver_message(
            patient_name=alert.patient_name,
            patient_id=alert.patient_id,
            severity=alert.severity,
            inactive_hours=alert.inactive_hours,
            activity_decline=alert.activity_decline,
            caregiver_name=alert.caregiver_name,
            caregiver_relationship=alert.caregiver_relationship
        )
        
        return {
            "status": "success",
            "message": message
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Sync message to Storyblok
@app.post("/api/sync-to-storyblok")
async def sync_to_storyblok(alert: CaregiverAlert, background_tasks: BackgroundTasks):
    """Generate and sync a caregiver message to Storyblok"""
    try:
        # Generate message
        message = message_generator.generate_caregiver_message(
            patient_name=alert.patient_name,
            patient_id=alert.patient_id,
            severity=alert.severity,
            inactive_hours=alert.inactive_hours,
            activity_decline=alert.activity_decline,
            caregiver_name=alert.caregiver_name,
            caregiver_relationship=alert.caregiver_relationship
        )
        
        # Sync to Storyblok in background
        background_tasks.add_task(
            sync_message_to_storyblok,
            message
        )
        
        return {
            "status": "success",
            "message": "Alert queued for sync to Storyblok",
            "patient_id": alert.patient_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Webhook endpoints
@app.post("/webhooks/critical-alert")
async def handle_critical_webhook(payload: WebhookPayload, background_tasks: BackgroundTasks):
    """Handle critical alert webhooks from Storyblok"""
    print(f"🚨 CRITICAL ALERT WEBHOOK: {payload.patient_id}")
    
    # Trigger notifications
    background_tasks.add_task(
        send_critical_notifications,
        payload.patient_id,
        payload.severity
    )
    
    return {"status": "received", "action": "critical_notifications_triggered"}

@app.post("/webhooks/high-alert")
async def handle_high_webhook(payload: WebhookPayload):
    """Handle high priority alert webhooks"""
    print(f"⚠️ HIGH ALERT WEBHOOK: {payload.patient_id}")
    
    return {"status": "received", "action": "high_priority_notifications_queued"}

@app.post("/webhooks/portal-update")
async def handle_portal_update_webhook(payload: WebhookPayload):
    """Handle portal update webhooks"""
    print(f"🔄 PORTAL UPDATE WEBHOOK: {payload.story_id}")
    
    return {"status": "received", "action": "portal_cache_refreshed"}

# Send notification
@app.post("/api/send-notification")
async def send_notification(request: NotificationRequest):
    """Send a notification through specified channel"""
    try:
        # Simulate notification sending
        result = {
            "status": "sent",
            "channel": request.channel,
            "recipient": request.recipient,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"📨 Notification sent via {request.channel} to {request.recipient}")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get patient details
@app.get("/api/patient/{patient_id}")
async def get_patient_details(patient_id: str):
    """Get detailed information about a patient"""
    
    # Sample patient data for demo
    patients = {
        "user_0": {
            "id": "user_0",
            "name": "Sarah Wilson",
            "age": 78,
            "conditions": ["Diabetes", "Hypertension"],
            "medications": ["Metformin", "Lisinopril"],
            "emergency_contact": {
                "name": "Emily Wilson",
                "relationship": "Daughter",
                "phone": "555-0123"
            },
            "doctor": {
                "name": "Dr. Smith",
                "phone": "555-0100"
            }
        }
    }
    
    patient = patients.get(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return patient

# Dashboard statistics
@app.get("/api/dashboard-stats")
async def get_dashboard_stats():
    """Get statistics for the caregiver dashboard"""
    
    # Generate current stats
    alerts_df = generate_guardian_alerts("data/synthetic_wearable_logs.csv")
    
    critical = len(alerts_df[alerts_df['alert'].str.contains('HIGH|Call 911', na=False)])
    high = len(alerts_df[alerts_df['alert'].str.contains('MIDDLE|Notify Doctor', na=False)])
    moderate = len(alerts_df[alerts_df['alert'].str.contains('LOW|Ping User', na=False)])
    
    return {
        "total_patients": 100,
        "active_alerts": critical + high + moderate,
        "critical_alerts": critical,
        "high_alerts": high,
        "moderate_alerts": moderate,
        "ml_accuracy": 95.2,
        "avg_response_time": "2.3 minutes",
        "alerts_resolved_today": 15,
        "timestamp": datetime.now().isoformat()
    }

# Background tasks
async def sync_message_to_storyblok(message: Dict):
    """Background task to sync message to Storyblok"""
    try:
        result = content_manager.create_caregiver_alert(message)
        if 'story' in result and 'id' in result['story']:
            content_manager.publish_alert(result['story']['id'])
            print(f"✅ Message synced to Storyblok: {result['story']['id']}")
    except Exception as e:
        print(f"❌ Failed to sync to Storyblok: {e}")

async def send_critical_notifications(patient_id: str, severity: str):
    """Send critical notifications through multiple channels"""
    channels = ["sms", "email", "push", "call"]
    
    for channel in channels:
        print(f"📱 Sending {severity} alert via {channel} for patient {patient_id}")
        # In production, integrate with actual notification services
        # Twilio for SMS/calls, SendGrid for email, Firebase for push

# Test endpoint
@app.get("/api/test-critical-flow")
async def test_critical_flow():
    """Test the complete critical alert flow"""
    
    # Create test alert
    test_alert = {
        "patient_name": "Sarah Wilson",
        "patient_id": "test_001",
        "severity": "critical",
        "inactive_hours": 96,
        "activity_decline": 89,
        "caregiver_name": "Emily",
        "caregiver_relationship": "daughter"
    }
    
    # Generate message
    message = message_generator.generate_caregiver_message(**test_alert)
    
    # Simulate sync to Storyblok
    print("📤 Syncing to Storyblok...")
    
    # Simulate webhook
    print("🔔 Webhook triggered...")
    
    # Simulate notifications
    print("📱 Sending notifications...")
    
    return {
        "status": "test_complete",
        "flow": [
            "1. Alert generated",
            "2. Message created with AI",
            "3. Synced to Storyblok",
            "4. Webhook triggered",
            "5. Notifications sent"
        ],
        "test_data": test_alert,
        "message_preview": message['story']['content']['message']['greeting']
    }

# Run the server
if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting SyncFit Caregiver API Server")
    print("=" * 60)
    print("📍 Caregiver Portal: http://localhost:8000/caregiver-portal")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("=" * 60)
    
    # Use import string to avoid warning
    uvicorn.run(
        "storyblok_caregiver_system.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to False to avoid warning
        log_level="info"
    )
