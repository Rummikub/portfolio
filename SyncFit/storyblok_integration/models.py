"""
Pydantic models for Storyblok content types
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import json

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class InterventionType(str, Enum):
    """Types of interventions"""
    USER_NOTIFICATION = "user_notification"
    CAREGIVER_ALERT = "caregiver_alert"
    DOCTOR_NOTIFICATION = "doctor_notification"
    EMERGENCY_CALL = "emergency_call"
    AUTOMATED_CHECKIN = "automated_checkin"

class UserProfile(BaseModel):
    """User profile content model for Storyblok"""
    user_id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    age: Optional[int] = None
    emergency_contact: Optional[Dict[str, str]] = None
    medical_conditions: Optional[List[str]] = []
    assigned_doctor: Optional[str] = None
    caregiver_ids: Optional[List[str]] = []
    device_id: Optional[str] = None
    enrollment_date: datetime
    is_active: bool = True
    churn_risk_score: Optional[float] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        
    def json_dict(self):
        """Return a JSON-serializable dictionary"""
        return json.loads(self.json())

class HealthMetrics(BaseModel):
    """Health metrics content model for Storyblok"""
    user_id: str
    timestamp: datetime
    steps: int
    heart_rate: int
    heart_rate_variability: Optional[float] = None
    calories_burned: Optional[int] = None
    distance_km: Optional[float] = None
    active_minutes: Optional[int] = None
    sleep_hours: Optional[float] = None
    sleep_quality: Optional[str] = None
    stress_level: Optional[int] = None
    blood_oxygen: Optional[float] = None
    temperature: Optional[float] = None
    class_attended: bool = False
    workout_type: Optional[str] = None
    mood_score: Optional[int] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Alert(BaseModel):
    """Alert content model for Storyblok"""
    alert_id: str
    user_id: str
    timestamp: datetime
    severity: AlertSeverity
    alert_type: str
    message: str
    metrics_snapshot: Dict[str, Any]
    inactive_hours: float
    activity_deviation: float
    recommended_action: str
    llm_summary: Optional[str] = None
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    intervention_triggered: bool = False
    intervention_id: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ChurnPrediction(BaseModel):
    """Churn prediction content model for Storyblok"""
    prediction_id: str
    user_id: str
    timestamp: datetime
    churn_probability: float
    churn_risk_level: str  # low, medium, high
    feature_importance: Dict[str, float]
    model_version: str
    prediction_horizon_days: int = 30
    contributing_factors: List[str]
    recommended_interventions: List[str]
    confidence_score: float
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Intervention(BaseModel):
    """Intervention content model for Storyblok"""
    intervention_id: str
    user_id: str
    alert_id: Optional[str] = None
    timestamp: datetime
    intervention_type: InterventionType
    severity: AlertSeverity
    description: str
    automated: bool = True
    recipient: str  # user, caregiver, doctor, emergency
    contact_method: str  # sms, email, phone, app_notification
    message_sent: str
    response_received: Optional[str] = None
    response_timestamp: Optional[datetime] = None
    outcome: Optional[str] = None
    follow_up_required: bool = False
    follow_up_date: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class StoryblokContent(BaseModel):
    """Wrapper for Storyblok content"""
    story: Dict[str, Any]
    cv: Optional[int] = None
    rels: Optional[List] = []
    links: Optional[List] = []

class StoryblokStory(BaseModel):
    """Storyblok story structure"""
    name: str
    slug: str
    content: Dict[str, Any]
    is_startpage: bool = False
    parent_id: Optional[int] = None
    
class WebhookPayload(BaseModel):
    """Webhook payload structure"""
    event_type: str
    timestamp: datetime
    user_id: str
    data: Dict[str, Any]
    source: str = "syncfit"
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AnalyticsReport(BaseModel):
    """Analytics report model"""
    report_id: str
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    total_users: int
    active_users: int
    churned_users: int
    at_risk_users: int
    total_alerts: int
    alerts_by_severity: Dict[str, int]
    interventions_triggered: int
    intervention_success_rate: float
    average_response_time_minutes: float
    top_risk_factors: List[Dict[str, Any]]
    recommendations: List[str]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
