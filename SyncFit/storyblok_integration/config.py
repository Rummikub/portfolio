"""
Storyblok Configuration for SyncFit
"""
import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class StoryblokConfig:
    """Configuration for Storyblok integration"""
    
    # Storyblok API credentials
    STORYBLOK_TOKEN = os.getenv("STORYBLOK_TOKEN", "")
    STORYBLOK_SPACE_ID = os.getenv("STORYBLOK_SPACE_ID", "")
    STORYBLOK_API_URL = "https://api.storyblok.com/v2/cdn"
    STORYBLOK_MANAGEMENT_API = "https://mapi.storyblok.com/v1"
    
    # Content types in Storyblok
    CONTENT_TYPES = {
        "user_profile": "user_profile",
        "health_metrics": "health_metrics",
        "alert": "alert",
        "churn_prediction": "churn_prediction",
        "intervention": "intervention"
    }
    
    # Redis configuration for caching
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    
    # OpenAI configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Alert thresholds
    ALERT_THRESHOLDS = {
        "inactive_hours": {
            "low": 3,
            "medium": 6,
            "high": 12,
            "critical": 24
        },
        "activity_deviation": {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "critical": 0.8
        }
    }
    
    # Webhook endpoints
    WEBHOOK_ENDPOINTS = {
        "data_update": "/webhook/data-update",
        "alert_trigger": "/webhook/alert-trigger",
        "intervention": "/webhook/intervention"
    }
    
    @classmethod
    def get_headers(cls) -> Dict[str, str]:
        """Get headers for Storyblok API requests"""
        return {
            "Authorization": cls.STORYBLOK_TOKEN,
            "Content-Type": "application/json"
        }
    
    @classmethod
    def get_management_headers(cls) -> Dict[str, str]:
        """Get headers for Storyblok Management API"""
        return {
            "Authorization": cls.STORYBLOK_TOKEN,
            "Content-Type": "application/json"
        }

# Initialize configuration
config = StoryblokConfig()
