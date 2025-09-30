"""
Storyblok API Client for SyncFit
"""
import requests
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from .config import config
from .models import (
    UserProfile, HealthMetrics, Alert, ChurnPrediction, 
    Intervention, StoryblokStory, StoryblokContent
)

logger = logging.getLogger(__name__)

class StoryblokClient:
    """Client for interacting with Storyblok API"""
    
    def __init__(self):
        self.token = config.STORYBLOK_TOKEN
        self.space_id = config.STORYBLOK_SPACE_ID
        self.api_url = config.STORYBLOK_API_URL
        self.management_api = config.STORYBLOK_MANAGEMENT_API
        self.headers = config.get_headers()
        self.management_headers = config.get_management_headers()
        
    def create_story(self, story_data: StoryblokStory, folder: str = "") -> Dict[str, Any]:
        """Create a new story in Storyblok"""
        url = f"{self.management_api}/spaces/{self.space_id}/stories/"
        
        payload = {
            "story": {
                "name": story_data.name,
                "slug": story_data.slug,
                "content": story_data.content,
                "is_startpage": story_data.is_startpage,
                "parent_id": story_data.parent_id
            }
        }
        
        if folder:
            payload["story"]["folder"] = folder
            
        try:
            response = requests.post(url, json=payload, headers=self.management_headers)
            response.raise_for_status()
            logger.info(f"Created story: {story_data.name}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create story: {e}")
            raise
            
    def update_story(self, story_id: int, story_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing story in Storyblok"""
        url = f"{self.management_api}/spaces/{self.space_id}/stories/{story_id}"
        
        payload = {"story": story_data}
        
        try:
            response = requests.put(url, json=payload, headers=self.management_headers)
            response.raise_for_status()
            logger.info(f"Updated story ID: {story_id}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to update story: {e}")
            raise
            
    def get_story(self, slug: str, version: str = "published") -> Optional[Dict[str, Any]]:
        """Get a story by slug"""
        url = f"{self.api_url}/stories/{slug}"
        params = {
            "token": self.token,
            "version": version
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get story: {e}")
            return None
            
    def get_stories(self, starts_with: str = "", per_page: int = 100) -> List[Dict[str, Any]]:
        """Get multiple stories using Management API"""
        # Use Management API instead of CDN API for Personal Access Tokens
        url = f"{self.management_api}/spaces/{self.space_id}/stories"
        params = {
            "starts_with": starts_with,
            "per_page": per_page
        }
        
        try:
            response = requests.get(url, params=params, headers=self.management_headers)
            if response.status_code == 200:
                return response.json().get("stories", [])
            else:
                logger.warning(f"Failed to get stories: {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get stories: {e}")
            return []
            
    def delete_story(self, story_id: int) -> bool:
        """Delete a story from Storyblok"""
        url = f"{self.management_api}/spaces/{self.space_id}/stories/{story_id}"
        
        try:
            response = requests.delete(url, headers=self.management_headers)
            response.raise_for_status()
            logger.info(f"Deleted story ID: {story_id}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to delete story: {e}")
            return False
            
    def search_stories(self, query: str, content_type: str = None) -> List[Dict[str, Any]]:
        """Search for stories"""
        url = f"{self.api_url}/stories"
        params = {
            "token": self.token,
            "search_term": query
        }
        
        if content_type:
            params["filter_query[component][in]"] = content_type
            
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json().get("stories", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to search stories: {e}")
            return []
            
    # Content-specific methods
    
    def create_user_profile(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Create a user profile in Storyblok"""
        # Convert to dict and handle datetime serialization
        profile_data = user_profile.dict()
        # Convert datetime to ISO format string
        for key, value in profile_data.items():
            if hasattr(value, 'isoformat'):
                profile_data[key] = value.isoformat()
        
        story = StoryblokStory(
            name=f"User: {user_profile.name}",
            slug=f"users/{user_profile.user_id}",
            content={
                "component": "user_profile",
                **profile_data
            }
        )
        return self.create_story(story, folder="users")
        
    def create_health_metrics(self, metrics: HealthMetrics) -> Dict[str, Any]:
        """Create health metrics entry in Storyblok"""
        timestamp_str = metrics.timestamp.strftime("%Y%m%d_%H%M%S")
        # Convert to dict and handle datetime serialization
        metrics_data = metrics.dict()
        for key, value in metrics_data.items():
            if hasattr(value, 'isoformat'):
                metrics_data[key] = value.isoformat()
        
        story = StoryblokStory(
            name=f"Metrics: {metrics.user_id} - {timestamp_str}",
            slug=f"metrics/{metrics.user_id}/{timestamp_str}",
            content={
                "component": "health_metrics",
                **metrics_data
            }
        )
        return self.create_story(story, folder="metrics")
        
    def create_alert(self, alert: Alert) -> Dict[str, Any]:
        """Create an alert in Storyblok"""
        # Convert to dict and handle datetime serialization
        alert_data = alert.dict()
        for key, value in alert_data.items():
            if hasattr(value, 'isoformat'):
                alert_data[key] = value.isoformat()
        
        story = StoryblokStory(
            name=f"Alert: {alert.user_id} - {alert.severity.value}",
            slug=f"alerts/{alert.alert_id}",
            content={
                "component": "alert",
                **alert_data
            }
        )
        return self.create_story(story, folder="alerts")
        
    def create_churn_prediction(self, prediction: ChurnPrediction) -> Dict[str, Any]:
        """Create a churn prediction in Storyblok"""
        # Convert to dict and handle datetime serialization
        prediction_data = prediction.dict()
        for key, value in prediction_data.items():
            if hasattr(value, 'isoformat'):
                prediction_data[key] = value.isoformat()
        
        story = StoryblokStory(
            name=f"Prediction: {prediction.user_id} - {prediction.churn_risk_level}",
            slug=f"predictions/{prediction.prediction_id}",
            content={
                "component": "churn_prediction",
                **prediction_data
            }
        )
        return self.create_story(story, folder="predictions")
        
    def create_intervention(self, intervention: Intervention) -> Dict[str, Any]:
        """Create an intervention record in Storyblok"""
        # Convert to dict and handle datetime serialization
        intervention_data = intervention.dict()
        for key, value in intervention_data.items():
            if hasattr(value, 'isoformat'):
                intervention_data[key] = value.isoformat()
        
        story = StoryblokStory(
            name=f"Intervention: {intervention.user_id} - {intervention.intervention_type.value}",
            slug=f"interventions/{intervention.intervention_id}",
            content={
                "component": "intervention",
                **intervention_data
            }
        )
        return self.create_story(story, folder="interventions")
        
    def get_user_metrics(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get health metrics for a specific user"""
        return self.get_stories(starts_with=f"metrics/{user_id}/", per_page=limit)
        
    def get_user_alerts(self, user_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get alerts for a specific user"""
        alerts = self.search_stories(user_id, content_type="alert")
        if active_only:
            alerts = [a for a in alerts if not a.get("content", {}).get("is_resolved", False)]
        return alerts
        
    def get_active_interventions(self) -> List[Dict[str, Any]]:
        """Get all active interventions"""
        interventions = self.get_stories(starts_with="interventions/", per_page=100)
        return [i for i in interventions if i.get("content", {}).get("follow_up_required", False)]
        
    def batch_create_metrics(self, metrics_list: List[HealthMetrics]) -> List[Dict[str, Any]]:
        """Batch create multiple health metrics"""
        results = []
        for metrics in metrics_list:
            try:
                result = self.create_health_metrics(metrics)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to create metrics for {metrics.user_id}: {e}")
        return results
        
    def update_alert_status(self, alert_id: str, resolved: bool = True, resolved_by: str = None):
        """Update alert resolution status"""
        story = self.get_story(f"alerts/{alert_id}")
        if story:
            story_data = story["story"]
            story_data["content"]["is_resolved"] = resolved
            if resolved:
                story_data["content"]["resolved_at"] = datetime.now().isoformat()
                if resolved_by:
                    story_data["content"]["resolved_by"] = resolved_by
            return self.update_story(story_data["id"], story_data)
        return None

# Singleton instance
storyblok_client = StoryblokClient()
