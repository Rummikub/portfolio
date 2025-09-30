"""
Storyblok Content Manager for Caregiver Communication
Manages rich content experiences for family caregivers through Storyblok CMS
"""

import os
import json
import requests
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CaregiverContentManager:
    """
    Manages caregiver content in Storyblok, transforming health data
    into rich, structured content experiences for families
    """
    
    def __init__(self):
        self.token = os.getenv("STORYBLOK_TOKEN")
        self.space_id = os.getenv("STORYBLOK_SPACE_ID")
        self.base_url = f"https://mapi.storyblok.com/v1/spaces/{self.space_id}"
        self.headers = {
            "Authorization": self.token,
            "Content-Type": "application/json"
        }
        
        # Content folders structure
        self.folders = {
            "critical": "caregiver-alerts/critical",
            "high": "caregiver-alerts/high",
            "moderate": "caregiver-alerts/moderate",
            "low": "caregiver-alerts/low"
        }
    
    def setup_caregiver_components(self):
        """
        Create custom Storyblok components for caregiver content
        """
        
        components = [
            {
                "name": "caregiver_alert",
                "display_name": "Caregiver Alert",
                "schema": {
                    "patient_name": {
                        "type": "text",
                        "display_name": "Patient Name",
                        "required": True
                    },
                    "patient_id": {
                        "type": "text",
                        "display_name": "Patient ID"
                    },
                    "caregiver_name": {
                        "type": "text",
                        "display_name": "Caregiver Name"
                    },
                    "caregiver_relationship": {
                        "type": "text",
                        "display_name": "Relationship"
                    },
                    "severity": {
                        "type": "option",
                        "display_name": "Severity Level",
                        "options": [
                            {"name": "Critical", "value": "critical"},
                            {"name": "High", "value": "high"},
                            {"name": "Moderate", "value": "moderate"},
                            {"name": "Low", "value": "low"}
                        ]
                    },
                    "alert_time": {
                        "type": "datetime",
                        "display_name": "Alert Time"
                    },
                    "message": {
                        "type": "bloks",
                        "display_name": "Alert Message",
                        "component_whitelist": ["message_content"]
                    },
                    "emergency_contacts": {
                        "type": "bloks",
                        "display_name": "Emergency Contacts",
                        "component_whitelist": ["contact_info"]
                    },
                    "resources": {
                        "type": "bloks",
                        "display_name": "Caregiver Resources",
                        "component_whitelist": ["resource_link"]
                    }
                }
            },
            {
                "name": "message_content",
                "display_name": "Message Content",
                "schema": {
                    "greeting": {
                        "type": "text",
                        "display_name": "Greeting"
                    },
                    "situation_summary": {
                        "type": "textarea",
                        "display_name": "Situation Summary"
                    },
                    "immediate_actions": {
                        "type": "textarea",
                        "display_name": "Immediate Actions",
                        "description": "List of actions for the caregiver"
                    },
                    "reassurance": {
                        "type": "textarea",
                        "display_name": "Reassurance Message"
                    },
                    "next_steps": {
                        "type": "textarea",
                        "display_name": "Next Steps"
                    }
                }
            },
            {
                "name": "contact_info",
                "display_name": "Contact Information",
                "schema": {
                    "name": {
                        "type": "text",
                        "display_name": "Contact Name"
                    },
                    "number": {
                        "type": "text",
                        "display_name": "Phone Number"
                    },
                    "available": {
                        "type": "text",
                        "display_name": "Availability"
                    },
                    "when_to_call": {
                        "type": "text",
                        "display_name": "When to Call"
                    }
                }
            },
            {
                "name": "resource_link",
                "display_name": "Resource Link",
                "schema": {
                    "title": {
                        "type": "text",
                        "display_name": "Resource Title"
                    },
                    "description": {
                        "type": "text",
                        "display_name": "Description"
                    },
                    "url": {
                        "type": "text",
                        "display_name": "URL"
                    },
                    "icon": {
                        "type": "asset",
                        "display_name": "Icon",
                        "filetypes": ["images"]
                    }
                }
            },
            {
                "name": "action_plan",
                "display_name": "Action Plan",
                "schema": {
                    "priority": {
                        "type": "option",
                        "display_name": "Priority",
                        "options": [
                            {"name": "Immediate", "value": "immediate"},
                            {"name": "Within 2 Hours", "value": "urgent"},
                            {"name": "Today", "value": "today"},
                            {"name": "This Week", "value": "week"}
                        ]
                    },
                    "action": {
                        "type": "text",
                        "display_name": "Action Item"
                    },
                    "details": {
                        "type": "textarea",
                        "display_name": "Action Details"
                    },
                    "completed": {
                        "type": "boolean",
                        "display_name": "Completed"
                    }
                }
            }
        ]
        
        print("Setting up Storyblok components for caregiver system...")
        for component in components:
            print(f"  - Creating component: {component['name']}")
            # In production, this would make API calls to create components
            # For demo, we'll simulate the setup
        
        return True
    
    def create_caregiver_alert(self, alert_data: Dict) -> Dict:
        """
        Create a caregiver alert story in Storyblok
        
        Args:
            alert_data: Dictionary containing alert information
            
        Returns:
            Created story data from Storyblok
        """
        
        # Determine folder based on severity
        severity = alert_data.get("story", {}).get("content", {}).get("severity", "moderate")
        folder = self.folders.get(severity.lower(), self.folders["moderate"])
        
        # Add folder to story data
        alert_data["story"]["parent_id"] = self._get_or_create_folder(folder)
        
        # Create the story in Storyblok
        try:
            response = requests.post(
                f"{self.base_url}/stories",
                headers=self.headers,
                json=alert_data
            )
            
            if response.status_code == 201:
                story = response.json()
                print(f"✅ Created caregiver alert: {story['story']['name']}")
                return story
            else:
                print(f"❌ Failed to create alert: {response.status_code}")
                return alert_data
                
        except Exception as e:
            print(f"Error creating Storyblok story: {e}")
            # Return the data for local storage/fallback
            return alert_data
    
    def _get_or_create_folder(self, folder_path: str) -> int:
        """
        Get or create a folder in Storyblok
        
        Args:
            folder_path: Path like "caregiver-alerts/critical"
            
        Returns:
            Folder ID
        """
        
        # In production, this would check if folder exists and create if needed
        # For demo, return a simulated folder ID
        folder_ids = {
            "caregiver-alerts/critical": 1001,
            "caregiver-alerts/high": 1002,
            "caregiver-alerts/moderate": 1003,
            "caregiver-alerts/low": 1004
        }
        
        return folder_ids.get(folder_path, 1000)
    
    def publish_alert(self, story_id: int) -> bool:
        """
        Publish a caregiver alert story
        
        Args:
            story_id: ID of the story to publish
            
        Returns:
            Success status
        """
        
        try:
            response = requests.put(
                f"{self.base_url}/stories/{story_id}/publish",
                headers=self.headers
            )
            
            if response.status_code == 200:
                print(f"✅ Published alert story: {story_id}")
                return True
            else:
                print(f"❌ Failed to publish: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error publishing story: {e}")
            return False
    
    def setup_webhooks(self):
        """
        Set up webhooks for real-time caregiver notifications
        """
        
        webhooks = [
            {
                "name": "Critical Alert Notification",
                "endpoint": "https://syncfit-api.com/webhooks/critical-alert",
                "actions": ["story.published"],
                "filter": "content.severity == 'critical'"
            },
            {
                "name": "High Priority Alert",
                "endpoint": "https://syncfit-api.com/webhooks/high-alert",
                "actions": ["story.published"],
                "filter": "content.severity == 'high'"
            },
            {
                "name": "Caregiver Portal Update",
                "endpoint": "https://syncfit-api.com/webhooks/portal-update",
                "actions": ["story.published", "story.updated"],
                "filter": "content.component == 'caregiver_alert'"
            }
        ]
        
        print("Setting up Storyblok webhooks for caregiver notifications...")
        for webhook in webhooks:
            print(f"  - Creating webhook: {webhook['name']}")
            # In production, this would make API calls to create webhooks
        
        return True
    
    def get_caregiver_alerts(self, caregiver_id: str = None, severity: str = None) -> List[Dict]:
        """
        Retrieve caregiver alerts from Storyblok
        
        Args:
            caregiver_id: Optional filter by caregiver
            severity: Optional filter by severity
            
        Returns:
            List of alert stories
        """
        
        params = {
            "token": self.token,
            "version": "published",
            "starts_with": "caregiver-alerts/",
            "per_page": 100
        }
        
        if severity:
            params["starts_with"] = f"caregiver-alerts/{severity}/"
        
        try:
            response = requests.get(
                f"https://api.storyblok.com/v2/cdn/stories",
                params=params
            )
            
            if response.status_code == 200:
                stories = response.json().get("stories", [])
                
                # Filter by caregiver if specified
                if caregiver_id:
                    stories = [
                        s for s in stories 
                        if s.get("content", {}).get("caregiver_id") == caregiver_id
                    ]
                
                return stories
            else:
                print(f"Failed to retrieve alerts: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error retrieving alerts: {e}")
            return []
    
    def create_caregiver_portal_content(self):
        """
        Create content structure for the caregiver portal
        """
        
        portal_pages = [
            {
                "name": "Caregiver Dashboard",
                "slug": "caregiver-dashboard",
                "content": {
                    "component": "page",
                    "title": "Your Loved Ones Dashboard",
                    "description": "Monitor and care for your family members",
                    "sections": [
                        {
                            "component": "alert_feed",
                            "title": "Recent Alerts",
                            "filter": "all"
                        },
                        {
                            "component": "patient_cards",
                            "title": "Your Loved Ones"
                        },
                        {
                            "component": "resource_grid",
                            "title": "Helpful Resources"
                        }
                    ]
                }
            },
            {
                "name": "Emergency Guide",
                "slug": "emergency-guide",
                "content": {
                    "component": "guide",
                    "title": "Emergency Response Guide",
                    "sections": [
                        {
                            "title": "When to Call 911",
                            "content": "Signs that require immediate emergency response..."
                        },
                        {
                            "title": "First Response Steps",
                            "content": "What to do while waiting for help..."
                        }
                    ]
                }
            },
            {
                "name": "Caregiver Resources",
                "slug": "caregiver-resources",
                "content": {
                    "component": "resource_library",
                    "title": "Caregiver Support Resources",
                    "categories": [
                        "Daily Care Tips",
                        "Medical Information",
                        "Support Groups",
                        "Legal & Financial"
                    ]
                }
            }
        ]
        
        print("Creating caregiver portal content structure...")
        for page in portal_pages:
            print(f"  - Creating page: {page['name']}")
            # In production, create these pages in Storyblok
        
        return True
    
    def generate_preview_url(self, story_id: int) -> str:
        """
        Generate a preview URL for a caregiver alert
        
        Args:
            story_id: ID of the story
            
        Returns:
            Preview URL
        """
        
        base_preview = "https://syncfit-caregiver-portal.com/preview"
        token = self.token
        
        return f"{base_preview}?story={story_id}&token={token}"


# Example usage
if __name__ == "__main__":
    manager = CaregiverContentManager()
    
    # Setup components
    manager.setup_caregiver_components()
    
    # Setup webhooks
    manager.setup_webhooks()
    
    # Create portal content
    manager.create_caregiver_portal_content()
    
    print("\n✅ Caregiver content management system ready!")
