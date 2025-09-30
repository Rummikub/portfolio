"""
AI-Powered Caregiver Message Generator
Transforms patient health data into compassionate, actionable communication for families
"""

import os
from typing import Dict, List, Optional
from datetime import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CaregiverMessageGenerator:
    """
    Generates personalized, empathetic messages for family caregivers
    using AI to transform medical data into understandable communication
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.use_ai = self.api_key and self.api_key != "your_openai_api_key_here"
        
    def generate_caregiver_message(
        self, 
        patient_name: str,
        patient_id: str,
        severity: str,
        inactive_hours: int,
        activity_decline: int,
        caregiver_name: str = "Family Member",
        caregiver_relationship: str = "family member"
    ) -> Dict:
        """
        Generate a personalized message for a caregiver based on patient status
        
        Args:
            patient_name: Name of the patient
            patient_id: Patient identifier
            severity: Alert severity (critical, high, moderate, low)
            inactive_hours: Hours of inactivity
            activity_decline: Percentage of activity decline
            caregiver_name: Name of the caregiver
            caregiver_relationship: Relationship to patient (daughter, son, spouse, etc.)
            
        Returns:
            Dictionary containing the structured caregiver message
        """
        
        if self.use_ai:
            return self._generate_ai_message(
                patient_name, patient_id, severity, 
                inactive_hours, activity_decline, 
                caregiver_name, caregiver_relationship
            )
        else:
            return self._generate_template_message(
                patient_name, patient_id, severity, 
                inactive_hours, activity_decline, 
                caregiver_name, caregiver_relationship
            )
    
    def _generate_ai_message(
        self, 
        patient_name: str,
        patient_id: str,
        severity: str,
        inactive_hours: int,
        activity_decline: int,
        caregiver_name: str,
        caregiver_relationship: str
    ) -> Dict:
        """Generate message using OpenAI API"""
        
        try:
            import time
            from openai import OpenAI
            
            # Add delay to avoid rate limits
            time.sleep(1)
            
            client = OpenAI(api_key=self.api_key)
            
            # Craft a compassionate prompt
            prompt = f"""
            You are a compassionate healthcare communication specialist helping families care for elderly loved ones.
            
            Create a caring, actionable message for {caregiver_name} ({caregiver_relationship}) about {patient_name}'s current health status.
            
            Patient Status:
            - Severity Level: {severity.upper()}
            - No activity detected for: {inactive_hours} hours
            - Activity level has declined by: {activity_decline}%
            
            The message should:
            1. Start with empathy and acknowledgment of the caregiver's concern
            2. Explain the situation in non-medical, easy-to-understand terms
            3. Provide 3-5 specific, actionable steps they can take right now
            4. Include reassurance while maintaining appropriate urgency
            5. End with support and next steps
            
            Keep the tone warm, supportive, and focused on helping the family member take action.
            Format the response as JSON with these fields:
            - greeting: Personalized opening
            - situation_summary: Clear explanation of what's happening
            - immediate_actions: List of 3-5 specific actions
            - reassurance: Supportive message
            - next_steps: What happens after immediate actions
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a compassionate healthcare communication specialist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            # Parse the AI response
            ai_content = response.choices[0].message.content
            
            # Try to parse as JSON, fallback to structured format
            try:
                message_data = json.loads(ai_content)
            except:
                # Fallback to template if JSON parsing fails
                return self._generate_template_message(
                    patient_name, patient_id, severity, 
                    inactive_hours, activity_decline, 
                    caregiver_name, caregiver_relationship
                )
            
            return self._structure_message(
                patient_name, patient_id, severity,
                inactive_hours, activity_decline,
                caregiver_name, caregiver_relationship,
                message_data
            )
            
        except Exception as e:
            print(f"AI generation failed: {e}")
            return self._generate_template_message(
                patient_name, patient_id, severity, 
                inactive_hours, activity_decline, 
                caregiver_name, caregiver_relationship
            )
    
    def _generate_template_message(
        self, 
        patient_name: str,
        patient_id: str,
        severity: str,
        inactive_hours: int,
        activity_decline: int,
        caregiver_name: str,
        caregiver_relationship: str
    ) -> Dict:
        """Generate message using templates when AI is not available"""
        
        templates = {
            "critical": {
                "greeting": f"Dear {caregiver_name}, we need your immediate attention regarding {patient_name}.",
                "situation_summary": f"{patient_name} hasn't shown any activity for {inactive_hours} hours, and their overall activity has dropped by {activity_decline}%. This is a critical situation that requires immediate action.",
                "immediate_actions": [
                    f"Call {patient_name} right now - if no answer, go to their location immediately",
                    "Check if they have fallen or need emergency medical attention",
                    "Call 911 if you cannot reach them or if they appear to be in distress",
                    "Contact their primary care physician to alert them of the situation",
                    "Stay with them until their condition stabilizes"
                ],
                "reassurance": "Your quick action can make all the difference. We're here to support you through this.",
                "next_steps": "After ensuring their immediate safety, schedule an urgent medical evaluation within 24 hours."
            },
            "high": {
                "greeting": f"Hello {caregiver_name}, we're concerned about {patient_name} and wanted to alert you.",
                "situation_summary": f"{patient_name} has been inactive for {inactive_hours} hours and their activity level has decreased by {activity_decline}%. This needs your attention today.",
                "immediate_actions": [
                    f"Call {patient_name} to check on their wellbeing",
                    "If they don't answer, visit them within the next 2 hours",
                    "Ask about any pain, discomfort, or changes in their routine",
                    "Review their medications to ensure they're taking them properly",
                    "Consider arranging a telehealth appointment today"
                ],
                "reassurance": "Early intervention often prevents more serious issues. You're doing great by staying involved.",
                "next_steps": "Schedule a check-up with their doctor this week and consider increasing daily check-ins."
            },
            "moderate": {
                "greeting": f"Hi {caregiver_name}, we wanted to update you about {patient_name}'s activity patterns.",
                "situation_summary": f"We've noticed {patient_name} has been less active recently - {inactive_hours} hours of inactivity and a {activity_decline}% decrease in movement. This might indicate they need some extra support.",
                "immediate_actions": [
                    f"Give {patient_name} a friendly call to see how they're feeling",
                    "Ask if they need help with groceries, medications, or appointments",
                    "Encourage them to take a short walk or do light exercises",
                    "Check if they're eating and sleeping well"
                ],
                "reassurance": "These changes might be temporary, but your attention helps ensure their wellbeing.",
                "next_steps": "Monitor their activity over the next few days and schedule a routine check-up if the pattern continues."
            },
            "low": {
                "greeting": f"Hello {caregiver_name}, here's a routine update about {patient_name}.",
                "situation_summary": f"{patient_name} has shown some minor changes in activity - about {activity_decline}% less movement than usual. This is likely normal variation but worth monitoring.",
                "immediate_actions": [
                    f"Check in with {patient_name} during your next regular call",
                    "Encourage them to maintain their daily routines",
                    "Remind them about any upcoming appointments"
                ],
                "reassurance": "Small fluctuations in activity are normal. Your regular check-ins help maintain their health.",
                "next_steps": "Continue your regular communication schedule and note any concerns for their next doctor visit."
            }
        }
        
        template = templates.get(severity.lower(), templates["moderate"])
        
        return self._structure_message(
            patient_name, patient_id, severity,
            inactive_hours, activity_decline,
            caregiver_name, caregiver_relationship,
            template
        )
    
    def _structure_message(
        self,
        patient_name: str,
        patient_id: str,
        severity: str,
        inactive_hours: int,
        activity_decline: int,
        caregiver_name: str,
        caregiver_relationship: str,
        message_content: Dict
    ) -> Dict:
        """Structure the message for Storyblok storage"""
        
        return {
            "story": {
                "name": f"Alert for {patient_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "slug": f"alert-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "content": {
                    "component": "caregiver_alert",
                    "patient_name": patient_name,
                    "patient_id": patient_id,
                    "caregiver_name": caregiver_name,
                    "caregiver_relationship": caregiver_relationship,
                    "severity": severity,
                    "alert_time": datetime.now().isoformat(),
                    "metrics": {
                        "inactive_hours": inactive_hours,
                        "activity_decline": activity_decline
                    },
                    "message": {
                        "greeting": message_content.get("greeting", ""),
                        "situation_summary": message_content.get("situation_summary", ""),
                        "immediate_actions": message_content.get("immediate_actions", []),
                        "reassurance": message_content.get("reassurance", ""),
                        "next_steps": message_content.get("next_steps", "")
                    },
                    "emergency_contacts": self._get_emergency_contacts(severity),
                    "resources": self._get_caregiver_resources(severity)
                }
            }
        }
    
    def _get_emergency_contacts(self, severity: str) -> List[Dict]:
        """Get relevant emergency contacts based on severity"""
        
        contacts = []
        
        if severity.lower() in ["critical", "high"]:
            contacts.append({
                "name": "Emergency Services",
                "number": "911",
                "available": "24/7",
                "when_to_call": "If patient is unresponsive or in immediate danger"
            })
        
        contacts.extend([
            {
                "name": "Primary Care Physician",
                "number": "555-0100",
                "available": "Mon-Fri 8am-5pm",
                "when_to_call": "For medical questions or to schedule urgent appointment"
            },
            {
                "name": "24/7 Nurse Hotline",
                "number": "555-0199",
                "available": "24/7",
                "when_to_call": "For immediate medical advice"
            },
            {
                "name": "SyncFit Support",
                "number": "555-0150",
                "available": "24/7",
                "when_to_call": "For questions about alerts or monitoring"
            }
        ])
        
        return contacts
    
    def _get_caregiver_resources(self, severity: str) -> List[Dict]:
        """Get helpful resources for caregivers"""
        
        resources = [
            {
                "title": "Understanding Activity Monitoring",
                "description": "Learn what activity patterns mean for elderly health",
                "url": "/resources/activity-monitoring-guide"
            },
            {
                "title": "Emergency Response Checklist",
                "description": "Step-by-step guide for health emergencies",
                "url": "/resources/emergency-checklist"
            },
            {
                "title": "Caregiver Support Community",
                "description": "Connect with other family caregivers",
                "url": "/community/caregiver-support"
            }
        ]
        
        if severity.lower() in ["critical", "high"]:
            resources.insert(0, {
                "title": "URGENT: What to Do Right Now",
                "description": "Immediate steps for critical situations",
                "url": "/resources/urgent-response-guide"
            })
        
        return resources
    
    def generate_batch_messages(self, patients: List[Dict]) -> List[Dict]:
        """Generate messages for multiple patients"""
        
        messages = []
        for patient in patients:
            message = self.generate_caregiver_message(
                patient_name=patient.get("name", "Patient"),
                patient_id=patient.get("id", "unknown"),
                severity=patient.get("severity", "moderate"),
                inactive_hours=patient.get("inactive_hours", 0),
                activity_decline=patient.get("activity_decline", 0),
                caregiver_name=patient.get("caregiver_name", "Family Member"),
                caregiver_relationship=patient.get("caregiver_relationship", "family member")
            )
            messages.append(message)
        
        return messages


# Example usage
if __name__ == "__main__":
    generator = CaregiverMessageGenerator()
    
    # Test with Sarah Wilson case
    message = generator.generate_caregiver_message(
        patient_name="Sarah Wilson",
        patient_id="patient_001",
        severity="critical",
        inactive_hours=96,
        activity_decline=89,
        caregiver_name="Emily",
        caregiver_relationship="daughter"
    )
    
    print(json.dumps(message, indent=2))
