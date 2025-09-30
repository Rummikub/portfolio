"""
Generate AI-Powered Caregiver Messages
Main script to generate and sync compassionate caregiver messages to Storyblok
"""

import argparse
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from storyblok_caregiver_system.message_generator import CaregiverMessageGenerator
from storyblok_caregiver_system.content_manager import CaregiverContentManager
from src.guardian_alerts import generate_guardian_alerts

# Sample caregiver relationships for demo
CAREGIVER_RELATIONSHIPS = {
    "user_0": {"name": "Emily", "relationship": "daughter"},
    "user_1": {"name": "Michael", "relationship": "son"},
    "user_2": {"name": "Margaret", "relationship": "spouse"},
    "user_3": {"name": "Jennifer", "relationship": "daughter"},
    "user_4": {"name": "Robert", "relationship": "son"},
    "user_5": {"name": "Linda", "relationship": "daughter"},
    "user_6": {"name": "David", "relationship": "son"},
    "user_7": {"name": "Susan", "relationship": "spouse"},
    "user_8": {"name": "Patricia", "relationship": "daughter"},
    "user_9": {"name": "James", "relationship": "son"},
    "user_10": {"name": "Barbara", "relationship": "spouse"}
}

# Sample patient names for demo
PATIENT_NAMES = {
    "user_0": "Sarah Wilson",
    "user_1": "John Martinez",
    "user_2": "Robert Thompson",
    "user_3": "Mary Johnson",
    "user_4": "William Davis",
    "user_5": "Elizabeth Brown",
    "user_6": "Richard Miller",
    "user_7": "Dorothy Anderson",
    "user_8": "Charles Taylor",
    "user_9": "Helen Moore",
    "user_10": "George Jackson"
}

def get_severity_from_alert(alert_text: str) -> str:
    """
    Determine severity level from alert text
    """
    if "Call 911" in alert_text or "HIGH" in alert_text:
        return "critical"
    elif "Notify Doctor" in alert_text or "MIDDLE" in alert_text:
        return "high"
    elif "Ping User" in alert_text or "LOW" in alert_text:
        return "moderate"
    else:
        return "low"

def generate_messages_for_all_critical():
    """
    Generate caregiver messages for all critical patients
    """
    print("=" * 60)
    print("GENERATING CAREGIVER MESSAGES FOR CRITICAL PATIENTS")
    print("=" * 60)
    
    # Load guardian alerts
    alerts_df = generate_guardian_alerts("data/synthetic_wearable_logs.csv")
    
    # Filter for critical alerts
    critical_alerts = alerts_df[alerts_df['alert'].str.contains('HIGH|Call 911', na=False)]
    
    if critical_alerts.empty:
        print("No critical alerts found.")
        return []
    
    print(f"\nFound {len(critical_alerts)} critical patients requiring immediate caregiver attention")
    
    # Initialize generators
    message_generator = CaregiverMessageGenerator()
    content_manager = CaregiverContentManager()
    
    messages = []
    
    for idx, row in critical_alerts.iterrows():
        user_id = row['user_id']
        
        # Get patient and caregiver info
        patient_name = PATIENT_NAMES.get(user_id, f"Patient {user_id}")
        caregiver_info = CAREGIVER_RELATIONSHIPS.get(
            user_id, 
            {"name": "Family Member", "relationship": "family member"}
        )
        
        print(f"\n📝 Generating message for {patient_name}'s {caregiver_info['relationship']} ({caregiver_info['name']})...")
        
        # Generate the caregiver message
        message = message_generator.generate_caregiver_message(
            patient_name=patient_name,
            patient_id=user_id,
            severity="critical",
            inactive_hours=int(row['inactive_hours']),
            activity_decline=int(row['deviation'] * 100),
            caregiver_name=caregiver_info['name'],
            caregiver_relationship=caregiver_info['relationship']
        )
        
        messages.append(message)
        
        # Display the generated message
        print(f"   ✅ Message generated for {caregiver_info['name']}")
        print(f"   📊 Severity: CRITICAL")
        print(f"   ⏰ Inactive: {row['inactive_hours']:.0f} hours")
        print(f"   📉 Activity decline: {row['deviation']*100:.0f}%")
    
    # Save messages locally
    with open("data/caregiver_messages.json", "w") as f:
        json.dump(messages, f, indent=2)
    
    print(f"\n✅ Generated {len(messages)} caregiver messages")
    print(f"📁 Saved to: data/caregiver_messages.json")
    
    return messages

def generate_message_for_patient(patient_name: str, severity: str):
    """
    Generate a caregiver message for a specific patient
    """
    print("=" * 60)
    print(f"GENERATING CAREGIVER MESSAGE FOR {patient_name.upper()}")
    print("=" * 60)
    
    # Find patient in the system
    patient_id = None
    for uid, name in PATIENT_NAMES.items():
        if name.lower() == patient_name.lower():
            patient_id = uid
            break
    
    if not patient_id:
        # Create a demo patient if not found
        patient_id = "demo_patient"
        print(f"⚠️  Patient '{patient_name}' not in system, creating demo message...")
    
    # Get or create caregiver info
    caregiver_info = CAREGIVER_RELATIONSHIPS.get(
        patient_id,
        {"name": "Emily", "relationship": "daughter"}  # Default for demo
    )
    
    # Set demo metrics based on severity
    severity_metrics = {
        "critical": {"hours": 96, "decline": 89},
        "high": {"hours": 48, "decline": 65},
        "moderate": {"hours": 24, "decline": 45},
        "low": {"hours": 6, "decline": 20}
    }
    
    metrics = severity_metrics.get(severity.lower(), severity_metrics["moderate"])
    
    # Initialize generator
    message_generator = CaregiverMessageGenerator()
    
    print(f"\n👤 Patient: {patient_name}")
    print(f"👨‍👩‍👧 Caregiver: {caregiver_info['name']} ({caregiver_info['relationship']})")
    print(f"⚠️  Severity: {severity.upper()}")
    print(f"📊 Metrics: {metrics['hours']}h inactive, {metrics['decline']}% decline")
    
    # Generate the message
    message = message_generator.generate_caregiver_message(
        patient_name=patient_name,
        patient_id=patient_id,
        severity=severity,
        inactive_hours=metrics['hours'],
        activity_decline=metrics['decline'],
        caregiver_name=caregiver_info['name'],
        caregiver_relationship=caregiver_info['relationship']
    )
    
    # Display key parts of the message
    print("\n" + "=" * 60)
    print("GENERATED CAREGIVER MESSAGE")
    print("=" * 60)
    
    content = message['story']['content']
    msg = content['message']
    
    print(f"\n📬 {msg['greeting']}")
    print(f"\n📋 {msg['situation_summary']}")
    
    print("\n⚡ IMMEDIATE ACTIONS:")
    for i, action in enumerate(msg['immediate_actions'], 1):
        print(f"   {i}. {action}")
    
    print(f"\n💚 {msg['reassurance']}")
    print(f"\n➡️  {msg['next_steps']}")
    
    # Save the message
    filename = f"data/caregiver_message_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(message, f, indent=2)
    
    print(f"\n✅ Message saved to: {filename}")
    
    return message

def sync_messages_to_storyblok(messages: List[Dict]):
    """
    Sync generated messages to Storyblok
    """
    print("\n" + "=" * 60)
    print("SYNCING MESSAGES TO STORYBLOK")
    print("=" * 60)
    
    content_manager = CaregiverContentManager()
    
    success_count = 0
    for message in messages:
        try:
            # Create the alert in Storyblok
            result = content_manager.create_caregiver_alert(message)
            
            # Try to publish if creation was successful
            if 'story' in result and 'id' in result['story']:
                story_id = result['story']['id']
                content_manager.publish_alert(story_id)
                success_count += 1
                
                # Generate preview URL
                preview_url = content_manager.generate_preview_url(story_id)
                print(f"   Preview: {preview_url}")
        except Exception as e:
            print(f"   ❌ Error syncing message: {e}")
    
    print(f"\n✅ Successfully synced {success_count}/{len(messages)} messages to Storyblok")

def main():
    """
    Main function to handle command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate AI-powered caregiver messages for SyncFit"
    )
    
    parser.add_argument(
        "--all-critical-patients",
        action="store_true",
        help="Generate messages for all critical patients"
    )
    
    parser.add_argument(
        "--patient",
        type=str,
        help="Generate message for a specific patient (e.g., 'sarah_wilson')"
    )
    
    parser.add_argument(
        "--severity",
        type=str,
        choices=["critical", "high", "moderate", "low"],
        default="critical",
        help="Severity level for the alert"
    )
    
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync generated messages to Storyblok"
    )
    
    args = parser.parse_args()
    
    messages = []
    
    if args.all_critical_patients:
        messages = generate_messages_for_all_critical()
    elif args.patient:
        # Format patient name
        patient_name = args.patient.replace("_", " ").title()
        message = generate_message_for_patient(patient_name, args.severity)
        messages = [message] if message else []
    else:
        print("Please specify either --all-critical-patients or --patient <name>")
        parser.print_help()
        return
    
    # Sync to Storyblok if requested
    if args.sync and messages:
        sync_messages_to_storyblok(messages)
    
    print("\n" + "=" * 60)
    print("CAREGIVER MESSAGE GENERATION COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review generated messages in data/caregiver_messages.json")
    print("2. Access Storyblok dashboard to see content structure")
    print("3. View caregiver portal at http://localhost:8000/caregiver-portal")
    print("4. Test webhook notifications for critical alerts")

if __name__ == "__main__":
    main()
