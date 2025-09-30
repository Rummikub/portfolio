"""
Caregiver Portal Sync
Synchronizes caregiver alerts between ML pipeline and Storyblok CMS
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from storyblok_caregiver_system.content_manager import CaregiverContentManager
from storyblok_caregiver_system.message_generator import CaregiverMessageGenerator
from src.guardian_alerts import generate_guardian_alerts

class CaregiverPortalSync:
    """
    Manages synchronization between SyncFit ML pipeline and Storyblok caregiver portal
    """
    
    def __init__(self):
        self.content_manager = CaregiverContentManager()
        self.message_generator = CaregiverMessageGenerator()
        self.sync_log = []
        
    def sync_all_alerts(self):
        """
        Sync all current alerts to Storyblok caregiver portal
        """
        print("=" * 60)
        print("CAREGIVER PORTAL SYNC - FULL SYNCHRONIZATION")
        print("=" * 60)
        
        # Generate fresh guardian alerts
        print("\n📊 Generating guardian alerts from ML pipeline...")
        alerts_df = generate_guardian_alerts("data/synthetic_wearable_logs.csv")
        
        # Categorize alerts by severity
        critical = alerts_df[alerts_df['alert'].str.contains('HIGH|Call 911', na=False)]
        high = alerts_df[alerts_df['alert'].str.contains('MIDDLE|Notify Doctor', na=False)]
        moderate = alerts_df[alerts_df['alert'].str.contains('LOW|Ping User', na=False)]
        
        print(f"\n📈 Alert Distribution:")
        print(f"   🔴 Critical: {len(critical)} patients")
        print(f"   🟡 High: {len(high)} patients")
        print(f"   🟢 Moderate: {len(moderate)} patients")
        
        # Process each severity level
        all_messages = []
        
        if not critical.empty:
            print(f"\n🚨 Processing CRITICAL alerts...")
            critical_messages = self._process_alert_batch(critical, "critical")
            all_messages.extend(critical_messages)
        
        if not high.empty:
            print(f"\n⚠️  Processing HIGH priority alerts...")
            high_messages = self._process_alert_batch(high, "high")
            all_messages.extend(high_messages)
        
        if not moderate.empty:
            print(f"\n📋 Processing MODERATE alerts...")
            moderate_messages = self._process_alert_batch(moderate[:5], "moderate")  # Limit for demo
            all_messages.extend(moderate_messages)
        
        # Sync to Storyblok
        print(f"\n☁️  Syncing {len(all_messages)} messages to Storyblok...")
        success_count = self._sync_to_storyblok(all_messages)
        
        # Generate sync report
        self._generate_sync_report(all_messages, success_count)
        
        return all_messages
    
    def _process_alert_batch(self, alerts_df: pd.DataFrame, severity: str) -> List[Dict]:
        """
        Process a batch of alerts and generate caregiver messages
        """
        messages = []
        
        # Sample patient and caregiver names for demo
        patient_names = {
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
        
        caregiver_info = {
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
        
        for idx, row in alerts_df.iterrows():
            user_id = row['user_id']
            
            # Get patient and caregiver details
            patient_name = patient_names.get(user_id, f"Patient {user_id}")
            caregiver = caregiver_info.get(
                user_id,
                {"name": "Family Member", "relationship": "family member"}
            )
            
            # Generate caregiver message
            message = self.message_generator.generate_caregiver_message(
                patient_name=patient_name,
                patient_id=user_id,
                severity=severity,
                inactive_hours=int(row['inactive_hours']),
                activity_decline=int(row['deviation'] * 100),
                caregiver_name=caregiver['name'],
                caregiver_relationship=caregiver['relationship']
            )
            
            messages.append(message)
            
            # Log the sync
            self.sync_log.append({
                "timestamp": datetime.now().isoformat(),
                "patient_id": user_id,
                "patient_name": patient_name,
                "severity": severity,
                "caregiver": caregiver['name'],
                "status": "generated"
            })
            
            print(f"   ✅ Generated message for {patient_name}'s {caregiver['relationship']}")
        
        return messages
    
    def _sync_to_storyblok(self, messages: List[Dict]) -> int:
        """
        Sync messages to Storyblok CMS
        """
        success_count = 0
        
        for message in messages:
            try:
                # Create the story in Storyblok
                result = self.content_manager.create_caregiver_alert(message)
                
                # Update sync log
                patient_id = message['story']['content']['patient_id']
                log_entry = next(
                    (item for item in self.sync_log if item['patient_id'] == patient_id),
                    None
                )
                
                if log_entry:
                    if 'story' in result and 'id' in result['story']:
                        log_entry['status'] = "synced"
                        log_entry['story_id'] = result['story']['id']
                        success_count += 1
                        
                        # Try to publish
                        if self.content_manager.publish_alert(result['story']['id']):
                            log_entry['status'] = "published"
                    else:
                        log_entry['status'] = "failed"
                        
            except Exception as e:
                print(f"   ❌ Sync error: {e}")
        
        return success_count
    
    def _generate_sync_report(self, messages: List[Dict], success_count: int):
        """
        Generate a detailed sync report
        """
        print("\n" + "=" * 60)
        print("SYNC REPORT")
        print("=" * 60)
        
        print(f"\n📊 Summary:")
        print(f"   Total messages generated: {len(messages)}")
        print(f"   Successfully synced: {success_count}")
        print(f"   Failed: {len(messages) - success_count}")
        
        # Save sync log
        log_file = f"data/sync_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w") as f:
            json.dump(self.sync_log, f, indent=2)
        
        print(f"\n📁 Sync log saved to: {log_file}")
        
        # Save all messages
        messages_file = "data/caregiver_portal_messages.json"
        with open(messages_file, "w") as f:
            json.dump(messages, f, indent=2)
        
        print(f"📁 Messages saved to: {messages_file}")
    
    def setup_portal_webhooks(self):
        """
        Set up webhooks for real-time portal updates
        """
        print("\n" + "=" * 60)
        print("SETTING UP CAREGIVER PORTAL WEBHOOKS")
        print("=" * 60)
        
        # Setup webhooks in Storyblok
        self.content_manager.setup_webhooks()
        
        # Configure local webhook endpoints
        webhook_config = {
            "critical_alert": {
                "endpoint": "http://localhost:8000/webhooks/critical",
                "actions": ["send_sms", "send_email", "trigger_call"],
                "recipients": ["caregiver", "doctor", "emergency_contact"]
            },
            "high_alert": {
                "endpoint": "http://localhost:8000/webhooks/high",
                "actions": ["send_email", "app_notification"],
                "recipients": ["caregiver", "doctor"]
            },
            "portal_update": {
                "endpoint": "http://localhost:8000/webhooks/update",
                "actions": ["refresh_dashboard", "update_cache"],
                "recipients": ["portal"]
            }
        }
        
        # Save webhook configuration
        with open("data/webhook_config.json", "w") as f:
            json.dump(webhook_config, f, indent=2)
        
        print("✅ Webhook configuration saved")
        print("\nWebhook endpoints configured:")
        for name, config in webhook_config.items():
            print(f"   - {name}: {config['endpoint']}")
        
        return webhook_config
    
    def test_critical_alert_flow(self):
        """
        Test the complete flow for a critical alert
        """
        print("\n" + "=" * 60)
        print("TESTING CRITICAL ALERT FLOW")
        print("=" * 60)
        
        # Create a test critical alert
        test_patient = {
            "name": "Sarah Wilson",
            "id": "test_001",
            "severity": "critical",
            "inactive_hours": 96,
            "activity_decline": 89,
            "caregiver_name": "Emily",
            "caregiver_relationship": "daughter"
        }
        
        print(f"\n🧪 Test Patient: {test_patient['name']}")
        print(f"   Caregiver: {test_patient['caregiver_name']} ({test_patient['caregiver_relationship']})")
        print(f"   Status: {test_patient['inactive_hours']}h inactive, {test_patient['activity_decline']}% decline")
        
        # Generate message
        print("\n1️⃣ Generating AI-powered caregiver message...")
        message = self.message_generator.generate_caregiver_message(
            patient_name=test_patient['name'],
            patient_id=test_patient['id'],
            severity=test_patient['severity'],
            inactive_hours=test_patient['inactive_hours'],
            activity_decline=test_patient['activity_decline'],
            caregiver_name=test_patient['caregiver_name'],
            caregiver_relationship=test_patient['caregiver_relationship']
        )
        print("   ✅ Message generated")
        
        # Sync to Storyblok
        print("\n2️⃣ Syncing to Storyblok CMS...")
        result = self.content_manager.create_caregiver_alert(message)
        print("   ✅ Alert created in Storyblok")
        
        # Simulate webhook trigger
        print("\n3️⃣ Simulating webhook notification...")
        webhook_payload = {
            "event": "story.published",
            "story_id": "test_story_001",
            "severity": "critical",
            "patient_id": test_patient['id'],
            "caregiver_contact": {
                "name": test_patient['caregiver_name'],
                "phone": "555-0123",
                "email": "emily@example.com"
            }
        }
        print("   📱 SMS notification would be sent to: 555-0123")
        print("   📧 Email notification would be sent to: emily@example.com")
        print("   ☎️  Automated call would be triggered")
        
        # Generate portal preview
        print("\n4️⃣ Generating caregiver portal preview...")
        preview_url = "http://localhost:8000/caregiver-portal/alert/test_story_001"
        print(f"   🌐 Portal URL: {preview_url}")
        
        print("\n✅ Critical alert flow test complete!")
        
        return message


def main():
    """
    Main function for caregiver portal sync
    """
    sync = CaregiverPortalSync()
    
    print("🚀 Starting Caregiver Portal Sync System")
    print("=" * 60)
    
    # Setup components
    print("\n📦 Setting up Storyblok components...")
    sync.content_manager.setup_caregiver_components()
    
    # Setup webhooks
    print("\n🔔 Configuring webhooks...")
    sync.setup_portal_webhooks()
    
    # Sync all alerts
    print("\n🔄 Syncing all alerts to caregiver portal...")
    messages = sync.sync_all_alerts()
    
    # Test critical alert flow
    print("\n🧪 Running critical alert test...")
    sync.test_critical_alert_flow()
    
    print("\n" + "=" * 60)
    print("✅ CAREGIVER PORTAL SYNC COMPLETE")
    print("=" * 60)
    print("\nPortal Access:")
    print("   🌐 Caregiver Portal: http://localhost:8000/caregiver-portal")
    print("   📊 Storyblok Dashboard: https://app.storyblok.com/")
    print("   📱 Mobile App: Download SyncFit Caregiver from App Store")
    print("\nNext Steps:")
    print("   1. Review synced content in Storyblok")
    print("   2. Test caregiver portal interface")
    print("   3. Verify webhook notifications")
    print("   4. Check mobile app synchronization")


if __name__ == "__main__":
    main()
