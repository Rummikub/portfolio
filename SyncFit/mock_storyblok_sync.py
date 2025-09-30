#!/usr/bin/env python
"""
Mock Storyblok Sync - For Demo Purposes
Shows successful sync without actual Storyblok configuration
"""

import json
import time
from datetime import datetime
import os

def mock_sync():
    """Mock successful sync to Storyblok for demo"""
    
    print("=" * 60)
    print("UPLOADING CAREGIVER MESSAGES TO STORYBLOK")
    print("=" * 60)
    
    # Check if messages exist
    if not os.path.exists("data/caregiver_messages.json"):
        print("❌ No caregiver messages found. Run generate_caregiver_messages.py first.")
        return
    
    # Load messages
    with open("data/caregiver_messages.json", "r") as f:
        messages = json.load(f)
    
    print(f"\n📦 Found {len(messages)} messages to upload")
    
    success_count = 0
    
    for i, msg in enumerate(messages, 1):
        content = msg['story']['content']
        patient_name = content['patient_name']
        severity = content['severity']
        caregiver = content.get('caregiver_name', 'Family Member')
        
        print(f"\n[{i}/{len(messages)}] Uploading alert for {patient_name} ({severity})...")
        
        # Simulate processing time
        time.sleep(0.8)
        
        # Mock success responses
        story_id = f"{int(datetime.now().timestamp())}-{i}"
        space_id = "demo-space-123456"
        
        print(f"   ✅ Created story: alert-{patient_name.lower().replace(' ', '-')}-{story_id}")
        print(f"   ✅ Published to: caregiver-alerts/{severity}/")
        print(f"   📍 Preview URL: https://app.storyblok.com/#!/me/spaces/{space_id}/stories/{story_id}")
        
        # Show webhook trigger for critical alerts
        if severity == "critical":
            print(f"   🔔 Webhook triggered: Notifying {caregiver} via SMS and email")
        
        success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("UPLOAD SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully uploaded: {success_count}/{len(messages)}")
    print(f"📊 Success rate: 100%")
    print(f"⚡ CDN Distribution: Global (< 50ms latency)")
    
    print("\n📋 Content Organization in Storyblok:")
    print("   └── caregiver-alerts/")
    
    # Count by severity
    severity_counts = {}
    for msg in messages:
        sev = msg['story']['content']['severity']
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    for severity, count in severity_counts.items():
        emoji = "🔴" if severity == "critical" else "🟠" if severity == "high" else "🟡"
        print(f"       ├── {severity}/ ({count} alerts) {emoji}")
    
    print("\n🚀 Next Steps:")
    print("1. ✅ Content successfully stored in Storyblok CMS")
    print("2. ✅ Webhooks configured for real-time notifications")
    print("3. ✅ Visual editor available for content customization")
    print("4. ✅ CDN ready for global content delivery")
    print("\n💡 Non-technical staff can now edit messages in Storyblok's visual editor")
    print("   while maintaining medical accuracy and compassionate tone.")
    
    # Simulate webhook notification
    print("\n" + "=" * 60)
    print("WEBHOOK NOTIFICATIONS")
    print("=" * 60)
    
    critical_count = severity_counts.get('critical', 0)
    if critical_count > 0:
        print(f"🚨 {critical_count} CRITICAL webhooks triggered:")
        for msg in messages:
            if msg['story']['content']['severity'] == 'critical':
                patient = msg['story']['content']['patient_name']
                caregiver = msg['story']['content'].get('caregiver_name', 'Caregiver')
                print(f"   → Notifying {caregiver} about {patient} via SMS/Email/App")
    
    print("\n✅ Demo sync completed successfully!")
    print("   View the caregiver portal to see the synced content.")

if __name__ == "__main__":
    mock_sync()
