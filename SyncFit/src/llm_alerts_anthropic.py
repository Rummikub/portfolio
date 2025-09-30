"""
Anthropic Claude Integration for SyncFit Health Alerts
Uses Claude API for intelligent healthcare recommendations
"""

import os
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_alert_message(user_id, deviation, hours):
    """
    Generate AI-powered health alert messages using Anthropic Claude.
    Falls back to structured alerts if API key is not configured.
    """
    
    # Check if API key is properly set
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key or api_key == "your_anthropic_api_key_here":
        # Return a structured fallback response when API key is not configured
        severity = _calculate_severity(deviation, hours)
        return _generate_fallback_alert(user_id, deviation, hours, severity)
    
    try:
        # Initialize Anthropic client
        client = Anthropic(api_key=api_key)
        
        # Calculate severity for context
        severity = _calculate_severity(deviation, hours)
        
        # Create the prompt for Claude
        prompt = f"""You are a healthcare monitoring AI assistant analyzing patient data for a hospital system.

Patient Alert Details:
- Patient ID: {user_id}
- Activity Deviation: {int(deviation*100)}% below baseline
- Inactive Duration: {int(hours)} hours
- Severity Level: {severity}

Based on this data, generate a concise, professional alert for healthcare providers that includes:

1. Risk Assessment (2-3 sentences)
2. Immediate Actions Required (3-5 bullet points)
3. Follow-up Protocol (2-3 steps)
4. Escalation Criteria (when to upgrade severity)

Keep the response under 200 words and focus on actionable medical interventions. Use clear, direct language suitable for emergency response teams."""

        # Call Claude API
        response = client.messages.create(
            model="claude-3-haiku-20240307",  # Fast and cost-effective
            # model="claude-3-sonnet-20240229",  # More capable, slightly slower
            # model="claude-3-opus-20240229",  # Most capable but expensive
            max_tokens=300,
            temperature=0.3,  # Lower temperature for consistent medical advice
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        # Extract the text from Claude's response
        return response.content[0].text
        
    except Exception as e:
        error_msg = str(e)
        
        # Handle rate limits
        if "rate_limit" in error_msg.lower():
            return (
                f"[Claude Rate Limit - Using Fallback System]\n\n"
                + _generate_fallback_alert(user_id, deviation, hours, _calculate_severity(deviation, hours))
                + f"\n\n💡 Note: Claude rate limit reached. Try again in a few seconds."
            )
        
        # Handle quota/billing issues
        elif "credit" in error_msg.lower() or "billing" in error_msg.lower():
            return (
                f"[Claude Billing Issue - Using Fallback System]\n\n"
                + _generate_fallback_alert(user_id, deviation, hours, _calculate_severity(deviation, hours))
                + f"\n\n💡 To restore AI alerts: Check Anthropic billing at console.anthropic.com"
            )
        
        # Other errors
        else:
            return (
                f"[Claude Error - Using Fallback System]\n\n"
                + _generate_fallback_alert(user_id, deviation, hours, _calculate_severity(deviation, hours))
                + f"\n\nError details: {error_msg[:100]}..."
            )


def _calculate_severity(deviation, hours):
    """Calculate alert severity based on metrics"""
    if deviation > 0.8 and hours > 72:
        return "CRITICAL"
    elif deviation > 0.6 or hours > 48:
        return "HIGH"
    elif deviation > 0.4 or hours > 24:
        return "MODERATE"
    else:
        return "LOW"


def _generate_fallback_alert(user_id, deviation, hours, severity):
    """Generate structured fallback alert when API is unavailable"""
    
    # Define protocols based on severity
    protocols = {
        "CRITICAL": {
            "risk": "Immediate life-threatening emergency likely. Patient may be unconscious, fallen, or experiencing acute medical crisis.",
            "actions": [
                "🚨 CALL 911 IMMEDIATELY",
                "Dispatch emergency medical services to patient location",
                "Contact next of kin/emergency contact",
                "Alert on-call physician for immediate consultation",
                "Prepare patient records for hospital admission"
            ],
            "followup": [
                "Document all intervention attempts",
                "Coordinate with EMS on arrival",
                "Brief receiving hospital on patient history"
            ],
            "escalation": "Already at maximum severity - maintain emergency protocol",
            "response_time": "IMMEDIATE - Within 15 minutes"
        },
        "HIGH": {
            "risk": "Significant health deterioration detected. Possible fall, stroke, or cardiac event. Urgent intervention required.",
            "actions": [
                "📞 Immediate phone contact attempt",
                "If no response within 30 min, dispatch welfare check",
                "Contact primary care physician urgently",
                "Review last 48h of health data for patterns",
                "Alert family member or caregiver"
            ],
            "followup": [
                "Schedule emergency telehealth within 2 hours",
                "Review medication compliance",
                "Arrange in-person check within 24 hours"
            ],
            "escalation": "Upgrade to CRITICAL if no contact within 2 hours",
            "response_time": "URGENT - Within 1 hour"
        },
        "MODERATE": {
            "risk": "Notable activity decline suggesting potential health issue. May indicate illness onset or medication issues.",
            "actions": [
                "Send automated wellness check message",
                "Schedule phone call within 4 hours",
                "Review recent vital signs and activity trends",
                "Check for environmental factors (weather, holidays)",
                "Update care team on status"
            ],
            "followup": [
                "Daily monitoring for next 3 days",
                "Adjust care plan if pattern continues",
                "Consider medication review"
            ],
            "escalation": "Upgrade to HIGH if no improvement in 24 hours",
            "response_time": "STANDARD - Within 4 hours"
        },
        "LOW": {
            "risk": "Minor activity variation detected. Likely within normal range but worth monitoring.",
            "actions": [
                "Log observation in patient record",
                "Continue routine monitoring",
                "Send encouraging activity reminder",
                "Review at next scheduled check-in"
            ],
            "followup": [
                "Re-evaluate in 24 hours",
                "No immediate intervention needed"
            ],
            "escalation": "Upgrade if inactivity exceeds 24 hours",
            "response_time": "ROUTINE - Within 24 hours"
        }
    }
    
    protocol = protocols[severity]
    
    return f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏥 HEALTHCARE ALERT - {severity} PRIORITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 PATIENT: {user_id}
📊 METRICS: {int(deviation*100)}% activity decline, {int(hours)} hours inactive
⏰ RESPONSE TIME: {protocol['response_time']}

🔍 RISK ASSESSMENT:
{protocol['risk']}

⚡ IMMEDIATE ACTIONS:
{chr(10).join(f'  {i+1}. {action}' for i, action in enumerate(protocol['actions']))}

📌 FOLLOW-UP PROTOCOL:
{chr(10).join(f'  • {step}' for step in protocol['followup'])}

⬆️ ESCALATION CRITERIA:
{protocol['escalation']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated by SyncFit Fallback Alert System"""


# Setup instructions
if __name__ == "__main__":
    print("=" * 60)
    print("ANTHROPIC CLAUDE SETUP INSTRUCTIONS")
    print("=" * 60)
    print("""
1. Get your Anthropic API key:
   - Go to: https://console.anthropic.com/
   - Sign in or create account
   - Navigate to API Keys section
   - Create new key or copy existing one
   
2. Add to your .env file:
   ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxx
   
3. Install Anthropic Python SDK:
   pip install anthropic
   
4. Available Claude models:
   - claude-3-haiku: Fast & cheap (default)
   - claude-3-sonnet: Balanced performance
   - claude-3-opus: Most capable
   
5. Usage in your code:
   from src.llm_alerts_anthropic import generate_alert_message
   message = generate_alert_message(user_id, deviation, hours)

6. Cost comparison (per 1M tokens):
   - Haiku: $0.25 input, $1.25 output
   - Sonnet: $3 input, $15 output
   - Opus: $15 input, $75 output
   - (OpenAI GPT-3.5: $0.50 input, $1.50 output)
   
Claude Haiku is recommended for this use case - it's fast, 
affordable, and excellent for structured healthcare alerts.
""")
    
    # Test with sample data
    print("\n" + "=" * 60)
    print("TESTING FALLBACK SYSTEM")
    print("=" * 60)
    
    # Test different severity levels
    test_cases = [
        ("user_001", 0.85, 96),  # Critical
        ("user_002", 0.65, 36),  # High
        ("user_003", 0.45, 18),  # Moderate
        ("user_004", 0.25, 6),   # Low
    ]
    
    for user_id, deviation, hours in test_cases:
        print(f"\nTest: {user_id} - {int(deviation*100)}% decline, {hours}h inactive")
        print("-" * 40)
        message = generate_alert_message(user_id, deviation, hours)
        print(message[:500] + "..." if len(message) > 500 else message)
