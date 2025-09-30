from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def generate_alert_message(user_id, deviation, hours):
    """
    Generate AI-powered health alert messages using OpenAI GPT.
    Returns a simulated message if API key is not configured.
    """
    
    # Check if API key is properly set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        # Return a simulated LLM response when API key is not configured
        severity = "critical" if deviation > 0.8 else "moderate"
        return (
            f"[Simulated AI Alert - Configure OpenAI API key for real responses]\n\n"
            f"Patient {user_id} Alert:\n"
            f"- Activity deviation: {int(deviation*100)}% below baseline\n"
            f"- Inactive duration: {int(hours)} hours\n"
            f"- Risk level: {severity.upper()}\n\n"
            f"Recommended Actions:\n"
            f"1. Immediate welfare check via phone call\n"
            f"2. Contact emergency contact if no response\n"
            f"3. Consider dispatching wellness team if inactive >48h\n"
            f"4. Review medication compliance and recent health events\n\n"
            f"Note: This is a simulated response. To enable real AI-powered alerts:\n"
            f"1. Get an API key from https://platform.openai.com/api-keys\n"
            f"2. Update OPENAI_API_KEY in .env file"
        )
    
    # Initialize OpenAI client with API key
    try:
        client = OpenAI(api_key=api_key)
        
        prompt = (
            f"You are a healthcare AI assistant. User {user_id} has shown an {int(deviation*100)}% drop in activity "
            f"and has been inactive for {int(hours)} hours. "
            f"Generate a concise, professional message for a healthcare provider with specific recommended next steps. "
            f"Include risk assessment and intervention priorities."
        )
        
        # Use the new OpenAI v1.0+ syntax
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a healthcare monitoring AI assistant providing actionable alerts for patient care."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        error_msg = str(e)
        
        # Handle quota exceeded error
        if "insufficient_quota" in error_msg or "429" in error_msg:
            severity = "critical" if deviation > 0.8 else "moderate"
            return (
                f"[OpenAI Quota Exceeded - Using Fallback Alert System]\n\n"
                f"⚠️ HEALTH ALERT - {severity.upper()} PRIORITY\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Patient ID: {user_id}\n"
                f"Status: {int(deviation*100)}% activity decline, {int(hours)} hours inactive\n\n"
                f"AUTOMATED INTERVENTION PROTOCOL:\n"
                f"────────────────────────────────\n"
            ) + (
                # Critical alerts (>80% deviation)
                f"🚨 IMMEDIATE ACTIONS REQUIRED:\n"
                f"1. Call 911 - Medical emergency likely\n"
                f"2. Dispatch welfare check immediately\n"
                f"3. Contact emergency contact NOW\n"
                f"4. Alert on-call physician\n"
                f"5. Prepare for hospital admission\n\n"
                f"⏰ Response Time: Within 30 minutes\n"
                if deviation > 0.8 else
                
                # High alerts (>60% deviation)
                f"⚠️ URGENT ACTIONS REQUIRED:\n"
                f"1. Contact patient via phone immediately\n"
                f"2. If no response, dispatch welfare check\n"
                f"3. Notify primary care physician\n"
                f"4. Review medication compliance\n"
                f"5. Schedule urgent telehealth visit\n\n"
                f"⏰ Response Time: Within 2 hours\n"
                if deviation > 0.6 else
                
                # Moderate alerts
                f"📋 STANDARD PROTOCOL:\n"
                f"1. Send automated check-in message\n"
                f"2. Schedule wellness call within 24h\n"
                f"3. Review activity history\n"
                f"4. Update care plan if needed\n"
                f"5. Document in patient record\n\n"
                f"⏰ Response Time: Within 24 hours\n"
            ) + (
                f"\n📊 RISK ASSESSMENT:\n"
                f"────────────────────\n"
                f"• Fall Risk: {'HIGH' if hours > 48 else 'MODERATE'}\n"
                f"• Medical Emergency: {'LIKELY' if deviation > 0.8 else 'POSSIBLE'}\n"
                f"• Hospitalization Risk: {int(deviation * 100)}%\n\n"
                f"💡 Note: OpenAI quota exceeded. To restore AI alerts:\n"
                f"   1. Add billing to OpenAI account\n"
                f"   2. Or wait for quota reset (monthly)\n"
                f"   3. Or use HuggingFace alternative (free)"
            )
        
        # Other errors
        return (
            f"[LLM Error]: {error_msg}\n\n"
            f"Fallback Alert for {user_id}:\n"
            f"• Activity: {int(deviation*100)}% below normal\n"
            f"• Inactive: {int(hours)} hours\n"
            f"• Action: Manual review required\n\n"
            f"To fix: Check your OpenAI API key and billing status."
        )
