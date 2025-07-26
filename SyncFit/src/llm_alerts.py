import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # Secure via .env or system var

def generate_alert_message(user_id, deviation, hours):
    prompt = (
        f"User {user_id} has shown an {int(deviation*100)}% drop in activity "
        f"and has been inactive for {int(hours)} hours. "
        f"Generate a message for a healthcare provider with recommended next steps."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM Error]: {e}"
