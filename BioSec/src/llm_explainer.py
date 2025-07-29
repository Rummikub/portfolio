import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize_alert(user_id, timestamp, action, system):
    """
    Summarize an alert by generating a human-readable sentence explaining why an action 
    by a user on a system at a given timestamp might be risky.

    Parameters
    ----------
    user_id : str
        Unique identifier of the user performing the action.
    timestamp : str
        Timestamp of when the action was performed.
    action : str
        Action performed (e.g. login, file access, etc.).
    system : str
        System on which the action was performed.

    Returns
    -------
    str
        A human-readable sentence summarizing the alert.

    Why gpt-3.5-turbo : the model is faster and cheaper with high-quality for generating short, structured explanaions 
    Fit for summarzing anomalies alerts without the overhead of GPT-4 (which is the best option for me..)
    """
    prompt = (
        f"User {user_id} performed a suspicious '{action}' action "
        f"on system '{system}' at {timestamp}. Why might this be risky?"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"[LLM Error] {e}"