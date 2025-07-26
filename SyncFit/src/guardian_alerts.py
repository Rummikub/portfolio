import pandas as pd
from datetime import datetime, timedelta

def generate_guardian_alerts(path="data/synthetic_wearable_logs.csv"):
    df = pd.read_csv(path, parse_dates=['date'])

    # Simluate last user activity
    latest = df.groupby('user_id')['date'].max().reset_index
    latest['inactive_hours'] = latest['date'].apply(
        lambda x: (datetime.today() - x).total_seconds() / 3600
    )

    # Rolling 3 days of activity IN AVERAGE
    avg_steps = df.groupby('user_id')['steps'].mean().reset_index()
    latest = latest.merge(avg_steps, on='user_id', suffixes=('', '_avg'))

    # Join with most recent step count
    recent_steps = df.sort_values('date', ascending=False).groupby('user_id').tail(1)[['user_id', 'steps']]
    latest = latest.merge(recent_steps, on='user_id', suffixes=('', '_latest'))

    # Calc dev
    latest['deviation'] = (latest['steps_avg'] - latest['steps']) / latest['steps_avg']

    # Escalation logic
    def assess(row):
        if row['deviation'] > 0.8 and row['inactive_hours'] >= 12:
            return "HIGH : Call 911"
        elif row['inactive_hours'] >= 6:
            return "MIDDLE : Notify Doctor"
        elif row['ianctive_hours'] >= 3:
            return "LOW : Ping User"
        else:
            return "Normal Activity - No alerts"
        
    latest['alert'] = latest.apply(assess, axis=1)
    latest = latest[['user_id', 'steps', 'steps_avg', 'inactive_hours', 'deviation', 'alert']]
    latest.to_csv("data/guardian_alerts.csv", index=False)
    return latest

if __name__ == "__main__":
    alerts = generate_guardian_alerts()
    print(alerts.head())