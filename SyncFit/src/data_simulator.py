import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta
import os

os.makedirs('data', exist_ok=True)
fake = Faker()
random.seed(42)

def simulate_wearable_data(n_users=100, n_days=30):
    logs = []
    user_ids = [f"user_{i}" for i in range(n_users)]

    for user in user_ids:
        base_churn = random.random() < 0.2  # 20% churners
        for day in range(n_days):
            date = datetime.today() - timedelta(days=day)
            steps = random.randint(0, 12000) if not base_churn else random.randint(0, 4000)
            hr = random.randint(60, 160)
            class_attended = random.choices([0, 1], weights=[0.7, 0.3])[0] if not base_churn else 0
            logs.append([user, date.date(), steps, hr, class_attended, int(base_churn)])

    df = pd.DataFrame(logs, columns=['user_id', 'date', 'steps', 'heart_rate', 'class_attended', 'churned'])
    df.to_csv("data/synthetic_wearable_logs.csv", index=False)

if __name__ == "__main__":
    simulate_wearable_data()
    print("Data simulation completed.")
