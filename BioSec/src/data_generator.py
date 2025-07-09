# Since there's no user log, created my own synthetic data
import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta
import os

fake = Faker()
random.seed(42) # Fixed Results!
os.makedirs('data', exist_ok=True)
def generate_logs(n=1000): # 1000 samples for now
    users = [f"user_{i:03}" for i in range(50)]
    ip_pool = [fake.ipv4() for _ in range(30)]
    zones = ['wet_lab','genome_db','compute_mode','sequencer','cloud_storage']

    logs = []
    for _ in range(n):
        user = random.choice(users)
        timestamp = fake.date_time_between(start_date='-10d',end_date = 'now')
        ip = random.choice(ip_pool)
        resource = random.choice(zones)
        action = random.choices(['read', 'write', 'delete', 'exfiltrate'], weights=[0.6,0.2,0.15,0.05])[0]
        label = 1 if action == 'exfiltrate' else 0
        logs.append([user, timestamp, ip, resource, action, label])


    df = pd.DataFrame(logs, columns=['user', 'timestamp', 'ip', 'resource', 'action', 'label'])
    # Save in local directory
    df.to_csv('data/synthetic_logs.csv', index=False)   
    return df

if __name__ == "__main__":
    df = generate_logs()
    print ("DOne with synthetic data generation", len(df))