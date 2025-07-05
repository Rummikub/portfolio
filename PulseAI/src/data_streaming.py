import json
import random
import time

def generate_synthetic_stream(output_file="data/synthetic_stream.jsonl", num_events=100):
    users = [f"user_{i:03}" for i in range(1, 11)]
    events = ["click", "search", "view", "add_to_cart", "checkout", "watch", "scroll"]
    device_types = ["mobile", "desktop", "tablet"]

    with open(output_file, "w") as f:
        for _ in range(num_events):
            record = {
                "user_id": random.choice(users),
                "event_type": random.choice(events),
                "device": random.choice(device_types),
                "timestamp": time.time() + random.uniform(0, 30)  # simulate recent traffic
            }
            f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    generate_synthetic_stream()
