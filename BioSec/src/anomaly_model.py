import pandas as pd
from pycaret.anomaly import setup, create_model, assign_model, save_model
from feature_extraction import preprocess_logs

def train_pycaret_model():
    X, _ = preprocess_logs("data/synthetic_logs.csv")
    X['id'] = range(len(X))

    exp = setup(data=X, index='id', session_id=42)
    model = create_model('iforest') #Isolation Forest will be used to detect anomalies
    results = assign_model(model)

    save_model(model, 'models/biosec_iforest')

    return results[['user_encoded','ip_encoded', 'is_anomaly','Anomaly']]

if __name__ == "__main__": 
    df = train_pycaret_model()
    print(df.head())