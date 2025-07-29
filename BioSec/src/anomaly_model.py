import pandas as pd
from pycaret.anomaly import setup, create_model, assign_model, save_model
from feature_extraction import preprocess_logs
from src.llm_explainer import summarize_alert # To run the anomaly detection product


def train_pycaret_model():
    """
    Trains an Isolation Forest model using PyCaret to detect anomalies in log data.

    The function preprocesses log data, sets up a PyCaret environment, creates an Isolation Forest model, 
    assigns the model to the data, and saves the trained model. It returns a DataFrame containing 
    user and IP encodings along with anomaly detection results.

    Returns:
        pd.DataFrame: A DataFrame with columns ['user_encoded', 'ip_encoded', 'is_anomaly', 'Anomaly'] 
        indicating the encoded user and IP information and anomaly detection results.
    """

    X, original_df = preprocess_logs("data/synthetic_logs.csv")
    X['id'] = range(len(X))

    exp = setup(data=X, index='id', session_id=42)
    model = create_model('iforest') #Isolation Forest will be used to detect anomalies
    results = assign_model(model)

    # Merge anomaly results back to original data
    output = pd.concat([original_df.reset_index(drop=True), results[['Anomaly', 'is_anomaly']]], axis=1)

    # Generate optional AI-based explanation per anomaly
    output['summary'] = output.apply(lambda row:
        summarize_alert(row['user_id'], row['timestamp'], row['action'], row['system'])
        if row['is_anomaly'] == 1 else "Normal", axis=1)

    output.to_csv("data/biosec_anomalies.csv", index=False)

    save_model(model, 'models/biosec_iforest')

    #return results[['user_encoded','ip_encoded', 'is_anomaly','Anomaly']]
    return output

if __name__ == "__main__": 
    df = train_pycaret_model()
    print(df[['user_id', 'action','system', 'is_anomaly', 'summary']].head())