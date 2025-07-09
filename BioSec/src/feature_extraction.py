import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_logs(path="data/synthetic_logs.csv"):
    df = pd.read_csv(path, parse_dates=['timestmp'])

    # Categorical Encoding
    le_user = LabelEncoder()
    le_ip = LabelEncoder()
    le_resource = LabelEncoder()
    le_action = LabelEncoder()

    # Separate Encoders will be needed !
    df['user'] = le_user.fit_transform(df['user'])
    df['ip'] = le_ip.fit_transform(df['ip'])
    df['resource'] = le_resource.fit_transform(df['resource'])
    df['action'] = le_action.fit_transform(df['action'])

    # Extract time related features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek  

    features = ['user_encoded','ip_encoded', 'resource_encoded', 'action_encoded', 'hour', 'dayofweek']
    labels = 'is_anomaly'

    X = df[features]
    y = df[labels]
    
    return X,y

