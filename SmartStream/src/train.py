import pandas as pd
from data_ingestion import load_data
from feature_engineering import engineer_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def train_model(data_path="data/mobility_data_sample.csv"):
    df = load_data(data_path)
    X, y = engineer_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/smartstream_model.pkl")
    print("Model saved to models/smartstream_model.pkl")

if __name__ == "__main__":
    train_model()
