import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from .feature_builder import build_user_features

def train_churn_model():
    X, y = build_user_features()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    joblib.dump(model, "models/syncfit_churn_model.pkl")
    return model

if __name__ == "__main__":
    train_churn_model()
    print("Churn model trained and saved.")
