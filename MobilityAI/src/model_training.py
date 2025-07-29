from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
import joblib

from src.model_training import train_model
model = train_model(X,y)

def train_model(X, y, save_path='models/GDB_model.pkl'):
    """
    Trains a Gradient Boosting Regressor model on the provided dataset and evaluates its performance.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    save_path (str, optional): File path to save the trained model. Defaults to 'models/GDB_model.pkl'.

    Returns:
    model: Trained Gradient Boosting Regressor model.

    This function splits the data into training and testing sets, trains a Gradient Boosting 
    Regressor model, computes the RMSE on the test set, prints the RMSE, saves the model to the 
    specified path, and returns the trained model.
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"Train Done ..RMSE: {rmse:.2f} seconds")

    joblib.dump(model, save_path)
    return model