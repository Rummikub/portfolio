import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle

def engineer_features(features):
    try:
        # Create a copy of the dataframe
        df = features.copy()
        print("Shape of input features:", df.shape)
        
        # 1. Temperature-Humidity Interactions
        df['temp_humidity_interaction'] = df['temp_mean'] * df['humidity_mean']
        df['temp_humidity_range'] = (df['temp_max'] - df['temp_min']) * (df['humidity_max'] - df['humidity_min'])
        
        # 2. Wind-related features
        df['wind_energy'] = df['wind_speed_mean'] ** 2
        df['wind_solar_interaction'] = df['wind_speed_mean'] * df['solar_flux_mean']
        
        # 3. Building-related features
        df['building_volume'] = df['building_density'] * df['avg_building_height']
        df['building_solar_exposure'] = df['building_coverage'] * df['solar_flux_mean']
        
        # 4. Temperature variations
        df['temp_range'] = df['temp_max'] - df['temp_min']
        df['temp_variation_coefficient'] = df['temp_std'] / df['temp_mean']
        
        # 5. Location-based features
        location_dummies = pd.get_dummies(df['location'], prefix='location')
        df = pd.concat([df, location_dummies], axis=1)
        df.drop('location', axis=1, inplace=True)
        
        return df
    
    except Exception as e:
        print(f"Error in feature engineering: {str(e)}")
        print(f"Error occurred at line: {e.__traceback__.tb_lineno}")
        raise

def create_and_train_model(X, y):
    try:
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train the model (without polynomial features for proper feature importance)
        model = ExtraTreesRegressor(
            n_estimators=500,
            bootstrap=True,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=20
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # First train on scaled data to get feature importance
        model.fit(X_train_scaled, y_train)
        
        # Get feature importance before polynomial transformation
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        print("\nTop 10 Most Important Features (before polynomial features):")
        print(feature_importance.head(10))
        
        # Now create polynomial features and train final model
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_val_poly = poly.transform(X_val_scaled)
        
        # Train final model with polynomial features
        model.fit(X_train_poly, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train_poly)
        val_pred = model.predict(X_val_poly)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        print(f"\nFinal Model Metrics:")
        print(f"Training R² Score: {train_r2:.4f}")
        print(f"Validation R² Score: {val_r2:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f}")
        
        return model, scaler, poly, feature_importance
    
    except Exception as e:
        print(f"Error in model training: {str(e)}")
        print(f"Error occurred at line: {e.__traceback__.tb_lineno}")
        raise

if __name__ == "__main__":
    try:
        # Load and prepare data
        print("Loading data...")
        train_df = pd.read_csv('Training_data_uhi_index_UHI2025-v2.csv')
        features = pd.read_pickle('train_features (1).pkl')
        y = train_df['UHI Index']
        
        # Engineer features
        print("\nEngineering features...")
        engineered_features = engineer_features(features)
        
        # Get all numeric columns for the model
        feature_columns = engineered_features.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Create final feature matrix
        X = engineered_features[feature_columns]
        
        # Train the model
        print("\nTraining model...")
        model, scaler, poly, feature_importance = create_and_train_model(X, y)
        
        # Save the model and components using protocol 4
        print("\nSaving model and components...")
        with open('uhi_model.pkl', 'wb') as f:
            pickle.dump(model, f, protocol=4)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f, protocol=4)
        with open('poly_features.pkl', 'wb') as f:
            pickle.dump(poly, f, protocol=4)
        with open('feature_columns.pkl', 'wb') as f:
            pickle.dump(feature_columns, f, protocol=4)
        
        # Save top 10 features
        top_10_features = feature_importance.head(10)
        top_10_features.to_csv('top_10_features.csv', index=False)
        print("Model and components saved successfully!")

    except Exception as e:
        print(f"Main error: {str(e)}")
        print(f"Error occurred at line: {e.__traceback__.tb_lineno}")
