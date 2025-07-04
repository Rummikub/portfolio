import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle

def engineer_features(features):
    try:
        df = features.copy()
        print("Shape of input features:", df.shape)
        
        # 1. Building-related ratios and interactions
        df['building_coverage_ratio'] = df['building_coverage'] / df['building_density']
        df['building_density_squared'] = df['building_density'] ** 2
        df['height_to_coverage_ratio'] = df['avg_building_height'] / df['building_coverage']
        
        # 2. Temperature and building interactions
        df['temp_building_interaction'] = df['temp_mean'] * df['building_density']
        df['temp_coverage_interaction'] = df['temp_mean'] * df['building_coverage']
        df['temp_std_coverage_ratio'] = df['temp_std'] * df['building_coverage_ratio']
        
        # 3. LST and building interactions
        df['lst_building_interaction'] = df['lst'] * df['building_density']
        df['lst_coverage_interaction'] = df['lst'] * df['building_coverage']
        
        # 4. Wind and building interactions
        df['wind_building_interaction'] = df['wind_direction_mean'] * df['building_density']
        df['wind_speed_coverage_ratio'] = df['wind_speed_mean'] * df['building_coverage_ratio']
        
        # 5. Surface exposure features
        df['surface_exposure_index'] = df['building_coverage'] * df['solar_flux_mean'] * df['building_density']
        
        # 6. Additional targeted interactions
        df['temp_wind_building'] = df['temp_mean'] * df['wind_speed_mean'] * df['building_density']
        df['solar_building_exposure'] = df['solar_flux_mean'] * df['building_coverage_ratio']
        
        # 7. Location encoding (if present)
        if 'location' in df.columns:
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
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Initialize model with optimized parameters
        model = ExtraTreesRegressor(
            n_estimators=2000,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        # Make predictions
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        print("\nModel Performance Metrics:")
        print(f"Training R² Score: {train_r2:.4f}")
        print(f"Validation R² Score: {val_r2:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f}")
        print(f"Cross-validation R² scores: {cv_scores}")
        print(f"Mean CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15))
        
        return model, scaler, feature_importance
    
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
        
        # Get numeric columns
        # feature_columns = engineered_features.select_dtypes(include=['float64', 'int64']).columns.tolist()
        feature_columns = ['lst', 'temp_mean', 'temp_max', 'temp_min',
       'temp_std', 'humidity_mean', 'humidity_max', 'humidity_min']

        X = engineered_features[feature_columns]
        
        # Train model
        print("\nTraining model...")
        model, scaler, feature_importance = create_and_train_model(X, y)
        
        # Save model and components
        print("\nSaving model and components...")
        with open('uhi_model.pkl', 'wb') as f:
            pickle.dump(model, f, protocol=4)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f, protocol=4)
        with open('feature_columns.pkl', 'wb') as f:
            pickle.dump(feature_columns, f, protocol=4)
        
        # Save top features
        top_features = feature_importance.head(15)
        top_features.to_csv('top_features.csv', index=False)
        print("Model and components saved successfully!")
        
    except Exception as e:
        print(f"Main error: {str(e)}")
        print(f"Error occurred at line: {e.__traceback__.tb_lineno}")