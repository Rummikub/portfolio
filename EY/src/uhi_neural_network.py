import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import rasterio
import xarray as xr

def calculate_ndvi(nir_band, red_band):
    """Calculate Normalized Difference Vegetation Index"""
    return (nir_band - red_band) / (nir_band + red_band + 1e-8)

def calculate_ndwi(nir_band, swir_band):
    """Calculate Normalized Difference Water Index"""
    return (nir_band - swir_band) / (nir_band + swir_band + 1e-8)

def load_and_preprocess_data(filepath, satellite_data_path=None):
    """
    Load and preprocess the data with additional features
    """
    # Load main dataset
    data = pd.read_csv(filepath)
    
    # If satellite data is available, add NDVI and NDWI
    if satellite_data_path:
        try:
            with rasterio.open(satellite_data_path) as src:
                # Extract bands (adjust band numbers based on your satellite data)
                nir_band = src.read(4)  # NIR band
                red_band = src.read(3)  # Red band
                swir_band = src.read(5)  # SWIR band
                
                # Calculate indices
                ndvi = calculate_ndvi(nir_band, red_band)
                ndwi = calculate_ndwi(nir_band, swir_band)
                
                # Add to dataframe (you'll need to match coordinates)
                data['NDVI'] = ndvi.flatten()
                data['NDWI'] = ndwi.flatten()
        except Exception as e:
            print(f"Could not load satellite data: {e}")
    
    # Add derived features
    data['Distance_to_center'] = np.sqrt(
        (data['Longitude'] - data['Longitude'].mean())**2 +
        (data['Latitude'] - data['Latitude'].mean())**2
    )
    
    return data

def create_model(input_dim):
    """
    Create a neural network model for UHI prediction
    """
    model = Sequential([
        # Input layer
        Dense(64, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.2),
        
        # Hidden layers
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(1, activation='linear')  # Linear activation for regression
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_model(X_train, y_train, X_val, y_val):
    """
    Train the neural network model
    """
    # Create model
    model = create_model(X_train.shape[1])
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return mse, r2

def main():
    # Load and preprocess data
    data = load_and_preprocess_data('Training_data_uhi_index_UHI2025-v2.csv')
    
    # Prepare features and target
    features = ['Longitude', 'Latitude', 'Distance_to_center']
    if 'NDVI' in data.columns:
        features.extend(['NDVI', 'NDWI'])
    
    X = data[features]
    y = data['UHI Index']
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model, history = train_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Evaluate model
    mse, r2 = evaluate_model(model, X_test_scaled, y_test)
    
    # Save model
    model.save('uhi_model.h5')
    
    # Save scaler
    import joblib
    joblib.dump(scaler, 'uhi_scaler.pkl')

if __name__ == "__main__":
    main()
