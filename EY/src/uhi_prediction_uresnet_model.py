import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.layers import UpSampling2D, Concatenate, Add, Dropout, Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import common functions for feature extraction from the CNN model file
try:
    from uhi_prediction_cnn_model import extract_advanced_features, create_enhanced_spatial_features, augment_data
    from math import radians, cos, sin, asin, sqrt
except ImportError:
    # Define the functions here if import fails
    print("Could not import functions from CNN model. Using local implementations.")
    
    def haversine(lon1, lat1, lon2, lat2):
        """Calculate the great circle distance between two points on the earth"""
        # Convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371  # Radius of earth in kilometers
        return c * r
    
    # Placeholder for the functions that would be imported
    # These will be defined if needed when running this file directly

# Define the U-ResNet architecture for UHI prediction
def residual_block(x, filters, kernel_size=3, strides=1, use_dropout=False):
    """
    Residual block with batch normalization and optional dropout
    """
    shortcut = x
    
    # First convolutional layer
    x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second convolutional layer
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # If stride > 1 or filter dimensions don't match, transform shortcut
    if strides > 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add skip connection
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    if use_dropout:
        x = Dropout(0.2)(x)
        
    return x

def encoder_block(x, filters, use_dropout=False):
    """
    Encoder block for the U-ResNet architecture
    """
    x = residual_block(x, filters, use_dropout=use_dropout)
    x = residual_block(x, filters, use_dropout=use_dropout)
    # Store for skip connection
    skip = x
    # Downsample
    x = MaxPooling2D(pool_size=(2, 2))(x)
    return x, skip

def decoder_block(x, skip, filters, use_dropout=False):
    """
    Decoder block for the U-ResNet architecture
    """
    # Upsample
    x = UpSampling2D(size=(2, 2))(x)
    # Concatenate with skip connection
    x = Concatenate()([x, skip])
    # Apply residual blocks
    x = residual_block(x, filters, use_dropout=use_dropout)
    x = residual_block(x, filters, use_dropout=use_dropout)
    return x

def create_uresnet_model(input_shape, output_dim=1):
    """
    Create a U-ResNet model for UHI prediction
    
    Args:
        input_shape: Tuple of input shape (height, width, channels)
        output_dim: Output dimension (1 for regression)
        
    Returns:
        Compiled U-ResNet model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv2D(64, 7, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Encoder path
    x, skip1 = encoder_block(x, 64)
    x, skip2 = encoder_block(x, 128)
    x, skip3 = encoder_block(x, 256, use_dropout=True)
    
    # Bridge
    x = residual_block(x, 512, use_dropout=True)
    x = residual_block(x, 512, use_dropout=True)
    
    # Decoder path
    x = decoder_block(x, skip3, 256, use_dropout=True)
    x = decoder_block(x, skip2, 128)
    x = decoder_block(x, skip1, 64)
    
    # Final convolution
    x = Conv2D(32, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Output layer - flatten and dense for regression
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(output_dim, activation='linear')(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='huber_loss',
        metrics=['mae', 'mse']
    )
    
    return model

def reshape_data_for_uresnet(X, img_size=(32, 32)):
    """
    Reshape 1D data into 2D images for U-ResNet processing
    
    Args:
        X: Input data, shape (n_samples, n_features)
        img_size: Size of the output images (height, width)
        
    Returns:
        Reshaped data with shape (n_samples, height, width, 1)
    """
    # Calculate if we need to pad the data
    n_samples = X.shape[0]
    n_features = X.shape[1]
    target_size = img_size[0] * img_size[1]
    
    if n_features < target_size:
        # Pad with zeros if we have fewer features than needed
        padding = np.zeros((n_samples, target_size - n_features))
        X_padded = np.hstack([X, padding])
    elif n_features > target_size:
        # If we have too many features, we could either:
        # 1. Truncate (as done here)
        X_padded = X[:, :target_size]
        # 2. Alternatively, resize to fit using PCA or feature selection
    else:
        X_padded = X
    
    # Reshape to 2D square images + single channel
    X_reshaped = X_padded.reshape((n_samples, img_size[0], img_size[1], 1))
    
    return X_reshaped

# Main execution
if __name__ == "__main__":
    try:
        print("Loading datasets...")
        # Check file existence before loading
        import os
        train_file = 'Training_data_uhi_index_2025-02-18.csv'
        test_file = 'Submission_template_UHI2025-v2.csv'
        lst_file = 'Landsat_LST.tiff'
        s2_file = 'S2_sample.tiff'
        
        # Verify all files exist
        missing_files = []
        for file in [train_file, test_file, lst_file, s2_file]:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            print(f"Error: The following files are missing: {', '.join(missing_files)}")
            print("Please ensure all required files are in the current directory.")
            exit(1)
            
        # Load training data
        train_data = pd.read_csv(train_file)
        train_data = train_data.drop('datetime', axis=1)
        
        # Load test data
        test_data = pd.read_csv(test_file)
        
        # Extract features from satellite imagery
        print("Extracting advanced features from Landsat LST...")
        lst_features_train = extract_advanced_features(
            lst_file, 
            train_data[['Longitude', 'Latitude']].values
        )
        lst_features_test = extract_advanced_features(
            lst_file, 
            test_data[['Longitude', 'Latitude']].values
        )
        
        print("Extracting advanced features from Sentinel-2...")
        s2_features_train = extract_advanced_features(
            s2_file, 
            train_data[['Longitude', 'Latitude']].values
        )
        s2_features_test = extract_advanced_features(
            s2_file, 
            test_data[['Longitude', 'Latitude']].values
        )
        
        print("Creating enhanced spatial features...")
        spatial_features_train = create_enhanced_spatial_features(
            train_data[['Longitude', 'Latitude']].values
        )
        spatial_features_test = create_enhanced_spatial_features(
            test_data[['Longitude', 'Latitude']].values
        )
        
        # Combine all features
        print("Combining features...")
        X_train = np.hstack([
            lst_features_train,
            s2_features_train,
            spatial_features_train,
            train_data[['Longitude', 'Latitude']].values  # Keep original coordinates
        ])
        X_test = np.hstack([
            lst_features_test,
            s2_features_test,
            spatial_features_test,
            test_data[['Longitude', 'Latitude']].values  # Keep original coordinates
        ])
        y_train = train_data['UHI Index'].values
        
        print(f"Initial feature set: {X_train.shape[1]} features")
        
        # Clean up the data
        print("Cleaning data...")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Data augmentation
        print("Performing advanced data augmentation...")
        X_train_aug, y_train_aug = augment_data(
            X_train, 
            y_train, 
            num_augmented=4000,  # More augmentation for deep learning
            strategies=['noise', 'jitter', 'smooth', 'adversarial']
        )
        print(f"Training data size after augmentation: {len(X_train_aug)} samples")
        
        # Split into training and validation sets
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train_aug, y_train_aug, test_size=0.2, random_state=42
        )
        
        # Preprocess features
        print("Preprocessing features with QuantileTransformer...")
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        X_train_scaled = scaler.fit_transform(X_train_split)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Reshape data for U-ResNet (2D format)
        print("Reshaping data for U-ResNet...")
        img_size = (32, 32)  # Choose an appropriate size based on feature count
        X_train_reshaped = reshape_data_for_uresnet(X_train_scaled, img_size)
        X_val_reshaped = reshape_data_for_uresnet(X_val_scaled, img_size)
        X_test_reshaped = reshape_data_for_uresnet(X_test_scaled, img_size)
        
        # Create and train U-ResNet model
        print("Creating U-ResNet model...")
        input_shape = X_train_reshaped.shape[1:]  # (height, width, channels)
        model = create_uresnet_model(input_shape)
        
        print(model.summary())
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
            ModelCheckpoint('uresnet_uhi_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        print("Training U-ResNet model...")
        history = model.fit(
            X_train_reshaped, y_train_split,
            validation_data=(X_val_reshaped, y_val),
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        val_loss, val_mae, val_mse = model.evaluate(X_val_reshaped, y_val)
        print(f"Validation MAE: {val_mae:.6f}")
        print(f"Validation MSE: {val_mse:.6f}")
        print(f"Validation RMSE: {np.sqrt(val_mse):.6f}")
        val_r2 = r2_score(y_val, model.predict(X_val_reshaped).flatten())
        print(f"Validation R²: {val_r2:.6f}")
        
        # Make predictions
        print("Generating predictions...")
        predictions = model.predict(X_test_reshaped).flatten()
        
        # Create submission file
        test_data['UHI Index'] = predictions
        test_data.to_csv('uresnet_submission.csv', index=False)
        print("Predictions saved to uresnet_submission.csv")
        
        # Save preprocessing components
        os.makedirs('uresnet_model', exist_ok=True)
        joblib.dump(scaler, 'uresnet_model/scaler.pkl')
        
        print("U-ResNet model and preprocessing components saved")
        
        # Calculate expected accuracy (using validation set as proxy)
        y_val_pred = model.predict(X_val_reshaped).flatten()
        mape = np.mean(np.abs((y_val - y_val_pred) / (y_val + 1e-10))) * 100
        expected_accuracy = 100 - mape
        print(f"Expected Accuracy: {expected_accuracy:.2f}%")
        
        print("\nU-ResNet model training and prediction complete!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
