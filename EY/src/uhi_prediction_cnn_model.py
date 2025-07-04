import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.ndimage import gaussian_filter, sobel, median_filter, laplace
from scipy.stats import skew, kurtosis
from math import radians, cos, sin, asin, sqrt
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Input, Dropout, BatchNormalization
from tensorflow.keras.layers import LSTM, GRU, Bidirectional, Concatenate, GlobalAveragePooling1D, TimeDistributed, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

# Utility functions from original model
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

def extract_advanced_features(tiff_path, coords, window_sizes=[3, 7, 11, 15, 21]):
    """Extract highly detailed features from satellite imagery"""
    with rasterio.open(tiff_path) as src:
        all_features = []
        num_coords = len(coords)
        
        for i, (lon, lat) in enumerate(coords):
            if i % 100 == 0:
                print(f"  Processing coordinate {i+1}/{num_coords}")
                
            try:
                # Convert geographic coordinates to pixel coordinates
                py, px = src.index(lon, lat)
                
                point_features = []
                
                # Extract features at different window sizes
                for size in window_sizes:
                    half_size = size // 2
                    # Ensure window is within image bounds
                    win_start_x = max(0, px - half_size)
                    win_start_y = max(0, py - half_size)
                    win_width = min(src.width - win_start_x, size)
                    win_height = min(src.height - win_start_y, size)
                    
                    window = Window(win_start_x, win_start_y, win_width, win_height)
                    
                    try:
                        # Read the window
                        pixel_array = src.read(1, window=window)
                        # Handle potential NaN values
                        pixel_array = np.nan_to_num(pixel_array, nan=0.0)
                        
                        # Skip empty or nearly empty windows
                        if pixel_array.size < 4 or np.all(pixel_array < 0.001):
                            point_features.extend([0] * 30)
                            continue
                        
                        # Apply multiple filters for advanced feature extraction
                        smoothed = gaussian_filter(pixel_array, sigma=1)
                        median_filtered = median_filter(pixel_array, size=3)
                        laplacian = laplace(gaussian_filter(pixel_array, sigma=1))
                        
                        # Calculate gradients
                        gradient_x = sobel(smoothed, axis=0)
                        gradient_y = sobel(smoothed, axis=1)
                        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                        
                        # Calculate distance from center to highest/lowest value
                        center_y, center_x = pixel_array.shape[0] // 2, pixel_array.shape[1] // 2
                        y_indices, x_indices = np.indices(pixel_array.shape)
                        
                        max_idx = np.unravel_index(np.argmax(pixel_array), pixel_array.shape)
                        min_idx = np.unravel_index(np.argmin(pixel_array), pixel_array.shape)
                        
                        dist_to_max = np.sqrt((max_idx[0] - center_y)**2 + (max_idx[1] - center_x)**2) / size
                        dist_to_min = np.sqrt((min_idx[0] - center_y)**2 + (min_idx[1] - center_x)**2) / size
                        
                        # Create a circular mask for radial analysis
                        y, x = np.indices(pixel_array.shape)
                        r = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                        r_int = r.astype(int)
                        
                        # Create radial profile with 8 bins
                        max_r = min(pixel_array.shape) // 2
                        step = max(1, max_r // 8)
                        radial_profile = [np.mean(pixel_array[r_int == i]) if np.sum(r_int == i) > 0 else 0 
                                         for i in range(0, max_r, step)]
                        # Pad to ensure consistent length
                        radial_profile = radial_profile[:8] + [0] * (8 - len(radial_profile))
                        
                        # Calculate texture features
                        texture_entropy = -np.sum(np.abs(np.diff(pixel_array, axis=0)) / pixel_array.size) - np.sum(np.abs(np.diff(pixel_array, axis=1)) / pixel_array.size)
                        texture_skewness = skew(pixel_array.flatten())
                        texture_kurtosis = kurtosis(pixel_array.flatten())
                        
                        # Compute quadrant statistics
                        half_y, half_x = pixel_array.shape[0] // 2, pixel_array.shape[1] // 2
                        q1 = pixel_array[:half_y, :half_x]
                        q2 = pixel_array[:half_y, half_x:]
                        q3 = pixel_array[half_y:, :half_x]
                        q4 = pixel_array[half_y:, half_x:]
                        
                        quadrant_means = [
                            np.mean(q) if q.size > 0 else 0 
                            for q in [q1, q2, q3, q4]
                        ]
                        
                        # Extract comprehensive set of features
                        features = [
                            # Basic statistics
                            np.mean(pixel_array),
                            np.std(pixel_array),
                            np.median(pixel_array),
                            np.percentile(pixel_array, 25),
                            np.percentile(pixel_array, 75),
                            np.max(pixel_array) - np.min(pixel_array),
                            
                            # Gradient and edge features
                            np.mean(gradient_magnitude),
                            np.std(gradient_magnitude),
                            np.mean(np.abs(laplacian)),
                            np.std(laplacian),
                            
                            # Spatial pattern features
                            dist_to_max,
                            dist_to_min,
                            
                            # Texture features
                            texture_entropy,
                            texture_skewness,
                            texture_kurtosis,
                            
                            # Filtered features
                            np.mean(smoothed),
                            np.mean(median_filtered),
                            
                            # Quadrant analysis
                            *quadrant_means,
                            
                            # Radial profile (captures circular patterns)
                            *radial_profile[:8],
                        ]
                        
                        point_features.extend(features)
                        
                    except Exception as e:
                        # If window reading fails, add zeros
                        point_features.extend([0] * 30)
                
                all_features.append(point_features)
                
            except Exception as e:
                # If coordinate conversion fails, add zeros
                all_features.append([0] * (30 * len(window_sizes)))
                
        return np.array(all_features)

def create_enhanced_spatial_features(coords):
    """Create advanced spatial features with urban morphology considerations"""
    # NYC reference points with UHI relevance
    landmarks = {
        'central_park': (-73.9654, 40.7829),      # Large green space
        'times_square': (-73.9855, 40.7580),      # High density urban
        'downtown': (-74.0060, 40.7128),          # Financial district
        'harlem': (-73.9465, 40.8116),            # Residential area
        'brooklyn': (-73.9442, 40.6782),          # Outer borough
        'east_river': (-73.9762, 40.7678),        # Water body
        'hudson_river': (-74.0099, 40.7258),      # Water body
        'prospect_park': (-73.9701, 40.6602),     # Large green space in Brooklyn
        'flushing_meadows': (-73.8458, 40.7463),  # Queens park
    }
    
    # Areas with expected high UHI
    uhi_hotspots = [
        (-73.9855, 40.7580),  # Times Square
        (-73.9844, 40.7484),  # Midtown
        (-74.0060, 40.7128),  # Downtown
        (-73.9712, 40.7831),  # Upper East Side
        (-73.9210, 40.7648),  # Long Island City
    ]
    
    # Green spaces (cooling effect)
    green_spaces = [
        (-73.9654, 40.7829),  # Central Park
        (-73.9701, 40.6602),  # Prospect Park
        (-73.8458, 40.7463),  # Flushing Meadows
        (-73.9555, 40.7821),  # Riverside Park
        (-73.8743, 40.8501),  # Van Cortlandt Park
    ]
    
    # Water bodies (cooling effect)
    water_bodies = [
        (-73.9762, 40.7678),  # East River
        (-74.0099, 40.7258),  # Hudson River
        (-74.0401, 40.6435),  # Upper Bay
        (-73.7949, 40.7829),  # Flushing Bay
        (-73.8340, 40.6473),  # Jamaica Bay
    ]
    
    features = []
    
    for lon, lat in coords:
        point_features = []
        
        # Distance to landmarks
        for name, (lm_lon, lm_lat) in landmarks.items():
            dist = haversine(lon, lat, lm_lon, lm_lat)
            point_features.append(dist)
            
            # Directional features (enables recognizing patterns in specific directions)
            dx = lon - lm_lon  # Longitude difference
            dy = lat - lm_lat  # Latitude difference
            angle = np.arctan2(dy, dx)  # Angle in radians
            
            # Add sin/cos of angle for circular representation
            point_features.append(np.sin(angle))
            point_features.append(np.cos(angle))
        
        # Distance to nearest UHI hotspot
        hotspot_dists = [haversine(lon, lat, h_lon, h_lat) for h_lon, h_lat in uhi_hotspots]
        point_features.append(min(hotspot_dists))
        
        # Average distance to top 3 closest hotspots (local UHI influence)
        if hotspot_dists:
            point_features.append(np.mean(sorted(hotspot_dists)[:3]))
        else:
            point_features.append(0)
        
        # Distance to nearest green space (cooling effect)
        green_dists = [haversine(lon, lat, g_lon, g_lat) for g_lon, g_lat in green_spaces]
        point_features.append(min(green_dists) if green_dists else 0)
        
        # Distance to nearest water body (cooling effect)
        water_dists = [haversine(lon, lat, w_lon, w_lat) for w_lon, w_lat in water_bodies]
        point_features.append(min(water_dists) if water_dists else 0)
        
        # Calculate "Urban Thermal Gradient" - ratio of distances to cooling vs heating elements
        cooling_dist = min(min(green_dists, default=10), min(water_dists, default=10))
        heating_dist = min(hotspot_dists, default=1)
        
        # Avoid division by zero
        if heating_dist > 0:
            thermal_gradient = cooling_dist / heating_dist
        else:
            thermal_gradient = cooling_dist
            
        point_features.append(thermal_gradient)
        
        # Urban vs. natural influence score
        urban_influence = np.sum([1 / (d + 0.1) for d in hotspot_dists])
        natural_influence = np.sum([1 / (d + 0.1) for d in green_dists + water_dists])
        urban_natural_ratio = urban_influence / (natural_influence + 0.1)
        point_features.append(urban_natural_ratio)
        
        # Calculate spatial density features
        density_1km = np.sum([1 for d in hotspot_dists if d < 1])
        density_3km = np.sum([1 for d in hotspot_dists if d < 3])
        point_features.extend([density_1km, density_3km])
        
        # Spatial pattern recognition features
        # Identify if point is between major hotspots (interpolation features)
        is_between_hotspots = 0
        if len(hotspot_dists) >= 2:
            # Get two closest hotspots
            closest_indices = np.argsort(hotspot_dists)[:2]
            h1_lon, h1_lat = uhi_hotspots[closest_indices[0]]
            h2_lon, h2_lat = uhi_hotspots[closest_indices[1]]
            
            # Calculate if point is between these hotspots using dot product
            v1 = (lon - h1_lon, lat - h1_lat)
            v2 = (h2_lon - h1_lon, h2_lat - h1_lat)
            
            # Normalize v2
            v2_norm = np.sqrt(v2[0]**2 + v2[1]**2)
            if v2_norm > 0:
                v2 = (v2[0]/v2_norm, v2[1]/v2_norm)
                
                # Dot product
                dot_prod = v1[0]*v2[0] + v1[1]*v2[1]
                
                # Check if point is along the line between hotspots
                if 0 < dot_prod < haversine(h1_lon, h1_lat, h2_lon, h2_lat):
                    # Calculate perpendicular distance to line
                    cross_prod = abs(v1[0]*v2[1] - v1[1]*v2[0])
                    perp_dist = cross_prod * v2_norm
                    
                    # If close to line connecting hotspots
                    if perp_dist < 0.5:  # Within 500m
                        is_between_hotspots = 1
        
        point_features.append(is_between_hotspots)
        
        features.append(point_features)
        
    return np.array(features)

def augment_data(X, y, num_augmented=1000, strategies=['noise', 'jitter', 'smooth', 'adversarial']):
    """Advanced data augmentation with multiple strategies"""
    X_aug_all = []
    y_aug_all = []
    
    # 1. Add noise - classic augmentation
    if 'noise' in strategies:
        indices = np.random.choice(len(X), size=num_augmented//4, replace=True)
        X_noise = X[indices].copy()
        y_noise = y[indices].copy()
        
        # Add scaled noise to each feature
        for i in range(X_noise.shape[1]):
            feature_std = np.std(X[:, i]) * 0.01  # 1% of std
            noise = np.random.normal(0, feature_std, size=len(X_noise))
            X_noise[:, i] += noise
            
        # Add small noise to targets
        y_noise += np.random.normal(0, np.std(y) * 0.004, size=len(y_noise))
        
        X_aug_all.append(X_noise)
        y_aug_all.append(y_noise)
    
    # 2. Coordinate jittering - move points slightly
    if 'jitter' in strategies:
        indices = np.random.choice(len(X), size=num_augmented//4, replace=True)
        X_jitter = X[indices].copy()
        y_jitter = y[indices].copy()
        
        # Only slightly modify coordinate features (assuming last 2 cols are coords)
        lon_std = np.std(X[:, -2]) * 0.001  # Very small shift
        lat_std = np.std(X[:, -1]) * 0.001
        
        X_jitter[:, -2] += np.random.normal(0, lon_std, size=len(X_jitter))
        X_jitter[:, -1] += np.random.normal(0, lat_std, size=len(X_jitter))
        
        # Adjust target based on local gradient (if we moved slightly in feature space)
        y_jitter += np.random.normal(0, np.std(y) * 0.003, size=len(y_jitter))
        
        X_aug_all.append(X_jitter)
        y_aug_all.append(y_jitter)
    
    # 3. Smoothed interpolation - create new samples as weighted combinations
    if 'smooth' in strategies:
        for _ in range(num_augmented//4):
            # Select two random samples
            idx1, idx2 = np.random.choice(len(X), size=2, replace=False)
            
            # Create a random interpolation
            alpha = np.random.beta(0.4, 0.4)  # Beta distribution for smoother interpolation
            
            # Create new sample as weighted combination
            X_interp = alpha * X[idx1] + (1 - alpha) * X[idx2]
            y_interp = alpha * y[idx1] + (1 - alpha) * y[idx2]
            
            X_aug_all.append(X_interp.reshape(1, -1))
            y_aug_all.append(np.array([y_interp]))
            
    # 4. Add adversarial examples (challenging samples)
    if 'adversarial' in strategies:
        indices = np.random.choice(len(X), size=num_augmented//4, replace=True)
        X_adv = X[indices].copy()
        y_adv = y[indices].copy()
        
        # Train a simple model to determine gradient direction
        from sklearn.ensemble import GradientBoostingRegressor
        simple_model = GradientBoostingRegressor(n_estimators=50, random_state=44)
        simple_model.fit(X, y)
        
        # Predict with the model
        preds = simple_model.predict(X_adv)
        
        # For samples with high error, move features in direction that would reduce error
        errors = y_adv - preds
        
        # Get feature importances to determine which features to perturb
        importances = simple_model.feature_importances_
        top_feature_indices = np.argsort(importances)[-5:]  # Top 5 important features
        
        # Perturb features in direction to reduce error (adversarial)
        for idx in top_feature_indices:
            # Scale perturbation by feature importance and error
            perturbation = np.sign(errors) * np.std(X[:, idx]) * 0.02 * importances[idx]
            X_adv[:, idx] += perturbation
            
        X_aug_all.append(X_adv)
        y_aug_all.append(y_adv)
    
    # Combine original and augmented data
    X_combined = np.vstack([X] + X_aug_all)
    y_combined = np.hstack([y] + y_aug_all)
    
    # Shuffle the combined data
    indices = np.arange(len(X_combined))
    np.random.seed(42)
    np.random.shuffle(indices)
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]
    
    return X_combined, y_combined

# CNN Model creation function
def create_cnn_model(input_shape, window_sizes=[5, 10, 15]):
    """Create a multi-scale CNN model for UHI prediction"""
    # Input layer
    inputs = Input(shape=(input_shape,))
    
    # Reshape for 1D CNN
    reshaped = Reshape((input_shape, 1))(inputs)
    
    # Multiple convolutional blocks with different window sizes
    conv_outputs = []
    
    for window_size in window_sizes:
        # Convolutional branch
        conv = Conv1D(64, kernel_size=window_size, padding='same', activation='relu')(reshaped)
        conv = BatchNormalization()(conv)
        conv = Conv1D(128, kernel_size=window_size//2 or 1, padding='same', activation='relu')(conv)
        conv = BatchNormalization()(conv)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Conv1D(256, kernel_size=window_size//3 or 1, padding='same', activation='relu')(conv)
        conv = BatchNormalization()(conv)
        conv = MaxPooling1D(pool_size=2)(conv)
        
        # Feature extraction
        gap = GlobalAveragePooling1D()(conv)
        conv_outputs.append(gap)
    
    # Add a recurrent branch to capture sequential patterns
    gru = Bidirectional(GRU(128, return_sequences=True))(reshaped)
    gru = GlobalAveragePooling1D()(gru)
    conv_outputs.append(gru)
    
    # Merge all branches
    merged = Concatenate()(conv_outputs)
    
    # Dense layers for regression
    dense = Dense(256, activation='relu')(merged)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.4)(dense)
    dense = Dense(128, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.3)(dense)
    dense = Dense(64, activation='relu')(dense)
    dense = Dropout(0.2)(dense)
    
    # Output layer
    output = Dense(1, activation='linear')(dense)
    
    # Create model
    model = Model(inputs=inputs, outputs=output)
    
    # Compile with appropriate loss and optimizer
    model.compile(
        loss='huber_loss',  # Robust to outliers
        optimizer=Adam(learning_rate=0.001),
        metrics=['mae', 'mse']
    )
    
    return model

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
            num_augmented=4000,  # More augmented data for CNN
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
        
        # Create and train CNN model
        print("Creating CNN model...")
        input_shape = X_train_scaled.shape[1]
        model = create_cnn_model(input_shape)
        
        print(model.summary())
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
            ModelCheckpoint('cnn_uhi_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        print("Training CNN model...")
        history = model.fit(
            X_train_scaled, y_train_split,
            validation_data=(X_val_scaled, y_val),
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        val_loss, val_mae, val_mse = model.evaluate(X_val_scaled, y_val)
        print(f"Validation MAE: {val_mae:.6f}")
        print(f"Validation MSE: {val_mse:.6f}")
        print(f"Validation RMSE: {np.sqrt(val_mse):.6f}")
        val_r2 = r2_score(y_val, model.predict(X_val_scaled).flatten())
        print(f"Validation R²: {val_r2:.6f}")
        
        # Make predictions
        print("Generating predictions...")
        predictions = model.predict(X_test_scaled).flatten()
        
        # Create submission file
        test_data['UHI Index'] = predictions
        test_data.to_csv('cnn_submission.csv', index=False)
        print("Predictions saved to cnn_submission.csv")
        
        # Save preprocessing components
        os.makedirs('cnn_model', exist_ok=True)
        joblib.dump(scaler, 'cnn_model/scaler.pkl')
        
        print("CNN model and preprocessing components saved")
        
        # Calculate expected accuracy (using validation set as proxy)
        y_val_pred = model.predict(X_val_scaled).flatten()
        mape = np.mean(np.abs((y_val - y_val_pred) / (y_val + 1e-10))) * 100
        expected_accuracy = 100 - mape
        print(f"Expected Accuracy: {expected_accuracy:.2f}%")
        
        print("\nCNN model training and prediction complete!")
    except Exception as e:
        print(f"An error occurred: {e}")
