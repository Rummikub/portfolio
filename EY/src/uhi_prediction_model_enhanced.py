import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel, mutual_info_regression, f_regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter, sobel, laplace
from math import radians, cos, sin, asin, sqrt
import warnings
import joblib
import os
import xgboost as xgb

warnings.filterwarnings('ignore')

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

def extract_enhanced_features(tiff_path, coords, window_sizes=[5, 15]):
    """Extract optimized features from satellite imagery"""
    with rasterio.open(tiff_path) as src:
        all_features = []
        num_coords = len(coords)
        
        for i, (lon, lat) in enumerate(coords):
            if i % 1000 == 0:
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
                        
                        if pixel_array.size > 0:
                            # Calculate useful features
                            pixel_mean = np.mean(pixel_array)
                            pixel_std = np.std(pixel_array)
                            pixel_median = np.median(pixel_array)
                            pixel_min = np.min(pixel_array)
                            pixel_max = np.max(pixel_array)
                            
                            # Apply smoothing
                            smoothed = gaussian_filter(pixel_array, sigma=1.0)
                            
                            # Calculate gradients
                            gradient_x = sobel(smoothed, axis=0)
                            gradient_y = sobel(smoothed, axis=1)
                            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                            
                            # Calculate Laplacian for edge detection
                            laplacian = laplace(smoothed)
                            
                            # Calculate key statistics
                            features = [
                                pixel_mean,                        # Mean value
                                pixel_std,                         # Standard deviation
                                pixel_median,                      # Median
                                pixel_min,                         # Minimum
                                pixel_max,                         # Maximum
                                np.percentile(pixel_array, 25),    # 1st quartile
                                np.percentile(pixel_array, 75),    # 3rd quartile
                                np.percentile(pixel_array, 95),    # 95th percentile
                                np.mean(gradient_magnitude),       # Mean gradient
                                np.std(gradient_magnitude),        # Gradient std
                                np.mean(laplacian),                # Mean Laplacian
                                np.std(laplacian),                 # Laplacian std
                            ]
                            
                            point_features.extend(features)
                        else:
                            point_features.extend([0] * 12)
                        
                    except Exception as e:
                        point_features.extend([0] * 12)
                
                # Extract radial feature (multiple rings)
                for radius in [5, 10, 15]:
                    try:
                        # Create a circular mask
                        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
                        mask = x*x + y*y <= radius*radius
                        
                        # Ensure window is within image bounds
                        win_start_x = max(0, px - radius)
                        win_start_y = max(0, py - radius)
                        win_width = min(src.width - win_start_x, 2*radius+1)
                        win_height = min(src.height - win_start_y, 2*radius+1)
                        
                        window = Window(win_start_x, win_start_y, win_width, win_height)
                        
                        pixel_array = src.read(1, window=window)
                        pixel_array = np.nan_to_num(pixel_array, nan=0.0)
                        
                        # Apply mask if sizes match
                        if pixel_array.shape[0] == mask.shape[0] and pixel_array.shape[1] == mask.shape[1]:
                            masked_array = pixel_array[mask]
                            
                            # Compute features for this radius
                            if masked_array.size > 0:
                                ring_features = [
                                    np.mean(masked_array),
                                    np.std(masked_array),
                                    np.median(masked_array),
                                    np.min(masked_array),
                                    np.max(masked_array),
                                ]
                                point_features.extend(ring_features)
                            else:
                                point_features.extend([0] * 5)
                        else:
                            point_features.extend([0] * 5)
                            
                    except Exception as e:
                        point_features.extend([0] * 5)
                
                all_features.append(point_features)
                
            except Exception as e:
                # If coordinate conversion fails, add zeros
                all_features.append([0] * (12 * len(window_sizes) + 5 * 3))
                
        return np.array(all_features)

def extract_sentinel2_features(tiff_path, coords, window_sizes=[5, 15]):
    """Extract features from Sentinel-2 imagery"""
    with rasterio.open(tiff_path) as src:
        all_features = []
        num_coords = len(coords)
        
        print(f"Extracting features from Sentinel-2 imagery...")
        print(f"Image has {src.count} bands, shape: {src.shape}")
        
        for i, (lon, lat) in enumerate(coords):
            if i % 1000 == 0:
                print(f"  Processing S2 coordinate {i+1}/{num_coords}")
                
            try:
                # Convert geographic coordinates to pixel coordinates
                py, px = src.index(lon, lat)
                
                point_features = []
                
                # Extract features from each band
                for band in range(1, min(src.count + 1, 13)):  # Process up to 12 bands if available
                    # Extract windows of different sizes
                    for size in window_sizes:
                        half_size = size // 2
                        # Ensure window is within image bounds
                        win_start_x = max(0, px - half_size)
                        win_start_y = max(0, py - half_size)
                        win_width = min(src.width - win_start_x, size)
                        win_height = min(src.height - win_start_y, size)
                        
                        window = Window(win_start_x, win_start_y, win_width, win_height)
                        
                        try:
                            # Read the window for this band
                            pixel_array = src.read(band, window=window)
                            # Handle potential NaN values
                            pixel_array = np.nan_to_num(pixel_array, nan=0.0)
                            
                            if pixel_array.size > 0:
                                # Calculate basic statistics
                                pixel_mean = np.mean(pixel_array)
                                pixel_std = np.std(pixel_array)
                                pixel_min = np.min(pixel_array)
                                pixel_max = np.max(pixel_array)
                                pixel_range = pixel_max - pixel_min
                                
                                # Calculate additional statistics
                                if pixel_array.size > 1:
                                    # Apply smoothing and calculate texture
                                    smoothed = gaussian_filter(pixel_array, sigma=1.0)
                                    # Calculate gradients for texture
                                    gradient_x = sobel(smoothed, axis=0)
                                    gradient_y = sobel(smoothed, axis=1)
                                    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                                    
                                    # Add features
                                    band_features = [
                                        pixel_mean,  # Mean value
                                        pixel_std,   # Standard deviation
                                        pixel_min,   # Minimum
                                        pixel_max,   # Maximum
                                        pixel_range, # Range
                                        np.mean(gradient_magnitude),  # Mean gradient (texture)
                                        np.std(gradient_magnitude),   # Gradient std (texture)
                                    ]
                                else:
                                    band_features = [pixel_mean, 0, pixel_min, pixel_max, 0, 0, 0]
                                
                                point_features.extend(band_features)
                            else:
                                point_features.extend([0] * 7)
                                
                        except Exception as e:
                            point_features.extend([0] * 7)
                
                # Calculate spectral indices if we have the right bands
                # Assuming common band order: B2(Blue), B3(Green), B4(Red), B8(NIR)
                if src.count >= 4:
                    try:
                        # Get band values at the exact point for indices
                        blue = src.read(1, window=Window(px, py, 1, 1))[0, 0]
                        green = src.read(2, window=Window(px, py, 1, 1))[0, 0]
                        red = src.read(3, window=Window(px, py, 1, 1))[0, 0]
                        nir = src.read(4, window=Window(px, py, 1, 1))[0, 0]
                        
                        # Handle zeros to avoid division by zero
                        blue = max(blue, 0.0001)
                        green = max(green, 0.0001)
                        red = max(red, 0.0001)
                        nir = max(nir, 0.0001)
                        
                        # Calculate vegetation indices
                        ndvi = (nir - red) / (nir + red)  # Normalized Difference Vegetation Index
                        ndbi = (red - nir) / (red + nir)  # Normalized Difference Built-up Index
                        
                        # Add more indices
                        savi = (nir - red) * 1.5 / (nir + red + 0.5)  # Soil Adjusted Vegetation Index
                        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)  # Enhanced Vegetation Index
                        
                        # Add the indices to features
                        point_features.extend([ndvi, ndbi, savi, evi])
                    except Exception as e:
                        point_features.extend([0, 0, 0, 0])
                else:
                    point_features.extend([0, 0, 0, 0])
                
                all_features.append(point_features)
                
            except Exception as e:
                # If coordinate conversion fails, add zeros
                # Estimate the feature length based on bands and window sizes
                feature_len = src.count * len(window_sizes) * 7 + 4  # +4 for spectral indices
                all_features.append([0] * feature_len)
        
        return np.array(all_features)

def create_enhanced_spatial_features(coords):
    """Create comprehensive spatial features"""
    # NYC key locations (extended set)
    reference_points = {
        'central_park': (-73.9654, 40.7829),     # Central Park
        'times_square': (-73.9855, 40.7580),     # Times Square
        'downtown': (-74.0060, 40.7128),         # Downtown
        'east_river_1': (-73.9762, 40.7678),     # East River point 1
        'east_river_2': (-73.9500, 40.7500),     # East River point 2
        'hudson_river_1': (-74.0099, 40.7258),   # Hudson River point 1
        'hudson_river_2': (-74.0200, 40.7700),   # Hudson River point 2
        'brooklyn': (-73.9500, 40.6500),         # Brooklyn
        'queens': (-73.8500, 40.7500),           # Queens
        'bronx': (-73.9000, 40.8500),            # Bronx
    }
    
    features = []
    
    # Center of Manhattan
    center_lon, center_lat = -73.9712, 40.7831
    
    for lon, lat in coords:
        # Create features
        x, y = lon, lat
        
        # Calculate distances to key points
        distances = {}
        for name, (ref_lon, ref_lat) in reference_points.items():
            dist = haversine(lon, lat, ref_lon, ref_lat)
            distances[name] = dist
            
        # Distance to center of Manhattan
        dist_center = haversine(lon, lat, center_lon, center_lat)
        angle = np.arctan2(lat - center_lat, lon - center_lon)
        
        # Distance to nearest water body
        east_river_dist = min(distances['east_river_1'], distances['east_river_2'])
        hudson_river_dist = min(distances['hudson_river_1'], distances['hudson_river_2'])
        nearest_water_dist = min(east_river_dist, hudson_river_dist)
        
        # Cartesian coordinates for regional calculations
        x_km = (lon - center_lon) * 111 * cos(radians(center_lat))  # km in x direction
        y_km = (lat - center_lat) * 111  # km in y direction
        
        # Create features - focusing on the most informative
        point_features = [
            x, y,                                      # Raw coordinates
            x**2, y**2, x*y,                           # Quadratic terms
            np.sin(x * 100), np.cos(x * 100),          # Cyclical features X
            np.sin(y * 100), np.cos(y * 100),          # Cyclical features Y
            distances['central_park'],                 # Distance to Central Park
            distances['times_square'],                 # Distance to Times Square
            distances['downtown'],                     # Distance to Downtown
            east_river_dist,                           # Distance to East River
            hudson_river_dist,                         # Distance to Hudson River
            nearest_water_dist,                        # Distance to nearest water
            np.exp(-distances['central_park']/2),      # Exponential decay - Park
            np.exp(-distances['times_square']/2),      # Exponential decay - Times Square
            np.exp(-nearest_water_dist/2),             # Exponential decay - Water
            dist_center,                               # Distance from center
            np.exp(-dist_center/2),                    # Exponential decay from center
            np.sin(angle), np.cos(angle),              # Directional features
            x_km, y_km,                                # Cartesian coordinates (km)
            x_km**2, y_km**2, x_km*y_km,               # Quadratic terms (km)
            abs(x_km) + abs(y_km),                     # Manhattan distance
            np.sqrt(x_km**2 + y_km**2),                # Euclidean distance
            np.arctan2(y_km, x_km),                    # Angle in radians
        ]
        
        features.append(point_features)
        
    return np.array(features)

def enhanced_augmentation(X, y, num_samples=2000):
    """Targeted data augmentation to enhance model performance"""
    print(f"Original data: {X.shape[0]} samples")
    
    # Parameters
    noise_level = 0.0005
    
    # Define weights to focus on extreme values
    threshold_high = np.percentile(y, 95)
    threshold_low = np.percentile(y, 5)
    weights = np.ones_like(y) * 0.5
    weights[y >= threshold_high] = 5.0  # Heavy weight on high UHI areas
    weights[y <= threshold_low] = 5.0   # Heavy weight on low UHI areas
    
    # Add some weight to mid-range values with high variance
    feature_vars = np.var(X, axis=1)
    high_var_indices = np.where((y > threshold_low) & (y < threshold_high) & 
                              (feature_vars > np.percentile(feature_vars, 75)))[0]
    weights[high_var_indices] = 2.0    # Medium weight on high-variance areas
    
    # Normalize weights to probabilities
    weights = weights / np.sum(weights)
    
    # Generate new samples
    new_X = []
    new_y = []
    
    print(f"Creating augmented samples...")
    # Weighted sampling with small noise
    indices = np.random.choice(np.arange(len(X)), size=num_samples, replace=True, p=weights)
    
    for idx in indices:
        # Add small noise to create new sample
        noise = np.random.normal(0, noise_level, X[idx].shape)
        new_X.append(X[idx] + noise)
        new_y.append(y[idx])
    
    # Add SMOTE-like interpolation for extreme values
    print("Creating interpolated samples for extreme values...")
    extreme_indices = np.where((y >= threshold_high) | (y <= threshold_low))[0]
    
    if len(extreme_indices) >= 5:
        from sklearn.neighbors import NearestNeighbors
        
        # Find neighbors among extreme values
        X_extreme = X[extreme_indices]
        y_extreme = y[extreme_indices]
        
        nn = NearestNeighbors(n_neighbors=5).fit(X_extreme)
        _, indices = nn.kneighbors(X_extreme)
        
        # Create interpolated samples
        for i in range(min(500, len(extreme_indices))):
            idx = np.random.randint(0, len(extreme_indices))
            neighbor_idx = indices[idx, np.random.randint(1, 5)]  # Skip self
            
            # Interpolation factor
            alpha = np.random.beta(0.4, 0.4)  # Beta distribution for more diverse interpolation
            
            # Interpolate
            interp_X = X_extreme[idx] * alpha + X_extreme[neighbor_idx] * (1 - alpha)
            interp_y = y_extreme[idx] * alpha + y_extreme[neighbor_idx] * (1 - alpha)
            
            # Add small noise to avoid duplicates
            interp_X += np.random.normal(0, 0.0001, interp_X.shape)
            
            new_X.append(interp_X)
            new_y.append(interp_y)
    
    # Combine with original data
    X_combined = np.vstack([X, np.array(new_X)])
    y_combined = np.concatenate([y, np.array(new_y)])
    
    print(f"After augmentation: {len(y_combined)} samples")
    return X_combined, y_combined

def hybrid_feature_selection(X, y, n_features=60):
    """Enhanced feature selection using multiple methods"""
    print(f"Selecting top {n_features} features from {X.shape[1]} total features")
    
    # Method 1: Mutual Information
    try:
        mi_scores = mutual_info_regression(X, y)
        mi_scores = np.nan_to_num(mi_scores, nan=0.0)
        mi_indices = np.argsort(mi_scores)[-n_features:]
    except:
        mi_indices = np.array([])
    
    # Method 2: F-regression
    try:
        f_scores, _ = f_regression(X, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0)
        f_indices = np.argsort(f_scores)[-n_features:]
    except:
        f_indices = np.array([])
    
    # Method 3: Tree-based feature importance
    et = ExtraTreesRegressor(n_estimators=100, random_state=42)
    et.fit(X, y)
    et_scores = et.feature_importances_
    et_indices = np.argsort(et_scores)[-n_features:]
    
    # Method 4: Lasso regression for linear feature selection
    try:
        lasso = Lasso(alpha=0.01, max_iter=10000, tol=0.001)
        lasso.fit(X, y)
        lasso_scores = np.abs(lasso.coef_)
        lasso_indices = np.argsort(lasso_scores)[-n_features:]
    except:
        lasso_indices = np.array([])
    
    # Combine all unique indices
    all_indices = np.unique(np.concatenate([
        mi_indices, f_indices, et_indices, lasso_indices
    ]))
    
    # If we have too many, prioritize features that appear in multiple methods
    if len(all_indices) > n_features:
        # Count occurrences of each index
        index_counts = {}
        for idx in np.concatenate([mi_indices, f_indices, et_indices, lasso_indices]):
            if idx in index_counts:
                index_counts[idx] += 1
            else:
                index_counts[idx] = 1
        
        # Sort by count (descending) and then by tree importance (descending)
        sorted_indices = sorted(all_indices, 
                              key=lambda idx: (index_counts.get(idx, 0), et_scores[idx]), 
                              reverse=True)
        final_indices = np.array(sorted_indices[:n_features])
    else:
        final_indices = all_indices
    
    print(f"Selected {len(final_indices)} best features")
    return X[:, final_indices], final_indices

def build_stacking_ensemble(X_train, y_train, X_test=None):
    """Build a powerful stacking ensemble model"""
    print("Training ensemble model...")
    
    # First level models
    base_models = [
        ('rf', RandomForestRegressor(
            n_estimators=1200, 
            max_depth=None,
            min_samples_split=2, 
            bootstrap=True,
            n_jobs=-1, 
            random_state=42
        )),
        ('et', ExtraTreesRegressor(
            n_estimators=1200, 
            max_depth=None,
            min_samples_split=2, 
            min_samples_leaf=1,
            bootstrap=True, 
            n_jobs=-1, 
            random_state=43
        )),
        ('gb', GradientBoostingRegressor(
            n_estimators=600, 
            learning_rate=0.01,
            max_depth=10, 
            subsample=0.8,
            random_state=44
        )),
        ('xgb', xgb.XGBRegressor(
            n_estimators=1200,
            learning_rate=0.01,
            max_depth=12,
            subsample=0.9,
            colsample_bytree=0.85,
            n_jobs=-1,
            random_state=45
        ))
    ]
    
    # Train first level models with cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_features = np.zeros((X_train.shape[0], len(base_models)))
    test_preds = np.zeros((X_test.shape[0], len(base_models)))
    
    for i, (name, model) in enumerate(base_models):
        print(f"Training {name}...")
        
        # Train model on full dataset
        model.fit(X_train, y_train)
        
        # Generate predictions for test data
        if X_test is not None:
            test_preds[:, i] = model.predict(X_test)
        
        # Generate cross-validated predictions for training data
        cv_preds = np.zeros(X_train.shape[0])
        for train_idx, val_idx in kf.split(X_train):
            # Split data
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train = y_train[train_idx]
            
            # Train model on this fold
            model.fit(X_fold_train, y_fold_train)
            
            # Predict on validation fold
            cv_preds[val_idx] = model.predict(X_fold_val)
        
        # Store CV predictions as meta-features
        meta_features[:, i] = cv_preds
        
        # Evaluate
        cv_r2 = r2_score(y_train, cv_preds)
        print(f"{name} CV R²: {cv_r2:.6f}")
    
    # Train meta-regressor (use an ensemble for meta-model too)
    print("Training meta-regressor...")
    
    # Try multiple meta-models and blend them
    ridge = Ridge(alpha=0.01)
    ridge.fit(meta_features, y_train)
    ridge_preds = ridge.predict(meta_features)
    
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=46)
    gbr.fit(meta_features, y_train)
    gbr_preds = gbr.predict(meta_features)
    
    # Blend meta-models with optimal weights
    meta_train_preds = 0.7 * ridge_preds + 0.3 * gbr_preds
    meta_train_r2 = r2_score(y_train, meta_train_preds)
    print(f"Meta-regressor R²: {meta_train_r2:.6f}")
    
    # Make test predictions
    if X_test is not None:
        ridge_test_preds = ridge.predict(test_preds)
        gbr_test_preds = gbr.predict(test_preds)
        meta_test_preds = 0.7 * ridge_test_preds + 0.3 * gbr_test_preds
    else:
        meta_test_preds = None
    
    # Return models and predictions
    models = {
        'base_models': {name: model for name, model in base_models},
        'meta_models': {'ridge': ridge, 'gbr': gbr},
        'base_test_preds': test_preds,
        'test_predictions': meta_test_preds,
        'cv_r2': meta_train_r2
    }
    
    return models

def calibrate_predictions(predictions, target_min=0.92, target_max=1.09):
    """Calibrate predictions to ensure they are within an appropriate range"""
    # Get current min/max
    current_min = np.min(predictions)
    current_max = np.max(predictions)
    
    # Apply linear scaling
    calibrated = (predictions - current_min) / (current_max - current_min) * (target_max - target_min) + target_min
    
    # Apply additional fine-tuning for better distribution
    # Find extreme values
    threshold_high = np.percentile(calibrated, 98)
    threshold_low = np.percentile(calibrated, 2)
    
    # Apply slight compression to extreme values (keep them extreme but reduce outliers)
    calibrated[calibrated > threshold_high] = threshold_high + 0.5 * (calibrated[calibrated > threshold_high] - threshold_high)
    calibrated[calibrated < threshold_low] = threshold_low - 0.5 * (threshold_low - calibrated[calibrated < threshold_low])
    
    return calibrated

# Main execution
print("Loading datasets...")
# Load training data
train_data = pd.read_csv('Training_data_uhi_index_2025-02-18.csv')
train_data = train_data.drop('datetime', axis=1)

# Load test data
test_data = pd.read_csv('Submission_template_UHI2025-v2.csv')

# Extract coordinates
train_coords = train_data[['Longitude', 'Latitude']].values
test_coords = test_data[['Longitude', 'Latitude']].values

# Step 1: Extract enhanced features from satellite imagery
print("Extracting enhanced features from Landsat LST...")
lst_features_train = extract_enhanced_features(
    'Landsat_LST.tiff', 
    train_coords
)
lst_features_test = extract_enhanced_features(
    'Landsat_LST.tiff', 
    test_coords
)

# Step 1b: Extract Sentinel-2 features
print("Extracting features from Sentinel-2 imagery...")
try:
    s2_features_train = extract_sentinel2_features(
        'S2_sample.tiff',
        train_coords
    )
    s2_features_test = extract_sentinel2_features(
        'S2_sample.tiff',
        test_coords
    )
    print(f"Sentinel-2 features shape: {s2_features_train.shape}")
    
    # Combine with other features
    X_train_s2 = np.hstack([lst_features_train, s2_features_train])
    X_test_s2 = np.hstack([lst_features_test, s2_features_test])
except Exception as e:
    print(f"Error extracting Sentinel-2 features: {e}")
    print("Proceeding without Sentinel-2 features")
    X_train_s2 = lst_features_train
    X_test_s2 = lst_features_test

# Step 2: Create enhanced spatial features
print("Creating enhanced spatial features...")
spatial_features_train = create_enhanced_spatial_features(train_coords)
spatial_features_test = create_enhanced_spatial_features(test_coords)

# Step 3: Combine all feature sets
print("Combining features...")
X_train = np.hstack([X_train_s2, spatial_features_train])
X_test = np.hstack([X_test_s2, spatial_features_test])
y_train = train_data['UHI Index'].values

print(f"Initial feature set: {X_train.shape[1]} features")

# Step 4: Clean the data
print("Cleaning data...")
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Step 5: Feature selection
print("Performing hybrid feature selection...")
X_train_selected, selected_indices = hybrid_feature_selection(X_train, y_train, n_features=60)
X_test_selected = X_test[:, selected_indices]

# Step 6: Add polynomial features (carefully)
print("Adding polynomial features...")
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_selected)
X_test_poly = poly.transform(X_test_selected)

print(f"Enhanced feature set: {X_train_poly.shape[1]} features")

# Step 7: Preprocessing - robust scaling to handle outliers
print("Scaling features...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# Step 8: Data augmentation for better generalization
print("Performing enhanced data augmentation...")
X_train_aug, y_train_aug = enhanced_augmentation(X_train_scaled, y_train, num_samples=2000)

# Step 9: Train stacking ensemble model
print("Building and training stacking ensemble...")
stacking_model = build_stacking_ensemble(X_train_aug, y_train_aug, X_test_scaled)

# Step 10: Fine-tune final predictions
final_predictions = stacking_model['test_predictions']

# Calibrate predictions if needed (ensure proper range)
print("Calibrating predictions...")
final_predictions = calibrate_predictions(final_predictions, target_min=0.92, target_max=1.09)

# Create submission file
print("Creating submission file...")
test_data['UHI Index'] = final_predictions
test_data.to_csv('submission_0317_s2.csv', index=False)
print("Predictions saved to submission_0317_s2.csv")

# Save models and preprocessing components
print("Saving model components...")
os.makedirs('enhanced_model_s2', exist_ok=True)
joblib.dump(selected_indices, 'enhanced_model_s2/selected_indices.pkl')
joblib.dump(poly, 'enhanced_model_s2/poly.pkl')
joblib.dump(scaler, 'enhanced_model_s2/scaler.pkl')
joblib.dump(stacking_model, 'enhanced_model_s2/stacking_model.pkl')

print("Model saved to 'enhanced_model_s2' directory")

# Evaluate final predictions
print("\nModel training complete. Evaluating predictions...")

# Compare with previous best model if available
try:
    prev_submission = pd.read_csv('submission_0316_efficient.csv')
    comparison = pd.DataFrame({
        'Longitude': test_data['Longitude'],
        'Latitude': test_data['Latitude'],
        'Previous_UHI': prev_submission['UHI Index'],
        'New_UHI': test_data['UHI Index'],
        'Difference': np.abs(prev_submission['UHI Index'] - test_data['UHI Index'])
    })
    
    # Calculate statistics
    mean_diff = comparison['Difference'].mean()
    max_diff = comparison['Difference'].max()
    std_diff = comparison['Difference'].std()
    
    print(f"Difference statistics:")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Std deviation: {std_diff:.6f}")
    
    # Save comparison for analysis
    comparison = comparison.sort_values('Difference', ascending=False)
    comparison.to_csv('enhanced_comparison.csv', index=False)
    print("Comparison saved to enhanced_comparison.csv")
    
    # Show top differences
    print("\nTop 10 samples with largest differences:")
    print(comparison.head(10))
except:
    print("Could not compare with previous predictions")

print("\nEnhanced model training and prediction complete!")
