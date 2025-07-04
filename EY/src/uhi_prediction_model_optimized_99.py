import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.utils import shuffle
from scipy.ndimage import gaussian_filter, sobel
from math import radians, cos, sin, asin, sqrt
import joblib
import warnings
import time

# Start timing
start_time = time.time()

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

def extract_enhanced_features(tiff_path, coords, window_sizes=[5, 11, 21]):
    """Extract enhanced features from satellite imagery with specific focus on UHI patterns"""
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
                            point_features.extend([0] * 15)
                            continue
                        
                        # Apply smoothing
                        smoothed = gaussian_filter(pixel_array, sigma=1)
                        
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
                        
                        # Extract key features - focused on most impactful for UHI
                        features = [
                            np.mean(pixel_array),            # Mean value
                            np.std(pixel_array),             # Standard deviation
                            np.percentile(pixel_array, 25),  # 1st quartile
                            np.percentile(pixel_array, 75),  # 3rd quartile
                            np.max(pixel_array) - np.min(pixel_array),  # Range
                            
                            # Gradient features
                            np.mean(gradient_magnitude),     # Mean gradient
                            np.percentile(gradient_magnitude, 90),  # High gradients
                            
                            # Edge vs center
                            np.mean(pixel_array[1:-1, 1:-1]) / (np.mean(pixel_array) + 0.001),  # Center to overall ratio
                            
                            # Spatial pattern features
                            dist_to_max,                     # Distance to max value
                            dist_to_min,                     # Distance to min value
                            
                            # Entropy-like measure (variety)
                            len(np.unique(pixel_array)) / (pixel_array.size + 0.001),
                            
                            # Quadrant analysis
                            np.mean(pixel_array[:center_y, :center_x]),  # Top-left
                            np.mean(pixel_array[:center_y, center_x:]),  # Top-right
                            np.mean(pixel_array[center_y:, :center_x]),  # Bottom-left
                            np.mean(pixel_array[center_y:, center_x:]),  # Bottom-right
                        ]
                        
                        point_features.extend(features)
                        
                    except Exception as e:
                        # If window reading fails, add zeros
                        point_features.extend([0] * 15)
                
                all_features.append(point_features)
                
            except Exception as e:
                # If coordinate conversion fails, add zeros
                all_features.append([0] * (15 * len(window_sizes)))
                
        return np.array(all_features)

def extract_sentinel2_features(tiff_path, coords):
    """Extract features specific to Sentinel-2 data"""
    print("Extracting Sentinel-2 features...")
    # Basic extraction with focused window sizes
    s2_features = extract_enhanced_features(tiff_path, coords, window_sizes=[5, 11])
    
    return s2_features

def create_advanced_location_features(coords):
    """Create enhanced spatial features focusing on urban heat island drivers"""
    # NYC reference points
    landmarks = {
        'central_park': (-73.9654, 40.7829),
        'times_square': (-73.9855, 40.7580),
        'downtown': (-74.0060, 40.7128),
        'harlem': (-73.9465, 40.8116),
        'brooklyn': (-73.9442, 40.6782),
    }
    
    # Areas with expected high UHI
    uhi_hotspots = [
        (-73.9855, 40.7580),  # Times Square
        (-73.9844, 40.7484),  # Midtown
        (-74.0060, 40.7128),  # Downtown
    ]
    
    # Green spaces (cooling effect)
    green_spaces = [
        (-73.9654, 40.7829),  # Central Park
        (-73.9771, 40.7695),  # Bryant Park
    ]
    
    # Water bodies (cooling effect)
    water_bodies = [
        (-74.0152, 40.7033),  # Hudson River
        (-73.9538, 40.7824),  # East River
    ]
    
    # Calculate Manhattan center
    center_lon, center_lat = -73.9712, 40.7831
    
    features = []
    for lon, lat in coords:
        # Create features
        x, y = lon, lat
        
        # Distances to key points
        landmark_dists = [haversine(lon, lat, point[0], point[1]) for point in landmarks.values()]
        
        # Distance to Manhattan center
        dist_center = haversine(lon, lat, center_lon, center_lat)
        
        # Minimum distance to UHI hotspots
        uhi_dists = [haversine(lon, lat, point[0], point[1]) for point in uhi_hotspots]
        min_uhi_dist = min(uhi_dists) if uhi_dists else 10.0
        
        # Minimum distance to green spaces
        green_dists = [haversine(lon, lat, point[0], point[1]) for point in green_spaces]
        min_green_dist = min(green_dists) if green_dists else 10.0
        
        # Minimum distance to water bodies
        water_dists = [haversine(lon, lat, point[0], point[1]) for point in water_bodies]
        min_water_dist = min(water_dists) if water_dists else 10.0
        
        # Decay functions (exponential decay with distance)
        green_effect = np.exp(-min_green_dist/1.0)  # Stronger decay for green spaces
        water_effect = np.exp(-min_water_dist/1.5)  # Moderate decay for water
        center_effect = np.exp(-dist_center/2.0)    # Slower decay for center distance
        
        # Directional features (angle from center)
        angle = np.arctan2(lat - center_lat, lon - center_lon)
        
        # Create feature vector
        point_features = [
            x, y,                        # Raw coordinates
            x**2, y**2, x*y,             # Quadratic terms
            np.sin(angle), np.cos(angle),  # Directional components
            *landmark_dists,             # Distance to landmarks
            dist_center,                 # Distance to Manhattan center
            min_uhi_dist,                # Distance to nearest UHI hotspot
            min_green_dist,              # Distance to nearest green space
            min_water_dist,              # Distance to nearest water
            green_effect,                # Green space cooling effect
            water_effect,                # Water cooling effect
            center_effect,               # Center heat effect
            min_green_dist / (min_uhi_dist + 0.001),  # Ratio of green to UHI distance
        ]
        
        features.append(point_features)
        
    return np.array(features)

def targeted_augmentation(X, y, num_samples=800):
    """Augment the training data focusing on areas with high UHI variability"""
    print(f"Original data: {X.shape[0]} samples")
    
    # Parameters
    noise_level = 0.001
    
    # Identify different UHI regimes for targeted augmentation
    high_uhi = np.percentile(y, 95)
    low_uhi = np.percentile(y, 5)
    
    # Set weights to prioritize extreme values and boundary areas
    weights = np.ones_like(y) * 0.2  # Base weight
    
    # High UHI areas (important for maximum accuracy)
    weights[y >= high_uhi] = 5.0  
    
    # Low UHI areas (green spaces, water influence)
    weights[y <= low_uhi] = 5.0
    
    # Boundary areas (moderate UHI) - critical for achieving 99% accuracy
    mid_uhi_mask = (y > low_uhi) & (y < high_uhi)
    
    # Find points with high variance in features (more complex areas)
    feature_vars = np.var(X, axis=1)
    high_var_indices = np.where(mid_uhi_mask & 
                               (feature_vars > np.percentile(feature_vars, 80)))[0]
    weights[high_var_indices] = 3.0
    
    # Normalize weights to probabilities
    weights = weights / np.sum(weights)
    
    # Generate new samples
    new_X = []
    new_y = []
    
    print("Creating augmented samples...")
    
    # Weighted sampling
    indices = np.random.choice(np.arange(len(X)), size=num_samples, replace=True, p=weights)
    
    for idx in indices:
        # Add small noise scaled by feature importance
        feature_importance = np.abs(X[idx]) / (np.max(np.abs(X[idx])) + 0.001)
        noise = np.random.normal(0, noise_level, X[idx].shape) * feature_importance
        new_X.append(X[idx] + noise)
        
        # Add slight noise to target
        target_noise = np.random.normal(0, 0.0005) 
        new_y.append(y[idx] + target_noise)
    
    # Combine with original data
    X_combined = np.vstack([X, np.array(new_X)])
    y_combined = np.concatenate([y, np.array(new_y)])
    
    # Shuffle the combined data
    X_combined, y_combined = shuffle(X_combined, y_combined, random_state=42)
    
    print(f"After augmentation: {X_combined.shape[0]} samples")
    return X_combined, y_combined

def select_optimal_features(X, y, k=120):
    """Select features using multiple methods with focus on UHI prediction accuracy"""
    print(f"Selecting top {k} features from {X.shape[1]} total features")
    
    # Method 1: Mutual information
    selector_mi = SelectKBest(mutual_info_regression, k=min(k, X.shape[1]))
    selector_mi.fit(X, y)
    mi_scores = selector_mi.scores_
    mi_indices = np.argsort(mi_scores)[-k:]
    
    # Method 2: Tree-based feature importance
    et = ExtraTreesRegressor(n_estimators=100, random_state=42)
    et.fit(X, y)
    et_scores = et.feature_importances_
    et_indices = np.argsort(et_scores)[-k:]
    
    # Combine unique features from both methods
    combined_indices = np.unique(np.concatenate([mi_indices, et_indices]))
    
    print(f"Selected {len(combined_indices)} features")
    return combined_indices, mi_scores

def build_stacked_ensemble(X_train, y_train, X_test=None):
    """Build a stacked ensemble optimized for UHI prediction accuracy"""
    print("Building stacked ensemble model...")
    
    # Base models
    base_models = [
        # ExtraTrees - good at capturing non-linear patterns
        ExtraTreesRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42
        ),
        
        # Random Forest - robust general model
        RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            bootstrap=True,
            n_jobs=-1,
            random_state=43
        ),
        
        # Gradient Boosting - sequential improvement of errors
        GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=44
        )
    ]
    
    # Train base models and generate predictions
    print("Training base models...")
    meta_features = np.zeros((X_train.shape[0], len(base_models)))
    
    if X_test is not None:
        test_preds = np.zeros((X_test.shape[0], len(base_models)))
    
    # Use cross-validation to create meta-features
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for i, model in enumerate(base_models):
        model_name = type(model).__name__
        print(f"Training {model_name}...")
        
        # Train model on full dataset for test predictions
        model.fit(X_train, y_train)
        
        # Generate predictions for test data
        if X_test is not None:
            test_preds[:, i] = model.predict(X_test)
        
        # Generate meta-features via cross-validation
        cv_preds = np.zeros(X_train.shape[0])
        for train_idx, val_idx in kf.split(X_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train = y_train[train_idx]
            
            model.fit(X_fold_train, y_fold_train)
            cv_preds[val_idx] = model.predict(X_fold_val)
        
        meta_features[:, i] = cv_preds
        
        # Evaluate model
        r2 = r2_score(y_train, cv_preds)
        rmse = np.sqrt(mean_squared_error(y_train, cv_preds))
        print(f"  {model_name} CV - R²: {r2:.6f}, RMSE: {rmse:.6f}")
    
    # Train a meta-model on the predictions
    print("Training meta-model...")
    meta_model = Ridge(alpha=0.01)
    meta_model.fit(meta_features, y_train)
    
    # Make final predictions
    if X_test is not None:
        ensemble_preds = meta_model.predict(test_preds)
    else:
        ensemble_preds = None
    
    # Print meta-model coefficients (model weights)
    print("Meta-model coefficients (weights):")
    for i, coef in enumerate(meta_model.coef_):
        model_name = type(base_models[i]).__name__
        print(f"  {model_name}: {coef:.4f}")
    
    return ensemble_preds, base_models, meta_model

def calibrate_predictions(predictions):
    """Calibrate predictions to ensure they reflect realistic UHI values"""
    # Expected range for UHI index
    target_min = 0.92
    target_max = 1.09
    
    # Get current range
    current_min = np.min(predictions)
    current_max = np.max(predictions)
    
    # Initial linear scaling to target range
    calibrated = (predictions - current_min) / (current_max - current_min) * (target_max - target_min) + target_min
    
    # Apply fine-tuning for extreme values
    high_threshold = np.percentile(calibrated, 98)
    low_threshold = np.percentile(calibrated, 2)
    
    # Apply a more nuanced adjustment to extreme values
    # Compress extreme values slightly to avoid unrealistic outliers
    calibrated[calibrated > high_threshold] = high_threshold + 0.5 * (calibrated[calibrated > high_threshold] - high_threshold)
    calibrated[calibrated < low_threshold] = low_threshold - 0.5 * (low_threshold - calibrated[calibrated < low_threshold])
    
    return calibrated

# Main execution
print("Starting UHI prediction model optimized for 99% accuracy...")

# Load datasets
print("Loading datasets...")
train_data = pd.read_csv('Training_data_uhi_index_2025-02-18.csv')
train_data = train_data.drop('datetime', axis=1)

test_data = pd.read_csv('Submission_template_UHI2025-v2.csv')

# Extract coordinates
train_coords = train_data[['Longitude', 'Latitude']].values
test_coords = test_data[['Longitude', 'Latitude']].values

# Step 1: Extract enhanced features from Landsat LST
print("Extracting enhanced features from Landsat LST...")
lst_features_train = extract_enhanced_features(
    'Landsat_LST.tiff', 
    train_coords
)
lst_features_test = extract_enhanced_features(
    'Landsat_LST.tiff', 
    test_coords
)

# Step 2: Extract Sentinel-2 features
print("Extracting Sentinel-2 features...")
try:
    s2_features_train = extract_sentinel2_features(
        'S2_sample.tiff', 
        train_coords
    )
    s2_features_test = extract_sentinel2_features(
        'S2_sample.tiff', 
        test_coords
    )
    using_s2 = True
    print(f"Sentinel-2 features extracted with shape: {s2_features_train.shape}")
except Exception as e:
    print(f"Error extracting Sentinel-2 features: {e}")
    s2_features_train = np.zeros((train_coords.shape[0], 0))
    s2_features_test = np.zeros((test_coords.shape[0], 0))
    using_s2 = False
    print("Proceeding without Sentinel-2 features")

# Step 3: Create advanced location features
print("Creating advanced location features...")
loc_features_train = create_advanced_location_features(train_coords)
loc_features_test = create_advanced_location_features(test_coords)

# Step 4: Combine all features
print("Combining all features...")
if using_s2:
    X_train = np.hstack([
        lst_features_train,
        s2_features_train,
        loc_features_train,
        train_data[['Longitude', 'Latitude']].values
    ])
    X_test = np.hstack([
        lst_features_test,
        s2_features_test,
        loc_features_test,
        test_data[['Longitude', 'Latitude']].values
    ])
else:
    X_train = np.hstack([
        lst_features_train,
        loc_features_train,
        train_data[['Longitude', 'Latitude']].values
    ])
    X_test = np.hstack([
        lst_features_test,
        loc_features_test,
        test_data[['Longitude', 'Latitude']].values
    ])

y_train = train_data['UHI Index'].values

print(f"Combined feature set: {X_train.shape[1]} features")

# Step 5: Clean data
print("Cleaning data...")
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Step 6: Select optimal features
print("Selecting optimal features...")
selected_indices, feature_scores = select_optimal_features(X_train, y_train, k=120)
X_train_selected = X_train[:, selected_indices]
X_test_selected = X_test[:, selected_indices]

print(f"Selected {X_train_selected.shape[1]} optimal features")

# Step 7: Perform targeted augmentation
print("Performing targeted data augmentation...")
X_train_aug, y_train_aug = targeted_augmentation(X_train_selected, y_train, num_samples=800)

# Step 8: Scale features
print("Scaling features...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_aug)
X_test_scaled = scaler.transform(X_test_selected)

# Step 9: Train stacked ensemble
print("Training ensemble model...")
ensemble_preds, base_models, meta_model = build_stacked_ensemble(X_train_scaled, y_train_aug, X_test_scaled)

# Step 10: Calibrate predictions
print("Calibrating predictions...")
final_predictions = calibrate_predictions(ensemble_preds)

# Step 11: Create submission file
print("Creating submission file...")
test_data['UHI Index'] = final_predictions
output_name = "submission_0317_s2_99percent.csv" if using_s2 else "submission_0317_99percent.csv"
test_data.to_csv(output_name, index=False)

# Calculate elapsed time
elapsed_time = time.time() - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

print(f"Predictions saved to {output_name}")
print("Model execution complete - optimized for 99% accuracy!")
