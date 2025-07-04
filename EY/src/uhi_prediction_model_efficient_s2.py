import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from scipy.ndimage import gaussian_filter
from sklearn.metrics import r2_score
import time
import warnings
import os
from math import radians, cos, sin, asin, sqrt

warnings.filterwarnings('ignore')

# Start timing
start_time = time.time()

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

def extract_raster_features(tiff_path, coords, window_size=7, print_interval=1000):
    """Extract features from raster imagery efficiently"""
    print(f"Extracting features from {os.path.basename(tiff_path)}...")
    
    with rasterio.open(tiff_path) as src:
        all_features = []
        num_coords = len(coords)
        
        for i, (lon, lat) in enumerate(coords):
            if i % print_interval == 0:
                print(f"  Processing coordinate {i+1}/{num_coords}")
                
            try:
                # Convert geographic coordinates to pixel coordinates
                py, px = src.index(lon, lat)
                
                half_size = window_size // 2
                # Ensure window is within image bounds
                win_start_x = max(0, px - half_size)
                win_start_y = max(0, py - half_size)
                win_width = min(src.width - win_start_x, window_size)
                win_height = min(src.height - win_start_y, window_size)
                
                window = Window(win_start_x, win_start_y, win_width, win_height)
                
                try:
                    # Read the window
                    pixel_array = src.read(1, window=window)
                    # Handle potential NaN values
                    pixel_array = np.nan_to_num(pixel_array, nan=0.0)
                    
                    if pixel_array.size > 0:
                        # Core statistics (fast to compute)
                        pixel_mean = np.mean(pixel_array)
                        pixel_std = np.std(pixel_array)
                        pixel_min = np.min(pixel_array)
                        pixel_max = np.max(pixel_array)
                        
                        # Percentiles for better distribution understanding
                        p25 = np.percentile(pixel_array, 25)
                        p75 = np.percentile(pixel_array, 75)
                        
                        # Simple smoothing
                        smoothed = gaussian_filter(pixel_array, sigma=1.0)
                        smoothed_mean = np.mean(smoothed)
                        
                        # Create feature vector
                        features = [
                            pixel_mean,         # Mean
                            pixel_std,          # Standard deviation
                            pixel_min,          # Minimum
                            pixel_max,          # Maximum
                            p25,                # 25th percentile
                            p75,                # 75th percentile
                            smoothed_mean,      # Smoothed mean
                            (pixel_max - pixel_min)  # Range
                        ]
                        
                        all_features.append(features)
                    else:
                        all_features.append([0] * 8)  # 8 features
                    
                except Exception as e:
                    all_features.append([0] * 8)
                
            except Exception as e:
                # If coordinate conversion fails, add zeros
                all_features.append([0] * 8)
                
        return np.array(all_features)

def extract_sentinel2_features(tiff_path, coords):
    """Extract specific features from Sentinel-2 imagery"""
    print("Extracting Sentinel-2 features...")
    
    # Get basic raster features with a smaller window for efficiency
    basic_features = extract_raster_features(tiff_path, coords, window_size=5, print_interval=1000)
    
    # For S2, we'll return just the basic features
    return basic_features

def create_spatial_features(coords):
    """Create optimized spatial features"""
    print("Creating spatial features...")
    
    # NYC key locations
    reference_points = {
        'central_park': (-73.9654, 40.7829),     # Central Park
        'times_square': (-73.9855, 40.7580),     # Times Square
        'downtown': (-74.0060, 40.7128),         # Downtown Manhattan
        'east_river': (-73.9762, 40.7678),       # East River
        'hudson_river': (-74.0099, 40.7258),     # Hudson River
    }
    
    features = []
    
    # Center of Manhattan
    center_lon, center_lat = -73.9712, 40.7831
    
    for lon, lat in coords:
        # Calculate distances to key points
        distances = {}
        for name, (ref_lon, ref_lat) in reference_points.items():
            dist = haversine(lon, lat, ref_lon, ref_lat)
            distances[name] = dist
            
        # Distance to center of Manhattan
        dist_center = haversine(lon, lat, center_lon, center_lat)
        
        # Calculate minimum distance to water (East River or Hudson River)
        min_water_dist = min(distances['east_river'], distances['hudson_river'])
        
        # Create optimized feature vector
        point_features = [
            lon, lat,                                   # Raw coordinates
            distances['central_park'],                  # Distance to Central Park
            distances['times_square'],                  # Distance to Times Square
            distances['downtown'],                      # Distance to Downtown Manhattan
            min_water_dist,                             # Distance to nearest water body
            dist_center,                                # Distance from center of Manhattan
            np.exp(-dist_center/2),                     # Exponential decay from center
            np.exp(-min_water_dist/2),                  # Exponential decay from water
        ]
        
        features.append(point_features)
        
    return np.array(features)
    
def selective_augmentation(X, y, num_samples=300):
    """Selectively augment data points in areas with high UHI variability"""
    print(f"Original data: {X.shape[0]} samples")
    
    # Identify high and low UHI areas
    high_uhi = np.percentile(y, 95)
    low_uhi = np.percentile(y, 5)
    
    # Set weights to prioritize extreme values
    weights = np.ones_like(y)
    weights[y >= high_uhi] = 5.0  # High weight for high UHI
    weights[y <= low_uhi] = 5.0   # High weight for low UHI
    
    # Normalize to form a probability distribution
    weights = weights / np.sum(weights)
    
    # Generate new samples
    new_X = []
    new_y = []
    
    print("Creating augmented samples...")
    
    # Use weighted sampling
    indices = np.random.choice(np.arange(len(X)), size=num_samples, replace=True, p=weights)
    
    # Set small noise level
    noise_level = 0.001
    
    for idx in indices:
        # Add small noise
        noise = np.random.normal(0, noise_level, X[idx].shape)
        new_X.append(X[idx] + noise)
        new_y.append(y[idx])
    
    # Combine with original data
    X_combined = np.vstack([X, np.array(new_X)])
    y_combined = np.concatenate([y, np.array(new_y)])
    
    print(f"After augmentation: {X_combined.shape[0]} samples")
    return X_combined, y_combined

def build_efficient_ensemble(X_train, y_train, X_test):
    """Build an efficient ensemble model for high accuracy"""
    print("Training optimized ensemble model...")
    
    # Base models with optimized hyperparameters
    models = [
        # Random Forest with moderate number of trees
        RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=4,
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        ),
        
        # Gradient Boosting with controlled complexity
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=44
        )
    ]
    
    # Train models and collect predictions
    all_preds = np.zeros((X_test.shape[0], len(models)))
    
    for i, model in enumerate(models):
        print(f"Training model {i+1}/{len(models)}...")
        model.fit(X_train, y_train)
        all_preds[:, i] = model.predict(X_test)
        
        # Validate on a holdout set
        X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        model.fit(X_t, y_t)
        val_preds = model.predict(X_v)
        val_score = r2_score(y_v, val_preds)
        print(f"  Validation R² score: {val_score:.6f}")
    
    # Use Ridge regression for final blending
    # First create a blending dataset from a subset
    X_blend, X_holdout, y_blend, y_holdout = train_test_split(X_train, y_train, test_size=0.3, random_state=43)
    
    # Train base models on blend portion
    blend_preds = np.zeros((X_holdout.shape[0], len(models)))
    for i, model in enumerate(models):
        model.fit(X_blend, y_blend)
        blend_preds[:, i] = model.predict(X_holdout)
    
    # Train a blender on predictions
    blender = Ridge(alpha=0.01)
    blender.fit(blend_preds, y_holdout)
    
    # Make final predictions
    final_preds = blender.predict(all_preds)
    
    # Add small corrections based on patterns in feature space
    feature_weights = np.mean(np.abs(X_test), axis=0)
    feature_weights = feature_weights / np.sum(feature_weights)
    
    # Fine-tune predictions with feature information
    feature_factor = 0.05  # Small correction factor
    feature_contribution = np.dot(X_test, feature_weights) * feature_factor
    feature_contribution = (feature_contribution - np.mean(feature_contribution)) / np.std(feature_contribution) * 0.01
    
    # Apply correction
    refined_preds = final_preds + feature_contribution
    
    return refined_preds

def calibrate_predictions(predictions):
    """Calibrate predictions to ensure they are within expected range"""
    # Expected range for UHI values
    target_min = 0.92
    target_max = 1.09
    
    # Get current min/max
    current_min = np.min(predictions)
    current_max = np.max(predictions)
    
    # Apply linear scaling
    calibrated = (predictions - current_min) / (current_max - current_min) * (target_max - target_min) + target_min
    
    # Apply specific adjustments to extreme values for better accuracy
    threshold_high = np.percentile(calibrated, 98)
    threshold_low = np.percentile(calibrated, 2)
    
    # Slightly compress extremes to avoid outliers
    calibrated[calibrated > threshold_high] = threshold_high + 0.5 * (calibrated[calibrated > threshold_high] - threshold_high)
    calibrated[calibrated < threshold_low] = threshold_low - 0.5 * (threshold_low - calibrated[calibrated < threshold_low])
    
    return calibrated

# Main execution
print("Starting UHI prediction model (Efficient S2 Version)...")

# Load datasets
print("Loading datasets...")
train_data = pd.read_csv('Training_data_uhi_index_2025-02-18.csv')
train_data = train_data.drop('datetime', axis=1)  # Remove datetime column

test_data = pd.read_csv('Submission_template_UHI2025-v2.csv')

# Extract coordinates
train_coords = train_data[['Longitude', 'Latitude']].values
test_coords = test_data[['Longitude', 'Latitude']].values

# Step 1: Extract features from Landsat imagery
lst_features_train = extract_raster_features('Landsat_LST.tiff', train_coords)
lst_features_test = extract_raster_features('Landsat_LST.tiff', test_coords)

# Step 2: Try to extract Sentinel-2 features
using_s2 = False
try:
    s2_features_train = extract_sentinel2_features('S2_sample.tiff', train_coords)
    s2_features_test = extract_sentinel2_features('S2_sample.tiff', test_coords)
    using_s2 = True
    print(f"Successfully extracted Sentinel-2 features with shape: {s2_features_train.shape}")
except Exception as e:
    print(f"Error extracting Sentinel-2 features: {e}")
    print("Continuing without Sentinel-2 features")
    s2_features_train = np.zeros((train_coords.shape[0], 0))
    s2_features_test = np.zeros((test_coords.shape[0], 0))

# Step 3: Create spatial features
spatial_features_train = create_spatial_features(train_coords)
spatial_features_test = create_spatial_features(test_coords)

# Step 4: Combine all features
print("Combining all features...")
if using_s2:
    X_train = np.hstack([lst_features_train, s2_features_train, spatial_features_train])
    X_test = np.hstack([lst_features_test, s2_features_test, spatial_features_test])
else:
    X_train = np.hstack([lst_features_train, spatial_features_train])
    X_test = np.hstack([lst_features_test, spatial_features_test])

y_train = train_data['UHI Index'].values

print(f"Combined feature set shape: {X_train.shape}")

# Step 5: Clean data
print("Cleaning data...")
# Replace any NaN or infinite values
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Step 6: Preprocess features
print("Preprocessing features...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Perform selective data augmentation
print("Performing data augmentation...")
X_train_aug, y_train_aug = selective_augmentation(X_train_scaled, y_train, num_samples=300)

# Step 8: Train the efficient ensemble model
print("Training model and generating predictions...")
predictions = build_efficient_ensemble(X_train_aug, y_train_aug, X_test_scaled)

# Step 9: Calibrate predictions
print("Calibrating predictions...")
final_predictions = calibrate_predictions(predictions)

# Step 10: Create submission file
print("Creating submission file...")
test_data['UHI Index'] = final_predictions
output_name = "submission_0317_s2_efficient.csv" if using_s2 else "submission_0317_efficient.csv"
test_data.to_csv(output_name, index=False)

# Calculate elapsed time
elapsed_time = time.time() - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

print(f"Predictions saved to {output_name}")
print("Model execution complete!")
