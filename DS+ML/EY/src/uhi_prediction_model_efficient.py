import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from scipy.ndimage import gaussian_filter, sobel
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import warnings
import joblib
import os
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

def extract_optimized_features(tiff_path, coords, window_sizes=[5, 15]):
    """Extract focused features from satellite imagery"""
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
                            
                            # Apply smoothing
                            smoothed = gaussian_filter(pixel_array, sigma=1.0)
                            
                            # Calculate gradients
                            gradient_x = sobel(smoothed, axis=0)
                            gradient_y = sobel(smoothed, axis=1)
                            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                            
                            # Calculate key statistics - only most important ones
                            features = [
                                pixel_mean,                        # Mean value
                                pixel_std,                         # Standard deviation
                                pixel_median,                      # Median
                                np.percentile(pixel_array, 75),    # 3rd quartile
                                np.max(pixel_array),               # Maximum
                                np.min(pixel_array),               # Minimum
                                np.mean(gradient_magnitude),       # Mean gradient
                                np.std(gradient_magnitude),        # Gradient std
                            ]
                            
                            point_features.extend(features)
                        else:
                            point_features.extend([0] * 8)
                        
                    except Exception as e:
                        point_features.extend([0] * 8)
                
                # Extract radial feature (only one ring size for efficiency)
                try:
                    radius = 10
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
                            ]
                            point_features.extend(ring_features)
                        else:
                            point_features.extend([0, 0])
                    else:
                        point_features.extend([0, 0])
                        
                except Exception as e:
                    point_features.extend([0, 0])
                
                all_features.append(point_features)
                
            except Exception as e:
                # If coordinate conversion fails, add zeros
                all_features.append([0] * (8 * len(window_sizes) + 2))
                
        return np.array(all_features)

def create_enhanced_spatial_features(coords):
    """Create optimized spatial features"""
    # NYC key locations
    reference_points = {
        'central_park': (-73.9654, 40.7829),  # Central Park
        'times_square': (-73.9855, 40.7580),  # Times Square
        'downtown': (-74.0060, 40.7128),      # Downtown
        'east_river': (-73.9762, 40.7678),    # East River
        'hudson_river': (-74.0099, 40.7258),  # Hudson River
    }
    
    features = []
    for lon, lat in coords:
        # Create features
        x, y = lon, lat
        
        # Calculate distances to key points
        distances = {}
        for name, (ref_lon, ref_lat) in reference_points.items():
            dist = haversine(lon, lat, ref_lon, ref_lat)
            distances[name] = dist
            
        # Calculate distance to center of Manhattan
        center_lon, center_lat = -73.9712, 40.7831
        dist_center = haversine(lon, lat, center_lon, center_lat)
        angle = np.arctan2(lat - center_lat, lon - center_lon)
        
        # Create features - focusing only on the most important ones
        point_features = [
            x, y,                                     # Raw coordinates
            x**2, y**2, x*y,                          # Quadratic terms
            distances['central_park'],                # Distance to Central Park
            distances['times_square'],                # Distance to Times Square
            distances['east_river'],                  # Distance to East River
            distances['hudson_river'],                # Distance to Hudson River
            np.exp(-distances['central_park']),       # Exponential decay for Central Park
            np.exp(-distances['times_square']),       # Exponential decay for Times Square
            min(distances['east_river'], distances['hudson_river']),  # Min distance to water
            np.exp(-min(distances['east_river'], distances['hudson_river'])),  # Exp decay for water
            dist_center,                              # Distance from center
            np.sin(angle),                            # Sine of angle
            np.cos(angle),                            # Cosine of angle
        ]
        
        features.append(point_features)
        
    return np.array(features)

def minimal_augmentation(X, y, num_samples=500):
    """Minimal data augmentation to avoid memory issues"""
    print(f"Original data: {X.shape[0]} samples")
    
    # Define weights to prioritize extreme UHI values
    threshold_high = np.percentile(y, 90)
    threshold_low = np.percentile(y, 10)
    weights = np.ones_like(y)
    weights[y >= threshold_high] = 3.0
    weights[y <= threshold_low] = 3.0
    
    # Normalize weights to probabilities
    weights = weights / np.sum(weights)
    
    # Randomly select samples based on weights
    selected_indices = np.random.choice(
        np.arange(len(X)), 
        size=num_samples, 
        replace=True, 
        p=weights
    )
    
    # Add small noise to create new samples
    noise_factor = 0.0005
    new_X = []
    new_y = []
    
    for idx in selected_indices:
        noise = np.random.normal(0, noise_factor, X[idx].shape)
        new_X.append(X[idx] + noise)
        new_y.append(y[idx])
    
    # Combine with original data
    X_combined = np.vstack([X, np.array(new_X)])
    y_combined = np.concatenate([y, np.array(new_y)])
    
    print(f"After augmentation: {len(y_combined)} samples")
    return X_combined, y_combined

def select_best_features(X, y, n_features=40):
    """Select best features using mutual information and tree importance"""
    print(f"Selecting {n_features} best features from {X.shape[1]} total")
    
    # Method 1: Mutual information
    try:
        mi_scores = mutual_info_regression(X, y)
    except:
        # Fallback if mutual info fails
        mi_scores = np.ones(X.shape[1])
    
    # Method 2: Tree-based feature importance
    et = ExtraTreesRegressor(n_estimators=100, random_state=42)
    et.fit(X, y)
    et_scores = et.feature_importances_
    
    # Combine scores
    combined_scores = 0.5 * (mi_scores / np.max(mi_scores)) + 0.5 * (et_scores / np.max(et_scores))
    
    # Select top features
    top_indices = np.argsort(combined_scores)[-n_features:]
    
    return X[:, top_indices], top_indices

# Main execution
print("Loading datasets...")
# Load training data
train_data = pd.read_csv('Training_data_uhi_index_2025-02-18.csv')
train_data = train_data.drop('datetime', axis=1)

# Load test data
test_data = pd.read_csv('Submission_template_UHI2025-v2.csv')

# Extract features from satellite imagery
print("Extracting features from Landsat LST...")
lst_features_train = extract_optimized_features(
    'Landsat_LST.tiff', 
    train_data[['Longitude', 'Latitude']].values
)
lst_features_test = extract_optimized_features(
    'Landsat_LST.tiff', 
    test_data[['Longitude', 'Latitude']].values
)

print("Creating spatial features...")
spatial_features_train = create_enhanced_spatial_features(
    train_data[['Longitude', 'Latitude']].values
)
spatial_features_test = create_enhanced_spatial_features(
    test_data[['Longitude', 'Latitude']].values
)

# Combine all features
print("Combining features...")
X_train = np.hstack([lst_features_train, spatial_features_train])
X_test = np.hstack([lst_features_test, spatial_features_test])
y_train = train_data['UHI Index'].values

print(f"Initial feature set: {X_train.shape[1]} features")

# Clean up the data
print("Cleaning data...")
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Feature selection
print("Performing feature selection...")
X_train_selected, selected_indices = select_best_features(X_train, y_train, n_features=40)
X_test_selected = X_test[:, selected_indices]

print(f"Selected {X_train_selected.shape[1]} features")

# Minimal data augmentation
print("Performing minimal data augmentation...")
X_train_aug, y_train_aug = minimal_augmentation(X_train_selected, y_train, num_samples=1000)

# Preprocessing
print("Preprocessing features...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_aug)
X_test_scaled = scaler.transform(X_test_selected)

# Define optimized models
print("Training models...")
models = {}

# ExtraTrees model
print("Training ExtraTrees model...")
et_model = ExtraTreesRegressor(
    n_estimators=1000,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
et_model.fit(X_train_scaled, y_train_aug)
models['et'] = et_model

# RandomForest model
print("Training RandomForest model...")
rf_model = RandomForestRegressor(
    n_estimators=1000,
    max_depth=None,
    min_samples_split=2,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=43
)
rf_model.fit(X_train_scaled, y_train_aug)
models['rf'] = rf_model

# GradientBoosting model
print("Training GradientBoosting model...")
gb_model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    subsample=0.8,
    max_depth=8,
    random_state=44
)
gb_model.fit(X_train_scaled, y_train_aug)
models['gb'] = gb_model

# Evaluate models
for name, model in models.items():
    train_pred = model.predict(X_train_scaled)
    train_r2 = r2_score(y_train_aug, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_aug, train_pred))
    print(f"{name} - R²: {train_r2:.6f}, RMSE: {train_rmse:.6f}")
    
    if hasattr(model, 'oob_score_'):
        print(f"{name} - OOB Score: {model.oob_score_:.6f}")

# Cross-validation for confidence
et_cv_scores = cross_val_score(et_model, X_train_scaled, y_train_aug, cv=5, scoring='r2')
print(f"ExtraTrees CV R² scores: {et_cv_scores}")
print(f"Mean CV R²: {np.mean(et_cv_scores):.6f}")

# Calculate model weights based on performance
# Higher R² gets more weight
et_weight = 0.5  # ExtraTrees often performs best
rf_weight = 0.3  # RandomForest second
gb_weight = 0.2  # GradientBoosting third

weights = {
    'et': et_weight,
    'rf': rf_weight,
    'gb': gb_weight
}
print("Model weights:", weights)

# Generate predictions
print("Generating ensemble predictions...")
predictions = {}
for name, model in models.items():
    predictions[name] = model.predict(X_test_scaled)

# Weighted ensemble prediction
ensemble_pred = np.zeros_like(predictions['et'])
for name, pred in predictions.items():
    ensemble_pred += weights[name] * pred

# Ensure predictions are within a reasonable range
ensemble_pred = np.clip(ensemble_pred, 0.85, 1.15)

# Create submission file
test_data['UHI Index'] = ensemble_pred
test_data.to_csv('submission_0316_efficient.csv', index=False)
print("Predictions saved to submission_0316_ultimate.csv")

# Save the models and preprocessing components
os.makedirs('efficient_model', exist_ok=True)
for name, model in models.items():
    joblib.dump(model, f'efficient_model/{name}_model.pkl')
joblib.dump(scaler, 'efficient_model/scaler.pkl')
joblib.dump(selected_indices, 'efficient_model/selected_indices.pkl')
joblib.dump(weights, 'efficient_model/weights.pkl')

print("Models and preprocessing components saved")

# Compare with previous predictions if available
try:
    prev_submission = pd.read_csv('optimized_submission.csv')
    comparison = pd.DataFrame({
        'Longitude': test_data['Longitude'],
        'Latitude': test_data['Latitude'],
        'Previous_UHI': prev_submission['UHI Index'],
        'New_UHI': test_data['UHI Index'],
        'Difference': np.abs(prev_submission['UHI Index'] - test_data['UHI Index'])
    })
    comparison = comparison.sort_values('Difference', ascending=False)
    comparison.to_csv('efficient_comparison.csv', index=False)
    print("Prediction comparison saved to efficient_comparison.csv")
    
    print("\nTop 10 samples with largest differences:")
    print(comparison.head(10))
except:
    print("Could not compare with previous predictions")
    
print("\nEfficient model training and prediction complete!")
