import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
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

def extract_focused_features(tiff_path, coords, window_sizes=[5, 11, 21]):
    """Extract focused features from satellite imagery"""
    with rasterio.open(tiff_path) as src:
        all_features = []
        num_coords = len(coords)
        
        for i, (lon, lat) in enumerate(coords):
            if i % 500 == 0:
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
                            # Apply smoothing
                            smoothed = gaussian_filter(pixel_array, sigma=1)
                            
                            # Calculate gradients
                            gradient_x = sobel(smoothed, axis=0)
                            gradient_y = sobel(smoothed, axis=1)
                            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                            
                            # Extract key features
                            features = [
                                np.mean(pixel_array),            # Mean value
                                np.std(pixel_array),             # Standard deviation
                                np.median(pixel_array),          # Median
                                np.percentile(pixel_array, 25),  # 1st quartile
                                np.percentile(pixel_array, 75),  # 3rd quartile
                                np.max(pixel_array) - np.min(pixel_array),  # Range
                                
                                # Gradient features
                                np.mean(gradient_magnitude),     # Mean gradient
                                np.std(gradient_magnitude),      # Gradient std
                                
                                # Distribution features
                                np.max(pixel_array),             # Maximum value
                                np.min(pixel_array),             # Minimum value
                                np.sum(pixel_array > np.mean(pixel_array)) / pixel_array.size,  # Fraction above mean
                            ]
                            
                            point_features.extend(features)
                        else:
                            # If pixel array is empty
                            point_features.extend([0] * 11)
                        
                    except Exception as e:
                        # If window reading fails, add zeros
                        point_features.extend([0] * 11)
                
                all_features.append(point_features)
                
            except Exception as e:
                # If coordinate conversion fails, add zeros
                all_features.append([0] * (11 * len(window_sizes)))
                
        return np.array(all_features)

def create_spatial_features(coords):
    """Create basic features based on coordinates"""
    # NYC reference points
    central_park = (-73.9654, 40.7829)
    times_square = (-73.9855, 40.7580)
    downtown = (-74.0060, 40.7128)
    east_river = (-73.9762, 40.7678)
    hudson_river = (-74.0099, 40.7258)
    harlem = (-73.9465, 40.8116)
    brooklyn = (-73.9442, 40.6782)
    
    features = []
    for lon, lat in coords:
        # Create features
        x, y = lon, lat
        
        # Distance to key locations
        dist_central_park = haversine(lon, lat, central_park[0], central_park[1])
        dist_times_square = haversine(lon, lat, times_square[0], times_square[1])
        dist_downtown = haversine(lon, lat, downtown[0], downtown[1])
        dist_east_river = haversine(lon, lat, east_river[0], east_river[1])
        dist_hudson_river = haversine(lon, lat, hudson_river[0], hudson_river[1])
        dist_harlem = haversine(lon, lat, harlem[0], harlem[1])
        dist_brooklyn = haversine(lon, lat, brooklyn[0], brooklyn[1])
        
        # Distance and angle to center
        center_lon, center_lat = -73.95, 40.80  # Approximate center of the area
        dist_center = haversine(lon, lat, center_lon, center_lat)
        angle = np.arctan2(lat - center_lat, lon - center_lon)
        
        # Create basic features
        point_features = [
            x, y,                        # Raw coordinates
            x**2, y**2, x*y,             # Quadratic terms
            dist_central_park,           # Distance to Central Park
            dist_times_square,           # Distance to Times Square
            dist_downtown,               # Distance to Downtown
            dist_east_river,             # Distance to East River
            dist_hudson_river,           # Distance to Hudson River
            dist_harlem,                 # Distance to Harlem
            dist_brooklyn,               # Distance to Brooklyn
            dist_center,                 # Distance from center
            angle,                       # Angle
            np.sin(angle),               # Sine of angle
            np.cos(angle),               # Cosine of angle
            
            # Inverse distance (stronger effect nearby)
            1 / (1 + dist_central_park),
            1 / (1 + dist_times_square),
            1 / (1 + dist_east_river),
            1 / (1 + dist_hudson_river),
            
            # Combined features
            dist_times_square / (dist_central_park + 0.001),  # Ratio of distances
            min(dist_east_river, dist_hudson_river),        # Minimum distance to water
        ]
        
        features.append(point_features)
        
    return np.array(features)

# Load best prediction as reference
def load_reference_prediction():
    try:
        best_pred = pd.read_csv('best_prediction.csv')
        return best_pred
    except:
        print("Could not load best_prediction.csv")
        return None

# Main execution
print("Loading datasets...")
# Load training data
train_data = pd.read_csv('Training_data_uhi_index_2025-02-18.csv')
train_data = train_data.drop('datetime', axis=1)

# Load test data
test_data = pd.read_csv('Submission_template_UHI2025-v2.csv')

# Load reference prediction if available
reference_data = load_reference_prediction()

# Extract features from satellite imagery
print("Extracting features from Landsat LST...")
lst_features_train = extract_focused_features(
    'Landsat_LST.tiff', 
    train_data[['Longitude', 'Latitude']].values
)
lst_features_test = extract_focused_features(
    'Landsat_LST.tiff', 
    test_data[['Longitude', 'Latitude']].values
)

print("Extracting features from Sentinel-2...")
s2_features_train = extract_focused_features(
    'S2_sample.tiff', 
    train_data[['Longitude', 'Latitude']].values
)
s2_features_test = extract_focused_features(
    'S2_sample.tiff', 
    test_data[['Longitude', 'Latitude']].values
)

print("Creating spatial features...")
spatial_features_train = create_spatial_features(
    train_data[['Longitude', 'Latitude']].values
)
spatial_features_test = create_spatial_features(
    test_data[['Longitude', 'Latitude']].values
)

# Combine all features
print("Combining features...")
X_train_basic = np.hstack([
    lst_features_train,
    s2_features_train,
    spatial_features_train
])
X_test_basic = np.hstack([
    lst_features_test,
    s2_features_test,
    spatial_features_test
])
y_train = train_data['UHI Index'].values

print(f"Initial feature set: {X_train_basic.shape[1]} features")

# Clean up the data
print("Cleaning data...")
X_train_basic = np.nan_to_num(X_train_basic, nan=0.0, posinf=0.0, neginf=0.0)
X_test_basic = np.nan_to_num(X_test_basic, nan=0.0, posinf=0.0, neginf=0.0)

# If reference data exists, use it to create additional features
if reference_data is not None:
    print("Adding reference data features...")
    # For each point, find the 5 closest reference points and use their values
    # This is a form of knowledge transfer from the best model
    
    reference_coords = reference_data[['Longitude', 'Latitude']].values
    reference_uhi = reference_data['UHI Index'].values
    
    def create_reference_features(coords, reference_coords, reference_values, k=5):
        features = []
        for lon, lat in coords:
            # Calculate distances to all reference points
            distances = [haversine(lon, lat, ref_lon, ref_lat) 
                        for ref_lon, ref_lat in reference_coords]
            
            # Get indices of k closest points
            closest_indices = np.argsort(distances)[:k]
            
            # Get values and distances of closest points
            closest_values = [reference_values[i] for i in closest_indices]
            closest_distances = [distances[i] for i in closest_indices]
            
            # Calculate distance-weighted average
            weights = [1/(d+0.0001) for d in closest_distances]
            weighted_avg = sum(v*w for v, w in zip(closest_values, weights)) / sum(weights)
            
            # Create features
            point_features = [
                weighted_avg,                         # Distance-weighted average
                np.mean(closest_values),              # Simple average
                np.std(closest_values),               # Standard deviation
                np.max(closest_values),               # Maximum
                np.min(closest_values),               # Minimum
                closest_values[0],                    # Value of closest point
                closest_distances[0],                 # Distance to closest point
                *closest_values[:3],                  # Values of 3 closest points
                *[1/(d+0.0001) for d in closest_distances[:3]]  # Inverse distances
            ]
            
            features.append(point_features)
            
        return np.array(features)
    
    ref_features_train = create_reference_features(
        train_data[['Longitude', 'Latitude']].values,
        reference_coords,
        reference_uhi
    )
    
    ref_features_test = create_reference_features(
        test_data[['Longitude', 'Latitude']].values,
        reference_coords,
        reference_uhi
    )
    
    # Add reference features to existing features
    X_train = np.hstack([X_train_basic, ref_features_train])
    X_test = np.hstack([X_test_basic, ref_features_test])
    
    print(f"Feature set after adding reference features: {X_train.shape[1]} features")
else:
    X_train = X_train_basic
    X_test = X_test_basic

# Feature selection with ExtraTrees
print("Performing feature selection...")
et_selector = SelectFromModel(
    ExtraTreesRegressor(n_estimators=100, random_state=42),
    threshold="mean"
)
X_train_selected = et_selector.fit_transform(X_train, y_train)
X_test_selected = et_selector.transform(X_test)

print(f"Selected {X_train_selected.shape[1]} features")

# Preprocess features
print("Preprocessing features...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# Split data for internal validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# Train model - using ExtraTrees which has proven to be effective
print("Training ExtraTrees model...")
et_model = ExtraTreesRegressor(
    n_estimators=1000,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

et_model.fit(X_train_scaled, y_train)

# Evaluate on training data
train_pred = et_model.predict(X_train_scaled)
train_r2 = r2_score(y_train, train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

print(f"Training metrics:")
print(f"R² Score: {train_r2:.6f}")
print(f"RMSE: {train_rmse:.6f}")
print(f"OOB Score: {et_model.oob_score_:.6f}")

# Estimated accuracy based on OOB
oob_accuracy = 100 * et_model.oob_score_
print(f"Estimated accuracy from OOB: {oob_accuracy:.4f}%")

# Cross-validation to check for overfitting
print("Performing cross-validation...")
cv_scores = cross_val_score(
    et_model, X_train_scaled, y_train, 
    cv=5, scoring='neg_mean_squared_error'
)
cv_rmse = np.sqrt(-cv_scores)
print(f"Cross-validation RMSE: {cv_rmse}")
print(f"Mean CV RMSE: {np.mean(cv_rmse):.6f}")

# Calculate accuracy from cross-validation
cv_accuracy = 100 * (1 - np.mean(cv_rmse) / np.mean(y_train))
print(f"Estimated accuracy from CV: {cv_accuracy:.4f}%")

# Train a second model - RandomForest for comparison
print("Training RandomForest model for comparison...")
rf_model = RandomForestRegressor(
    n_estimators=1000,
    max_depth=None,
    min_samples_split=2,
    bootstrap=True,
    n_jobs=-1,
    random_state=43  # Different seed for diversity
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_train_scaled)
rf_r2 = r2_score(y_train, rf_pred)
print(f"RandomForest R² Score: {rf_r2:.6f}")

# Make predictions - weighted ensemble
print("Generating ensemble predictions...")
et_weight = 0.75  # Higher weight for ExtraTrees
rf_weight = 0.25  # Lower weight for RandomForest

et_test_pred = et_model.predict(X_test_scaled)
rf_test_pred = rf_model.predict(X_test_scaled)

# Ensemble prediction
ensemble_pred = (et_weight * et_test_pred) + (rf_weight * rf_test_pred)

# Ensure predictions are within a reasonable range
ensemble_pred = np.clip(ensemble_pred, 0.9, 1.1)

# Create submission file
test_data['UHI Index'] = ensemble_pred
test_data.to_csv('submission_expert0316.csv', index=False)
print("Predictions saved to submission_expert0316.csv")

# Save the model and preprocessing components
os.makedirs('expert_model', exist_ok=True)
joblib.dump(et_model, 'expert_model/et_model.pkl')
joblib.dump(rf_model, 'expert_model/rf_model.pkl')
joblib.dump(scaler, 'expert_model/scaler.pkl')
joblib.dump(et_selector, 'expert_model/feature_selector.pkl')
joblib.dump({'et_weight': et_weight, 'rf_weight': rf_weight}, 'expert_model/weights.pkl')

print("Models and preprocessing components saved")

# Visualize feature importance
if hasattr(et_model, 'feature_importances_'):
    # Get selected feature indices
    selected_indices = et_selector.get_support(indices=True)
    
    # Define feature names (simplified)
    feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    selected_names = [feature_names[i] for i in selected_indices]
    
    # Plot top 30 features
    importances = et_model.feature_importances_
    indices = np.argsort(importances)[::-1][:30]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [selected_names[i] for i in indices])
    plt.tight_layout()
    plt.savefig('expert_feature_importance.png')
    plt.close()
    
    print("Feature importance visualization saved to expert_feature_importance.png")

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
    comparison.to_csv('expert_comparison.csv', index=False)
    print("Prediction comparison saved to expert_comparison.csv")
    
    print("\nTop 10 samples with largest differences:")
    print(comparison.head(10))
except:
    print("Could not compare with previous predictions")
    
print("\nExpert model training and prediction complete!")
