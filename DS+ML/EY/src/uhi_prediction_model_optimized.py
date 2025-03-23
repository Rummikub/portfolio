import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.utils import shuffle
from scipy.ndimage import gaussian_filter, sobel, median_filter
from scipy.interpolate import griddata
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import warnings
import joblib
warnings.filterwarnings('ignore')

# Utility functions
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

def extract_window_features(tiff_path, coords, window_sizes=[5, 11, 21]):
    """Extract focused features from satellite imagery around each coordinate"""
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
                            point_features.extend([0] * 20)
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
                        
                        # Calculate radial profile (average value at different distances from center)
                        y_center, x_center = pixel_array.shape[0] // 2, pixel_array.shape[1] // 2
                        y, x = np.indices(pixel_array.shape)
                        r = np.sqrt((y - y_center)**2 + (x - x_center)**2)
                        r_int = r.astype(int)
                        
                        # Create radial profile with 5 bins
                        max_r = min(pixel_array.shape) // 2
                        step = max(1, max_r // 5)
                        radial_profile = [np.mean(pixel_array[r_int == i]) if np.sum(r_int == i) > 0 else 0 
                                        for i in range(0, max_r, step)]
                        # Pad to ensure consistent length
                        radial_profile = radial_profile[:5] + [0] * (5 - len(radial_profile))
                        
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
                            np.median(gradient_magnitude),   # Gradient median
                            
                            # Spatial pattern features
                            dist_to_max,                     # Distance to max value
                            dist_to_min,                     # Distance to min value
                            
                            # Radial profile (captures circular patterns)
                            *radial_profile,
                        ]
                        
                        point_features.extend(features)
                        
                    except Exception as e:
                        # If window reading fails, add zeros
                        point_features.extend([0] * 20)
                
                all_features.append(point_features)
                
            except Exception as e:
                # If coordinate conversion fails, add zeros
                all_features.append([0] * (20 * len(window_sizes)))
                
        return np.array(all_features)

def create_location_features(coords):
    """Create basic features based on coordinates and distances to key points"""
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
    
    features = []
    for lon, lat in coords:
        # Create features
        x, y = lon, lat
        
        # Distances to key points
        landmark_dists = [haversine(lon, lat, point[0], point[1]) for point in landmarks.values()]
        
        # Minimum distance to UHI hotspots
        uhi_dists = [haversine(lon, lat, point[0], point[1]) for point in uhi_hotspots]
        min_uhi_dist = min(uhi_dists) if uhi_dists else 0
        
        # Minimum distance to green spaces
        green_dists = [haversine(lon, lat, point[0], point[1]) for point in green_spaces]
        min_green_dist = min(green_dists) if green_dists else 0
        
        # Create basic features
        point_features = [
            x, y,                        # Raw coordinates
            x**2, y**2, x*y,             # Quadratic terms
            np.sin(x*100), np.cos(y*100),  # Oscillatory features
            *landmark_dists,             # Distance to landmarks
            min_uhi_dist,                # Distance to nearest UHI hotspot
            min_green_dist,              # Distance to nearest green space
            min_green_dist / (min_uhi_dist + 0.001),  # Ratio of distances
        ]
        
        features.append(point_features)
        
    return np.array(features)

def augment_data(X, y, num_augmented=500, noise_level=0.001):
    """Augment the training data with slightly noisy copies of existing points"""
    # Select random indices to augment
    indices = np.random.choice(len(X), size=num_augmented, replace=True)
    
    # Create augmented features and targets
    X_aug = X[indices].copy()
    y_aug = y[indices].copy()
    
    # Add small noise to features
    for i in range(X_aug.shape[1]):
        # Scale noise by feature std
        noise = np.random.normal(0, noise_level * np.std(X[:, i]), size=num_augmented)
        X_aug[:, i] += noise
    
    # Add small noise to targets
    y_aug += np.random.normal(0, noise_level * np.std(y), size=num_augmented)
    
    # Combine original and augmented data
    X_combined = np.vstack([X, X_aug])
    y_combined = np.hstack([y, y_aug])
    
    # Shuffle the combined data
    X_combined, y_combined = shuffle(X_combined, y_combined, random_state=42)
    
    return X_combined, y_combined

def select_best_features(X, y, k=100):
    """Select the most informative features using multiple methods"""
    # Use mutual information (non-linear relationship measure)
    selector_mi = SelectKBest(mutual_info_regression, k=min(k, X.shape[1]))
    X_mi = selector_mi.fit_transform(X, y)
    mi_scores = selector_mi.scores_
    
    # Use f_regression (linear relationship measure)
    selector_f = SelectKBest(f_regression, k=min(k, X.shape[1]))
    X_f = selector_f.fit_transform(X, y)
    f_scores = selector_f.scores_
    
    # Use ExtraTrees feature importance
    et = ExtraTreesRegressor(n_estimators=100, random_state=42)
    et.fit(X, y)
    et_scores = et.feature_importances_
    
    # Combine scores from different methods
    # Normalize scores
    if mi_scores is not None:
        mi_scores = mi_scores / np.sum(mi_scores)
    else:
        mi_scores = np.zeros(X.shape[1])
        
    if f_scores is not None:
        f_scores = f_scores / np.sum(f_scores)
    else:
        f_scores = np.zeros(X.shape[1])
        
    et_scores = et_scores / np.sum(et_scores)
    
    # Average scores
    combined_scores = (mi_scores + f_scores + et_scores) / 3
    
    # Get indices of top features
    top_indices = np.argsort(combined_scores)[-k:]
    
    return top_indices, combined_scores

def optimize_model(X, y):
    """Find optimal hyperparameters for the ExtraTrees model"""
    print("Optimizing ExtraTrees model hyperparameters...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [None, 15, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create the model
    et = ExtraTreesRegressor(random_state=42, n_jobs=-1)
    
    # Use cross-validation to find the best parameters
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=et, 
        param_grid=param_grid,
        cv=kf, 
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Small subset for hyperparameter tuning to speed things up
    X_sample, X_, y_sample, y_ = train_test_split(X, y, train_size=min(1000, len(X)), random_state=42)
    
    # Find the best parameters
    grid_search.fit(X_sample, y_sample)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.6f}")
    
    return grid_search.best_params_

def plot_feature_importance(scores, feature_names, title="Feature Importance", top_n=30):
    """Plot feature importance scores"""
    # Get indices of top N features
    top_indices = np.argsort(scores)[-top_n:]
    
    # Sort in ascending order for plot
    top_indices = top_indices[::-1]
    
    # Extract names and scores
    top_names = [feature_names[i] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_names)), top_scores, align='center')
    plt.yticks(range(len(top_names)), top_names)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# Main execution
print("Loading datasets...")
# Load training data
train_data = pd.read_csv('Training_data_uhi_index_2025-02-18.csv')
train_data = train_data.drop('datetime', axis=1)

# Load test data
test_data = pd.read_csv('Submission_template_UHI2025-v2.csv')

# Extract features from satellite imagery
print("Extracting features from Landsat LST...")
lst_features_train = extract_window_features(
    'Landsat_LST.tiff', 
    train_data[['Longitude', 'Latitude']].values
)
lst_features_test = extract_window_features(
    'Landsat_LST.tiff', 
    test_data[['Longitude', 'Latitude']].values
)

print("Extracting features from Sentinel-2...")
s2_features_train = extract_window_features(
    'S2_sample.tiff', 
    train_data[['Longitude', 'Latitude']].values
)
s2_features_test = extract_window_features(
    'S2_sample.tiff', 
    test_data[['Longitude', 'Latitude']].values
)

print("Creating location-based features...")
loc_features_train = create_location_features(
    train_data[['Longitude', 'Latitude']].values
)
loc_features_test = create_location_features(
    test_data[['Longitude', 'Latitude']].values
)

# Combine all features
print("Combining features...")
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
y_train = train_data['UHI Index'].values

print(f"Initial feature set: {X_train.shape[1]} features")

# Clean up the data
print("Cleaning data...")
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Select best features
print("Selecting best features...")
k = min(100, X_train.shape[1])  # Cap at 100 features
top_indices, feature_scores = select_best_features(X_train, y_train, k=k)
X_train_selected = X_train[:, top_indices]
X_test_selected = X_test[:, top_indices]

print(f"Selected {len(top_indices)} best features")

# Feature names for visualization
feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
plot_feature_importance(feature_scores, feature_names, top_n=min(30, len(feature_scores)))

# Data augmentation
print("Augmenting training data...")
X_train_aug, y_train_aug = augment_data(X_train_selected, y_train, num_augmented=500, noise_level=0.001)
print(f"Training data size after augmentation: {len(X_train_aug)} samples")

# Preprocess features
print("Preprocessing features...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_aug)
X_test_scaled = scaler.transform(X_test_selected)

# Optimize model parameters
best_params = optimize_model(X_train_scaled, y_train_aug)

# Train the final model
print("Training final model with optimized parameters...")
et_model = ExtraTreesRegressor(
    **best_params,
    n_jobs=-1,
    random_state=42
)

et_model.fit(X_train_scaled, y_train_aug)

# Evaluate on training data
train_pred = et_model.predict(X_train_scaled)
train_r2 = r2_score(y_train_aug, train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train_aug, train_pred))
train_mae = mean_absolute_error(y_train_aug, train_pred)

print(f"Training metrics:")
print(f"R² Score: {train_r2:.6f}")
print(f"RMSE: {train_rmse:.6f}")
print(f"MAE: {train_mae:.6f}")

# Cross-validation to check for overfitting
print("Performing cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(et_model, X_train_scaled, y_train_aug, 
                           cv=kf, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print(f"Cross-validation RMSE: {cv_rmse}")
print(f"Mean CV RMSE: {np.mean(cv_rmse):.6f}")
print("Generating predictions...")
test_pred = et_model.predict(X_test_scaled)

# Ensure predictions are within a reasonable range
test_pred = np.clip(test_pred, 0.9, 1.1)

# Create submission file
test_data['UHI Index'] = test_pred
test_data.to_csv('optimized_submission.csv', index=False)
print("Predictions saved to optimized_submission.csv")

# Save the model and preprocessing components
joblib.dump(et_model, 'optimized_et_model.pkl')
joblib.dump(scaler, 'optimized_scaler.pkl')
joblib.dump(top_indices, 'optimized_feature_indices.pkl')

print("Model and preprocessing components saved")

# Find samples with different predictions
try:
    prev_submission = pd.read_csv('submission_0316.csv')
    comparison = pd.DataFrame({
        'Longitude': test_data['Longitude'],
        'Latitude': test_data['Latitude'],
        'Previous_UHI': prev_submission['UHI Index'],
        'New_UHI': test_data['UHI Index'],
        'Difference': np.abs(prev_submission['UHI Index'] - test_data['UHI Index'])
    })
    comparison = comparison.sort_values('Difference', ascending=False)
    comparison.to_csv('prediction_comparison.csv', index=False)
    print("Prediction comparison saved to prediction_comparison.csv")
    
    print("\nTop 10 samples with largest differences:")
    print(comparison.head(10))
except:
    print("Could not compare with previous predictions")
    
print("\nModel training and prediction complete!")
