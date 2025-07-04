import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy.ndimage import gaussian_filter, sobel, median_filter, laplace
from scipy.interpolate import griddata
from scipy.stats import skew, kurtosis
from math import radians, cos, sin, asin, sqrt
import warnings
import joblib
import os
from pycaret.regression import *
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
        (-73.9771, 40.7695),  # Bryant Park
        (-74.0046, 40.7029),  # Battery Park
        (-73.9701, 40.6602),  # Prospect Park
        (-73.8458, 40.7463),  # Flushing Meadows
    ]
    
    # Water bodies (cooling effect)
    water_bodies = [
        (-73.9762, 40.7678),  # East River
        (-74.0099, 40.7258),  # Hudson River
        (-74.0401, 40.6701),  # Upper Bay
    ]
    
    features = []
    for lon, lat in coords:
        # Base coordinates
        x, y = lon, lat
        
        # Distance to landmarks
        landmark_dists = [haversine(lon, lat, landmark[0], landmark[1]) 
                         for landmark in landmarks.values()]
        
        # Distance to UHI hotspots
        uhi_dists = [haversine(lon, lat, hs[0], hs[1]) for hs in uhi_hotspots]
        min_uhi_dist = min(uhi_dists)
        avg_uhi_dist = np.mean(uhi_dists)
        inv_uhi_influence = 1 / (1 + min_uhi_dist)  # Inverse distance for stronger effect nearby
        
        # Distance to cooling features
        green_dists = [haversine(lon, lat, gs[0], gs[1]) for gs in green_spaces]
        min_green_dist = min(green_dists)
        avg_green_dist = np.mean(green_dists)
        inv_green_influence = 1 / (1 + min_green_dist)
        
        water_dists = [haversine(lon, lat, wb[0], wb[1]) for wb in water_bodies]
        min_water_dist = min(water_dists)
        avg_water_dist = np.mean(water_dists)
        inv_water_influence = 1 / (1 + min_water_dist)
        
        # Geographic coordinate transformations
        x_centered = x - (-73.95)  # Center around approximate NYC center
        y_centered = y - 40.75
        r = np.sqrt(x_centered**2 + y_centered**2)  # Distance from center
        theta = np.arctan2(y_centered, x_centered)  # Angle from center
        
        # Compute thermal exposure index - higher exposure to hotspots, less exposure to cooling
        thermal_index = inv_uhi_influence / (inv_green_influence + inv_water_influence + 0.1)
        
        # Create advanced features
        point_features = [
            # Raw and transformed coordinates
            x, y,
            x_centered, y_centered,
            r, theta,
            x**2, y**2, x*y,
            
            # Distance metrics
            *landmark_dists,
            
            # UHI influence factors
            min_uhi_dist, avg_uhi_dist,
            inv_uhi_influence,
            
            # Cooling influence factors
            min_green_dist, avg_green_dist,
            inv_green_influence,
            min_water_dist, avg_water_dist,
            inv_water_influence,
            
            # Combined influence index
            thermal_index,
            np.log1p(thermal_index),
            
            # Oscillatory features to capture periodic patterns
            np.sin(x*100), np.cos(y*100),
            np.sin(x*50), np.cos(y*50),
            np.sin(theta*4), np.cos(theta*4),
        ]
        
        features.append(point_features)
        
    return np.array(features)

def augment_data(X, y, num_augmented=1000, strategies=['noise', 'jitter', 'smooth']):
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
    X_combined, y_combined = shuffle(X_combined, y_combined, random_state=42)
    
    return X_combined, y_combined

def best_feature_selection(X, y, strategy='combined'):
    """Advanced feature selection targeting the most predictive features for UHI prediction"""
    n_features = X.shape[1]
    # Increase from 10 to 15 features for better model capacity
    n_select = 15
    
    # Initialize weights for each feature
    feature_weights = np.zeros(n_features)
    
    # 1. Mutual information (non-linear relationship measure) - most important for UHI prediction
    try:
        mi_selector = SelectKBest(mutual_info_regression, k=min(60, n_features))
        mi_selector.fit(X, y)
        mi_scores = mi_selector.scores_
        mi_scores = mi_scores / np.sum(mi_scores) if np.sum(mi_scores) > 0 else np.zeros_like(mi_scores)
        feature_weights += mi_scores * 0.45  # 45% weight - increase weight for non-linear relationships
    except:
        print("Error in mutual information selection")
    
    # 2. F-regression (linear relationship measure)
    try:
        f_selector = SelectKBest(f_regression, k=min(60, n_features))
        f_selector.fit(X, y)
        f_scores = f_selector.scores_
        f_scores = f_scores / np.sum(f_scores) if np.sum(f_scores) > 0 else np.zeros_like(f_scores)
        feature_weights += f_scores * 0.15  # 15% weight
    except:
        print("Error in f_regression selection")
    
    # 3. ExtraTrees feature importance - specifically good for UHI patterns
    try:
        et = ExtraTreesRegressor(n_estimators=300, random_state=42)
        et.fit(X, y)
        et_scores = et.feature_importances_
        feature_weights += et_scores * 0.25  # 25% weight
    except:
        print("Error in ExtraTrees selection")
        
    # 4. Add Gradient Boosting feature importance
    try:
        gb = GradientBoostingRegressor(n_estimators=200, random_state=43)
        gb.fit(X, y)
        gb_scores = gb.feature_importances_
        feature_weights += gb_scores * 0.15  # 15% weight
    except:
        print("Error in GradientBoosting selection")
    
    # Get indices of top features
    top_indices = np.argsort(feature_weights)[-n_select:]
    
    # Print top feature info for debugging
    print(f"Selected top {n_select} features with weights:")
    for i, idx in enumerate(top_indices):
        print(f"  Feature {idx}: Weight {feature_weights[idx]:.6f}")
    
    return top_indices, feature_weights

def save_feature_importance(feature_weights, feature_names, top_n=50):
    """Print feature importance sorted by weight"""
    # Get indices of top features
    top_indices = np.argsort(feature_weights)[-top_n:]
    top_indices = top_indices[::-1]  # Reverse to get descending order
    
    # Print feature importance
    print(f"\nTop {top_n} Feature Importance:")
    for i, idx in enumerate(top_indices):
        if i < top_n:
            print(f"  {i+1}. {feature_names[idx]}: {feature_weights[idx]:.6f}")

# Main execution
print("Loading datasets...")
# Load training data
train_data = pd.read_csv('Training_data_uhi_index_2025-02-18.csv')
train_data = train_data.drop('datetime', axis=1)

# Load test data
test_data = pd.read_csv('Submission_template_UHI2025-v2.csv')

# Extract features from satellite imagery
print("Extracting advanced features from Landsat LST...")
lst_features_train = extract_advanced_features(
    'Landsat_LST.tiff', 
    train_data[['Longitude', 'Latitude']].values
)
lst_features_test = extract_advanced_features(
    'Landsat_LST.tiff', 
    test_data[['Longitude', 'Latitude']].values
)

print("Extracting advanced features from Sentinel-2...")
s2_features_train = extract_advanced_features(
    'S2_sample.tiff', 
    train_data[['Longitude', 'Latitude']].values
)
s2_features_test = extract_advanced_features(
    'S2_sample.tiff', 
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

# Feature selection
print("Performing advanced feature selection...")
top_indices, feature_scores = best_feature_selection(X_train, y_train)
X_train_selected = X_train[:, top_indices]
X_test_selected = X_test[:, top_indices]

print(f"Selected {len(top_indices)} best features")

# Create feature names for visualization
feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
selected_feature_names = [f"Feature_{i}" for i in top_indices]

# Print feature importance information
save_feature_importance(feature_scores, feature_names, top_n=30)

# Data augmentation
print("Performing advanced data augmentation...")
X_train_aug, y_train_aug = augment_data(
    X_train_selected, 
    y_train, 
    num_augmented=3000,
    strategies=['noise', 'jitter', 'smooth', 'adversarial']
)
print(f"Training data size after augmentation: {len(X_train_aug)} samples")

# Preprocess features
print("Preprocessing features with QuantileTransformer...")
scaler = QuantileTransformer(output_distribution='normal', random_state=42)
X_train_scaled = scaler.fit_transform(X_train_aug)
X_test_scaled = scaler.transform(X_test_selected)

# Apply PCA to reduce dimensionality while preserving variance
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=0.9995, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"Reduced to {X_train_pca.shape[1]} PCA components")

# Create PyCaret-ready dataframes
print("Setting up data for PyCaret...")
pca_feature_names = [f'PCA_{i}' for i in range(X_train_pca.shape[1])]
train_df = pd.DataFrame(X_train_pca, columns=pca_feature_names)
train_df['UHI_Index'] = y_train_aug
test_df = pd.DataFrame(X_test_pca, columns=pca_feature_names)

# Initialize PyCaret setup
print("Initializing PyCaret for regression modeling...")
pycaret_setup = setup(
    data=train_df,
    target='UHI_Index',
    session_id=42,
    normalize=False,  # Already normalized
    transformation=False,  # Already transformed
    feature_selection=False,  # Already selected
    remove_outliers=True,
    outliers_threshold=0.05,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.9,
    log_experiment=False,  # Set to False to avoid mlflow dependency
    experiment_name='uhi_prediction',
    verbose=True,
    n_jobs=-1
)

# Compare all models to find the best performers
print("Comparing models in PyCaret...")
best_models = compare_models(
    sort='MAPE',  # Sort by Mean Absolute Percentage Error
    n_select=5,   # Select top 5 models
    budget_time=1 # Time budget (hours) for comparison
)

# Ensure best_models is a list
if not isinstance(best_models, list):
    best_models = [best_models]
    
# Create a blended model with optimal weights
print("Creating an optimized blended model...")
blender = blend_models(
    estimator_list=best_models,
    optimize='MAPE',
    weights=None,    # Auto-determine weights
    choose_better=True
)

# Fine-tune the blended model
print("Fine-tuning the blended model...")
tuned_blender = tune_model(
    blender,
    optimize='MAPE',
    n_iter=100,  # Number of iterations for tuning
    search_library='optuna',
    search_algorithm='tpe',
    early_stopping='Median'
)

# Create a stacked ensemble of the best tuned models
print("Creating stacked ensemble for maximum accuracy...")
stacked_model = stack_models(
    estimator_list=[model for model in best_models],  # Ensure it's a list
    meta_model=tuned_blender,
    optimize='MAPE',
    method='auto',
    restack=True,
    choose_better=True
)

# Final predictions
print("Generating final predictions...")
final_model = finalize_model(stacked_model)
final_pred = predict_model(final_model, data=test_df)['Label'].values

# Create submission file
test_data['UHI Index'] = final_pred
test_data.to_csv('pycaret_submission.csv', index=False)
print("Predictions saved to pycaret_submission.csv")

# Calculate expected performance
cv_results = pull()
print(f"\nExpected model performance (CV results):")
print(f"MAE: {cv_results.loc[cv_results['Model'] == 'Stacking Regressor', 'MAE'].values[0]:.6f}")
print(f"MSE: {cv_results.loc[cv_results['Model'] == 'Stacking Regressor', 'MSE'].values[0]:.6f}")
print(f"RMSE: {cv_results.loc[cv_results['Model'] == 'Stacking Regressor', 'RMSE'].values[0]:.6f}")
print(f"R²: {cv_results.loc[cv_results['Model'] == 'Stacking Regressor', 'R2'].values[0]:.6f}")
expected_accuracy = 100 * (1 - cv_results.loc[cv_results['Model'] == 'Stacking Regressor', 'MAPE'].values[0]/100)
print(f"Expected Accuracy: {expected_accuracy:.2f}%")

# Save the models and preprocessing components
os.makedirs('pycaret_model', exist_ok=True)
save_model(final_model, 'pycaret_model/final_model')
joblib.dump(scaler, 'pycaret_model/scaler.pkl')
joblib.dump(pca, 'pycaret_model/pca.pkl')
joblib.dump(top_indices, 'pycaret_model/selected_features.pkl')

print("Model and preprocessing components saved to pycaret_model directory")

# Find samples with different predictions
try:
    prev_submission = pd.read_csv('refined_submission_99pct.csv')
    comparison = pd.DataFrame({
        'Longitude': test_data['Longitude'],
        'Latitude': test_data['Latitude'],
        'Previous_UHI': prev_submission['UHI Index'],
        'New_UHI': test_data['UHI Index'],
        'Difference': np.abs(prev_submission['UHI Index'] - test_data['UHI Index'])
    })
    comparison = comparison.sort_values('Difference', ascending=False)
    comparison.to_csv('pycaret_comparison.csv', index=False)
    print("Prediction comparison saved to pycaret_comparison.csv")
    
    print("\nTop 10 samples with largest differences:")
    print(comparison.head(10))
except Exception as e:
    print(f"Could not compare with previous predictions: {e}")
    
print("\nPyCaret model training and prediction complete!")
