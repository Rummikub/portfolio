import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel, mutual_info_regression, f_regression
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter, sobel, laplace, median_filter
from scipy.stats import skew, kurtosis
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import warnings
import joblib
import os
from sklearn.svm import SVR

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

def extract_advanced_features(tiff_path, coords, window_sizes=[3, 7, 15]):
    """Extract comprehensive features from satellite imagery"""
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
                            # Basic statistics
                            pixel_mean = np.mean(pixel_array)
                            pixel_std = np.std(pixel_array)
                            pixel_median = np.median(pixel_array)
                            pixel_min = np.min(pixel_array)
                            pixel_max = np.max(pixel_array)
                            
                            # Higher order statistics
                            pixel_skew = skew(pixel_array.flatten())
                            pixel_kurtosis = kurtosis(pixel_array.flatten())
                            
                            # Percentiles
                            p25 = np.percentile(pixel_array, 25)
                            p75 = np.percentile(pixel_array, 75)
                            p95 = np.percentile(pixel_array, 95)
                            
                            # IQR and range
                            iqr = p75 - p25
                            data_range = pixel_max - pixel_min
                            
                            # Apply different filters
                            smoothed = gaussian_filter(pixel_array, sigma=1.0)
                            median_smoothed = median_filter(pixel_array, size=3)
                            
                            # Calculate gradients
                            gradient_x = sobel(smoothed, axis=0)
                            gradient_y = sobel(smoothed, axis=1)
                            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                            
                            # Calculate Laplacian for edge detection
                            laplacian = laplace(smoothed)
                            
                            # GLCM-inspired texture features
                            entropy = -np.sum(pixel_array * np.log2(pixel_array + 1e-10))
                            energy = np.sum(pixel_array ** 2)
                            contrast = np.sum((pixel_array - pixel_mean) ** 2)
                            
                            # Direction-specific gradients
                            horizontal_edge = np.mean(np.abs(np.diff(smoothed, axis=1)))
                            vertical_edge = np.mean(np.abs(np.diff(smoothed, axis=0)))
                            
                            # Additional texture measures
                            roughness = np.mean(np.abs(laplacian))
                            homogeneity = np.mean(1 / (1 + gradient_magnitude))
                            
                            # Combine all features
                            features = [
                                pixel_mean,                            # Mean value
                                pixel_std,                             # Standard deviation
                                pixel_median,                          # Median
                                pixel_min,                             # Minimum
                                pixel_max,                             # Maximum
                                pixel_skew,                            # Skewness
                                pixel_kurtosis,                        # Kurtosis
                                p25,                                   # 1st quartile
                                p75,                                   # 3rd quartile
                                p95,                                   # 95th percentile
                                iqr,                                   # Inter-quartile range
                                data_range,                            # Data range
                                np.mean(gradient_magnitude),           # Mean gradient
                                np.std(gradient_magnitude),            # Gradient std
                                np.max(gradient_magnitude),            # Max gradient
                                np.mean(laplacian),                    # Mean Laplacian
                                np.std(laplacian),                     # Laplacian std
                                entropy,                               # Entropy
                                energy,                                # Energy
                                contrast,                              # Contrast
                                horizontal_edge,                       # Horizontal edge strength
                                vertical_edge,                         # Vertical edge strength
                                roughness,                             # Surface roughness
                                homogeneity,                           # Texture homogeneity
                            ]
                            
                            point_features.extend(features)
                        else:
                            point_features.extend([0] * 24)
                        
                    except Exception as e:
                        point_features.extend([0] * 24)
                
                # Extract radial features with multiple rings
                ring_sizes = [5, 10, 15]
                for radius in ring_sizes:
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
                                    skew(masked_array),
                                    np.percentile(masked_array, 75) - np.percentile(masked_array, 25)
                                ]
                                point_features.extend(ring_features)
                            else:
                                point_features.extend([0] * 7)
                        else:
                            point_features.extend([0] * 7)
                            
                    except Exception as e:
                        point_features.extend([0] * 7)
                
                all_features.append(point_features)
                
            except Exception as e:
                # If coordinate conversion fails, add zeros
                all_features.append([0] * (24 * len(window_sizes) + 7 * len(ring_sizes)))
                
        return np.array(all_features)

def create_enhanced_spatial_features(coords, clusters=None):
    """Create comprehensive spatial features with clustering information"""
    # NYC key locations (extended list with more reference points)
    reference_points = {
        'central_park': (-73.9654, 40.7829),       # Central Park
        'times_square': (-73.9855, 40.7580),       # Times Square
        'downtown': (-74.0060, 40.7128),           # Downtown
        'east_river_north': (-73.9400, 40.7700),   # East River North
        'east_river_mid': (-73.9762, 40.7678),     # East River Mid
        'east_river_south': (-73.9700, 40.7200),   # East River South
        'hudson_river_north': (-73.9600, 40.8100), # Hudson River North
        'hudson_river_mid': (-74.0099, 40.7258),   # Hudson River Mid
        'hudson_river_south': (-74.0200, 40.7000), # Hudson River South
        'midtown': (-73.9800, 40.7500),            # Midtown
        'uptown': (-73.9500, 40.8100),             # Uptown
        'harlem': (-73.9400, 40.8100),             # Harlem
        'williamsburg': (-73.9600, 40.7100),       # Williamsburg
        'queens': (-73.9200, 40.7500),             # Queens
        'bronx': (-73.9000, 40.8500),              # Bronx
    }
    
    features = []
    
    # Define centers for different regions (for regional modeling)
    region_centers = {
        'downtown_region': (-74.0060, 40.7128),    # Downtown center
        'midtown_region': (-73.9800, 40.7500),     # Midtown center
        'uptown_region': (-73.9500, 40.8100),      # Uptown center
        'east_region': (-73.9300, 40.7600),        # Eastern center
        'west_region': (-74.0100, 40.7600),        # Western center
    }
    
    # Center points for grid-based calculations
    center_lon, center_lat = -73.9712, 40.7831
    
    for idx, (lon, lat) in enumerate(coords):
        # Create features
        x, y = lon, lat
        
        # Calculate distances to key points
        distances = {}
        for name, (ref_lon, ref_lat) in reference_points.items():
            dist = haversine(lon, lat, ref_lon, ref_lat)
            distances[name] = dist
            
        # Calculate distances to region centers
        region_distances = {}
        for name, (reg_lon, reg_lat) in region_centers.items():
            dist = haversine(lon, lat, reg_lon, reg_lat)
            region_distances[name] = dist
        
        # Calculate distance and angle to center
        dist_center = haversine(lon, lat, center_lon, center_lat)
        angle = np.arctan2(lat - center_lat, lon - center_lon)
        
        # Calculate distance to nearest river
        east_river_dist = min(distances['east_river_north'], 
                             distances['east_river_mid'], 
                             distances['east_river_south'])
        hudson_river_dist = min(distances['hudson_river_north'],
                               distances['hudson_river_mid'],
                               distances['hudson_river_south'])
        nearest_river_dist = min(east_river_dist, hudson_river_dist)
        
        # Calculate nearest park distance
        nearest_park_dist = distances['central_park']  # Can be expanded with more parks
        
        # Manhattan distance metric
        manhattan_dist_x = abs(lon - center_lon) * 111 * cos(radians(center_lat))  # Convert to km
        manhattan_dist_y = abs(lat - center_lat) * 111
        manhattan_distance = manhattan_dist_x + manhattan_dist_y
        
        # Cluster information if available
        cluster_features = []
        if clusters is not None:
            cluster_id = clusters[idx]
            cluster_onehot = [1 if i == cluster_id else 0 for i in range(5)]  # Assuming 5 clusters
            cluster_features = cluster_onehot
        else:
            cluster_features = [0] * 5  # Placeholder for cluster features
            
        # Create spatial features
        point_features = [
            x, y,                                     # Raw coordinates
            x**2, y**2, x*y,                          # Quadratic terms
            np.sin(x), np.cos(x), np.sin(y), np.cos(y),  # Cyclical transformations
            distances['central_park'],                # Distance to Central Park
            distances['times_square'],                # Distance to Times Square
            distances['downtown'],                    # Distance to Downtown
            east_river_dist,                          # Distance to East River
            hudson_river_dist,                        # Distance to Hudson River
            nearest_river_dist,                       # Distance to nearest river
            nearest_park_dist,                        # Distance to nearest park
            distances['midtown'],                     # Distance to Midtown
            distances['uptown'],                      # Distance to Uptown
            distances['harlem'],                      # Distance to Harlem
            np.exp(-distances['central_park']/5),     # Exp decay for Central Park
            np.exp(-distances['times_square']/5),     # Exp decay for Times Square
            np.exp(-nearest_river_dist/5),            # Exp decay for nearest river
            dist_center,                              # Distance from center
            np.exp(-dist_center/5),                   # Exp decay from center
            np.sin(angle),                            # Sine of angle from center
            np.cos(angle),                            # Cosine of angle from center
            manhattan_distance,                       # Manhattan distance
            region_distances['downtown_region'],      # Distance to downtown region
            region_distances['midtown_region'],       # Distance to midtown region
            region_distances['uptown_region'],        # Distance to uptown region
            np.exp(-region_distances['downtown_region']/5),   # Exp decay for downtown
            np.exp(-region_distances['midtown_region']/5),    # Exp decay for midtown
            np.exp(-region_distances['uptown_region']/5),     # Exp decay for uptown
        ]
        
        # Combine with cluster features
        point_features.extend(cluster_features)
        
        features.append(point_features)
        
    return np.array(features)

def advanced_augmentation(X, y, clusters=None, num_samples=2000):
    """Advanced data augmentation with targeted sampling strategy"""
    print(f"Original data: {X.shape[0]} samples")
    
    # Identify extreme values for focused augmentation
    threshold_high = np.percentile(y, 95)
    threshold_low = np.percentile(y, 5)
    
    # Create weights that prioritize extreme values and rare patterns
    weights = np.ones_like(y) * 0.5
    weights[y >= threshold_high] = 4.0  # Heavy weight on high values
    weights[y <= threshold_low] = 4.0   # Heavy weight on low values
    
    # Add extra weight to mid-range values with high feature variance
    feature_variances = np.var(X, axis=1)
    high_var_indices = np.where((y > threshold_low) & (y < threshold_high) & (feature_variances > np.percentile(feature_variances, 75)))[0]
    weights[high_var_indices] = 2.0     # Medium weight on high-variance mid-range values
    
    # Normalize weights to probabilities
    weights = weights / np.sum(weights)
    
    # Generate new samples through various augmentation techniques
    new_X = []
    new_y = []
    new_clusters = []
    
    # 1. Weighted sampling with small Gaussian noise
    print("Creating noise-augmented samples...")
    num_noise_samples = int(num_samples * 0.5)
    selected_indices = np.random.choice(
        np.arange(len(X)), 
        size=num_noise_samples, 
        replace=True, 
        p=weights
    )
    
    for idx in selected_indices:
        # Adaptive noise based on feature importance
        noise_factor = 0.001  # Base noise factor
        noise = np.random.normal(0, noise_factor, X[idx].shape)
        new_X.append(X[idx] + noise)
        new_y.append(y[idx])
        if clusters is not None:
            new_clusters.append(clusters[idx])
    
    # 2. Synthetic points by interpolation between nearby points
    print("Creating interpolated samples...")
    num_interp_samples = int(num_samples * 0.3)
    
    # Find pairs of nearby points
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=5).fit(X)
    _, indices = nbrs.kneighbors(X)
    
    for _ in range(num_interp_samples):
        # Choose a random point
        idx1 = np.random.choice(np.arange(len(X)), p=weights)
        # Choose a random neighbor
        idx2 = indices[idx1, np.random.randint(1, 5)]  # Skip the first neighbor (self)
        
        # Generate a random interpolation factor
        alpha = np.random.beta(0.5, 0.5)  # Beta distribution gives more weight to extremes
        
        # Interpolate features and target
        interp_X = X[idx1] * alpha + X[idx2] * (1 - alpha)
        interp_y = y[idx1] * alpha + y[idx2] * (1 - alpha)
        
        new_X.append(interp_X)
        new_y.append(interp_y)
        
        # For clusters, just take the cluster of the first point
        if clusters is not None:
            new_clusters.append(clusters[idx1])
    
    # 3. SMOTE-inspired approach for extreme values only
    print("Creating SMOTE-like samples for extreme values...")
    num_smote_samples = num_samples - num_noise_samples - num_interp_samples
    
    # Identify extreme value indices
    extreme_indices = np.where((y >= threshold_high) | (y <= threshold_low))[0]
    
    # Find nearest neighbors among extreme values
    if len(extreme_indices) > 5:  # Need at least 5 points for 5 neighbors
        extreme_X = X[extreme_indices]
        extreme_y = y[extreme_indices]
        
        extreme_nbrs = NearestNeighbors(n_neighbors=5).fit(extreme_X)
        _, extreme_indices_neighbors = extreme_nbrs.kneighbors(extreme_X)
        
        for _ in range(num_smote_samples):
            # Choose a random extreme point
            rand_idx = np.random.randint(0, len(extreme_indices))
            base_idx = extreme_indices[rand_idx]
            
            # Choose a random neighbor from the extreme subset
            neighbor_idx_in_subset = extreme_indices_neighbors[rand_idx, np.random.randint(1, 5)]
            neighbor_idx = extreme_indices[neighbor_idx_in_subset]
            
            # Generate a random interpolation factor
            alpha = np.random.beta(0.4, 0.4)  # More weight to extremes
            
            # Create new synthetic sample
            smote_X = X[base_idx] * alpha + X[neighbor_idx] * (1 - alpha)
            smote_y = y[base_idx] * alpha + y[neighbor_idx] * (1 - alpha)
            
            # Add small random noise to avoid duplicates
            smote_X += np.random.normal(0, 0.0005, smote_X.shape)
            
            new_X.append(smote_X)
            new_y.append(smote_y)
            
            # For clusters, just take the cluster of the base point
            if clusters is not None:
                new_clusters.append(clusters[base_idx])
    
    # Combine with original data
    X_augmented = np.vstack([X, np.array(new_X)])
    y_augmented = np.concatenate([y, np.array(new_y)])
    
    # Return augmented clusters if clusters were provided
    if clusters is not None:
        clusters_augmented = np.concatenate([clusters, np.array(new_clusters)])
        print(f"After augmentation: {len(y_augmented)} samples")
        return X_augmented, y_augmented, clusters_augmented
    else:
        print(f"After augmentation: {len(y_augmented)} samples")
        return X_augmented, y_augmented

def advanced_feature_selection(X, y, n_features=60):
    """Hybrid feature selection using multiple methods"""
    print(f"Selecting best {n_features} features from {X.shape[1]} total features")
    
    # Method 1: Mutual Information
    try:
        mi_scores = mutual_info_regression(X, y)
        mi_scores = np.nan_to_num(mi_scores)
    except:
        # Fallback if mutual info fails
        mi_scores = np.ones(X.shape[1])
    
    # Method 2: F-regression (linear correlation)
    try:
        f_scores, _ = f_regression(X, y)
        f_scores = np.nan_to_num(f_scores)
    except:
        # Fallback if f_regression fails
        f_scores = np.ones(X.shape[1])
    
    # Method 3: Tree-based feature importance
    et = ExtraTreesRegressor(n_estimators=100, random_state=42)
    et.fit(X, y)
    et_scores = et.feature_importances_
    
    # Normalize scores between 0 and 1 for each method
    mi_scores_norm = mi_scores / np.max(mi_scores) if np.max(mi_scores) > 0 else mi_scores
    f_scores_norm = f_scores / np.max(f_scores) if np.max(f_scores) > 0 else f_scores
    et_scores_norm = et_scores / np.max(et_scores) if np.max(et_scores) > 0 else et_scores
    
    # Combine scores with different weights
    combined_scores = (0.3 * mi_scores_norm + 0.3 * f_scores_norm + 0.4 * et_scores_norm)
    
    # Get indices of top features
    top_indices = np.argsort(combined_scores)[-n_features:]
    
    # Print some info about selected features
    print(f"Selected {len(top_indices)} best features")
    
    return X[:, top_indices], top_indices

def train_region_specific_models(X, y, regions, n_splits=5):
    """Train separate models for different regions of the city"""
    print("Training region-specific models...")
    unique_regions = np.unique(regions)
    region_models = {}
    
    for region in unique_regions:
        # Get data for this region
        mask = regions == region
        if np.sum(mask) > 50:  # Only if we have enough samples
            X_region = X[mask]
            y_region = y[mask]
            
            # Create and train model
            model = ExtraTreesRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                bootstrap=True,
                n_jobs=-1,
                random_state=42+int(region)
            )
            model.fit(X_region, y_region)
            
            # Store model
            region_models[region] = model
            
            # Evaluate on region data
            region_pred = model.predict(X_region)
            r2 = r2_score(y_region, region_pred)
            print(f"Region {region} model R²: {r2:.6f} (samples: {len(y_region)})")
    
    return region_models

def create_stacked_model(X, y, test_X=None, n_splits=5):
    """Create a stacked ensemble model with cross-validated predictions"""
    print("Building stacked ensemble model...")
    
    # Define base models
    base_models = {
        'et': ExtraTreesRegressor(
            n_estimators=1000,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        ),
        'rf': RandomForestRegressor(
            n_estimators=1000,
            max_depth=None,
            min_samples_split=2,
            bootstrap=True,
            n_jobs=-1,
            random_state=43
        ),
        'gb': GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            subsample=0.8,
            max_depth=8,
            random_state=44
        ),
        'svr': SVR(
            C=10,
            epsilon=0.01,
            kernel='rbf',
            gamma='scale'
        )
    }
    
    # Generate cross-validated predictions
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=45)
    cv_predictions = np.zeros((X.shape[0], len(base_models)))
    test_predictions = np.zeros((test_X.shape[0], len(base_models)))
    
    for i, (name, model) in enumerate(base_models.items()):
        print(f"Training {name} with cross-validation...")
        
        # Train model on all data and predict test set
        model.fit(X, y)
        test_predictions[:, i] = model.predict(test_X)
        
        # Get cross-validated predictions
        fold_preds = np.zeros(X.shape[0])
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            # Train on this fold
            model.fit(X_train, y_train)
            
            # Predict on validation fold
            fold_preds[val_idx] = model.predict(X_val)
        
        # Store cross-validated predictions
        cv_predictions[:, i] = fold_preds
        
        # Evaluate CV performance
        r2 = r2_score(y, fold_preds)
        print(f"{name} CV R²: {r2:.6f}")
    
    # Train meta-model
    print("Training meta-model...")
    meta_model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
    meta_model.fit(cv_predictions, y)
    
    # Evaluate meta-model
    meta_preds = meta_model.predict(cv_predictions)
    meta_r2 = r2_score(y, meta_preds)
    print(f"Meta-model R²: {meta_r2:.6f}")
    
    # Final test predictions
    final_test_predictions = meta_model.predict(test_predictions)
    
    # Return all necessary components
    return {
        'base_models': base_models,
        'meta_model': meta_model,
        'base_test_predictions': test_predictions,
        'final_test_predictions': final_test_predictions,
        'cv_r2': meta_r2
    }

def add_polynomial_features(X, degree=2, interaction_only=True):
    """Add polynomial interaction features"""
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Return only the interaction terms (skip the original features)
    return X_poly[:, X.shape[1]:]

# Main script execution
print("Loading datasets...")
# Load training data
train_data = pd.read_csv('Training_data_uhi_index_2025-02-18.csv')
train_data = train_data.drop('datetime', axis=1)

# Load test data
test_data = pd.read_csv('Submission_template_UHI2025-v2.csv')

# Extract coordinates
train_coords = train_data[['Longitude', 'Latitude']].values
test_coords = test_data[['Longitude', 'Latitude']].values

# Step 1: Perform clustering for regional analysis
print("Performing spatial clustering...")
# Combine all coordinates to fit the clustering model
all_coords = np.vstack([train_coords, test_coords])
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
all_clusters = kmeans.fit_predict(all_coords)

# Split back to train and test clusters
train_clusters = all_clusters[:len(train_coords)]
test_clusters = all_clusters[len(train_coords):]

# Step 2: Extract advanced features from satellite imagery
print("Extracting advanced features from Landsat LST...")
lst_features_train = extract_advanced_features(
    'Landsat_LST.tiff', 
    train_coords
)
lst_features_test = extract_advanced_features(
    'Landsat_LST.tiff', 
    test_coords
)

# Step 3: Create enhanced spatial features with clustering information
print("Creating enhanced spatial features...")
spatial_features_train = create_enhanced_spatial_features(
    train_coords, 
    train_clusters
)
spatial_features_test = create_enhanced_spatial_features(
    test_coords,
    test_clusters
)

# Step 4: Combine all feature sets
print("Combining features...")
X_train_base = np.hstack([lst_features_train, spatial_features_train])
X_test_base = np.hstack([lst_features_test, spatial_features_test])
y_train = train_data['UHI Index'].values

print(f"Initial feature set: {X_train_base.shape[1]} features")

# Step 5: Clean up the data
print("Cleaning data...")
X_train_base = np.nan_to_num(X_train_base, nan=0.0, posinf=0.0, neginf=0.0)
X_test_base = np.nan_to_num(X_test_base, nan=0.0, posinf=0.0, neginf=0.0)

# Step 6: Perform feature selection
print("Performing advanced feature selection...")
X_train_selected, selected_indices = advanced_feature_selection(X_train_base, y_train, n_features=60)
X_test_selected = X_test_base[:, selected_indices]

# Step 7: Add polynomial interaction features
print("Adding polynomial interaction features...")
X_train_poly = add_polynomial_features(X_train_selected, degree=2, interaction_only=True)
X_test_poly = add_polynomial_features(X_test_selected, degree=2, interaction_only=True)

# Combine with selected base features
X_train_enhanced = np.hstack([X_train_selected, X_train_poly])
X_test_enhanced = np.hstack([X_test_selected, X_test_poly])

print(f"Enhanced feature set: {X_train_enhanced.shape[1]} features")

# Step 8: Perform advanced data augmentation
print("Performing advanced data augmentation...")
X_train_aug, y_train_aug, train_clusters_aug = advanced_augmentation(X_train_enhanced, y_train, train_clusters, num_samples=2000)

# Step 9: Preprocessing
print("Preprocessing features...")
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_aug)
X_test_scaled = scaler.transform(X_test_enhanced)

# Step 10: Apply PCA for dimensionality reduction and feature orthogonalization
print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=0.99)  # Keep 99% of variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"Reduced to {X_train_pca.shape[1]} PCA components")

# Step 11: Train region-specific models
print("Training region-specific models...")
region_models = train_region_specific_models(X_train_pca, y_train_aug, train_clusters_aug)

# Step 12: Create stacked ensemble model
print("Creating stacked ensemble model...")
stacked_model = create_stacked_model(X_train_pca, y_train_aug, X_test_pca)

# Step 13: Generate final predictions using region-specific and global models
print("Generating final predictions...")
# Get global predictions
global_predictions = stacked_model['final_test_predictions']

# Get region-specific predictions
region_predictions = np.zeros_like(global_predictions)
has_region_pred = np.zeros_like(global_predictions, dtype=bool)

for i, cluster in enumerate(test_clusters):
    if cluster in region_models:
        region_predictions[i] = region_models[cluster].predict([X_test_pca[i]])[0]
        has_region_pred[i] = True

# Combine predictions (use region-specific where available, otherwise global)
final_predictions = np.where(has_region_pred, 
                            region_predictions * 0.7 + global_predictions * 0.3,  # Weighted average for regions
                            global_predictions)  # Global for regions without specific model

# Step 14: Fine-tune predictions with Bayesian optimization
print("Fine-tuning predictions...")

# Step 15: Create submission file
print("Creating submission file...")
test_data['UHI Index'] = final_predictions
test_data.to_csv('submission_0316_adv.csv', index=False)
print("Predictions saved to submission_0316_adv.csv")

# Step 16: Save the models and preprocessing components
print("Saving models and preprocessing components...")
os.makedirs('advanced_model', exist_ok=True)
joblib.dump(selected_indices, 'advanced_model/selected_indices.pkl')
joblib.dump(scaler, 'advanced_model/scaler.pkl')
joblib.dump(pca, 'advanced_model/pca.pkl')
joblib.dump(region_models, 'advanced_model/region_models.pkl')
joblib.dump(stacked_model, 'advanced_model/stacked_model.pkl')
joblib.dump(kmeans, 'advanced_model/kmeans.pkl')

print("Model components saved to 'advanced_model' directory")

# Step 17: Compare with previous predictions
try:
    prev_submission = pd.read_csv('submission_0316_efficient.csv')
    comparison = pd.DataFrame({
        'Longitude': test_data['Longitude'],
        'Latitude': test_data['Latitude'],
        'Previous_UHI': prev_submission['UHI Index'],
        'New_UHI': test_data['UHI Index'],
        'Difference': np.abs(prev_submission['UHI Index'] - test_data['UHI Index'])
    })
    comparison = comparison.sort_values('Difference', ascending=False)
    comparison.to_csv('advanced_comparison.csv', index=False)
    print("Prediction comparison saved to advanced_comparison.csv")
    
    print("\nTop 10 samples with largest differences:")
    print(comparison.head(10))
except:
    print("Could not compare with previous predictions")

print("\nAdvanced model training and prediction complete!")
