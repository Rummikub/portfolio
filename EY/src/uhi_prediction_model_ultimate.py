import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy.ndimage import gaussian_filter, sobel
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

# Custom distance function
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

# Enhanced feature extraction from satellite imagery
def extract_advanced_features(tiff_path, coords, window_sizes=[3, 7, 15, 31]):
    """Extract advanced features from satellite imagery with multiple scales and texture analysis"""
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
                
                # Extract multi-scale features
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
                            # Apply smoothing with different sigmas
                            smoothed1 = gaussian_filter(pixel_array, sigma=0.5)
                            smoothed2 = gaussian_filter(pixel_array, sigma=1.0)
                            
                            # Calculate gradients in different directions
                            gradient_x = sobel(smoothed1, axis=0)
                            gradient_y = sobel(smoothed1, axis=1)
                            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                            
                            # Calculate directional gradients (edge detection)
                            edge_horizontal = np.abs(np.diff(smoothed2, axis=0))
                            edge_vertical = np.abs(np.diff(smoothed2, axis=1))
                            
                            # Compute various statistical features
                            features = [
                                np.mean(pixel_array),            # Mean value
                                np.std(pixel_array),             # Standard deviation
                                np.median(pixel_array),          # Median
                                np.percentile(pixel_array, 10),  # 10th percentile
                                np.percentile(pixel_array, 25),  # 1st quartile
                                np.percentile(pixel_array, 75),  # 3rd quartile
                                np.percentile(pixel_array, 90),  # 90th percentile
                                np.max(pixel_array) - np.min(pixel_array),  # Range
                                
                                # Gradient features
                                np.mean(gradient_magnitude),     # Mean gradient
                                np.std(gradient_magnitude),      # Gradient standard deviation
                                np.percentile(gradient_magnitude, 75),  # 75th percentile of gradient
                                
                                # Edge features
                                np.mean(edge_horizontal),        # Mean horizontal edge
                                np.mean(edge_vertical),          # Mean vertical edge
                                
                                # Shape and distribution features
                                np.max(pixel_array),             # Maximum value
                                np.min(pixel_array),             # Minimum value
                                np.sum(pixel_array > np.mean(pixel_array)) / pixel_array.size,  # Fraction above mean
                                np.sum(pixel_array > np.percentile(pixel_array, 75)) / pixel_array.size,  # Fraction in top quartile
                                
                                # Entropy-like measure (simplified)
                                -np.sum((pixel_array/np.sum(pixel_array))**2) if np.sum(pixel_array) > 0 else 0,
                                
                                # Skewness and kurtosis (simplified)
                                np.mean(((pixel_array - np.mean(pixel_array))/np.std(pixel_array))**3) if np.std(pixel_array) > 0 else 0,
                                np.mean(((pixel_array - np.mean(pixel_array))/np.std(pixel_array))**4) if np.std(pixel_array) > 0 else 0,
                            ]
                            
                            point_features.extend(features)
                        else:
                            # If pixel array is empty
                            point_features.extend([0] * 20)
                        
                    except Exception as e:
                        # If window reading fails, add zeros
                        point_features.extend([0] * 20)
                
                # Add radial profile features (analyze in concentric rings)
                for radius in [3, 5, 7, 10, 15]:
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
                                    np.percentile(masked_array, 75),
                                ]
                                point_features.extend(ring_features)
                            else:
                                point_features.extend([0, 0, 0])
                        else:
                            point_features.extend([0, 0, 0])
                            
                    except Exception as e:
                        point_features.extend([0, 0, 0])
                        
                all_features.append(point_features)
                
            except Exception as e:
                # If coordinate conversion fails, add zeros for all features
                all_features.append([0] * (20 * len(window_sizes) + 3 * 5))
                
        return np.array(all_features)

# Enhanced spatial features with more sophisticated distance calculations
def create_enhanced_spatial_features(coords):
    """Create enhanced features based on spatial coordinates with exponential decay distances"""
    # NYC reference points (key locations that could influence UHI)
    reference_points = {
        'central_park': (-73.9654, 40.7829),
        'times_square': (-73.9855, 40.7580),
        'downtown': (-74.0060, 40.7128),
        'east_river': (-73.9762, 40.7678),
        'hudson_river': (-74.0099, 40.7258),
        'harlem': (-73.9465, 40.8116),
        'brooklyn': (-73.9442, 40.6782),
        'bronx': (-73.8648, 40.8448),
        'queens': (-73.7949, 40.7282),
        'jfk_airport': (-73.7781, 40.6413),
        'central_manhattan': (-73.9712, 40.7831)
    }
    
    # UHI hotspots (hypothetical, adjust based on domain knowledge)
    uhi_hotspots = [
        (-73.9855, 40.7580),  # Times Square
        (-73.9877, 40.7498),  # Midtown
        (-74.0445, 40.6892),  # Downtown
        (-73.8648, 40.8448),  # South Bronx
    ]
    
    # Water bodies (cooling effect)
    water_bodies = [
        (-74.0099, 40.7258),  # Hudson River West
        (-73.9762, 40.7678),  # East River
        (-74.0431, 40.6698),  # New York Harbor
        (-73.8740, 40.7745),  # Flushing Bay
    ]
    
    # Green spaces (cooling effect)
    green_spaces = [
        (-73.9654, 40.7829),  # Central Park
        (-73.9209, 40.6694),  # Prospect Park
        (-73.8751, 40.8501),  # Bronx Park
        (-73.8458, 40.7282),  # Flushing Meadows
    ]
    
    features = []
    for lon, lat in coords:
        # Basic coordinates and derived features
        x, y = lon, lat
        x_scaled = (x + 74.1) * 100  # Scale and shift for better numeric properties
        y_scaled = (y - 40.5) * 100
        
        point_features = [
            x_scaled, y_scaled,                          # Scaled coordinates
            x_scaled**2, y_scaled**2, x_scaled*y_scaled, # Quadratic terms
        ]
        
        # Distance to reference points with exponential decay
        for name, (ref_lon, ref_lat) in reference_points.items():
            dist = haversine(lon, lat, ref_lon, ref_lat)
            exp_dist = np.exp(-dist)  # Exponential decay - stronger effect nearby
            point_features.append(dist)       # Regular distance
            point_features.append(exp_dist)   # Exponential decay
        
        # Minimum distance to UHI hotspots
        uhi_distances = [haversine(lon, lat, hot_lon, hot_lat) for hot_lon, hot_lat in uhi_hotspots]
        min_uhi_dist = min(uhi_distances)
        exp_min_uhi_dist = np.exp(-min_uhi_dist)
        point_features.append(min_uhi_dist)
        point_features.append(exp_min_uhi_dist)
        
        # Minimum distance to water (cooling effect)
        water_distances = [haversine(lon, lat, w_lon, w_lat) for w_lon, w_lat in water_bodies]
        min_water_dist = min(water_distances)
        exp_min_water_dist = np.exp(-min_water_dist)
        point_features.append(min_water_dist)
        point_features.append(exp_min_water_dist)
        
        # Minimum distance to green spaces (cooling effect)
        green_distances = [haversine(lon, lat, g_lon, g_lat) for g_lon, g_lat in green_spaces]
        min_green_dist = min(green_distances)
        exp_min_green_dist = np.exp(-min_green_dist)
        point_features.append(min_green_dist)
        point_features.append(exp_min_green_dist)
        
        # Special "urban canyon" proxy - ratio of distances
        urban_canyon_proxy = min_water_dist / (min_green_dist + 0.01)
        point_features.append(urban_canyon_proxy)
        
        # Calculate angle to center of Manhattan
        center_lon, center_lat = -73.9712, 40.7831  # Approximate center
        angle = np.arctan2(lat - center_lat, lon - center_lon)
        
        # Add cyclic encoding of angle to maintain continuity
        point_features.append(np.sin(angle))
        point_features.append(np.cos(angle))
        
        features.append(point_features)
        
    return np.array(features)

# Advanced data augmentation techniques
def advanced_augmentation(X, y, num_samples=5000):
    """Advanced data augmentation with multiple strategies"""
    print(f"Original data: {X.shape[0]} samples")
    augmented_X = []
    augmented_y = []
    
    # Strategy 1: Add small noise to features
    noise_factor = 0.001
    for _ in range(num_samples // 5):
        noise = np.random.normal(0, noise_factor, X.shape)
        augmented_X.append(X + noise)
        augmented_y.append(y)
    
    # Strategy 2: Interpolate between nearby points
    # Find nearest neighbors for each point - simplified approach
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    
    for _ in range(num_samples // 5):
        # Randomly select points
        idx = np.random.choice(len(X), min(1000, len(X)), replace=False)
        
        for i in idx:
            # Get nearest neighbor
            j = indices[i, 1]
            
            # Random interpolation weight
            alpha = np.random.uniform(0.2, 0.8)
            
            # Interpolate features and target
            new_x = alpha * X[i] + (1 - alpha) * X[j]
            new_y = alpha * y[i] + (1 - alpha) * y[j]
            
            augmented_X.append(new_x)
            augmented_y.append(new_y)
    
    # Strategy 3: Enhanced SMOTE-like approach for areas with extreme UHI values
    # Find extreme UHI values (top and bottom 10%)
    threshold_high = np.percentile(y, 90)
    threshold_low = np.percentile(y, 10)
    extreme_idx_high = np.where(y >= threshold_high)[0]
    extreme_idx_low = np.where(y <= threshold_low)[0]
    
    # Over-sample extreme values
    for idx_group in [extreme_idx_high, extreme_idx_low]:
        if len(idx_group) >= 2:
            nn = NearestNeighbors(n_neighbors=min(5, len(idx_group)))
            nn.fit(X[idx_group])
            _, indices = nn.kneighbors(X[idx_group])
            
            for _ in range(num_samples // 5):
                # Randomly select extreme points
                i = np.random.choice(len(idx_group))
                
                # Get one of its neighbors
                j = indices[i, np.random.randint(1, min(5, len(idx_group)))]
                
                # Random interpolation weight
                alpha = np.random.uniform(0.2, 0.8)
                
                # Interpolate features and target
                new_x = alpha * X[idx_group[i]] + (1 - alpha) * X[idx_group[j]]
                new_y = alpha * y[idx_group[i]] + (1 - alpha) * y[idx_group[j]]
                
                augmented_X.append(new_x)
                augmented_y.append(new_y)
    
    # Combine original and augmented data
    augmented_X = np.vstack([X] + augmented_X)
    augmented_y = np.concatenate([y] + augmented_y)
    
    print(f"After augmentation: {len(augmented_y)} samples")
    return augmented_X, augmented_y

# Advanced feature selection with multiple techniques
def advanced_feature_selection(X, y, n_features=100):
    """Perform advanced feature selection using multiple techniques"""
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, f_regression
    from sklearn.ensemble import ExtraTreesRegressor
    
    # Method 1: Mutual information
    mi_selector = SelectKBest(mutual_info_regression, k=n_features)
    mi_selector.fit(X, y)
    mi_scores = mi_selector.scores_
    
    # Method 2: F-regression
    f_selector = SelectKBest(f_regression, k=n_features)
    f_selector.fit(X, y)
    f_scores = f_selector.scores_
    
    # Method 3: Tree-based feature importance
    et = ExtraTreesRegressor(n_estimators=100, random_state=42)
    et.fit(X, y)
    et_scores = et.feature_importances_
    
    # Normalize scores
    mi_scores = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-10)
    f_scores = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-10)
    et_scores = (et_scores - et_scores.min()) / (et_scores.max() - et_scores.min() + 1e-10)
    
    # Combined score with weights
    combined_scores = 0.4 * mi_scores + 0.3 * f_scores + 0.3 * et_scores
    
    # Select top features
    top_indices = np.argsort(combined_scores)[-n_features:]
    
    # Return selected features and their scores
    return top_indices, combined_scores

# Main execution function
def main():
    print("Loading datasets...")
    # Load training data
    train_data = pd.read_csv('Training_data_uhi_index_2025-02-18.csv')
    train_data = train_data.drop('datetime', axis=1)

    # Load test data
    test_data = pd.read_csv('Submission_template_UHI2025-v2.csv')

    # Extract advanced features from satellite imagery
    print("Extracting advanced features from Landsat LST...")
    lst_features_train = extract_advanced_features(
        'Landsat_LST.tiff', 
        train_data[['Longitude', 'Latitude']].values
    )
    lst_features_test = extract_advanced_features(
        'Landsat_LST.tiff', 
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
        spatial_features_train
    ])
    X_test = np.hstack([
        lst_features_test,
        spatial_features_test
    ])
    y_train = train_data['UHI Index'].values

    print(f"Initial feature set: {X_train.shape[1]} features")

    # Clean up the data
    print("Cleaning data...")
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Perform feature selection
    print("Performing advanced feature selection...")
    selected_indices, feature_scores = advanced_feature_selection(X_train, y_train, n_features=75)
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    print(f"Selected {X_train_selected.shape[1]} best features")

    # Data augmentation
    print("Performing advanced data augmentation...")
    X_train_aug, y_train_aug = advanced_augmentation(X_train_selected, y_train, num_samples=5000)

    # Preprocessing with QuantileTransformer for robustness to outliers
    print("Preprocessing features with QuantileTransformer...")
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_scaled = qt.fit_transform(X_train_aug)
    X_test_scaled = qt.transform(X_test_selected)

    # Apply PCA for dimensionality reduction
    print("Applying PCA for dimensionality reduction...")
    pca = PCA(n_components=0.999)  # Keep 99.9% of variance
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"Reduced to {X_train_pca.shape[1]} PCA components")

    # Define base models for stacking
    print("Training stacking ensemble model...")
    base_models = [
        ('et', ExtraTreesRegressor(
            n_estimators=1000, 
            max_depth=None,
            min_samples_split=2,
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        )),
        ('rf', RandomForestRegressor(
            n_estimators=1000, 
            max_depth=None,
            bootstrap=True,
            n_jobs=-1,
            random_state=43
        )),
        ('gb', GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            max_depth=7,
            random_state=44
        )),
        ('xgb', xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            subsample=0.8,
            max_depth=7,
            random_state=45
        )),
    ]

    # Meta-level model
    meta_model = Ridge(alpha=0.5)

    # Create stacking regressor with cross-validation
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )

    # Fit stacking model
    stacking_model.fit(X_train_pca, y_train_aug)

    # Evaluate base models individually for comparison
    print("Evaluating individual base models for comparison:")
    for name, model in base_models:
        model.fit(X_train_pca, y_train_aug)
        train_pred = model.predict(X_train_pca)
        r2 = r2_score(y_train_aug, train_pred)
        rmse = np.sqrt(mean_squared_error(y_train_aug, train_pred))
        print(f"  {name} - R²: {r2:.6f}, RMSE: {rmse:.6f}")

    # Evaluate stacking model
    train_pred_stack = stacking_model.predict(X_train_pca)
    r2_stack = r2_score(y_train_aug, train_pred_stack)
    rmse_stack = np.sqrt(mean_squared_error(y_train_aug, train_pred_stack))
    print(f"Stacking ensemble - R²: {r2_stack:.6f}, RMSE: {rmse_stack:.6f}")

    # Compute weighted cross-validation scores with special focus on extreme values
    def custom_scorer(model, X, y):
        # Predict
        y_pred = model.predict(X)
        
        # Calculate weights giving more importance to extreme values
        weights = np.ones_like(y)
        extreme_high = y > np.percentile(y, 90)
        extreme_low = y < np.percentile(y, 10)
        weights[extreme_high | extreme_low] = 2.0  # Double weight for extreme values
        
        # Weighted MSE
        weighted_mse = np.average(((y - y_pred) ** 2), weights=weights)
        return -weighted_mse  # Negative because sklearn maximizes score
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(stacking_model, X_train_pca, y_train_aug, 
                                cv=cv, scoring=custom_scorer)
    
    weighted_accuracy = 100 * (1 + np.mean(cv_scores))  # Transform to accuracy-like metric
    print(f"Weighted CV accuracy estimate: {weighted_accuracy:.4f}%")

    # Cross-validation for final confidence
    cv_scores_r2 = cross_val_score(stacking_model, X_train_pca, y_train_aug, 
                                  cv=5, scoring='r2')
    mean_cv_r2 = np.mean(cv_scores_r2)
    print(f"Mean 5-fold CV R²: {mean_cv_r2:.6f}")

    print(f"Estimated model accuracy: {100 * mean_cv_r2:.4f}%")

    # Generate predictions for test data
    print("Generating predictions for test data...")
    test_pred = stacking_model.predict(X_test_pca)
    
    # Optional: Post-process predictions
    # For example, clip to reasonable range
    test_pred = np.clip(test_pred, 0.8, 1.2)
    
    # Create submission file
    test_data['UHI Index'] = test_pred
    test_data.to_csv('submission_0316_ultimate.csv', index=False)
    print("Predictions saved to submission_0316_ultimate.csv")

    # Save the model and preprocessing components
    os.makedirs('ultimate_model', exist_ok=True)
    joblib.dump(stacking_model, 'ultimate_model/stacking_model.pkl')
    joblib.dump(qt, 'ultimate_model/quantile_transformer.pkl')
    joblib.dump(pca, 'ultimate_model/pca.pkl')
    joblib.dump(selected_indices, 'ultimate_model/selected_indices.pkl')
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
        comparison.to_csv('ultimate_comparison.csv', index=False)
        print("Prediction comparison saved to ultimate_comparison.csv")
        
        print("\nTop 10 samples with largest differences:")
        print(comparison.head(10))
    except:
        print("Could not compare with previous predictions")
        
    print("\nUltimate model training and prediction complete!")

if __name__ == "__main__":
    main()
