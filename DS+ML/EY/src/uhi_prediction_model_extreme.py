import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFECV, VarianceThreshold
import xgboost as xgb
from scipy.ndimage import gaussian_filter, sobel, median_filter, laplace
from scipy.stats import skew, kurtosis
from math import radians, cos, sin, asin, sqrt
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

try:
    import fiona
    from shapely.geometry import shape, Point
    HAS_FIONA = True
except ImportError:
    print("Warning: fiona or shapely not installed. Building footprint features will not be used.")
    HAS_FIONA = False

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # Radius of earth in kilometers
    return c * r

def load_building_footprints(kml_path):
    """Load building footprint geometries from KML file"""
    if not HAS_FIONA:
        return None
        
    try:
        with fiona.open(kml_path) as src:
            footprints = [shape(feature['geometry']) for feature in src]
        return footprints
    except Exception as e:
        print(f"Error loading building footprints: {e}")
        return None

def extract_advanced_features(tiff_path, coords, window_sizes=[3, 7, 11, 15, 21]):
    """
    Extract comprehensive multi-scale features with advanced filtering
    """
    with rasterio.open(tiff_path) as src:
        all_features = []
        height, width = src.height, src.width
        
        for lon, lat in coords:
            try:
                # Convert geographic coordinates to pixel coordinates
                py, px = src.index(lon, lat)
                
                point_features = []
                # Extract features at different window sizes
                for size in window_sizes:
                    padding = size // 2
                    # Ensure window is within image bounds
                    x_start = max(0, px - padding)
                    y_start = max(0, py - padding)
                    x_end = min(width, px + padding + 1)
                    y_end = min(height, py + padding + 1)
                    
                    window = Window(x_start, y_start, x_end - x_start, y_end - y_start)
                    
                    try:
                        # Read the window
                        pixel_array = src.read(1, window=window)
                        # Handle potential NaN values
                        pixel_array = np.nan_to_num(pixel_array)
                        
                        if pixel_array.size > 0:
                            # Apply multiple filters
                            smoothed = gaussian_filter(pixel_array, sigma=1)
                            median = median_filter(pixel_array, size=3)
                            laplacian = laplace(smoothed)
                            
                            # Calculate gradients
                            gradient_x = sobel(smoothed, axis=0)
                            gradient_y = sobel(smoothed, axis=1)
                            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                            
                            # Multi-scale smoothing
                            multi_smoothed = [
                                gaussian_filter(pixel_array, sigma=s) 
                                for s in [0.5, 1, 2]
                            ]
                            
                            # Compute local maxima count
                            local_max_count = np.sum((pixel_array > np.roll(pixel_array, 1, axis=0)) & 
                                                    (pixel_array > np.roll(pixel_array, -1, axis=0)) &
                                                    (pixel_array > np.roll(pixel_array, 1, axis=1)) &
                                                    (pixel_array > np.roll(pixel_array, -1, axis=1)))
                            
                            # Extract extensive feature set
                            features = [
                                # Basic statistics
                                np.mean(pixel_array),            # Mean
                                np.std(pixel_array),             # Standard deviation
                                np.median(pixel_array),          # Median
                                np.min(pixel_array),             # Minimum
                                np.max(pixel_array),             # Maximum
                                np.percentile(pixel_array, 25),  # 1st quartile
                                np.percentile(pixel_array, 75),  # 3rd quartile
                                np.percentile(pixel_array, 10),  # 10th percentile
                                np.percentile(pixel_array, 90),  # 90th percentile
                                
                                # Smoothed features
                                np.mean(smoothed),               # Smoothed mean
                                np.std(smoothed),                # Smoothed std
                                np.median(smoothed),             # Smoothed median
                                
                                # Median filter features
                                np.mean(median),                 # Median filter mean
                                np.std(median),                  # Median filter std
                                
                                # Multi-scale smoothing features
                                *[np.mean(ms) for ms in multi_smoothed],
                                *[np.std(ms) for ms in multi_smoothed],
                                
                                # Gradient features
                                np.mean(gradient_magnitude),     # Mean gradient magnitude
                                np.std(gradient_magnitude),      # Std of gradient
                                np.max(gradient_magnitude),      # Max gradient
                                np.sum(gradient_magnitude) / pixel_array.size,  # Total gradient normalized
                                
                                # Laplacian features (edge detection)
                                np.mean(laplacian),              # Mean laplacian
                                np.std(laplacian),               # Std of laplacian
                                np.sum(np.abs(laplacian)) / pixel_array.size,  # Total edge strength
                                
                                # Texture features
                                skew(pixel_array.flatten()),     # Skewness
                                kurtosis(pixel_array.flatten()), # Kurtosis
                                
                                # Pattern features
                                local_max_count / pixel_array.size,  # Local maxima density
                                
                                # Range features
                                np.ptp(pixel_array),             # Peak-to-peak
                                
                                # Energy features
                                np.sum(pixel_array**2) / pixel_array.size,  # Energy
                                
                                # Entropy approximation
                                np.sum(np.abs(np.diff(pixel_array, axis=0))) / pixel_array.size,  # Horizontal changes
                                np.sum(np.abs(np.diff(pixel_array, axis=1))) / pixel_array.size,  # Vertical changes
                                
                                # Histogram features - capture distribution
                                *[h / pixel_array.size for h in np.histogram(pixel_array, bins=10)[0]],
                            ]
                            
                            point_features.extend(features)
                        else:
                            # If pixel array is empty
                            point_features.extend([0] * 43)  # Match the number of features above
                        
                    except Exception as e:
                        # If window reading fails, add zeros
                        point_features.extend([0] * 43)
                
                all_features.append(point_features)
                
            except Exception as e:
                # If coordinate conversion fails, add zeros for all window sizes
                all_features.append([0] * (43 * len(window_sizes)))
                
        return np.array(all_features)

def create_enhanced_spatial_features(coords, footprints=None):
    """
    Create advanced spatial features with building proximity and other environmental factors
    """
    # NYC reference points
    landmarks = {
        'central_park': (-73.9654, 40.7829),
        'times_square': (-73.9855, 40.7580),
        'downtown': (-74.0060, 40.7128),
        'east_river': (-73.9762, 40.7678),
        'hudson_river': (-74.0099, 40.7258),
        'harlem': (-73.9465, 40.8116),
        'bronx': (-73.8648, 40.8448),
        'queens': (-73.7949, 40.7282),
        'brooklyn': (-73.9442, 40.6782),
    }
    
    # Urban heat island reference points (estimated high UHI areas)
    uhi_hotspots = [
        (-73.9855, 40.7580),  # Times Square (high density)
        (-73.9844, 40.7484),  # Midtown
        (-74.0060, 40.7128),  # Downtown
        (-73.9712, 40.7831),  # Upper East Side
    ]
    
    # Green spaces (cooling effect)
    green_spaces = [
        (-73.9654, 40.7829),  # Central Park
        (-73.9771, 40.7695),  # Bryant Park
        (-74.0046, 40.7029),  # Battery Park
    ]
    
    features = []
    for lon, lat in coords:
        # Distance and angle to center
        center_lon, center_lat = -73.95, 40.80  # Approximate center of the area
        dist_center = haversine(lon, lat, center_lon, center_lat)
        angle = np.arctan2(lat - center_lat, lon - center_lon)
        
        # Distance to landmarks
        landmark_dists = [
            haversine(lon, lat, landmark[0], landmark[1]) 
            for landmark in landmarks.values()
        ]
        
        # Distance to UHI hotspots (minimum and average)
        uhi_dists = [haversine(lon, lat, hs[0], hs[1]) for hs in uhi_hotspots]
        min_uhi_dist = min(uhi_dists)
        avg_uhi_dist = np.mean(uhi_dists)
        
        # Distance to green spaces (minimum and average)
        green_dists = [haversine(lon, lat, gs[0], gs[1]) for gs in green_spaces]
        min_green_dist = min(green_dists)
        avg_green_dist = np.mean(green_dists)
        
        # Coordinates and transformations
        x, y = lon, lat
        x_scaled = (x - (-74.05)) / ((-73.85) - (-74.05))  # Normalize to 0-1
        y_scaled = (y - 40.65) / (40.90 - 40.65)           # Normalize to 0-1
        
        # Building proximity features
        building_features = [0, 0, 0, 0, 0]  # Default placeholders
        if footprints and HAS_FIONA:
            try:
                point = Point(lon, lat)
                # Distance to nearest building
                min_dist = float('inf')
                buildings_within_100m = 0
                buildings_within_200m = 0
                building_area_within_200m = 0
                building_perimeter_within_200m = 0
                
                # Check only a subset of buildings for performance
                for building in footprints[:100]:  # Limit to first 100 for performance
                    try:
                        d = point.distance(building)
                        min_dist = min(min_dist, d)
                        
                        # Count buildings within 100m
                        if d < 0.001:  # approx 100m
                            buildings_within_100m += 1
                        
                        # Count and measure buildings within 200m
                        if d < 0.002:  # approx 200m
                            buildings_within_200m += 1
                            building_area_within_200m += building.area
                            building_perimeter_within_200m += building.length
                    except:
                        continue
                
                if min_dist == float('inf'):
                    min_dist = 0.01  # Default if no buildings found
                    
                building_features = [
                    min_dist * 100,                    # Distance to nearest building in meters
                    buildings_within_100m,             # Number of buildings within 100m
                    buildings_within_200m,             # Number of buildings within 200m
                    building_area_within_200m * 10000, # Building area within 200m in sq meters
                    building_perimeter_within_200m * 100, # Building perimeter within 200m in meters
                ]
            except Exception as e:
                print(f"Error processing building features: {e}")
        
        # Advanced spatial features
        point_features = [
            # Distance metrics
            dist_center,                        # Distance from center
            min_uhi_dist,                       # Distance to nearest UHI hotspot
            avg_uhi_dist,                       # Average distance to UHI hotspots
            min_green_dist,                     # Distance to nearest green space
            avg_green_dist,                     # Average distance to green spaces
            np.exp(-dist_center),               # Exponential decay of distance
            1/(1 + dist_center),                # Inverse distance
            
            # Angular features
            angle,                              # Angle
            np.sin(angle),                      # Sine of angle
            np.cos(angle),                      # Cosine of angle
            np.sin(2*angle),                    # Harmonic components
            np.cos(2*angle),
            np.sin(4*angle),
            np.cos(4*angle),
            
            # Coordinate features
            x, y,                               # Raw coordinates
            x**2, y**2,                         # Quadratic terms
            x*y,                                # Interaction
            x_scaled, y_scaled,                 # Scaled coordinates (0-1)
            x_scaled**2, y_scaled**2,           # Scaled quadratic
            x_scaled*y_scaled,                  # Scaled interaction
            
            # Distance transformations
            dist_center * np.sin(angle),        # Polar transform
            dist_center * np.cos(angle),        # Polar transform
            dist_center**2,                     # Squared distance
            np.sqrt(dist_center),               # Square root distance
            np.log1p(dist_center),              # Log distance
            
            # Oscillatory features
            np.sin(lon * 20),                   # Capture cyclic patterns
            np.cos(lat * 20),
            np.sin(lon * 50),
            np.cos(lat * 50),
            
            # Heat island potential indicators
            min_green_dist / (min_uhi_dist + 0.001),  # Ratio of green to UHI proximity
            np.exp(-min_green_dist) * np.exp(min_uhi_dist),  # Combined effect
        ]
        
        # Add landmark distances
        point_features.extend(landmark_dists)
        
        # Add building features
        point_features.extend(building_features)
        
        features.append(point_features)
        
    return np.array(features)

def create_kernel_features(X, n_components=100):
    """Create non-linear features using kernel approximation"""
    feature_map = Nystroem(
        kernel='rbf',
        gamma=0.1,
        n_components=n_components,
        random_state=42
    )
    return feature_map.fit_transform(X)

# Main execution
print("Loading datasets...")
# Load training data
train_data = pd.read_csv('Training_data_uhi_index_2025-02-18.csv')
train_data = train_data.drop('datetime', axis=1)

# Load test data
test_data = pd.read_csv('Submission_template_UHI2025-v2.csv')

# Try to load building footprints
print("Loading building footprints...")
building_footprints = None
if HAS_FIONA:
    try:
        building_footprints = load_building_footprints('Building_Footprint.kml')
        if building_footprints:
            print(f"Loaded {len(building_footprints)} building footprints")
    except Exception as e:
        print(f"Error loading building footprints: {e}")

# Extract advanced features from both satellite images
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
    train_data[['Longitude', 'Latitude']].values, 
    building_footprints
)
spatial_features_test = create_enhanced_spatial_features(
    test_data[['Longitude', 'Latitude']].values, 
    building_footprints
)

# Add raw coordinates as additional features
coord_features_train = train_data[['Longitude', 'Latitude']].values
coord_features_test = test_data[['Longitude', 'Latitude']].values

# Combine all features
print("Combining features...")
X_train = np.hstack([
    lst_features_train,
    s2_features_train,
    spatial_features_train,
    coord_features_train
])
X_test = np.hstack([
    lst_features_test,
    s2_features_test,
    spatial_features_test,
    coord_features_test
])
y_train = train_data['UHI Index'].values

print(f"Initial feature set: {X_train.shape[1]} features")

# Handle any NaN or inf values
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Preprocess features using quantile transformation for robustness against outliers
print("Preprocessing features...")
# Use QuantileTransformer for more robust scaling of features
# This helps to handle different distributions in the features
scaler = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Remove low variance features
print("Removing low variance features...")
selector = VarianceThreshold(threshold=0.001)
X_train_var = selector.fit_transform(X_train_scaled)
X_test_var = selector.transform(X_test_scaled)
print(f"After variance threshold: {X_train_var.shape[1]} features")

# Dimension reduction with PCA
print("Performing PCA...")
pca = PCA(n_components=0.99)  # Keep 99% of variance
X_train_pca = pca.fit_transform(X_train_var)
X_test_pca = pca.transform(X_test_var)
print(f"After PCA: {X_train_pca.shape[1]} features")

# Add kernel features for non-linear relationships
print("Creating kernel features...")
X_train_kernel = create_kernel_features(X_train_pca[:, :20], n_components=50)  # Use top PCA components
X_test_kernel = create_kernel_features(X_test_pca[:, :20], n_components=50)

# Combine PCA features with kernel features
X_train_final = np.hstack([X_train_pca, X_train_kernel])
X_test_final = np.hstack([X_test_pca, X_test_kernel])
print(f"Final feature count: {X_train_final.shape[1]} features")

# Split data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_final, y_train, test_size=0.2, random_state=42
)

# Train ExtraTreesRegressor with optimized parameters
print("Training ExtraTreesRegressor...")
et_model = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)

et_model.fit(X_train_final, y_train)

# Train a diverse set of models for stacking
print("Training ensemble models...")
models = {}

# RandomForest with different parameters
models['rf'] = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=2,
    max_features='sqrt',
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

# GradientBoosting with careful tuning
models['gb'] = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=8,
    min_samples_split=3,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)

# XGBoost with regularization
models['xgb'] = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=7,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    gamma=0.01,
    random_state=42,
    n_jobs=-1
)

# Neural Network for complex patterns
models['mlp'] = MLPRegressor(
    hidden_layer_sizes=(200, 100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size='auto',
    learning_rate='adaptive',
    max_iter=2000,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)

# Train all models and evaluate
model_scores = {}
model_predictions_train = {}
model_predictions_test = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_final, y_train)
    
    # Get predictions
    train_pred = model.predict(X_train_final)
    model_predictions_train[name] = train_pred
    model_predictions_test[name] = model.predict(X_test_final)
    
    # Evaluate
    train_r2 = r2_score(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    model_scores[name] = train_r2
    
    print(f"  {name} - R²: {train_r2:.6f}, RMSE: {train_rmse:.6f}")

# Evaluate ExtraTrees model
et_train_pred = et_model.predict(X_train_final)
et_train_r2 = r2_score(y_train, et_train_pred)
et_train_rmse = np.sqrt(mean_squared_error(y_train, et_train_pred))
model_scores['et'] = et_train_r2
model_predictions_train['et'] = et_train_pred
model_predictions_test['et'] = et_model.predict(X_test_final)

print(f"ExtraTrees - R²: {et_train_r2:.6f}, RMSE: {et_train_rmse:.6f}, OOB Score: {et_model.oob_score_:.6f}")

# Estimate accuracy
et_accuracy = 100 * (1 - et_train_rmse/np.mean(y_train))
print(f"Estimated ExtraTrees accuracy: {et_accuracy:.4f}%")

# Find the best model
best_model_name = max(model_scores, key=model_scores.get)
best_r2 = model_scores[best_model_name]
print(f"Best model: {best_model_name} with R² = {best_r2:.6f}")

# Create stacking meta-features
meta_train = np.column_stack([model_predictions_train[name] for name in models.keys()])
meta_test = np.column_stack([model_predictions_test[name] for name in models.keys()])

# Train a meta-regressor on the stacked predictions
print("Training meta-regressor for stacking...")
meta_regressor = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.005,  # Lower learning rate for meta-model
    max_depth=5,
    subsample=0.8,
    random_state=42
)
meta_regressor.fit(meta_train, y_train)

# Generate stacked predictions
stacked_train_pred = meta_regressor.predict(meta_train)
stacked_train_r2 = r2_score(y_train, stacked_train_pred)
stacked_train_rmse = np.sqrt(mean_squared_error(y_train, stacked_train_pred))
stacked_accuracy = 100 * (1 - stacked_train_rmse/np.mean(y_train))

print(f"Stacked ensemble - R²: {stacked_train_r2:.6f}, RMSE: {stacked_train_rmse:.6f}")
print(f"Estimated stacked ensemble accuracy: {stacked_accuracy:.4f}%")

# FOR THIS VERSION: SPECIFICALLY USE EXTRATREES MODEL ONLY
print("Using ExtraTrees model as requested for final predictions")
final_test_pred = model_predictions_test['et']
final_accuracy = et_accuracy

print(f"Final estimated accuracy: {final_accuracy:.4f}%")

# Handle any potential issues with predictions
final_test_pred = np.clip(final_test_pred, 0.9, 1.1)  # Realistic range for UHI Index

# Create submission file
test_data['UHI Index'] = final_test_pred
test_data.to_csv('extreme_submission_et.csv', index=False)
print("Predictions saved to extreme_submission_et.csv")

# Save model performance metrics
model_performance = pd.DataFrame({
    'Model': list(model_scores.keys()) + ['stacked'],
    'R2_Score': list(model_scores.values()) + [stacked_train_r2],
    'Accuracy': [100 * (1 - np.sqrt(mean_squared_error(y_train, model_predictions_train[m]))/np.mean(y_train)) 
                for m in model_scores.keys()] + [stacked_accuracy]
})
model_performance.to_csv('extreme_model_performance.csv', index=False)
print("Model performance metrics saved to extreme_model_performance.csv")

# Save the ExtraTrees model for future use
joblib.dump(et_model, 'extreme_et_model.pkl')
joblib.dump(scaler, 'extreme_scaler.pkl')
joblib.dump(selector, 'extreme_variance_selector.pkl')
joblib.dump(pca, 'extreme_pca.pkl')

print("ExtraTrees model saved for future use.")
