import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
import xgboost as xgb
from scipy.ndimage import gaussian_filter, sobel, median_filter
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

try:
    import fiona
    from shapely.geometry import shape, Point
    HAS_FIONA = True
except ImportError:
    print("Warning: fiona or shapely not installed. Building footprint features will not be used.")
    HAS_FIONA = False

from math import radians, cos, sin, asin, sqrt

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

print("Loading datasets...")
# Load training data
train_data = pd.read_csv('Training_data_uhi_index_2025-02-18.csv')
train_data = train_data.drop('datetime', axis=1)  # Drop datetime as requested

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

def extract_multi_scale_features(tiff_path, coords, window_sizes=[3, 7, 15]):
    """
    Extract features at multiple scales to capture different spatial patterns
    """
    with rasterio.open(tiff_path) as src:
        all_features = []
        
        for lon, lat in coords:
            try:
                # Convert geographic coordinates to pixel coordinates
                py, px = src.index(lon, lat)
                
                point_features = []
                # Extract features at different window sizes
                for size in window_sizes:
                    padding = size // 2
                    window = Window(
                        px - padding, py - padding,
                        size, size
                    )
                    
                    try:
                        # Read the window
                        pixel_array = src.read(1, window=window)
                        # Handle potential NaN values
                        pixel_array = np.nan_to_num(pixel_array)
                        
                        # Apply multiple filters
                        smoothed = gaussian_filter(pixel_array, sigma=1)
                        median = median_filter(pixel_array, size=3)
                        
                        # Calculate gradients
                        gradient_x = sobel(smoothed, axis=0)
                        gradient_y = sobel(smoothed, axis=1)
                        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                        
                        # Basic statistical features
                        features = [
                            np.mean(pixel_array),            # Mean
                            np.std(pixel_array),             # Standard deviation
                            np.median(pixel_array),          # Median
                            np.min(pixel_array),             # Minimum
                            np.max(pixel_array),             # Maximum
                            np.percentile(pixel_array, 25),  # 1st quartile
                            np.percentile(pixel_array, 75),  # 3rd quartile
                            
                            # Smoothed features
                            np.mean(smoothed),               # Smoothed mean
                            np.std(smoothed),                # Smoothed std
                            
                            # Median filter features
                            np.mean(median),                 # Median filter mean
                            np.std(median),                  # Median filter std
                            
                            # Gradient features
                            np.mean(gradient_magnitude),     # Mean gradient magnitude
                            np.std(gradient_magnitude),      # Std of gradient
                            np.max(gradient_magnitude),      # Max gradient
                            
                            # Texture features
                            skew(pixel_array.flatten()),     # Skewness
                            kurtosis(pixel_array.flatten()), # Kurtosis
                            
                            # Edge features
                            np.sum(gradient_magnitude) / (size * size),  # Edge density
                            
                            # Range features
                            np.ptp(pixel_array),             # Peak-to-peak
                            
                            # Histogram-based features
                            np.histogram(pixel_array, bins=5)[0][0] / pixel_array.size,  # Histogram bin 1
                            np.histogram(pixel_array, bins=5)[0][1] / pixel_array.size,  # Histogram bin 2
                            np.histogram(pixel_array, bins=5)[0][2] / pixel_array.size,  # Histogram bin 3
                            np.histogram(pixel_array, bins=5)[0][3] / pixel_array.size,  # Histogram bin 4
                            np.histogram(pixel_array, bins=5)[0][4] / pixel_array.size,  # Histogram bin 5
                        ]
                        
                        point_features.extend(features)
                        
                    except Exception:
                        # If window reading fails, add zeros
                        point_features.extend([0] * 23)
                
                all_features.append(point_features)
                
            except Exception:
                # If coordinate conversion fails, add zeros for all window sizes
                all_features.append([0] * (23 * len(window_sizes)))
                
        return np.array(all_features)

def create_enhanced_spatial_features(coords, footprints=None):
    """
    Create advanced spatial features including building proximity if available
    """
    # NYC reference points
    landmarks = {
        'central_park': (-73.9654, 40.7829),
        'times_square': (-73.9855, 40.7580),
        'downtown': (-74.0060, 40.7128),
        'east_river': (-73.9762, 40.7678),
        'hudson_river': (-74.0099, 40.7258),
    }
    
    features = []
    for lon, lat in coords:
        # Base spatial features
        center_lon, center_lat = -73.95, 40.80  # Approximate center of the area
        dist = np.sqrt((lon - center_lon)**2 + (lat - center_lat)**2)
        angle = np.arctan2(lat - center_lat, lon - center_lon)
        
        # Distance to key landmarks (using Haversine distance)
        landmark_dists = [
            haversine(lon, lat, landmark[0], landmark[1]) 
            for landmark in landmarks.values()
        ]
        
        # Building proximity features
        building_features = [0, 0, 0]  # Default placeholders
        if footprints and HAS_FIONA:
            try:
                point = Point(lon, lat)
                # Distance to nearest building
                min_dist = float('inf')
                buildings_within_100m = 0
                building_area_within_200m = 0
                
                # Check only a subset of buildings for performance
                for building in footprints[:100]:  # Limit to first 100 for performance
                    try:
                        d = point.distance(building)
                        min_dist = min(min_dist, d)
                        
                        # Count buildings within 100m
                        if d < 0.001:  # approx 100m
                            buildings_within_100m += 1
                        
                        # Sum area of buildings within 200m
                        if d < 0.002:  # approx 200m
                            building_area_within_200m += building.area
                    except:
                        continue
                
                if min_dist == float('inf'):
                    min_dist = 0.01  # Default if no buildings found
                    
                building_features = [
                    min_dist * 100,  # Convert to approximate meters
                    buildings_within_100m,
                    building_area_within_200m * 10000  # Convert to approx sq meters
                ]
            except Exception as e:
                print(f"Error processing building features: {e}")
        
        # Combine all spatial features
        point_features = [
            dist,                          # Distance from center
            angle,                         # Angle
            np.sin(angle),                 # Sine of angle
            np.cos(angle),                 # Cosine of angle
            lon * lat,                     # Interaction
            dist * np.sin(angle),          # Polar transform
            dist * np.cos(angle),          # Polar transform
            lon**2,                        # Quadratic terms
            lat**2,
            np.sin(lon * 20),              # Oscillatory features
            np.cos(lat * 20),
        ]
        
        # Add landmark distances
        point_features.extend(landmark_dists)
        
        # Add building features
        point_features.extend(building_features)
        
        features.append(point_features)
        
    return np.array(features)

# Extract multi-scale features from both satellite images
print("Extracting multi-scale features from Landsat LST...")
lst_features_train = extract_multi_scale_features(
    'Landsat_LST.tiff', 
    train_data[['Longitude', 'Latitude']].values
)
lst_features_test = extract_multi_scale_features(
    'Landsat_LST.tiff', 
    test_data[['Longitude', 'Latitude']].values
)

print("Extracting multi-scale features from Sentinel-2...")
s2_features_train = extract_multi_scale_features(
    'S2_sample.tiff', 
    train_data[['Longitude', 'Latitude']].values
)
s2_features_test = extract_multi_scale_features(
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
    coord_features_train  # Add raw coordinates
])
X_test = np.hstack([
    lst_features_test,
    s2_features_test,
    spatial_features_test,
    coord_features_test  # Add raw coordinates
])
y_train = train_data['UHI Index'].values

print(f"Total number of features: {X_train.shape[1]}")

# Handle any NaN values
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

# Preprocess features
print("Preprocessing features...")
# Use robust scaler to handle outliers better
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add polynomial interactions for key features
print("Adding polynomial features...")
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print(f"Number of features after polynomial expansion: {X_train_poly.shape[1]}")

# Dimension reduction with PCA to avoid overfitting
print("Performing PCA...")
pca = PCA(n_components=0.99)  # Keep 99% of variance
X_train_pca = pca.fit_transform(X_train_poly)
X_test_pca = pca.transform(X_test_poly)
print(f"Reduced dimensions from {X_train_poly.shape[1]} to {X_train_pca.shape[1]} using PCA")

# Feature selection using XGBoost importance
print("Performing feature selection...")
xgb_selector = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
selector = SelectFromModel(xgb_selector, threshold='median')
X_train_selected = selector.fit_transform(X_train_pca, y_train)
X_test_selected = selector.transform(X_test_pca)
print(f"Selected {X_train_selected.shape[1]} features")

# Split data for cross-validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_selected, y_train, test_size=0.2, random_state=42
)

# Define base models
print("Building models...")
base_models = [
    ('rf', RandomForestRegressor(
        n_estimators=1000,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )),
    ('gb', GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=8,
        min_samples_split=3,
        min_samples_leaf=1,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )),
    ('xgb', xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        random_state=42,
        n_jobs=-1
    )),
    ('svr', SVR(
        C=10,
        gamma='scale',
        epsilon=0.01
    )),
    ('ada', AdaBoostRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(200, 100, 50),  # Deeper network
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=1000,  # Increased max iterations
        random_state=42
    ))
]

# Train separate models first
print("Training individual models...")
models = {}
model_scores = {}
for name, model in base_models:
    print(f"Training {name}...")
    model.fit(X_train_selected, y_train)
    models[name] = model
    
    # Evaluate each model
    train_pred = model.predict(X_train_selected)
    train_r2 = r2_score(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    model_scores[name] = train_r2
    print(f"  {name} - R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")

# Only include models with positive R² scores in the ensemble
good_models = [(name, model) for name, model in models.items() if model_scores[name] > 0.5]
print(f"Using only well-performing models: {[name for name, _ in good_models]}")

# Create a voting ensemble with specific weights based on individual performance
print("Building voting ensemble...")
if len(good_models) >= 2:
    # Create weights proportional to R² scores
    weights = [max(0.1, model_scores[name]) for name, _ in good_models]
    # Normalize weights
    weights = [w/sum(weights) for w in weights]
    
    voting_regressor = VotingRegressor(good_models, weights=weights)
    
    # Train the voting ensemble
    print("Training ensemble model...")
    voting_regressor.fit(X_train_selected, y_train)
    
    # Evaluate on training data
    train_pred = voting_regressor.predict(X_train_selected)
    train_r2 = r2_score(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_accuracy = 100 * (1 - train_rmse/np.mean(y_train))
    
    print(f"Final model - Training R² Score: {train_r2:.4f}")
    print(f"Final model - Training RMSE: {train_rmse:.4f}")
    print(f"Final model - Training Accuracy: {train_accuracy:.2f}%")
    
    # Perform cross-validation to check for overfitting
    print("Performing cross-validation...")
    cv_scores = cross_val_score(voting_regressor, X_train_selected, y_train, 
                               cv=5, scoring='r2')
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R² score: {np.mean(cv_scores):.4f}")
    
    # Make predictions on test data
    print("Generating predictions...")
    test_predictions = voting_regressor.predict(X_test_selected)
    
    # Creating final prediction based on only the well-performing models
    print("Creating ensemble blend...")
    final_predictions = test_predictions
    
    # If both RF and GB performed well, use them for the final blend
    if 'rf' in dict(good_models) and 'gb' in dict(good_models):
        rf_pred = models['rf'].predict(X_test_selected)
        gb_pred = models['gb'].predict(X_test_selected)
        # Blend with weights based on training R²
        rf_weight = max(0.2, model_scores['rf'] / (model_scores['rf'] + model_scores['gb']))
        gb_weight = 1 - rf_weight
        print(f"Using RF (weight={rf_weight:.2f}) and GB (weight={gb_weight:.2f}) for final blend")
        final_predictions = rf_weight * rf_pred + gb_weight * gb_pred
else:
    # If no good ensemble can be formed, use the best single model
    best_model_name = max(model_scores, key=model_scores.get)
    print(f"Using best single model: {best_model_name} (R²: {model_scores[best_model_name]:.4f})")
    final_predictions = models[best_model_name].predict(X_test_selected)

# Create submission file
test_data['UHI Index'] = final_predictions
test_data.to_csv('submission_0315.csv', index=False)
print("Predictions saved to submission_0315.csv")

# Save the model performance metrics for reference
model_performance = pd.DataFrame({
    'Model': list(model_scores.keys()),
    'R2_Score': list(model_scores.values())
})
model_performance.to_csv('model_performance_0315.csv', index=False)
print("Model performance metrics saved to model_performance_0315.csv")
