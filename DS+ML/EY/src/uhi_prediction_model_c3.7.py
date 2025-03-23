import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from scipy.ndimage import gaussian_filter, sobel
import warnings
warnings.filterwarnings('ignore')

print("Loading datasets...")
# Load training data
train_data = pd.read_csv('Training_data_uhi_index_2025-02-18.csv')
train_data = train_data.drop('datetime', axis=1)  # Drop datetime as requested

# Load test data
test_data = pd.read_csv('Submission_template_UHI2025-v2.csv')

def extract_image_features(tiff_path, coords, window_size=7):
    """
    Extract advanced features from satellite imagery at given coordinates
    """
    with rasterio.open(tiff_path) as src:
        features_list = []
        
        for lon, lat in coords:
            try:
                # Convert geographic coordinates to pixel coordinates
                py, px = src.index(lon, lat)
                
                # Extract a window around the point
                padding = window_size // 2
                window = Window(
                    px - padding, py - padding,
                    window_size, window_size
                )
                
                # Read the window
                pixel_array = src.read(1, window=window)
                
                # Handle potential NaN values
                pixel_array = np.nan_to_num(pixel_array)
                
                # Apply Gaussian smoothing for noise reduction
                smoothed = gaussian_filter(pixel_array, sigma=1)
                
                # Calculate gradient (edge detection)
                gradient_x = sobel(smoothed, axis=0)
                gradient_y = sobel(smoothed, axis=1)
                gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                
                # Extract statistical features
                features = [
                    np.mean(pixel_array),               # Mean
                    np.std(pixel_array),                # Standard deviation
                    np.median(pixel_array),             # Median
                    np.min(pixel_array),                # Minimum
                    np.max(pixel_array),                # Maximum
                    np.percentile(pixel_array, 25),     # 1st quartile
                    np.percentile(pixel_array, 75),     # 3rd quartile
                    np.percentile(pixel_array, 90),     # 90th percentile
                    np.percentile(pixel_array, 10),     # 10th percentile
                    np.mean(smoothed),                  # Smoothed mean
                    np.std(smoothed),                   # Smoothed std
                    np.mean(gradient_magnitude),        # Mean gradient magnitude
                    np.std(gradient_magnitude),         # Std of gradient magnitude
                    np.max(gradient_magnitude),         # Max gradient magnitude
                    np.sum(gradient_x),                 # Sum of x gradients
                    np.sum(gradient_y),                 # Sum of y gradients
                    # Texture features
                    np.mean(np.abs(pixel_array - np.mean(pixel_array))),  # Mean absolute deviation
                    np.var(pixel_array),                # Variance
                    np.sum(np.abs(np.diff(pixel_array.flatten()))),  # Total variation
                ]
                
                features_list.append(features)
                
            except Exception as e:
                # If any error occurs, fill with zeros
                features_list.append([0] * 19)
                
        return np.array(features_list)

def create_spatial_features(coords):
    """
    Create spatial features from geographic coordinates
    """
    features = []
    # Approximate center of the area (Manhattan/NYC)
    center_lon, center_lat = -73.95, 40.80
    
    for lon, lat in coords:
        # Distance from center
        dist = np.sqrt((lon - center_lon)**2 + (lat - center_lat)**2)
        
        # Angular position (in radians)
        angle = np.arctan2(lat - center_lat, lon - center_lon)
        
        # Create features
        features.append([
            dist,                          # Distance from center
            angle,                         # Angle
            np.sin(angle),                 # Sine of angle
            np.cos(angle),                 # Cosine of angle
            lon * lat,                     # Interaction term
            dist * np.sin(angle),          # Polar coordinate transform
            dist * np.cos(angle),          # Polar coordinate transform
            lon**2,                        # Quadratic terms
            lat**2,
            lon**3,                        # Cubic terms
            lat**3,
            np.sin(lon * 10),              # Periodic features
            np.cos(lat * 10),
        ])
        
    return np.array(features)

# Extract features from both satellite images
print("Extracting features from Landsat LST...")
lst_features_train = extract_image_features('Landsat_LST.tiff', train_data[['Longitude', 'Latitude']].values)
lst_features_test = extract_image_features('Landsat_LST.tiff', test_data[['Longitude', 'Latitude']].values)

print("Extracting features from Sentinel-2...")
s2_features_train = extract_image_features('S2_sample.tiff', train_data[['Longitude', 'Latitude']].values)
s2_features_test = extract_image_features('S2_sample.tiff', test_data[['Longitude', 'Latitude']].values)

print("Creating spatial features...")
spatial_features_train = create_spatial_features(train_data[['Longitude', 'Latitude']].values)
spatial_features_test = create_spatial_features(test_data[['Longitude', 'Latitude']].values)

# Combine all features
X_train = np.hstack([
    lst_features_train,
    s2_features_train,
    spatial_features_train
])
X_test = np.hstack([
    lst_features_test,
    s2_features_test,
    spatial_features_test
])
y_train = train_data['UHI Index'].values

# Scale features
print("Preprocessing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add polynomial features for interactions between features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Feature selection to reduce dimensionality
print("Performing feature selection...")
# Use XGBoost for feature selection
selector_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
selector = SelectFromModel(selector_model, threshold="median")
X_train_selected = selector.fit_transform(X_train_poly, y_train)
X_test_selected = selector.transform(X_test_poly)

print(f"Features reduced from {X_train_poly.shape[1]} to {X_train_selected.shape[1]}")

# Split data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_selected, y_train, test_size=0.2, random_state=42
)

# Define base models with optimized hyperparameters
base_models = [
    ('rf', RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )),
    ('gb', GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )),
    ('xgb', xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        random_state=42,
        n_jobs=-1
    ))
]

# Meta-model for stacking
meta_model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Create stacking ensemble
print("Training stacking ensemble model...")
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

# Train the model
stacking_model.fit(X_train_selected, y_train)

# Evaluate on training data
train_pred = stacking_model.predict(X_train_selected)
train_r2 = r2_score(y_train, train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
train_accuracy = 100 * (1 - train_rmse/np.mean(y_train))

print(f"Training R² Score: {train_r2:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Training Accuracy: {train_accuracy:.2f}%")

# Make predictions on test data
test_predictions = stacking_model.predict(X_test_selected)

# Ensure predictions are reasonable values for UHI index (typically between 0 and 2)
test_predictions = np.clip(test_predictions, 0.8, 1.5)

# Create submission file
test_data['UHI Index'] = test_predictions
test_data.to_csv('submission_0314_v3.csv', index=False)
print("Predictions saved to submission_0314_v3.csv")
