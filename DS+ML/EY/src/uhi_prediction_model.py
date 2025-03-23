import pandas as pd
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# Load training data
train_data = pd.read_csv('Training_data_uhi_index_2025-02-18.csv')
train_data = train_data.drop('datetime', axis=1)  # Drop datetime as requested

# Load test data
test_data = pd.read_csv('Submission_template_UHI2025-v2.csv')

# Function to extract advanced pixel values from GeoTIFF
def extract_advanced_pixel_values(tiff_path, coords, window_size=5):
    with rasterio.open(tiff_path) as src:
        pixel_values = []
        for lon, lat in coords:
            try:
                # Get pixel coordinates from geographic coordinates
                py, px = src.index(lon, lat)
                # Read a larger window for more context
                padding = window_size // 2
                window = rasterio.windows.Window(
                    px - padding, py - padding,
                    window_size, window_size
                )
                pixel_array = src.read(1, window=window)
                
                # Apply Gaussian smoothing
                smoothed = gaussian_filter(pixel_array, sigma=1)
                
                features = [
                    np.mean(pixel_array),     # Mean value
                    np.std(pixel_array),      # Standard deviation
                    np.max(pixel_array),      # Maximum value
                    np.min(pixel_array),      # Minimum value
                    np.median(pixel_array),   # Median value
                    np.percentile(pixel_array, 25),  # 25th percentile
                    np.percentile(pixel_array, 75),  # 75th percentile
                    np.mean(smoothed),        # Smoothed mean
                    np.std(smoothed),         # Smoothed std
                    np.sum(np.gradient(pixel_array)[0]**2),  # Gradient magnitude X
                    np.sum(np.gradient(pixel_array)[1]**2),  # Gradient magnitude Y
                ]
                pixel_values.append(features)
            except:
                pixel_values.append([0] * 11)  # Handle edge cases
    return np.array(pixel_values)

# Create spatial features
def create_spatial_features(coords):
    features = []
    for lon, lat in coords:
        # Calculate distance from center point
        center_lon, center_lat = -73.95, 40.80  # Approximate center of the area
        dist = np.sqrt((lon - center_lon)**2 + (lat - center_lat)**2)
        
        # Calculate angular position
        angle = np.arctan2(lat - center_lat, lon - center_lon)
        
        # Add features
        features.append([
            dist,
            angle,
            np.sin(angle),
            np.cos(angle),
            lon * lat,  # Interaction
            lon**2,     # Quadratic
            lat**2
        ])
    return np.array(features)

print("Extracting advanced features from Landsat LST...")
train_lst_features = extract_advanced_pixel_values('Landsat_LST.tiff', train_data[['Longitude', 'Latitude']].values)
test_lst_features = extract_advanced_pixel_values('Landsat_LST.tiff', test_data[['Longitude', 'Latitude']].values)

print("Extracting advanced features from Sentinel-2...")
train_s2_features = extract_advanced_pixel_values('S2_sample.tiff', train_data[['Longitude', 'Latitude']].values)
test_s2_features = extract_advanced_pixel_values('S2_sample.tiff', test_data[['Longitude', 'Latitude']].values)

# Create spatial features
print("Creating spatial features...")
train_spatial = create_spatial_features(train_data[['Longitude', 'Latitude']].values)
test_spatial = create_spatial_features(test_data[['Longitude', 'Latitude']].values)

# Combine all features
X_train = np.hstack([
    train_data[['Longitude', 'Latitude']].values,
    train_lst_features,
    train_s2_features,
    train_spatial
])
X_test = np.hstack([
    test_data[['Longitude', 'Latitude']].values,
    test_lst_features,
    test_s2_features,
    test_spatial
])
y_train = train_data['UHI Index'].values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Split training data for validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_poly, y_train, test_size=0.2, random_state=42
)

# Define optimized base models
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

gb = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

xgb = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Define base models
base_models = [
    ('rf', rf),
    ('gb', gb),
    ('xgb', xgb)
]

# Define meta-model
meta_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Create stacking regressor
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

# Train the model
print("Training enhanced stacking model...")
stacking_model.fit(X_train_poly, y_train)

# Make predictions
train_pred = stacking_model.predict(X_train_poly)
train_r2 = r2_score(y_train, train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
print(f"Training R² Score: {train_r2:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")

# Generate predictions for test set
test_predictions = stacking_model.predict(X_test_poly)

# Create submission file
test_data['UHI Index'] = test_predictions
test_data.to_csv('submission_0314_v2.csv', index=False)
print("Predictions saved to submission_0314_v2.csv")
