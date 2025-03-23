import pandas as pd
import numpy as np
import pickle
from new_model import engineer_features  # Import from your model file

def make_predictions():
    try:
        # Load the saved model pipeline and components
        print("Loading model pipeline and components...")
        with open('uhi_model.pkl', 'rb') as f:
            model_pipeline = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
            
        # Load test data
        print("Loading test data...")
        test_df = pd.read_csv('Submission_template_UHI2025-v2.csv')
        test_features = pd.read_pickle('submit_features.pkl')
        
        # Engineer features for test data
        print("Engineering features for test data...")
        engineered_test_features = engineer_features(test_features)
        
        if engineered_test_features is None:
            raise ValueError("Feature engineering failed for test data")
        
        # Select the same features used in training
        print("Preparing features for prediction...")
        X_test = engineered_test_features[feature_columns]
        
        # Make predictions using the pipeline
        print("Making predictions...")
        predictions = model_pipeline.predict(X_test)
        
        # Ensure predictions are within reasonable bounds
        print("Validating predictions...")
        predictions = np.clip(predictions, 0, 2)  # Adjust min/max based on your domain knowledge
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'Latitude': test_df['Latitude'],
            'Longitude': test_df['Longitude'],
            'UHI Index': predictions
        })
        
        # Add some basic validation checks
        print("\nValidation Checks:")
        print(f"Number of predictions: {len(predictions)}")
        print(f"Number of null values: {submission_df['UHI Index'].isnull().sum()}")
        print("\nPrediction Statistics:")
        print(f"Mean UHI Index: {predictions.mean():.4f}")
        print(f"Median UHI Index: {np.median(predictions):.4f}")
        print(f"Std UHI Index: {predictions.std():.4f}")
        print(f"Min UHI Index: {predictions.min():.4f}")
        print(f"Max UHI Index: {predictions.max():.4f}")
        
        # Check for anomalies
        z_scores = np.abs((predictions - predictions.mean()) / predictions.std())
        anomalies = z_scores > 3
        if np.any(anomalies):
            print(f"\nWarning: Found {np.sum(anomalies)} potential anomalies in predictions")
            print("Anomaly locations:")
            anomaly_df = submission_df[anomalies][['Longitude', 'Latitude', 'UHI Index']]
            print(anomaly_df)
        
        # Save predictions
        output_file = 'new_preds_v1.csv'
        submission_df.to_csv(output_file, index=False)
        print(f"\nPredictions saved to {output_file}")
        
        # Create a simple visualization of predictions if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.hist(predictions, bins=50, edgecolor='black')
            plt.title('Distribution of UHI Index Predictions')
            plt.xlabel('UHI Index')
            plt.ylabel('Frequency')
            plt.savefig('prediction_distribution.png')
            plt.close()
            print("Prediction distribution plot saved as 'prediction_distribution.png'")
            
            # Create a scatter plot of predictions vs location
            plt.figure(figsize=(12, 8))
            plt.scatter(submission_df['Longitude'], submission_df['Latitude'], 
                       c=submission_df['UHI Index'], cmap='hot', s=50)
            plt.colorbar(label='UHI Index')
            plt.title('UHI Index by Location')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.savefig('prediction_map.png')
            plt.close()
            print("Prediction map saved as 'prediction_map.png'")
            
        except ImportError:
            print("Matplotlib not available - skipping visualizations")
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(f"Error occurred at line: {e.__traceback__.tb_lineno}")
        raise

if __name__ == "__main__":
    make_predictions()