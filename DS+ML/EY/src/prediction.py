import pandas as pd
import pickle

def make_predictions():
    try:
        # Load the saved model and components
        print("Loading model and components...")
        
        with open('uhi_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('poly_features.pkl', 'rb') as f:
            poly = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        # Load test data
        print("Loading test data...")
        test_df = pd.read_csv('Submission_template_UHI2025-v2.csv')
        test_features = pd.read_pickle('submit_features.pkl')
        
        # Engineer features for test data
        print("Engineering features for test data...")
        from improved_model import engineer_features
        engineered_test_features = engineer_features(test_features)
        
        if engineered_test_features is None:
            raise ValueError("Feature engineering failed for test data")
        
        # Select the same features used in training
        X_test = engineered_test_features[feature_columns]
        
        # Scale and create polynomial features
        X_test_scaled = scaler.transform(X_test)
        X_test_poly = poly.transform(X_test_scaled)
        
        # Make predictions
        print("Making predictions...")
        predictions = model.predict(X_test_poly)
        
        # Create submission dataframe
        submission_df = pd.DataFrame({
            'Longitude': test_df['Longitude'],
            'Latitude': test_df['Latitude'],
            'UHI Index': predictions
        })
        
        # Save predictions
        output_file = 'uhi_predictions_v12.csv'
        submission_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        # Print some statistics about the predictions
        print("\nPrediction Statistics:")
        print(f"Mean UHI Index: {predictions.mean():.4f}")
        print(f"Min UHI Index: {predictions.min():.4f}")
        print(f"Max UHI Index: {predictions.max():.4f}")
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(f"Error occurred at line: {e.__traceback__.tb_lineno}")
        raise

if __name__ == "__main__":
    make_predictions()