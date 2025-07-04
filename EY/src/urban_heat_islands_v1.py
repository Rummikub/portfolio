import pandas as pd
import numpy as np
# Predict by using K-means clustering
# Pycaret
from pycaret.clustering import *

def load_data(filepath):
    raw_data = pd.read_csv(filepath)
    data = pd.DataFrame(raw_data)
    print("How many data:",len(data))
    return data

def model(data):
    exp = CluesteringExperiment()
    exp.setup(data=data[['Longitude','Latitude']], normalize=True, silent=True, session_id=123)
    kmeans = create_model('kmeans')
    kmeans_cluster = assign_model(kmeans)
    plot_model(kmeans_cluster, plot ='elbow') #kmeans
    return kmeans
    
def evaluate(model, val_data):
    evaluate_model(model)   
    kmeans_pred = predict_model(model, data=val_data)
    save_model(model, 'UHI_model_v1')
    save_experiment('exp_1')
    
def main():
    #train_data = load_data('Training_data_uhi_index_UHI2025-v2.csv')
    #kmeans = model(train_data)
    fm(pd.DataFrame(pd.read_csv('Submission_template_UHI2025-v2.csv')))

def fm(df):
    # Create feature dataset for clustering
    features = df[['Longitude', 'Latitude']]

    # Initialize PyCaret setup
    cluster_setup = setup(data=features, 
                        normalize=True,
                        silent=True, 
                        html=False)

    # Create and train KMeans model
    kmeans = create_model('kmeans', num_clusters=5)

    # Get cluster labels
    clusters = predict_model(kmeans)

    # Scale clusters to UHI Index range (typically 0-10 for urban heat islands)
    min_max_scaler = lambda x: ((x - x.min()) / (x.max() - x.min())) * 10
    df['UHI Index'] = min_max_scaler(clusters['Cluster'])

    # Round to 2 decimal places
    df['UHI Index'] = df['UHI Index'].round(2)

    # Save the results
    df.to_csv('UHI_submission.csv', index=False)

    print("Distribution of UHI Index values:")
    print(df['UHI Index'].describe())

if __name__ == '__main__':
    main()