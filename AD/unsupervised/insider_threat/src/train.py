# General Libs
import pandas as pd
import numpy as np
import os
import time

# Neural Networks Related
from keras import optimizers, regularizers
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Input
from keras.models import Model
from keras.callbacks import EarlyStopping, TensorBoard

# Sickit-learn Related
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


# Data Viz
import matplotlib.pyplot as plt

# Timesteps to build loop
lookback = 3

# Create Local Directory to save model
os.makedirs("./insider_threat/model/", exist_ok=True)


# Reads and preprocesses the dataset from CSV File
def read_data(path):
    df = pd.read_csv(path)
    df = df.drop(['duration_ts'], axis=1)
    df.fillna("missing")
    df['target'] = 0
    return df


# Creates temporal sequences for time series data
def temporalize(in_X, lb):
    X = []
    for i in range(lb, len(in_X)+1):
        X.append(in_X[i - lb:i, :])
    X = np.array(X)
    return X


# Similar to temporalize() but handles test data with labels
def temporalize_test(in_X, in_y, lb):
    X = []
    y = []
    for i in range(lb, len(in_X) + 1):
        X.append(in_X[i - lb:i, :])
        y.append(in_y[i-1])
    X = np.array(X)
    y = np.array(y)
    return X, y


# Prepares train and test datset by user
def get_train_test_data(train, test, lb):
    X_train = []
    X_train = np.array(X_train)
    X_test = []
    X_test = np.array(X_train)
    y_test = []
    y_test = np.array(y_test)

    users = list(train.Account_Name.unique())
    flag = 1
    for usr in users:
        u_train = train.loc[train['Account_Name'] == usr]
        u_test = test.loc[test['Account_Name'] == usr]
        u_y_test = u_test['target'].values
        u_test = u_test.drop(['target'], axis=1)

        # Data Normalization
        sc = MinMaxScaler(feature_range=(0, 1))
        u_train = sc.fit_transform(u_train)
        u_test = sc.transform(u_test)
        t1 = temporalize(u_train, lb) # To create a sequence
        t3, t4 = temporalize_test(u_test, u_y_test, lb)

        if flag == 1:
            X_train = t1
            X_test = t3
            y_test = t4
            flag = 0
        else:
            if t1.size != 0:
                X_train = np.concatenate([X_train, t1])
            if t3.size != 0:
                X_test = np.concatenate([X_test, t3])
            if t4.size != 0:
                y_test = np.concatenate([y_test, t4])
    return X_train, X_test, y_test


# Extracts the last timestep from each sequence
def flatten_data(X):
    lb = X.shape[1]
    flattened_X = np.empty((X.shape[0], X.shape[2]))
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, lb-1, :]
    return flattened_X


# Temporal Post-Preprocess for categorical data
def temp_postprep(df):
    # Extract Categorical Data Only
    cat_data = df[['Account_Name', 'title', 'department', 'company']]

    # LabelEncoder
    le = LabelEncoder()
    for col in cat_data:
        encoded_data = le.fit_transform(df[col])
        df[col] = encoded_data
    return df


# Function to build LSTM model
def train_model(X_train, X_test):

    # Data Prep
    n_features = X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], lookback, n_features)
    X_test = X_test.reshape(X_test.shape[0], lookback, n_features)

    epochs = 100
    batch_size = 256
    lr = 0.0001
    timesteps = lookback
    encoding_dimensions = 16
    hidden_dimensions = 8

    # Add Layers for LSTM
    input_layer = Input(shape = (timesteps, n_features))
    L1 = LSTM(encoding_dimensions, activation='relu', return_sequences=True,
              kernel_regularizer=regularizers.l2(0.00))(input_layer)
    L2 = LSTM(hidden_dimensions, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(timesteps)(L2)
    L4 = LSTM(hidden_dimensions, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(encoding_dimensions, activation='relu', return_sequences=True)(L4)
    output_layer = TimeDistributed(Dense(n_features))(L5) 
    lstm_model = Model(input_layer, output_layer)

    # Compile, Optimize, Fit
    adam = optimizers.Adam(lr)
    lstm_model.compile(loss='mse', optimizer=adam)

    # Keep record
    # Neee
    # cp = ModelCheckpoint(filepath='model/rnn.h5', save_best_only=True, verbose=0)
    tb = TensorBoard(log_dir='model/logs', histogram_freq=0, write_graph=True, write_images=True)

    # Early Stopping
    es = EarlyStopping(monitor='val_loss', min_delta=.0001, patience=2)

    # History to check general model performance
    history=lstm_model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, 
                            verbose=1, shuffle=False, validation_split=.12,
                            callbacks=[es, tb]).history

    plt.plot(history['loss'], linewidth=2, label='Train')
    plt.plot(history['val_loss'], linewidth=2, label='Validation')
    plt.legend(loc='upper right')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('insider_threat/model/train_model_loss.png')
    return lstm_model


def predict_model(lstm_model, X_test, y_test):
    preds = lstm_model.predict(X_test)
    mse = np.mean(np.power(flatten_data(X_test) - flatten_data(preds), 2), axis=1)
    error_df = pd.DataFrame({'Reconstruction_error': mse,
                            'True_class': y_test})
    threshold = .02  # Reassign it at some point

    # Calculate Reconstruction Error
    pred_y = [1 if e > threshold else 0 for e in error_df['Reconstruction_error'].values]

    # 1. Confusion Matrix
    conf_matrix = confusion_matrix(error_df['True_class'], pred_y)
    print("Confusion Matrix\n", conf_matrix)

    # 2. ROC Curve
    false_pos_rate, true_pos_rate, threshold = roc_curve(error_df.True_class, error_df.Reconstruction_error)
    roc_auc = auc(false_pos_rate, true_pos_rate, )

    plt.plot(false_pos_rate, true_pos_rate, linewidth=3, label='AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], linewidth=3)

    plt.xlim([-0.01, 1])
    plt.ylim([0, 1.01])
    plt.legend(loc='lower right')
    plt.title('Receiver operating characteristic curve (ROC)')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('insider_threat/model/ROC_curve.png')

    # 3. TPR, TNR Curve
    threshold = 0.02  # starting point
    increment = 0.001 # increased by 1%
    stopping = 0.06   # ending point
    th, prec, rec, accuracy, f1, TPR, FPR, TNR = [], [], [], [], [], [], [], []

    error_df = pd.DataFrame({'Reconstruction_error': mse,
                            'True_class': y_test})

    while threshold < stopping:
        pred_y = [1 if e > threshold else 0 for e in error_df['Reconstruction_error'].values]
        conf_matrix = confusion_matrix(error_df['True_class'], pred_y)
        TP = conf_matrix[1][1]
        FP = conf_matrix[0][1]
        TN = conf_matrix[0][0]
        FN = conf_matrix[1][0]
        th.append(threshold)
        p = TP/(TP+FP)
        r = TP/(TP+FN)
        prec.append(p)
        rec.append(r)
        accuracy.append((TN+TP)/(TN+TP+FN+FP))
        TPR.append(TP/(TP+FN))
        FPR.append(FP/(TN+FP))
        TNR.append(TN/(TN+FP))
        f1.append(2*p*r/(p+r))
        threshold = threshold + increment

    plt.plot(th, TPR, linewidth=2, label='TPR', color ='r')
    plt.plot(th, TNR, linewidth=2, label='TNR', color='b')
    plt.title('TPR, TNR Curve')
    plt.ylabel('TPR/TNR')
    plt.xlabel('Threshold')
    plt.legend(loc='upper right')
    plt.savefig('insider_threat/model/TPRxTNR.png')

def main():

    df = read_data("insider_threat/data/train_data.csv")

    encoded_df = temp_postprep(df)
    train_data, test_data = train_test_split(encoded_df, test_size=.3, random_state=55)

    # drop target from train_data
    train_data = train_data.drop('target', axis=1)

    X_train, X_test, y_test = get_train_test_data(train_data, test_data, lookback)

    X_train = X_train[:, :, 1:]
    X_test = X_test[:, :, 1:]

    print("Model Building begins.....")
    s = time.time()
    model = train_model(X_train, X_test)
    print("Done: ", s-time.time())

    print("Predict and Generating Visualization starts...")
    predict_model(model, X_test, y_test)
    print("Plotting Done...")
    
main()