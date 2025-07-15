import requests
import os
import boto3
from botocore.exceptions import ClientError
import datetime as dt
import pytz
import pickle
import MLDB as ml
import logging
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

# S3 Credentials
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
region_name = os.environ.get("AWS_DEFAULT_REGION")
bucket_name = os.environ.get("AWS_S3_BUCKET")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    endpoint_url=endpoint_url,
    region_name=region_name,
)


def read_pickle(pickle_name, s3):
    timestamp = dt.datetime.now(pytz.timezone("America/New_York")).strftime("%m%d%y_%H")
    file_key = f"pickles/{timestamp}/{pickle_name}.pickle"
    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        pickle_data = response["Body"].read()
        data = pickle.loads(pickle_data)
        return data
    except ClientError as e:
        logging.error(e)
        # return empty dictionary
        return dict()


def get_internal_url(user):
    """
    Get internal URL for model inference.
    """
    deployed_model_name = f"u{user}" if str(user[0]).isdigit() else user
    rest_url = "http://modelmesh-serving.your_model_url:8008"
    return f"{rest_url}/v2/models/{deployed_model_name}/infer"


def drop_columns(data):
    """
    Drop unnecessary columns from the data.
    """
    filtered_data = {}
    for user, df in data.items():
        filtered_data[user] = df.drop(
            ["Time", "Date_Time", "Date", "Account_Name", "ComputerName", "day"],
            axis=1,
        )
    return filtered_data


def predict(user, df, infer_url):
    """
    Generate predictions for each row of the dataframe.
    """
    pred_list = []

    # Get prediction one row at a time
    for data in df.values.tolist():
        # Convert Non-float values to float
        data = [float(x) if not isinstance(x, bool) else int(x) for x in data]

        json_data = {
            "inputs": [
                {
                    "name": "X",
                    "shape": [1, df.values.shape[1]],
                    "datatype": "FP32",
                    "data": data,
                }
            ]
        }

        response = requests.post(infer_url, json=json_data)
        response_dict = response.json()

        # Append prediction for each row
        pred_list.append(response_dict["outputs"][0]["data"][0])

    return pred_list


def add_columns(data, preds):
    """
    Add predictions to the original data.
    """
    for user, df in data.items():
        for pred_user, pred in preds.items():
            if user == pred_user:
                df["prediction"] = pred
    return data


def extract_filter_data(data):
    """
    Extract and filter abnormal data.
    """
    abnorm_dct = {}
    for user, df in data.items():
        # Extract anomalies data only
        abnorm_df = df.loc[df["prediction"] == 1]
        # Rename columns
        abnorm_df.rename(
            columns={"EncodedDay": "DayFreq", "EncodedComputerName": "CompFreq"},
            inplace=True,
        )
        # Rearrange column orders
        columns = list(abnorm_df.columns)
        abnorm_df = abnorm_df[columns[-1:] + columns[0:-1]]

        # No anomalies check
        if not abnorm_df.empty:
            abnorm_dct[user] = abnorm_df

    for user, df in abnorm_dct.items():
        print(f"Does {user} has benign data? {(df['prediction'] == 0).all()} ")

    return abnorm_dct


def analyze_cases(data, status):
    """
    Check data entries against abnormal scenarios defined in syndatagen.py
    """
    results = []  # To store all result DataFrames

    for user, df in data.items():
        # Rename columns if needed
        if status == "postprep":
            df = df.rename(
                columns={"EncodedDay": "DayFreq", "EncodedComputerName": "CompFreq"}
            )

        # Extract needed columns
        res_df = df[["Account_Name", "day", "Time", "CompFreq"]].copy()

        # Define conditions and corresponding cases
        conditions = [
            (df["workingday"]) & (~df["workinghour"]) & (df["CompFreq"] == 0),
            (~df["workingday"]) & (df["workinghour"]) & (df["CompFreq"] == 0),
            (df["workingday"]) & (df["workinghour"]) & (df["CompFreq"] == 0),
            (~df["workingday"]) & (~df["workinghour"]) & (df["CompFreq"] != 0),
            (df["workingday"]) & (~df["workinghour"]) & (df["CompFreq"] != 0),
            (~df["workingday"]) & (df["workinghour"]) & (df["CompFreq"] != 0),
            (~df["workingday"]) & (~df["workinghour"]) & (df["CompFreq"] == 0),
        ]

        cases = ["Case 1", "Case 2", "Case 3", "Case 4", "Case 5", "Case 6", "Case 7"]

        # Apply conditions to DataFrame for the specified status column
        res_df[status] = np.select(conditions, cases, default="Normal")

        results.append(res_df)

    # Concatenate all DataFrames if results is not empty
    return pd.concat(results) if results else pd.DataFrame()


def merge_and_save(postprep_df, predict_df, s3, suffix):
    """
    Merge postprep and predict DataFrames if there anomaly was detected.
    Then save to the S3 bucket under pickles/timestamp.
    """
    merged_df = pd.merge(
        postprep_df,
        predict_df[["Account_Name", "day", "Time", "predict"]],
        on=["Account_Name", "day", "Time"],
        how="left",  # Keep all rows from postprep_df
    )

    # Fill NaN with Normal
    merged_df["predict"] = merged_df["predict"].fillna("Normal")

    # Save merged data to the S3 bucket
    try:
        pickle_data = pickle.dumps(merged_df)
        s3.put_object(
            Body=pickle_data,
            Bucket=bucket_name,
            Key=f"pickles/{suffix}/predict_analysis.pickle",
        )
    except ClientError:
        err_msg = "S3 upload error on analysis"
        ml.write_log(err_msg, 3, "Predict.py", ml.lineno())


if __name__ == "__main__":
    # Load from the bucket
    user_with_models_df = read_pickle("users_with_models", s3_client)
    post_prep_data = read_pickle("post_preprocessed_dicts_models", s3_client)

    # Save post prep data before dropping columns
    copied_data = post_prep_data.copy()

    # Drop columns cast boolean to integers
    filtered_data = drop_columns(post_prep_data)

    # Get predictions using model server
    users_list = list(user_with_models_df.keys())
    model_predictions = {}

    # Iterate each dataframe and if the user and data user are matching, get url
    for data_user, df in filtered_data.items():
        for user in users_list:
            if user == data_user:
                url = get_internal_url(user)
                # Generate predictions for the user
                predictions = predict(user, df, url)
                model_predictions[user] = predictions

    # Add predicted labels as new column in original data
    final_data = add_columns(copied_data, model_predictions)

    # Sanity Check : compare final_df with prediction labels, see if indices are matching
    for user, df in final_data.items():
        for user2, pred_arr in model_predictions.items():
            if user == user2:
                setA = set(df.index[df["prediction"] == 1])
                setB = set(n for n, x in enumerate(pred_arr) if x == 1)
                print(f"User {user} indices are matching? {setA == setB}")

    # Extract and filter abnormal data
    abnorm_dct = extract_filter_data(final_data)
    # Empty Dictionary Check
    if abnorm_dct == {}:
        info_msg = "No data tagged as abnormal"
        ml.write_log(info_msg, 1, "Predict.py", ml.lineno())

    else:
        """
        About the format conversion:
        Since dictionary values are non-iterable, we first convert them into a list.
        Convert the list of values to a DataFrame using pd.concat().
        To send it back to Splunk, convert the DataFrame to JSON using to_json().
        """
        # Convert Dictionary values to DataFrame
        converted_df = pd.concat(list(abnorm_dct.values()))
        # Convert dataframe to JSON format
        json_data = converted_df.to_json(
            orient="records", date_format="iso", lines=True
        )
        suffix = dt.datetime.now(pytz.timezone("America/New_York")).strftime(
            "%m%d%y_%H"
        )

        # Save consolidated data to the S3 bucket
        try:
            s3_client.put_object(
                Body=json_data,
                Bucket=bucket_name,
                Key=f"predictions/predict{suffix}.json",
            )
        except ClientError as error:
            err_msg = "S3 upload error"
            print(err_msg, error)
            ml.write_log(err_msg, 3, "Predict.py", ml.lineno())

        postprep_df = analyze_cases(post_prep_data, "postprep")
        predict_df = analyze_cases(abnorm_dct, "predict")
        merge_and_save(postprep_df, predict_df, s3_client, suffix)
