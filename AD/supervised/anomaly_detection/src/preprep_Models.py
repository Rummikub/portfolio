# Pipeline Libs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# General Libs
import pandas as pd
import pickle
import datetime as dt
import pytz
import logging
import sys
import MLDB as ml

# S3 libs
import boto3
from botocore.exceptions import ClientError
import os

# s3 connection parameters:
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
# has models load.py -> preprep_Models.py -> postprep_models.py -> predict.py


def read_pickle(pickle_name, s3):
    timestamp = dt.datetime.now(pytz.timezone("America/New_York")).strftime("%m%d%y_%H")
    file_key = f"pickles/{timestamp}/{pickle_name}.pickle"
    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        pickle_data = response["Body"].read()

        # Unpickle the data
        data = pickle.loads(pickle_data)
        return data
    except ClientError as e:
        logging.error(e)
        # return empty dictionary
        return dict()


def write_pickle(data, pickle_name, s3):
    # Pickle the data
    pickle_data = pickle.dumps(data)
    timestamp = dt.datetime.now(pytz.timezone("America/New_York")).strftime("%m%d%y_%H")
    file_key = f"pickles/{timestamp}/{pickle_name}.pickle"
    # Upload the pickle data to S3
    try:
        s3.put_object(Bucket=bucket_name, Key=file_key, Body=pickle_data)
    except ClientError as e:
        logging.error(e)
        return False
    return True


class CreatedDateTime(BaseEstimator, TransformerMixin):
    """
    Creates a new column named 'Date_Time' by combining the two columns 'Date'
    and 'Time' in order to sort the users data in order in a later step
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for user, df in X.items():
            df["Date"] = df["Date"].astype(str)
            df["Time"] = df["Time"].astype(str)
            df["Date_Time"] = pd.to_datetime(
                df["Date"] + " " + df["Time"], format="%Y-%m-%d %H:%M:%S"
            )

        return X


class SortedValues(BaseEstimator, TransformerMixin):
    """
    Sorts the users data using the column 'Date_Time'
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for user, df in X.items():
            X[user] = df.sort_values(by="Date_Time")

        return X


class CastedTimeDate(BaseEstimator, TransformerMixin):
    """
    Casts the values in 'Time' column as a datetime time object
    Casts the values in 'Date' column as a datetime date object
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for user, df in X.items():
            df["Time"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.time
            df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d").dt.date

        return X


class RearrangedColumns(BaseEstimator, TransformerMixin):
    """
    Rearranges the dataframe for future processing.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for user, df in X.items():
            # X[user] = df[df.columns[[7,0,1,2,3,4,8,9,10]]]
            X[user] = df[
                ["Date_Time", "Date", "Time", "day", "Account_Name", "ComputerName"]
            ]

        return X


Pre_Preprocess_Pipe = Pipeline(
    [
        ("Create DateTime", CreatedDateTime()),
        ("Sorting Values", SortedValues()),
        ("Casting Date and Time", CastedTimeDate()),
        ("Rarrange Columns", RearrangedColumns()),
    ]
)


if __name__ == "__main__":
    try:
        # Load from the bucket
        dct = read_pickle("users_with_models", s3_client)
        if len(dct) == 0:
            err_msg = "No pickle file was found"
            ml.write_log(err_msg, 3, "Preprep_Models.py", ml.lineno())
            sys.exit(3)
        # Passing the newly loaded data to preprocess through pipeline
        pre_preprocessed_dicts = Pre_Preprocess_Pipe.fit_transform(dct.copy())
        # Printing the total amount of data
        for user, df in pre_preprocessed_dicts.items():
            print(
                user,
                "has",
                len(df),
                "amounts of data entries after being put through part 1 \
                of preprocess pipeline.",
            )
        # Save to the bucket
        write_pickle(pre_preprocessed_dicts, "pre_preprocessed_dicts_models", s3_client)

    except Exception as e:
        err_msg = "Error running the request." + format(e)
        print(err_msg)
        ml.write_log(err_msg, 3, "Preprep_Models.py", ml.lineno())
        sys.exit(3)
