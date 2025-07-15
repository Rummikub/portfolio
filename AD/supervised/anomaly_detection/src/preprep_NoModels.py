# Pipeline Libs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# General Libs
import pandas as pd
import numpy as np
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

"""
Order of scripts for pipeline without models
no models load.py -> preprep_noModels.py -> syndatagen.py ->
postprep_noModels.py -> train.py -> savemodels.py
"""


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


class DropLessThanWeekUsers(BaseEstimator, TransformerMixin):
    """
    Drops users from the dictionary when they do not have one week's
    worth of range in the dataset.
    For example, if user Bob has the start date 03-07-2024 and an end date of
    03-09-2024, they would be dropped.
    But if a user like Alex, who has a start date 02-24-2024 and an end date
    of 03-03-2024, they would be kept.

    Variables:
        users_to_drop: It is a list that would contain the users that do not
        pass the condition check of having a weeks worth of date range.
        It would then be used to remove the users that are in the list from
        the dictionary too.

    ! The reasoning for this is because it would break everything else in the
    pipeline if they don't have sufficient data.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        users_to_drop = []

        for user, df in X.items():
            start_date = df["Date_Time"].iloc[0]
            end_date = start_date + dt.timedelta(days=6)
            print("Lower Limit:", start_date, "Upper Limit:", end_date)

            if start_date <= df["Date_Time"].iloc[-1] <= end_date:
                users_to_drop.append(user)

            if df.shape[0] <= 100:
                users_to_drop.append(user)

        users_to_drop = set(users_to_drop)

        if users_to_drop:
            for user, df in X.items():
                print(
                    user,
                    "has",
                    len(df),
                    "amounts of data entries before being dropped.",
                )
            for i in users_to_drop:
                X.pop(i)

                print(
                    "Dropped User:",
                    i,
                    "because they didn't have data with an one week range!",
                )

        if not users_to_drop:
            print("No users need to be dropped!")

        return X


class UnlocksAndLocks(BaseEstimator, TransformerMixin):
    """
    Creates two new columns, 'Unlocks' and 'Locks' based on the entry inside
    the column, 'EventCode'.
    'Unlocks' would be 1 if the Event Code is 4801, otherwise it would be 0
    'Locks' would be 1 if the Event Code is 4800, otherwise it would be 1
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for user, df in X.items():
            df["Unlocks"] = np.where(df["EventCode"] == 4801, 1, 0)
            df["Locks"] = np.where(df["EventCode"] == 4800, 1, 0)

        return X


class Target(BaseEstimator, TransformerMixin):
    """
    Creates a new column named 'Target' for the model to have a class.
    Because of the nature of our dataset, all the entries are assigned '0' for
    'Target' because we are working with the assumption that to the best of
    our knowledge, it is all normal data.
    ! It is very important to note that we are working with the assumption
    that all the data currently, as of 6/4/2024, are considered normal data.
    So in the future, when we have this in production, and actually have
    properly labeled data, this function might be deprecated or removed
    completely since the model would have more accurate data and labels
    to work with.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for user, df in X.items():
            df["Target"] = 0

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
                [
                    "Date_Time",
                    "Date",
                    "Time",
                    "day",
                    "Account_Name",
                    "ComputerName",
                    "Unlocks",
                    "Locks",
                    "Target",
                ]
            ]

        return X


Pre_Preprocess_Pipe = Pipeline(
    [
        ("Create DateTime", CreatedDateTime()),
        ("Sorting Values", SortedValues()),
        ("Casting Date and Time", CastedTimeDate()),
        ("Drop users with less than one week range", DropLessThanWeekUsers()),
        ("Creating Unlock and Locks Column", UnlocksAndLocks()),
        ("Creating Target Column", Target()),
        ("Rearranging Dataframe", RearrangedColumns()),
    ]
)

if __name__ == "__main__":
    try:
        # Load from the bucket
        dct = read_pickle("users_with_no_models", s3_client)
        if len(dct) == 0:
            err_msg = "No pickle file was found"
            ml.write_log(err_msg, 3, "Preprep_NoModels.py", ml.lineno())
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
        write_pickle(
            pre_preprocessed_dicts, "pre_preprocessed_dicts_no_models", s3_client
        )

    except Exception as e:
        err_msg = "Error running the request." + format(e)
        print(err_msg)
        ml.write_log(err_msg, 3, "Preprep_NoModels.py", ml.lineno())
        sys.exit(3)
