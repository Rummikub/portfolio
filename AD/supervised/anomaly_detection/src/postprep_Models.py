# Pipeline Libs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# General libs
import pandas as pd
import numpy as np
import re
import pickle
import MLDB as ml
import sqlutils as su
import datetime as dt
import pytz
import logging
import sys

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


# Function to get values related DayFreq, workingday and workinghours from database
def split_workingdata(user):
    """
    This function will find the user id from the database and then get
    all working-related data, i.e. workingdays and working hours.

    Args:
        user (str): Account_Name also known as model name or user name

    Returns:
        - wd_dict (dict): Values related to DayFreq and working days.
        - wh_dict (dict): Values related to working hours.
    """
    print(user)
    usr = su.get_user_id(user)
    _, db_dict = ml.get_work_data(usr)
    wd_dict = {
        k: v for k, v in db_dict.items() if not isinstance(v, int)
    }  # if value is float = workingday
    wh_dict = {
        k: v for k, v in db_dict.items() if isinstance(v, int)
    }  # value is int = workinghour
    return wd_dict, wh_dict


class CompFreq(BaseEstimator, TransformerMixin):
    """
    The purpose of this function is to frequency encode all the ComputerNames
    after it had synthetic data added. The reason of needing to do so is
    because majority of the models we plan to use currently can not handle
    strings, so we need to convert any important string columns into some
    numerical value for the model to interpret

    Example: Let's say a user has 20 entries of Computer1 and 5 entries of
    Computer2. Since the parameters has normalize = True, it would return a
    float between 0.1
    Rows with 'Computer1' would be assigned 0.8 since it is 20/25
    Rows with 'Computer2' would be assigned 0.2 since it is 5/25

    Another example would be let's say for a certain user, we have 3 computers.
    Computer1 with 10 entries, Computer2 with 10 entries,
    and Computer3 with 5 entries
    Rows with 'Computer1' and 'Computer2' would be assigned 0.4
    because both Computers have 10/25 entries.
    Rows with 'Computer3' would be assigned 0.2 since it is 5/25
    for the dataset, which is the same as the last example.

    Currently, this is the best performing encoding method so far.
        The methods I have tried so far are:
        1) Hash Encoding with 7-10 n_features created: This method performed
        consistently with our typical pattern of recall score being high
        and precision score being low when
        the models were tested on the validation set.
            > We should consider revisiting this method with more n_features
            due to how it compared to Frequency Encoding once we figure out
            how to make the run time faster.
            The main problem is how long it takes to run Hash Encoding.
        2) Binary Encoding: This one performed better than the Hash Encoding
        method but the models could not have been tested on the validation set
        because it was not creating the same amount of columns as seen in the
        fit without adding 1 to 1 synthetic data to every dataset again.
        3) One Hot Encoding: Not feasible currently because we do not have a
        concrete dataset to work on and when using it on ComputerNames would
        on average create 100+ columns because of the newly added
        computernames from synthetic data.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for user, df in X.items():
            # Connect with database
            _, freq_ldict = ml.get_frequencies(user)
            df["EncodedComputerName"] = np.nan

            for freq_dict in freq_ldict:
                computername = list(freq_dict.keys())[0]
                frequency = freq_dict[computername]
                mask = df["ComputerName"] == computername
                df.loc[mask, "EncodedComputerName"] = frequency

                # Avoid NaN values from frequency of new computername
                if df["EncodedComputerName"].isna().values.any():
                    df["EncodedComputerName"] = df["EncodedComputerName"].fillna(0)
        return X


class DayFreq(BaseEstimator, TransformerMixin):
    """
    The purpose of this function is to frequency encode all the day of week
    after synthetic data has been added. The reason of needing to do so is
    to introduce the concept of abnormality day of week to the machine and
    expect it detects somewhat least frequent day of week as an important feature
    to flag as abnormal based on our synthetic data scenarios

    From a database standpoint, each value will be from the working_days table. For example, if
    historical data saved in the database is as follows:
        Monday: 0.18 , Tuesday: 0.22, Wednesday: 0.17,
        Thursday: 0.23, Friday: 0.1,
        Saturday: 0.05 , Sunday: 0.05
    And the new incoming data's day of the week is Tuesday,
    it will retrieve 0.22 as the value.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for user, df in X.items():
            wd_dict, _ = split_workingdata(user)
            capitalized_dict = {
                key.capitalize(): value for key, value in wd_dict.items()
            }
            for day in capitalized_dict:
                df.loc[df["day"] == day, "EncodedDay"] = capitalized_dict[day]
        return X


class OneHotEncodedDay(BaseEstimator, TransformerMixin):
    """
    The purpose of this function is to One-Hot Encode days of the week for the
    models to properly use the 'day' column in our dataset
    Since there are only 7 days in a week, we can guarantee to consistently
    create a set amount of columns for the dataset.
    It checks for one by one for everyday of the week to see if it should put
    in 0 or 1 on the current day.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for user, df in X.items():
            df["Monday"] = np.where(df["day"] == "Monday", 1, 0)
            df["Tuesday"] = np.where(df["day"] == "Tuesday", 1, 0)
            df["Wednesday"] = np.where(df["day"] == "Wednesday", 1, 0)
            df["Thursday"] = np.where(df["day"] == "Thursday", 1, 0)
            df["Friday"] = np.where(df["day"] == "Friday", 1, 0)
            df["Saturday"] = np.where(df["day"] == "Saturday", 1, 0)
            df["Sunday"] = np.where(df["day"] == "Sunday", 1, 0)
        return X


class WorkingDay(BaseEstimator, TransformerMixin):
    """
    The purpose of this function is to give a machine a hint
    about what an abnormal working day looks like.
    Since it assumes the first 5 frequent days as normal working days,
    it will mark the remaining 2 days as False based on user behavior
    by only extracting the original data.

    It checks each day of the week to determine whether it should be marked
    as True or False based on their working day.

    From a database standpoint, each freq_values will be from working_days table
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for user, df in X.items():
            wd_dict, _ = split_workingdata(user)

            freq_sorted = sorted((v, k) for k, v in wd_dict.items())
            # Sort them with descending
            sorted_data = sorted(freq_sorted, key=lambda x: x[0], reverse=True)
            # Extract the highest five values
            top_five = sorted_data[:5]

            # Create a new list with the day names
            norm = [day for value, day in top_five]
            norm = [day.capitalize() for day in norm]
            df["workingday"] = np.where(df["day"].isin(norm), True, False)

        return X


class WorkingHour(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for user, df in X.items():
            res, wh_dict = split_workingdata(user)
            days = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]

            df["hour"] = df["Time"].astype(str).str[:2].astype(int)
            doW_list = []

            df["workinghour"] = False
            # Create Temporary List of list [[Monday, s,e],...]
            for idx, day in enumerate(days):
                num = 3
                if day.startswith("Tues") or day.startswith("Thur"):
                    num = 4

                start_key = f"{day[:num].lower()}_s"
                end_key = f"{day[:num].lower()}_e"
                doW_list.append([day, wh_dict[start_key], wh_dict[end_key]])

            for sub in doW_list:
                day = sub[0]
                start = sub[1]
                end = sub[2]
                # Mark True if in between start and end time for a given day
                mask = (df["day"] == day) & (df["hour"] <= end) & (df["hour"] >= start)
                df.loc[mask, "workinghour"] = True

            # Delete afteruse
            df.drop("hour", axis=1, inplace=True)

        return X


class BinnedTime(BaseEstimator, TransformerMixin):
    """
    This function One Hot Encodes times into their respective hours.
    If separating 'Time' into it's own column didn't help, then maybe
    one hot encoding it would be better.
    The reason why we focus on time is that every user will have set patterns
    they tend to follow. Such as working between 10AM to 6PM or 8AM to 4PM.
    Therefore, when there are data entries outside of those typical ranges,
    the model should ideally recognize this as unusual.

    Example of how this function should work:
    Assuming we have a the 'Date_Time' entry of 4/15/2024 10:24:59,
    It would be assigned a '1' the newly created column, '10:00-10:59'.
    The other columns such as '00:00-00:59', '13:00-13:59', etc. would be '0'
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for user, df in X.items():
            # define the bins
            bins = [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
            ]
            # bins = range(0,23,1)

            # add custom labels if desired
            labels = [
                "00:00-00:59",
                "01:00-01:59",
                "02:00-02:59",
                "03:00-03:59",
                "04:00-04:59",
                "05:00-05:59",
                "06:00-06:59",
                "07:00-07:59",
                "08:00-08:59",
                "09:00-09:59",
                "10:00-10:59",
                "11:00-11:59",
                "12:00-12:59",
                "13:00-13:59",
                "14:00-14:59",
                "15:00-15:59",
                "16:00-16:59",
                "17:00-17:59",
                "18:00-18:59",
                "19:00-19:59",
                "20:00-20:59",
                "21:00-21:59",
                "22:00-22:59",
                "23:00-23:59",
            ]

            # add the bins to the dataframe
            df["Time_Bin"] = pd.cut(
                df["Date_Time"].dt.hour, bins, labels=labels, right=False
            )
            # df['Time_Bin'] = df['Date_Time'].dt.hour
            # time_bins = labels.tolist()
            X[user] = pd.get_dummies(data=df, columns=["Time_Bin"])

            # For some preprocessing, we need to add in Time_Bin column again.
            # (Maybe unnecessary for now)
            df["Time_Bin"] = pd.cut(
                df["Date_Time"].dt.hour, bins, labels=labels, right=False
            )
        return X


class JsonEncoding(BaseEstimator, TransformerMixin):
    """
    The purpose of this function is to encode JSON characters in the feature,
    due to LightGBM cannot support certain characters in the feature names.
    Especially, return error if feature appears more than one time.
    For example, Time_Bin 12:00-12:59 and Time_Bin 13:00-13:59 etc. in one df.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for user, df in X.items():
            new_cols = {col: re.sub(r"[^A-Za-z0-9_]+", "", col) for col in df.columns}
            new_nList = list(new_cols.values())
            new_cols = {
                col: f"{new_col}_{i}" if new_col in new_nList[:i] else new_col
                for i, (col, new_col) in enumerate(new_cols.items())
            }
            X[user] = df.rename(columns=new_cols)
        return X


Post_Preprocess_Pipe = Pipeline(
    [
        ("Frequency Encoding ComputerName", CompFreq()),
        ("Frequency Encoding Day of Week", DayFreq()),
        ("One Hot Encoding Day of Week", OneHotEncodedDay()),
        ("One Hot Encoding WorkingDay", WorkingDay()),
        ("One Hot Encoding WorkingHour", WorkingHour()),
        ("One Hot Encoding Time Bins", BinnedTime()),
        ("JSON characters encoding", JsonEncoding()),
    ]
)


if __name__ == "__main__":
    try:
        # Load from the bucket
        concated_data = read_pickle("pre_preprocessed_dicts_models", s3_client)
        if len(concated_data) == 0:
            err_msg = "No pickle file was found"
            ml.write_log(err_msg, 3, "Postprep_Models.py", ml.lineno())
            sys.exit(3)
        # Passing the newly loaded data to preprocess through pipeline
        post_preprocessed_dicts = Post_Preprocess_Pipe.fit_transform(
            concated_data.copy()
        )
        # Printing all columns to see if df has been transformed correctly
        for user, df in post_preprocessed_dicts.items():
            print(user, "has", df.columns.tolist())
            print("Does user:", user, "have nulls?", df.isnull().values.any())
            print("  ")

        # Save to the bucket
        write_pickle(
            post_preprocessed_dicts, "post_preprocessed_dicts_models", s3_client
        )

    except Exception as e:
        err_msg = "Error running the request." + format(e)
        print(err_msg)
        ml.write(err_msg, 3, "Postprep_Models.py", ml.lineno())
        sys.exit(3)
