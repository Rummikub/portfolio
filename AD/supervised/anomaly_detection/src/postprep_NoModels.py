# Pipeline Libs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# General libs
import pandas as pd
import numpy as np
import pickle
import re
import MLDB as ml
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


# Funtion to read pickle files from the s3 bucket
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


# Function to write pickle files to the s3 bucket
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


# Function to save frequency encoded computername to database
def save_computer_frequency(df):
    norm_df = df.loc[df["Target"] == 0]
    cList = norm_df["ComputerName"].unique()
    res = []
    total = 0
    norm_freq = norm_df["ComputerName"].value_counts(normalize=True)
    norm_df["EncodedComputerName"] = norm_df["ComputerName"].map(norm_freq)
    norm_df["EncodedComputerName"] = norm_df["EncodedComputerName"].replace("", np.nan)
    if norm_df["EncodedComputerName"].isna().values.any():
        norm_df["EncodedComputerName"] = norm_df["EncodedComputerName"].fillna(0)

    # Append Frequency
    for c in cList:
        new_freq = norm_df.loc[
            norm_df["ComputerName"] == c, "EncodedComputerName"
        ].values[0]
        total += new_freq
        res.append({c: new_freq})
    return res, total


# Function to get average start and end time of user for their original data
def avg_start_end(df, day):
    """
    The purpose of this function is to get the average start time and end time.
    > It initializes two variables, calc_avg_start and calc_avg_end set to 0.
    > Then it assigns a value to calc_avg_start by averaging the hour of the
    first entry of every 'Date'.
    Then it is casted into an integer to get rid of the float.
    > Afterwards, it assigns a value to calc_avg_start by averaging the hour
    of the last entry of every 'Date'. Then it is casted into an integer to
    get rid of the float.
    > Finally, it returns both calc_avg_start and calc_avg_end.
    """
    norm_df = df.loc[(df["Target"] == 0) & (df["day"] == day)]
    calc_avg_start, calc_avg_end = 0, 0
    calc_avg_start = norm_df.groupby(["Date"])["Date_Time"].first().dt.hour.mean()
    calc_avg_end = norm_df.groupby(["Date"])["Date_Time"].last().dt.hour.mean()

    return calc_avg_start, calc_avg_end


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
            # Save Historical Data on frequency table
            result, total_freq = save_computer_frequency(df)
            ml.new_multi_frequency(user, result)
            print(f"DB Saved: {user} total {len(result)} as {total_freq}")

            # Frequency Encoding on Concated Data
            frequency = df["ComputerName"].value_counts(normalize=True)
            df["EncodedComputerName"] = df["ComputerName"].map(frequency)
            df["EncodedComputerName"] = df["EncodedComputerName"].replace("", np.nan)
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

    From a database standpoint, each value will be saved in the working_days table.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for user, df in X.items():
            frequency = df["day"].value_counts(normalize=True)
            df["EncodedDay"] = df["day"].map(frequency)
            df["EncodedDay"] = df["EncodedDay"].replace("", np.nan)
            if df["EncodedDay"].isna().values.any():
                df["EncodedDay"] = df["EncodedDay"].fillna(0)
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

    From a database standpoint, each freq_values will be saved in working_days table
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for user, df in X.items():
            normal_df = df.loc[df["Target"] == 0]
            abnorm_dotW, norm_dotW = list(), list()
            frequency = {
                "Monday": 0,
                "Tuesday": 0,
                "Wednesday": 0,
                "Thursday": 0,
                "Friday": 0,
                "Saturday": 0,
                "Sunday": 0,
            }
            # Save Historical Data on woking_days table with DayFreq values
            freq_values = normal_df["day"].value_counts(normalize=True).to_dict()
            frequency = {
                key: freq_values.get(key, val) for key, val in frequency.items()
            }
            freq_sorted = sorted(((v, k) for k, v in frequency.items()))

            for key, value in frequency.items():
                if value <= freq_sorted[-6][0] or value == 0:
                    abnorm_dotW.append(key)
                else:
                    norm_dotW.append(key)

            wDays = {day: v for v, day in freq_sorted}
            wDays = {k.lower(): v for k, v in wDays.items()}
            ml.new_working_days(user, wDays)
            total = 0
            for k, v in wDays.items():
                total += v
            print(f"DB Saved: user {user} workingday frequency as {total}")
            df["workingday"] = np.where(df["day"].isin(norm_dotW), True, False)

        return X


class WorkingHour(BaseEstimator, TransformerMixin):
    """
    The purpose of this function is to give a machine a hint
    about range of normal working hours.

    It first filters to only normal (non-target) records and
    then calculates the average start and end hours for each day of the week
    by finding the mean first and last hour of the day across different dates.

    It creates a new column WorkingHour which marks
    whether each record falls within the calculated normal working hours,
    defaulting to a full day (0-23 hours) if no data is available for a specific day.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for user, df in X.items():
            dotW_list = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]

            res = {}
            for current_day in dotW_list:
                # Calculate Average Start and EndTime for a given day
                start, end = avg_start_end(df, current_day)
                if np.isnan(start) and np.isnan(end):
                    start, end = 0, 23
                else:
                    start, end = int(start), int(end)

                # Save Historical Data on work_hours table
                dayKey = current_day
                if current_day == "Tuesday" or current_day == "Thursday":
                    dayKey = current_day[:4]
                else:
                    dayKey = current_day[:3]
                res.update({f"{dayKey}_s": start, f"{dayKey}_e": end})
                res = {k.lower(): v for k, v in res.items()}

                # Extract certain day of week data on concated_df
                day_mask = df["day"] == current_day
                # Edgecase start time 0 and end time 23 is abnormal no matter what
                if start == 0 and end == 23:
                    df.loc[day_mask, "workinghour"] = False
                else:
                    normtime = [
                        hour for hour in range(start, end + 1) if 0 <= hour <= 23
                    ]

                    df.loc[day_mask, "workinghour"] = np.where(
                        df.loc[day_mask, "Date_Time"].dt.hour.isin(normtime),
                        True,
                        False,
                    )
            ml.new_work_hours(user, res)
            print(
                f"DB Saved: user {user} as {' '.join(f'{k}:{v}' for k,v in res.items())}"
            )
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
        concated_data = read_pickle("concated_syn_dicts", s3_client)
        if len(concated_data) == 0:
            err_msg = "No pickle file was found"
            ml.write_log(err_msg, 3, "Postprep_NoModels.py", ml.lineno())
            sys.exit(3)
        # Passing the newly loaded data to preprocess through pipeline
        post_preprocessed_dicts = Post_Preprocess_Pipe.fit_transform(
            concated_data.copy()
        )
        # Printing null check
        for user, df in post_preprocessed_dicts.items():
            print(user, "has", df.columns.tolist())
            print("Does user:", user, "have nulls?", df.isnull().values.any())
        # Save to the bucket
        write_pickle(
            post_preprocessed_dicts, "post_preprocessed_dicts_no_models", s3_client
        )

    except Exception as e:
        err_msg = "Error running the request." + format(e)
        print(err_msg)
        ml.write(err_msg, 3, "Postprep_NoModels.py", ml.lineno())
        sys.exit(3)
