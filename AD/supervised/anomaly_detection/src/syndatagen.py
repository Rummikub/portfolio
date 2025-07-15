# Pipeline Libs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# General libs
import pandas as pd
import numpy as np
import pickle
from hashlib import md5
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

# globally declare an instance of the md5 function
md5 = md5()


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


def md5_hash(string):
    """
    Function to take a string as input, encode it into bytes, and
    then apply an MD5 hash to it.
    Returns the md5 hashed string in hexidecimal.
    """
    md5.update(string.encode())
    string_hash = md5.hexdigest()
    return string_hash


def spec_avg_start_end(start_end_df):
    """
    Function for returning a dataframe that contains 3 column
    DayOfWeek, Avg_Start_Time, and Avg_End_Time
    > DayOfWeek would Monday to Sunday
    > StartTime would contain the average first Time entry of given day of week
    > EndTime would contain the average last Time entry of given day of week
    > If there is a DayOfWeek that has not shown up at all in the dataset
    or in other words, if StartTime and EndTime are nan,
    then it would set Start_Time as 0, and End_Time as 23
    """
    dotW_list = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    start_time = []
    end_time = []

    for current_dotW in dotW_list:
        spec_dotW_avg_start = (
            start_end_df[start_end_df["day"] == current_dotW]
            .groupby(["Date"])["Date_Time"]
            .first()
            .dt.hour.mean()
        )
        spec_dotW_avg_end = (
            start_end_df[start_end_df["day"] == current_dotW]
            .groupby(["Date"])["Date_Time"]
            .last()
            .dt.hour.mean()
        )

        if np.isnan(spec_dotW_avg_start) | np.isnan(spec_dotW_avg_end):
            spec_dotW_avg_start, spec_dotW_avg_end = 0, 23

        else:
            spec_dotW_avg_start = int(spec_dotW_avg_start)
            spec_dotW_avg_end = int(spec_dotW_avg_end)

        print(
            "Function Start Time",
            spec_dotW_avg_start,
            "Function End Time",
            spec_dotW_avg_end,
        )

        start_time.append(spec_dotW_avg_start)
        end_time.append(spec_dotW_avg_end)

    dotW_Start_End_df = pd.DataFrame(
        {"DayOfWeek": dotW_list, "StartTime": start_time, "EndTime": end_time}
    )
    print(dotW_Start_End_df)
    return dotW_Start_End_df


# Function for getting two lists, one that contains abnormal days of the week
# and the other that contains normal days of the week
def dotW_select(dotW_df):
    """
    This function returns two lists, abnorm_dotW and norm_dotW.
    > We are working with the assumption that every user works at least 5 days
    throughout a week so typically, they would have a minimum of 2 days having
    little amount of data entries or none at all.
    > First it initializes a dictionary where the keys are the Days of the
    Week and the values are all 0.
    > Then it would grab the frequency of appearance throughout the entire
    dataset for every day of the week and save it.
    > Finally, it would add the days that had the smallest count or has a
    count of 0 to abnorm_dotW and the rest into norm_dotw
    """
    # Creating some empty lists inside the function to return later
    abnorm_dotW = []
    norm_dotW = []

    # Preassigning 0s to all the days of the week to deal with users
    # that won't use their computers at all on certain days
    frequency = {
        "Monday": 0,
        "Tuesday": 0,
        "Wednesday": 0,
        "Thursday": 0,
        "Friday": 0,
        "Saturday": 0,
        "Sunday": 0,
    }

    # Storing results into a variable for later use
    freq_values = dotW_df["day"].value_counts(normalize=True).to_dict()

    # Updating original dictionary with new values gained from .value_counts()
    frequency = {key: freq_values.get(key, val) for key, val in frequency.items()}
    freq_sorted = sorted(((v, k) for k, v in frequency.items()))

    # Filling the lists on what would be an anomalous or normal day of week
    # based on the users habits via a set threshold
    # Which is the lowest frequencies of the dotW or if value is equal to 0.
    for key, value in frequency.items():
        if value <= freq_sorted[-6][0] or value == 0:
            abnorm_dotW.append(key)

        else:
            norm_dotW.append(key)

    return abnorm_dotW, norm_dotW


# Function for getting a specific date based on day of week selected
def specific_date_select(date_df, day_selected):
    # Assigning variables to the start and end dates for readability
    starting_date = np.datetime64(date_df["Date"].iloc[0])
    ending_date = np.datetime64(date_df["Date"].iloc[-1])

    date_selected = pd.to_datetime(
        np.random.choice(
            pd.date_range(
                starting_date, ending_date, freq="W-" + str(day_selected[0:3]).upper()
            )
        )
    ).date()

    return date_selected


def Unlock_Lock_Ratio(lock_unlock_df):
    """
    Despite the misleading name, in reality it returns the fraction of unlocks
    compared to the total unlocks and locks of the user.
    The variable returned then is used to try to mimic the pattern of the user
    by giving the two potential choices their appropriate weights based on the
    current user.
    """
    # .sum() is a brillant way that Harini did to get the total of the Unlock
    # and Lock column because it only contains 1 or 0. So in this case
    # every Unlock or Lock = 1 would be counted correctly regardless of the
    # number of zeroes inside the column
    allUnlocks = lock_unlock_df["Unlocks"].sum()
    allLocks = lock_unlock_df["Locks"].sum()

    unlock_lock_ratio = allUnlocks / (allUnlocks + allLocks)
    print("Unlock == 1 Percentage:", unlock_lock_ratio)

    return unlock_lock_ratio


# Function for getting normal time
def norm_time_select(norm_user_avg_start_time, norm_user_avg_end_time):
    """
    Function for returning a list of "normal" times for the user.
    Essentially it returns norm_user_avg_start_time and norm_user_avg_end_time

    It handles the currently known edge cases which are:
        If norm_end is less than norm_user_avg_start_time,
        then it would set norm_end equal to norm_start + 7
        because a person typically only works for around 7 hours a day.
        The second edge case is when the new norm_end is
        greater than or equal to 24 because of the previous edge case.
        It would set norm_end to 23.

    After checking for those two edge cases, it would then cast both of them
    as strings and plug it into pd.date_range
    to get a list of times between the two variables and return it.
    """
    norm_start = norm_user_avg_start_time
    norm_end = norm_user_avg_end_time

    if norm_end < norm_start:
        norm_end = norm_start + 7

        if norm_end >= 24:
            norm_end = 23

        norm_end = str(norm_end)

    norm_start = str(norm_start)
    norm_end = str(norm_end)

    norm_time = pd.date_range(norm_start + ":00", norm_end + ":59", freq="s").time
    # print("Normal Times for this user us:",norm_time)

    return norm_time


# Function for getting abnormal time
def abnorm_time_select(user_avg_start_time, user_avg_end_time):
    """
    Function for returning a list of "abnormal" times for the user.
    It returns values are NOT between norm_user_avg_start_time and end_time
    It handles the currently known edge cases which are:
        If abnorm_start is less than or equal to 0,
        it would set abnorm_start to 0.
        The second edge case is when the abnorm_end is
        greater than or equal to 24, then it would set abnorm_end to 23

    After checking for those two edge cases,
    it would then cast both of them as strings and plug it
    into pd.date_range to get a list of times that are NOT
    between the two variables and return it.
        It does this by only getting the difference between a pd.date_range
        with inputs 0:00 to 23:59 and a pd.date_range
        with inputs abnorm_start and abnorm_end.
    """

    abnorm_start = user_avg_start_time - 1
    abnorm_end = user_avg_end_time + 1

    # To make sure they are not out of bounds for pd.date_range
    if abnorm_start <= 0:
        abnorm_start = 0
    if abnorm_end >= 24:
        abnorm_end = 23

    abnorm_start = str(abnorm_start)
    abnorm_end = str(abnorm_end)

    abnorm_time = (
        pd.date_range("0:00", "23:59", freq="s")
        .difference(pd.date_range(abnorm_start + ":00", abnorm_end + ":59", freq="s"))
        .time
    )
    # print("Abnormal Times for this user are:", abnorm_time)

    return abnorm_time


def time_dictionary(dotW_df):
    """
    Function for creating a dictionary that has the key DayOfWeek
    and values Normal Times and Abnormal Times
    > Normal Times would be anytime between the Start and End Time
    > Abnormal Times would be anytime outside of the Start and End Time

    > If there is the case where StartTime = 0 AND EndTime = 23,
    then both Normal Time and Abnormal Time would contain
    all the values between 0:00 and 23:59 with seconds precision
    because we are working with the assumption that for this specific case,
    that means the user has not Unlock or Lock on that day at all.
    Therefore, because we are creating synthetic data
    that is considered abnormal no matter what,
    we can just say that for those days, both NormalTime and
    AbnormalTimes are the same for convenience.
    """
    func_dotW_dictionary = {}
    dotW = dotW_df["DayOfWeek"].unique()

    for current_dotW in dotW:
        spec_start_time = dotW_df.loc[
            dotW_df["DayOfWeek"] == current_dotW, "StartTime"
        ].iloc[0]
        spec_end_time = dotW_df.loc[
            dotW_df["DayOfWeek"] == current_dotW, "EndTime"
        ].iloc[0]

        if (spec_start_time == 0) & (spec_end_time == 23):
            spec_start_time = str(spec_start_time)
            spec_end_time = str(spec_end_time)

            times_list = pd.date_range(
                spec_start_time + ":00", spec_end_time + ":59", freq="s"
            ).time

            func_dotW_dictionary[current_dotW] = {
                "NormalTimes": times_list,
                "AbnormalTimes": times_list,
            }

        else:
            norm_times_list = norm_time_select(spec_start_time, spec_end_time)
            abnorm_times_list = abnorm_time_select(spec_start_time, spec_end_time)

            func_dotW_dictionary[current_dotW] = {
                "NormalTimes": norm_times_list,
                "AbnormalTimes": abnorm_times_list,
            }

    return func_dotW_dictionary


def get_range(index, ranges):
    """
    This function would iterate all ranges once we've gone past all ranges
    will be returning the last range
    """
    for idx, rng in enumerate(ranges):
        if index < rng:
            return idx + 1
    return len(ranges)


# To generate concated dataset
class concat_dfs(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        > The purpose of concat_dts is to add 1 to 1 abnormal synthetic data
        to every user.
        > For every user, it will first call dotW_select to return two
        variables, 'abnormal_dotW' and 'normal_dotW' to be used for selecting
        different cases of 'day'
        > Then it'll call Unlock_Lock_Ratio and return the users unlock percent
        > The third function it'll call is spec_avg_start_end, which returns a
        dataframe where it contains 7 rows and 3 columns: DayOfTheWeek,
        AvgStartTime, and AvgEndTime
        > The fourth function called is time_dictionary where it returns a
        dictionary where the keys are DayOfTheWeek, and two values,
        AbnormalTime and NormalTime. In
        AbnormalTime, it contains all the times that are outside of the
        average start and end times of that day of the week for the current
        user. For NormalTime, it contains all the times with one second
        precision that is between the average start and end times of
        that day for the user.

        > The next part of concat_dfs is that it creates data entries for
        every user depending on a base range per total amount of data.
        There are a total of 7 cases.
            1. If the current range is 1: It would
            generate a data entry with normal day of the week, abnormal time,
            and abnormal computer
            2. If the current range is 2: It would generate a data entry with
            abnormal day of week, normal time, and abnormal computer.
            3. If the current range is 3: It would generate a data entry with
            normal day of week, normal time,
            and abnormal computer.
            4. If the current range is 4: It would generate a data entry with
            abnormal day of week, abnormal time, and normal computer.
            5. If the current range is 5: It would generate a data entry with
            normal day of week, abnormal time, and normal computer.
            6. If the current range is 6: It would generate a data entry with
            abnormal day of week, normal time, and normal computer.
            7. If the current range is 7: It would generate a data entry with
            abnormal day of week, abnormal time, and abnormal computer.
        After it has generated enough synthetic data for the user, it would
        take the list and then concatenate it into a single dataframe and sort
        it by Date_Time before moving onto the next user.
        It would repeat this process until it has
        doing this for every user.
        """
        for user, df in X.items():
            abnormal_dotW, normal_dotW = dotW_select(df)
            unlockPercent = Unlock_Lock_Ratio(df)
            dotW_df = spec_avg_start_end(df)
            dotW_dict = time_dictionary(dotW_df)

            columns = [
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

            syn_data = []
            concated_syn_data = []

            length = len(df)
            # Divide each range as similar portion
            ranges = [round(length / 7 * i) for i in range(1, 8)]

            # Proportion Check
            print("Proportion per Scenario:", ranges)
            print(f"Range 1: {ranges[0]}")
            for i in range(1, len(ranges)):
                print(f"Range {i+1}: {ranges[i] - ranges[i-1]}")

            for x in range(0, length):
                current_range = get_range(x, ranges)
                Account_Name = user

                # Initializing what is needed for Unlock and Locks to have a
                # somewhat consistent ratio with each other based on user
                options = [1, 0]
                choice = np.random.choice(
                    options, p=[unlockPercent, (1 - unlockPercent)]
                )
                Unlocks = choice
                Locks = 1 - Unlocks
                Target = 1

                if current_range == 1:
                    # If the random number is less than or equal to 24
                    # It would generate a data entry with normal day of the
                    # week, abnormal time, and abnormal computer
                    day = np.random.choice(normal_dotW)
                    date = specific_date_select(df, day)
                    time = np.random.choice(dotW_dict[day]["AbnormalTimes"])
                    date_time = dt.datetime.combine(date, time)
                    ComputerName = (
                        "W151WBWY"
                        + str((np.random.randint(0, 9)))
                        + str((np.random.randint(0, 9)))
                        + "J0"
                        + str((np.random.randint(0, 9)))
                        + str((np.random.randint(0, 9)))
                        + ".windows.nyc.hra.nycnet"
                    )
                    ComputerName = md5_hash(ComputerName)

                    syn_data.append(
                        [
                            date_time,
                            date,
                            time,
                            day,
                            Account_Name,
                            ComputerName,
                            Unlocks,
                            Locks,
                            Target,
                        ]
                    )

                elif current_range == 2:
                    # It would generate a data entry with abnormal day of week,
                    # normal time, and abnormal computer
                    day = np.random.choice(abnormal_dotW)
                    date = specific_date_select(df, day)
                    time = np.random.choice(dotW_dict[day]["NormalTimes"])
                    date_time = dt.datetime.combine(date, time)
                    ComputerName = (
                        "W151WBWY"
                        + str((np.random.randint(0, 9)))
                        + str((np.random.randint(0, 9)))
                        + "J0"
                        + str((np.random.randint(0, 9)))
                        + str((np.random.randint(0, 9)))
                        + ".windows.nyc.hra.nycnet"
                    )
                    ComputerName = md5_hash(ComputerName)

                    syn_data.append(
                        [
                            date_time,
                            date,
                            time,
                            day,
                            Account_Name,
                            ComputerName,
                            Unlocks,
                            Locks,
                            Target,
                        ]
                    )

                elif current_range == 3:
                    # It would generate a data entry with normal day of week,
                    # normal time, and abnormal computer
                    day = np.random.choice(normal_dotW)
                    date = specific_date_select(df, day)
                    time = np.random.choice(dotW_dict[day]["NormalTimes"])
                    date_time = dt.datetime.combine(date, time)
                    ComputerName = (
                        "W151WBWY"
                        + str((np.random.randint(0, 9)))
                        + str((np.random.randint(0, 9)))
                        + "J0"
                        + str((np.random.randint(0, 9)))
                        + str((np.random.randint(0, 9)))
                        + ".windows.nyc.hra.nycnet"
                    )
                    ComputerName = md5_hash(ComputerName)

                    syn_data.append(
                        [
                            date_time,
                            date,
                            time,
                            day,
                            Account_Name,
                            ComputerName,
                            Unlocks,
                            Locks,
                            Target,
                        ]
                    )

                elif current_range == 4:
                    # It would generate a data entry with abnormal day of week,
                    # abnormal time, and normal computer
                    day = np.random.choice(abnormal_dotW)
                    date = specific_date_select(df, day)
                    time = np.random.choice(dotW_dict[day]["AbnormalTimes"])
                    date_time = dt.datetime.combine(date, time)
                    ComputerName = np.random.choice(df["ComputerName"].tolist())

                    syn_data.append(
                        [
                            date_time,
                            date,
                            time,
                            day,
                            Account_Name,
                            ComputerName,
                            Unlocks,
                            Locks,
                            Target,
                        ]
                    )

                elif current_range == 5:
                    # It would generate a data entry with normal day of week,
                    # abnormal time, and normal computer
                    day = np.random.choice(normal_dotW)
                    date = specific_date_select(df, day)
                    time = np.random.choice(dotW_dict[day]["AbnormalTimes"])
                    date_time = dt.datetime.combine(date, time)
                    ComputerName = np.random.choice(df["ComputerName"].tolist())

                    syn_data.append(
                        [
                            date_time,
                            date,
                            time,
                            day,
                            Account_Name,
                            ComputerName,
                            Unlocks,
                            Locks,
                            Target,
                        ]
                    )

                elif current_range == 6:
                    # It would generate a data entry with abnormal day of week,
                    # normal time, and normal computer
                    day = np.random.choice(abnormal_dotW)
                    date = specific_date_select(df, day)
                    time = np.random.choice(dotW_dict[day]["NormalTimes"])
                    date_time = dt.datetime.combine(date, time)
                    ComputerName = np.random.choice(df["ComputerName"].tolist())

                    syn_data.append(
                        [
                            date_time,
                            date,
                            time,
                            day,
                            Account_Name,
                            ComputerName,
                            Unlocks,
                            Locks,
                            Target,
                        ]
                    )

                elif current_range == 7:
                    # It would generate a data entry with abnormal day of week,
                    # abnormal time, and abnormal computer
                    day = np.random.choice(abnormal_dotW)
                    date = specific_date_select(df, day)
                    time = np.random.choice(dotW_dict[day]["AbnormalTimes"])
                    date_time = dt.datetime.combine(date, time)
                    ComputerName = (
                        "W151WBWY"
                        + str((np.random.randint(0, 9)))
                        + str((np.random.randint(0, 9)))
                        + "J0"
                        + str((np.random.randint(0, 9)))
                        + str((np.random.randint(0, 9)))
                        + ".windows.nyc.hra.nycnet"
                    )
                    ComputerName = md5_hash(ComputerName)

                    syn_data.append(
                        [
                            date_time,
                            date,
                            time,
                            day,
                            Account_Name,
                            ComputerName,
                            Unlocks,
                            Locks,
                            Target,
                        ]
                    )
            print("Synthetic Data Generating Done..")
            syn_df = pd.DataFrame(syn_data, columns=columns)
            concated_syn_data = pd.concat([df, syn_df], axis=0)
            concated_syn_data = concated_syn_data.sort_values(by="Date_Time")
            concated_syn_data.reset_index(inplace=True, drop=True)
            X[user] = concated_syn_data
            print(f"{user}'s Synthetic Data Has Been Created....!'")

        return X


synData_Pipe = Pipeline(
    [
        ("Generating Synthetic Data and Concatenating it", concat_dfs()),
    ]
)


if __name__ == "__main__":
    try:
        # Load from the bucket
        pre_preprocessed_data = read_pickle(
            "pre_preprocessed_dicts_no_models", s3_client
        )
        if len(pre_preprocessed_data) == 0:
            err_msg = "No pickle file was found"
            ml.write_log(err_msg, 3, "Syndatagen.py", ml.lineno())
            sys.exit(3)
        concated_dicts = synData_Pipe.fit_transform(pre_preprocessed_data.copy())

        # Sanity check for null values
        for user, df in concated_dicts.items():
            print("Does user:", user, "have nulls?", df.isnull().values.any())

            print(
                "Unlock percentage after synthetic data for user:",
                user,
                "is:",
                df["Unlocks"].sum() / (df["Unlocks"].sum() + df["Locks"].sum()),
            )

        # Save to the bucket
        write_pickle(concated_dicts, "concated_syn_dicts", s3_client)

    except Exception as e:
        err_msg = "Error running the request." + format(e)
        print(err_msg)
        ml.write_log(err_msg, 3, "Syndatagen.py", ml.lineno())
        sys.exit(3)
