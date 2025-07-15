import requests
import os
import urllib3
from defusedxml.minidom import parseString
import time
import json
import boto3
from botocore.exceptions import ClientError
import logging
import pandas as pd
import glob
import gc

# For later use
# import sys
# import MLDB as ml

session = requests.Session()

urllib3.disable_warnings()

PYTHONWARNINGS = "ignore"

base_url = "https://splunkcloud_url:8089"

proxies = {"http": "http://proxy_url:8080", "https": "http://proxy_url_2:8080"}

# For Later Use - specify whether the data gets saved locally ("local") or to a "bucket"
save_dest = "local"

# For Later Use - specify whether the output is "parquet" or "csv"
OUTPUT_FILE = "csv"

# save to bucket if save_dest var equals bucket
if save_dest == "bucket":
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
    region_name = os.environ.get("AWS_DEFAULT_REGION")
    bucket_name = os.environ.get("AWS_S3_BUCKET")

    if not all(
        [
            aws_access_key_id,
            aws_secret_access_key,
            endpoint_url,
            region_name,
            bucket_name,
        ]
    ):
        v_msg = "One or more data connection variables are empty. \n"
        v_msg = v_msg + "Please check your data connection to an S3 bucket."
        # ml.write_log(v_msg, 3, "Ingest.py", "ValueError raised")
        raise ValueError(v_msg)

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url,
        region_name=region_name,
    )

# Create Local Directory
os.makedirs("./insider_threat/data/", exist_ok=True)
os.makedirs("./insider_threat/temp/", exist_ok=True)

# Truncation Warning
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


timeframe = "-90d"  # enter the date range/timeframe for the query

# Query 1 - Logon with unlock and lock
logon_query = f"""
        search index=[customized_index]EventCode IN (4624,4672,4800,4801)
        [inputlookup mlusers.csv] earliest={timeframe}
        | eval Timestamp=strftime(_time, "%m/%d/%y %H:%M:%S")
        | eval Activity=case(EventCode == 4624, "Logon", EventCode == 4672,"Privileged Logon",EventCode == 4800, "Lock",EventCode == 4801, "Unlock", true(), "Error")
        | eval Details=name
        | rex field=ComputerName "(?<computername>[^.]+)\.[your_domain]"
        | eval Account_Name = lower(Account_Name)
        | eval ComputerName = upper(computername)
        | dedup Timestamp Account_Name ComputerName Activity Details
        | sort limit=0 _time asc
        | table Timestamp Account_Name ComputerName Activity Details
        """

# Query 2 - Http with web browsing
http_query = f"""
        search index=[customized_index] earliest={timeframe}
        | lookup mlusers.csv Account_Name as user OUTPUT Account_Name
        | search Account_Name=*
        | eval Timestamp=strftime(_time, "%m/%d/%y %H:%M:%S")
        | eval ComputerName=upper(src_nt_host), Details=url
        | eval Activity=
                    case(match(category, ".*(Suspicious|Phishing|Malicious|Proxy Avoidance|Spam|Compromised Sites).*"),"Browsing Hacking Sites",
                        ```Hacking Criteria```
                    like(category, "%Job Search%"),"Browsing Job Search Sites",
                        ```Job Search Criteria```
                        true(),"Browsing Neutral Sites")
                        ```If all else fails (1=1),return Neutral```
        | dedup Timestamp Account_Name ComputerName Activity Details
        | table Timestamp Account_Name ComputerName Activity Details
        """


# Query 3 - File data from CrowdStrike
file_query = f"""
        search index=[customized_index]earliest={timeframe}
            | lookup mlusers.csv Account_Name as user OUTPUT Account_Name
            | search Account_Name=*
            | eval Timestamp=strftime(_time, "%m/%d/%y %H:%M:%S")
            | fillnull value="" *
            | dedup Timestamp Account_Name ComputerName Activity Details
            | table Timestamp Account_Name ComputerName Activity Details
    """

# Query 4 - Device connection data (don't have user field yet)
device_query = f"""
        search index=[customized_index]
        [inputlookup mlusers.csv] earliest={timeframe}
        | eval Timestamp=strftime(_time, "%m/%d/%y %H:%M:%S")
        | search event_simpleName IN ("*DcUsbDeviceDisconnected*","*DcUsbDeviceConnected*")
        | eval Details=mvzip(DeviceProduct,DeviceManufacturer," - ")
        | rename event_simpleName as Activity, aid_computer_name as ComputerName
        | dedup Timestamp Activity ComputerName Details
        | table TimeStamp Activity ComputerName Details
        """

# Query 5 - Categorical data, search on 12 hours basis -> 5 days (need to discuss)
cat_query = """
        search index=[customized_index] earliest="-5d"
        | eval sAMAccountName=lower(sAMAccountName)
        | lookup mlusers.csv Account_Name as sAMAccountName OUTPUT Account_Name
        | search Account_Name != "*$*"
        | eval Role=title, Team=department, Department=company
        | fillnull value="" title department company
        | dedup Account_Name title department company
        | table Account_Name title department company
    """

# Global variable - Ascending order based on the data volume
queries = [
    ("cat_data", cat_query),
    ("logon_data", logon_query),
    ("http_data", http_query),
    ("file_data", file_query),
    ("device_data", device_query),
]


# retrieve auth token from environment variable
def get_token():
    # return os.environ.get("SPLUNK2", "ENV Variable Not Found")

    # local token
    try:
        with open("C://your//directory//token//SPLUNK.txt", "r") as f:
            token = f.readline().strip()
            return token
    except FileNotFoundError:
        print("Error: File not found")
        return None


# POST request to run a search job
def post_request(token, query):
    path = "/services/search/v2/jobs/"
    post_url = base_url + path
    query = query.strip()
    payload = {"search": query}
    response = session.post(
        post_url,
        data=payload,
        headers={"Authorization": f"Bearer {token}"},
        verify=False,
        proxies=proxies,
    )
    print("Returning response......")
    print(f"Status Code: {response.status_code}")
    sid = parseString(response.text).getElementsByTagName("sid")[0].firstChild.nodeValue
    return sid, token


# check job status
def job_status(sid, token):
    done = False
    while not done:
        path = f"/services/search/v2/jobs/{sid}"
        url = base_url + path
        r = session.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            verify=False,
            proxies=proxies,
        )
        response = parseString(r.text)
        for node in response.getElementsByTagName("s:key"):
            if (
                node.hasAttribute("name")
                and node.getAttribute("name") == "dispatchState"
            ):
                dispatchState = node.firstChild.nodeValue
                print("Search Status: ", dispatchState)
                if dispatchState == "DONE":
                    done = True
                elif dispatchState == "FAILED":
                    done = "FAILED"
                    break
                else:  # QUEUED
                    time.sleep(5)
    return done


# For Later Use - write data to json or csv local file
def write_file(data, file_location):
    # write file to json
    if OUTPUT_FILE == "json":
        with open(file=file_location, mode="w", encoding="utf-8") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
    # write file to csv
    elif OUTPUT_FILE == "csv":
        df = pd.DataFrame(data)
        df.to_csv(file_location, index=False)


# For Later Use - write to S3 bucket
def upload_data_to_s3(data, file_location):
    """Upload data to an S3 bucket

    :param data: pass data that is returned from the get_request
    :return: True if file was uploaded, else False
    """
    # Upload the data
    try:
        s3_client.put_object(
            Body=json.dumps(data),
            Bucket=bucket_name,
            Key=file_location,
        )
    except ClientError as e:
        logging.error(e)
        return False
    return True


# Function to create session based on logon type
def create_sessions(df):
    # Sort by Timestamp, Account_Name and ComputerName
    df = df.sort_values(["Account_Name", "ComputerName", "Timestamp"])

    # Find first login
    unlocks = df[
        df["Activity"].isin(
            ["Logon", "Privileged Logon", "Unlock", "UserLoggedIn", "Connected"]
        )
    ].copy()
    locks = df[df["Activity"] == "Lock"].copy()

    locks_renamed = locks.rename(columns={"Timestamp": "lock_time"})
    unlocks_renamed = unlocks.rename(columns={"Timestamp": "unlock_time"})

    # Matched Session _ merge_asof forward would select the first row in the right dataframe whose key is greater than left key
    sessions = pd.merge_asof(
        unlocks_renamed.sort_values("unlock_time"),
        locks_renamed.sort_values("lock_time"),
        by=["Account_Name", "ComputerName"],
        left_on="unlock_time",
        right_on="lock_time",
        direction="forward",
        allow_exact_matches=False,
    )

    # Filter out rows where no lock match was found
    matched_sessions = sessions[sessions["lock_time"].notna()].copy()

    # Ensure that if multiple unlock events match to the same lock_time keep first only
    matched_sessions = matched_sessions.sort_values("unlock_time").drop_duplicates(
        subset=["Account_Name", "ComputerName", "lock_time"], keep="first"
    )

    matched_sessions["duration_ts"] = (
        (matched_sessions["lock_time"] - matched_sessions["unlock_time"])
        .dt.total_seconds()
        .astype(int)
    )
    matched_sessions["day_unlock"] = matched_sessions["unlock_time"].dt.day_name()
    matched_sessions["day_lock"] = matched_sessions["lock_time"].dt.day_name()
    session_df = matched_sessions[
        [
            "Account_Name",
            "ComputerName",
            "unlock_time",
            "lock_time",
            "duration_ts",
            "day_unlock",
            "day_lock",
        ]
    ]

    # Dangling sessions
    unmatched_unlocks = sessions[sessions["lock_time"].isna()].copy()
    unmatched_unlocks = unmatched_unlocks[
        ["Account_Name", "ComputerName", "unlock_time"]
    ]
    unmatched_unlocks["day_unlock"] = unmatched_unlocks["unlock_time"].dt.day_name()

    # Do reverse match: for each lock, find latest unlock BEFORE it
    unlocks_renamed = unlocks.rename(columns={"Timestamp": "unlock_time"})
    locks_renamed = locks.rename(columns={"Timestamp": "lock_time"})
    reverse_matches = pd.merge_asof(
        locks_renamed.sort_values("lock_time"),
        unlocks_renamed.sort_values("unlock_time"),
        by=["Account_Name", "ComputerName"],
        left_on="lock_time",
        right_on="unlock_time",
        direction="backward",
        allow_exact_matches=False,
    )

    unmatched_locks = reverse_matches[reverse_matches["unlock_time"].isna()].copy()
    unmatched_locks = unmatched_locks[["Account_Name", "ComputerName", "lock_time"]]
    unmatched_locks["day_lock"] = unmatched_locks["lock_time"].dt.day_name()

    session_df.to_csv("./insider_threat/temp/session_data.csv", index=False)
    unmatched_unlocks.to_csv("./insider_threat/temp/unmatched_unlocks.csv", index=False)
    unmatched_locks.to_csv("./insider_threat/temp/unmatched_locks.csv", index=False)

    return session_df, unmatched_unlocks, unmatched_locks


# GET request - assign batch size and pagination
def get_request(sid, token):
    path = f"/services/search/v2/jobs/{sid}/results"
    get_url = base_url + path
    headers = {
        "content-type": "application/x-www-form-urlencoded",
        "Authorization": f"Bearer {token}",
    }

    all_data = []
    offset = 0
    batch_size = 50000  # Fetch by chunks

    # Pagination
    while True:
        payload = {
            "output_mode": "json",
            "count": batch_size,
            "offset": offset,
            "exec_mode": "oneshot",
        }

        res = session.get(
            get_url,
            headers=headers,
            params=payload,
            verify=False,
            proxies=proxies,
        )

        batch_data = json.loads(res.text)["results"]

        if not batch_data or len(batch_data) == 0:
            break

        all_data.extend(batch_data)
        print(f"Retrieved {len(batch_data)} records (Total: {len(all_data)})")

        # Handle small size data
        if len(batch_data) < batch_size:
            break
        offset += batch_size

    return all_data


# Function to handle list type data
def flatten_func(df):
    # For some reason it cannot save list data type
    def flatten_list(x):
        if isinstance(x, list):
            return x[1] if len(x) > 0 else ""
        return x

    # List columns then flatten them
    list_cols = [
        col
        for col in df.columns
        if df[col].dtype == "object"
        and any(isinstance(val, list) for val in df[col].dropna())
    ]
    for col in list_cols:
        df[col] = df[col].apply(flatten_list)

    return df


# Run query and save as Parquet and CSV format in case
def run_save_parquet(query_name, query_string, chunk_size=10000):
    print(f"Running {query_name} query…")
    token = get_token()
    if token is None:
        print(f"Could not retrieve token for {query_name}.")
        return False

    try:
        sid, token = post_request(token, query_string)
        status = job_status(sid, token)
        if status == "FAILED":
            print(f"Job failed for {query_name}.")
            return False

        # Data retrieval with chunking
        data = get_request(sid, token)
        if len(data) == 0:
            print(f"No results returned for {query_name}.")
            return False
        print(f"Retrieved {len(data)} results for {query_name}.")

        # Ensure directory path
        os.makedirs("insider_threat/data", exist_ok=True)
        os.makedirs("insider_threat/temp", exist_ok=True)

        # Chunk & flatten
        if len(data) > chunk_size:
            parts = []
            for i in range(0, len(data), chunk_size):
                df_chunk = flatten_func(pd.DataFrame(data[i : i + chunk_size]))
                parts.append(df_chunk)
            df = pd.concat(parts, ignore_index=True)
            del parts
            gc.collect()
        else:
            df = flatten_func(pd.DataFrame(data))

        # Save
        os.makedirs("insider_threat/data", exist_ok=True)
        os.makedirs("insider_threat/temp", exist_ok=True)

        if query_name == "cat_data":
            path = f"insider_threat/temp/{query_name}.csv"
            df.to_csv(path, index=False)
        else:
            path = f"insider_threat/data/{query_name}.parquet"
            try:
                df.to_parquet(path, index=False)
            except Exception:  # In case fail to save in parquet
                df.to_pickle(path.replace(".parquet", ".pkl"))
                path = path.replace(".parquet", ".pkl")
        print(f"Saved {query_name} → {path}")
        return True

    except Exception as error:
        print(f"Error running {query_name} query: {error}")
        return False


# Read all Parquet format files and concatenate as one
def read_combine_file():
    try:
        # Collect data except categorical data
        file_path = glob.glob("./insider_threat/data/*.parquet")
        file_path = [f for f in file_path if "cat_data.parquet" not in f]
        logger.info(f"Found {len(file_path)} parquet files")

        files = []
        for i, file in enumerate(file_path):
            try:
                logger.info(f"Reading file {i+1}/{len(file_path)}: {file}")
                current_file = pd.read_parquet(file)
                logger.info(f"File {file} has {len(current_file)} rows")
                files.append(current_file)

                # Memory Allocation - reset every 2 files
                if len(files) % 2 == 0:
                    gc.collect()

            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
                continue
        if not files:
            raise ValueError("No files were successfully read")

        # Concatenate them by same column name
        logger.info("Concatenating dataframes...")
        combined_df = pd.concat(files, ignore_index=True)
        logger.info(f"Combined dataframe has {len(combined_df)} rows")

        # Sort by ascending order with error handling
        if "Timestamp" in combined_df.columns:
            try:
                combined_df["Timestamp"] = pd.to_datetime(
                    combined_df["Timestamp"],
                    format="%m/%d/%y %H:%M:%S",
                    errors="coerce",
                )
                combined_df = combined_df.sort_values(
                    by="Timestamp", ascending=True
                ).reset_index(drop=True)
                logger.info("Data sorted by timestamp")
            except Exception as e:
                logger.warning(f"Could not sort by timestamp: {e}")
        return combined_df

    except Exception as e:
        logger.error(f"Error on read_and_combine : {e}")
        raise


def main():
    token = get_token()
    if token is None:
        print("Token missing!")
        return "fail"

    # Run Queries
    results = []
    exec_time = []
    for q_name, q_string in queries:
        s = time.time()
        ok = run_save_parquet(q_name, q_string, chunk_size=10000)
        print("Delay 10 to start next query...")
        time.sleep(600)
        elapsed = time.time() - s
        print(f"Elapsed time for {q_name}: {elapsed:.2f} secs")
        print("....Next Query will run")
        results.append(ok)
        exec_time.append((q_name, elapsed))

    os.makedirs("insider_threat/temp", exist_ok=True)
    with open("insider_threat/temp/query_times.txt", "w") as f:
        for name, t in exec_time:
            f.write(f"{name}: {t:.2f} secs\n")

    # Combine and write temporary files for preprocess
    combined_df = read_combine_file()
    session_df, _, _ = create_sessions(combined_df)

    combined_df.to_csv("./insider_threat/temp/combined_data.csv", index=False)
    # In case (over 1M rows)
    combined_df.to_parquet("./insider_threat/temp/combined_data.parquet", index=False)

    print("Gather Data and Generate Session Data Done..")
    print("How Many Sessions Were Created? ", len(session_df))

    # Final return
    return "success"


# Run main function
main()
