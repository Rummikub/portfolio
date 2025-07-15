import requests
import os
import urllib3
from defusedxml.minidom import parseString
import time
import json
import datetime as dt
import boto3
from botocore.exceptions import ClientError
import logging
import pandas as pd
import sys
import MLDB as ml
import pytz

session = requests.Session()

urllib3.disable_warnings()

PYTHONWARNINGS = "ignore"
# Splunk on-prem load balancer
base_url = "https://base_url:8089"

# specify whether the data gets saved locally ("local") or to a "bucket"
save_dest = "bucket"

# specify whether the output is "json" or "csv"
output_file = "json"

# specify file name and location
timestamp = dt.datetime.now(pytz.timezone("America/New_York")).strftime("%m%d%y_%H")

# assign the file path
file_location = f"data/splunkdata{timestamp}.{output_file}"

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
        ml.write_log(v_msg, 3, "Ingest.py", "ValueError raised")
        raise ValueError(v_msg)

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url,
        region_name=region_name,
    )

# query to check if the lookup tables are duplicating data
verify_query = """
    | inputlookup ad_hashk.csv
        | rename Lan_ID as lookup_field
        | eval source="ad_hashk.csv"
    | append
        [| inputlookup ad_asset_hash.csv
    | rename ComputerName as lookup_field
    | eval source ="ad_asset_hash.csv"]
    | stats count as source_count values(source) as source by lookup_field
    | where source_count>1
        """

# Splunk query to pull lock and unlock data
timeframe = "-2h"  # enter the date range/timeframe for the query
data_query = f"""
        search index=[customized-index] EventCode IN (4801,4800)
        [inputlookup mlusers.csv] earliest={timeframe}
        | eval Date=strftime(_time,"%m/%d/%y")
        | eval Time=strftime(_time,"%H:%M:%S")
        | eval day=strftime(_time,"%A")
        | rex field=ComputerName "(?<computername>[^.]+)\[personal/domain]"
        | eval Account_Name = lower(Account_Name)
        | eval ComputerName = upper(computername)
        | lookup ad_hashk.csv Lan_ID AS Account_Name OUTPUTNEW user_hash
        | lookup ad_asset_hash.csv ComputerName OUTPUTNEW nt_host_MD5
        | rename user_hash AS Account_Name, nt_host_MD5 AS ComputerName
        | dedup Date Time day Account_Name ComputerName EventCode name
        | table Date Time day Account_Name ComputerName EventCode name
        """


# retrieve auth token from environment variable
def get_token():
    return os.environ.get("SPLUNK", "ENV Variable Not Found")


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
        r = session.get(url, headers={"Authorization": f"Bearer {token}"}, verify=False)
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
                else:
                    print("Search Status: ", dispatchState)
                    time.sleep(5)
    return done


# GET request
def get_request(sid, token):
    path = f"/services/search/v2/jobs/{sid}/results"
    get_url = base_url + path
    headers = {
        "content-type": "application/x-www-form-urlencoded",
        "Authorization": f"Bearer {token}",
    }

    payload = {"output_mode": "json", "count": 0}
    res = session.get(get_url, headers=headers, params=payload, verify=False)
    data = json.loads(res.text)["results"]

    return data


# write data to json or csv local file
def write_file(data):
    # write file to json
    if output_file == "json":
        with open(file=file_location, mode="w", encoding="utf-8") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
    # write file to csv
    elif output_file == "csv":
        df = pd.DataFrame(data)
        df.to_csv(file_location, index=False)


# write to S3 bucket
def upload_data_to_s3(data):
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


# main function for all functions
def main():
    """
    This is a driver function the runs the above get and post request functions
    with error checking and handling built-in.
    If any step of the process is false, fails, or returns an error then the
    function stops running.
    """
    token = get_token()
    # exit if token can't be found
    if token == "ENV Variable Not Found":
        ml.write_log(str(token), 3, "Ingest.py", ml.lineno())
    else:
        print("Token retrieved...")
        # run the rest of the functions if token is retrieved
        try:
            # verify that the lookup tables have unique data
            sid, token = post_request(token, verify_query)
            print("Running lookup table integriy check...")
            status = job_status(sid, token)
            # if the job failed, then stop running the rest of the function
            if status == "FAILED":
                f_msg = "Job failed. Please review the code and try again."
                ml.write_log(f_msg, 4, "Ingest.py", ml.lineno())
            elif status:
                data = get_request(sid, token)
                # if zero results, then lookup tables are unique
                if len(data) == 0:
                    print("Lookup tables have no issues.")
                    print("                             ")
                    # post the query for the lock/unlock data
                    print("Running unlock/lock data query...")
                    sid, token = post_request(token, data_query)
                    status = job_status(sid, token)
                    # if the job failed, then stop running the function
                    if status == "FAILED":
                        f_msg = "Job failed. Please review "
                        f_msg = f_msg + "the code and try again."
                        ml.write_log(f_msg, 3, "Ingest.py", ml.lineno())
                        sys.exit(3)
                    elif status:
                        data = get_request(sid, token)
                        # if zero results are returned, then exit
                        if len(data) == 0:
                            n_msg = "There were no results returned. "
                            n_msg = n_msg + "Please check the query."
                            ml.write_log(n_msg, 2, "Ingest.py", ml.lineno())
                            sys.exit(2)
                        print("Data retrieved.")
                        # save results to bucket
                        capture = ml.get_all("users", "username")
                        if capture == []:
                            ml.write_log(capture, 2, "Ingest.py", ml.lineno())
                            sys.exit(2)
                        for item in data:
                            p_user = item["Account_Name"]
                            # if not p_user in capture:
                            if p_user not in capture:
                                _, msg = ml.new_user(p_user, "", "")
                                capture.append(p_user)
                                ml.write_log(msg, 1, "Ingest.py", ml.lineno())
                        if save_dest == "bucket":
                            result = upload_data_to_s3(data)
                            if result:
                                res_msg = "Data was successfully "
                                res_msg = res_msg + "uploaded to "
                                res_msg = res_msg + str(bucket_name)
                                res_msg = res_msg + "."
                                ml.write_log(res_msg, 1, "Ingest.py", ml.lineno())
                            else:
                                res_msg = "The data was not uploaded to"
                                res_msg = res_msg + bucket_name + "."
                                ml.write_log(res_msg, 2, "Ingest.py", ml.lineno())
                        # save results to a local file
                        elif save_dest == "local":
                            write_file(data)
                            wf_msg = "Data saved to "
                            wf_msg = wf_msg + file_location + "."
                            ml.write_log(wf_msg, 1, "Ingest.py", ml.lineno())
                else:
                    tmsg = "Lookup tables have duplicates. Exiting..."
                    ml.write_log(tmsg, 2, "Ingest.py", ml.lineno())
                    sys.exit(2)
                print("Ingest process completed")
                ml.write_log("Ingest succeeded", 1, "Ingest.py", "")
            else:
                msg = "Lookup tables have duplicates. Exiting..."
                ml.write_log(msg, 2, "Ingest.py", ml.lineno())
        except Exception as error:
            ex_msg = "Error running the request." + format(error)
            print(ex_msg)
            ml.write_log(ex_msg, 3, "Ingest.py", ml.lineno())


# run main function
main()
