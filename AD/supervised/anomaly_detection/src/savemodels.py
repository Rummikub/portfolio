# General Libs
import pickle
import os
import MLDB as ml
import datetime as dt
import pytz
import logging
import sys

# S3 Libs
import boto3
from botocore.exceptions import ClientError

# ONNX Lib
import onnx

# S3 Credentials
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
region_name = os.environ.get("AWS_DEFAULT_REGION")
bucket_name = os.environ.get("AWS_S3_BUCKET")

# Create Client for bucket connection
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


def save_model_in_DB(model):
    res, nsg = ml.update_model(model, "G", "status")


# Model Type Check
def model_type_checker(models):
    if not isinstance(models, onnx.ModelProto):
        raise TypeError(f"Error: Not in ONNX format, got{type(models)}")
    else:
        print("It is in ONNX format")


# Metadata Check
def version_checker(models):
    print(f"Model IR version    :{models.ir_version}")
    print(f"Opset import version: {models.opset_import[0].version}", "\n")


def save_models_to_s3(user_models, model_scores):
    """
    Saves a model for each user inside the specified root directory,
     if it satisfied the following conditions.
        Precision score >= 0.85
        Recall Score >= 0.85
        F1-score >= 0.85
    Args:
        user_models: a dictionary contains the keys, user, and best models.
        model_scores: a dictionary with users and evaluation metrics as values.
    """
    for user, models in user_models.items():
        for user_scores, m_scores in model_scores.items():
            if user == user_scores:
                if (
                    (m_scores["precision"] >= 0.85)
                    & (m_scores["recall"] >= 0.85)
                    & (m_scores["f1_score"] >= 0.85)
                ):
                    file_key = f"models/{user}/{user}.onnx"
                    print(f"Model for {user}")
                    try:
                        s3_client.put_object(
                            Body=models.SerializeToString(),
                            Bucket=bucket_name,
                            Key=file_key,
                        )
                        # Verify model with value
                        model_type_checker(models)
                        # Check ModelProto metadata
                        version_checker(models)
                        save_model_in_DB(user)
                        m_id = ml.get_id("models", user)
                        if m_id == -1:
                            # id not found
                            ret_val, msg = ml.new_user(user, user, "")
                            if ret_val != 1:
                                ml.write_log(msg, 2, "Save_models.py", "")
                            else:
                                ml.write_log(msg, 1, "Save_models.py", "")
                                m_id = ml.get_id("models", user)
                        ret_val, msg = ml.update_model(m_id, "G", "status")
                        if ret_val == 1:
                            ml.write_log(msg, 1, "Save_models.py", "")

                    except ClientError as e:
                        raise print(f"S3 upload error: {user}" + format(e))

                else:
                    print(f"Did not save model for {user}")


if __name__ == "__main__":
    try:
        # Load from the bucket
        model_dct = read_pickle("user_model_dicts", s3_client)
        model_scores_dct = read_pickle("model_scores", s3_client)

        if (len(model_dct) == 0) or (len(model_scores_dct) == 0):
            err_msg = "No pickle file was found"
            ml.write_log(err_msg, 3, "Savemodels.py", ml.lineno())
            sys.exit(3)

        # Save to the bucket
        save_models_to_s3(model_dct, model_scores_dct)

    except Exception as e:
        err_msg = "Error running the request." + format(e)
        print(err_msg)
        ml.write_log(err_msg, 3, "Savemodels.py", ml.lineno())
        sys.exit(3)
