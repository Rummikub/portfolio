# General libs
import pickle
import timeit
import MLDB as ml
import datetime as dt
import pytz
import logging
import sys

# S3 libs
import boto3
from botocore.exceptions import ClientError
import os

# ML Libs for train and fine-tuning
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from pycaret.classification import (
    setup,
    get_config,
    blend_models,
    tune_model,
    finalize_model,
    automl,
    predict_model,
)


# ONNX Libs for converting sklearn model
from onnx.helper import get_attribute_value
from skl2onnx import to_onnx, update_registered_converter
from skl2onnx.common.data_types import (
    Int64TensorType,
    FloatTensorType,
    guess_tensor_type,
)
from skl2onnx._parse import _apply_zipmap, _get_sklearn_operator_name
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost.utils import convert_to_onnx_object


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


""" Although basic scikit-learn models were covered on skl2onnx package,
certain boosters required extra converters and parsers for conversion.
CatBoostClassifier
    > skl2onnx_parser_catboost_classifier : a manual CatBoostClassifier parser
    > skl2onnx_convert_catboost : a manually defined converter
XGBoostClassifier
LGBMClassifier
    > converters and parsers are provided by Onnxmltools package.
These following functions to handle these boosters during conversion."""


# This function is parser matching the catboost classifier format with onnx.
def skl2onnx_parser_catboost_classifier(scope, model, inputs, custom_parsers=None):
    options = scope.get_options(model, dict(zipmap=True))
    no_zipmap = isinstance(options["zipmap"], bool) and not options["zipmap"]
    alias = _get_sklearn_operator_name(type(model))
    this_operator = scope.declare_local_operator(alias, model)
    this_operator.inputs = inputs
    label_variable = scope.declare_local_variable("label", Int64TensorType())
    prob_dtype = guess_tensor_type(inputs[0].type)
    probability_tensor_variable = scope.declare_local_variable(
        "probabilities", prob_dtype
    )
    this_operator.outputs.append(label_variable)
    this_operator.outputs.append(probability_tensor_variable)
    probability_tensor = this_operator.outputs
    if no_zipmap:
        return probability_tensor
    return _apply_zipmap(
        options["zipmap"], scope, model, inputs[0].type, probability_tensor
    )


# This function is converter to convert catboost to onnx format
def skl2onnx_convert_catboost(scope, operator, container):
    """
    CatBoost returns an ONNX graph with a single node.
    This function enable to add certain nodes to the main graph.
    """
    onx = convert_to_onnx_object(operator.raw_operator)
    opsets = {d.domain: d.version for d in onx.opset_import}
    if "" in opsets and opsets[""] >= container.target_opset:
        raise RuntimeError("CatBoost uses an opset recent one than target")
    if len(onx.graph.initializer) > 0 or len(onx.graph.sparse_initializer) > 0:
        raise NotImplementedError(
            "CatBoost returns a model initializers. \
             This option is not implemented yet."
        )
    if (
        len(onx.graph.node) not in (1, 2)
        or not onx.graph.node[0].op_type.startswith("TreeEnsemble")
        or (len(onx.graph.node) == 2 and onx.graph.node[1].op_type != "ZipMap")
    ):
        types = ", ".join(map(lambda n: n.op_type, onx.graph.node))
        raise NotImplementedError(
            f"CatBoost returns {len(onx.graph.node)} != 1 (types={types}). "
            f"This option is not implemented yet."
        )
    node = onx.graph.node[0]
    atts = {}
    for att in node.attribute:
        atts[att.name] = get_attribute_value(att)
    container.add_node(
        node.op_type,
        [operator.inputs[0].full_name],
        [operator.outputs[0].full_name, operator.outputs[1].full_name],
        op_domain=node.domain,
        op_version=opsets.get(node.domain, None),
        **atts,
    )


"""
This code is for registering customized converters which were assigned above.
ONNX format will support complicated network design,
by generating computational graph, where nodes represent operations,
and edges represent the flow of data.
Each node corresponds to a specific operation, such as convolution, pooling,
or activation, and includes attributes that define its behavior.
"""

for classifier, name, convert_func in [
    (CatBoostClassifier, "CatBoostClassifier", skl2onnx_convert_catboost),
    (XGBClassifier, "XGBoostXGBClassifier", convert_xgboost),
    (LGBMClassifier, "LGBMClassifier", convert_lightgbm),
]:
    update_registered_converter(
        classifier,
        name,
        calculate_linear_classifier_output_shapes,
        convert_func,
        parser=skl2onnx_parser_catboost_classifier,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )


def pycaret_model(X):
    """The purpose of this function is to reduce the running time on modeling,
    while implementing PyCaret classification libray.
    This process includes z-score feature scaling, cross validation,
    ensemble methods and evaluation metrics on precision, recall, f1-score.
    Args:
        X: Dataframe of all post_pre_processed dataframe inside dictionary.

    Returns:
        best_models: dictionary of best model and its best average f1 score.
        model_metrics: dictionary of scores on precision, recall, f1.
    """
    model_metrics, best_models = {}, {}
    for user, df in X.items():
        user_best_model = None
        best_model_score = -9999999
        df = df.drop(
            [
                "Date_Time",
                "Date",
                "Time",
                "day",
                "Account_Name",
                "ComputerName",
                "Unlocks",
                "Locks",
            ],
            axis=1,
        )
        # Convert column names fn format
        columns = [df.columns.values]
        target_n = f"f{df.columns.get_loc('Target')}"
        df.columns = columns
        df.columns = [f"f{i}" for i in range(len(df.columns))]
        """Data will be splitted as 95% of train data 5% of validate data.
        > train_data : data to train pycaret model splitted 70/30 as train/test
        > val_data : data to retrieve evaluation metrics for validation."""
        train_data, val_data = train_test_split(
            df, train_size=0.95, test_size=0.05, random_state=30
        )
        exp = setup(
            data=train_data,
            target=target_n,
            session_id=30,
            normalize=True,
            fold=5,
            n_jobs=None,
        )
        X_sample = get_config("X_train")[:1]
        best_baseline = exp.compare_models(
            verbose=False, sort="Precision", n_select=3, exclude=["qda", "gbc"]
        )
        blender = blend_models(estimator_list=best_baseline, method="hard")
        tuned_model = tune_model(blender)
        finalize_model(tuned_model)
        user_best_model = automl(optimize="f1")
        """
        Troubleshoot while ONNX conversion
        > Votingclassifier : default onnx converter can't handle
                            flatten_transform=True, set parameter as False
        > XGBoostClassifier : converter requires certain format in features.
                            For example, can't read string format featurenames,
                            read only 0,1,2 or f0,f1,f2.
                            In that regard, convert 'Target' column to 'fn'.
        """
        if (
            _get_sklearn_operator_name(type(user_best_model))
            == "SklearnVotingClassifier"
        ):
            # Set parameter flatten_transform as False
            user_best_model = user_best_model.set_params(flatten_transform=False)
            user_best_model = user_best_model.set_params(weights=None)

        else:
            print(
                f"{user} best model is \
                {_get_sklearn_operator_name(type(user_best_model))}"
            )

        # Convert to ONNX format
        model_onnx = to_onnx(
            user_best_model,
            initial_types=[("X", FloatTensorType([None, X_sample.shape[1]]))],
            target_opset={"": 15, "ai.onnx.ml": 3},
            options={id(user_best_model): {"zipmap": False}},
        )

        # Model.py
        pred_holdout = predict_model(user_best_model)
        best_model_score = f1_score(
            pred_holdout[target_n], pred_holdout["prediction_label"]
        )
        # Predict.py
        pred_unseen = predict_model(user_best_model, data=val_data)
        user_precision = precision_score(
            pred_unseen[target_n], pred_unseen["prediction_label"]
        )
        user_recall = recall_score(
            pred_unseen[target_n], pred_unseen["prediction_label"]
        )
        user_f1_score = f1_score(pred_unseen[target_n], pred_unseen["prediction_label"])

        # Revert column names as normal for less confusion
        train_data.columns, val_data.columns = columns, columns

        # Sanity check1
        best_models[user] = {
            "best_model": model_onnx,
            "model_score": best_model_score,
        }

        # Score.py
        model_metrics[user] = {
            "precision": user_precision,
            "recall": user_recall,
            "f1_score": user_f1_score,
        }
    return best_models, model_metrics


if __name__ == "__main__":
    try:
        # Load from the bucket
        post_preprocessed_data = read_pickle(
            "post_preprocessed_dicts_no_models", s3_client
        )
        if len(post_preprocessed_data) == 0:
            err_msg = "No pickle file was found"
            ml.write_log(err_msg, 3, "Train.py", ml.lineno())
            sys.exit(3)

        start_timeit = timeit.default_timer()
        user_best_models, user_model_metrics = pycaret_model(
            post_preprocessed_data.copy()
        )
        print(timeit.default_timer() - start_timeit)

        # Sanity check 1
        for user, model in user_best_models.items():
            print(
                "For",
                user,
                "their best model is:",
                model["best_model"],
                "and their average prediction score is:",
                model["model_score"],
            )

        # Sanity check 2
        for user, model_score in user_model_metrics.items():
            print(
                "For",
                user,
                "their best model scored on precision",
                model_score["precision"],
            )
            print(user, "'s best model scored on recall", model_score["recall"])
            print(user, "'s best model scored on f1", model_score["f1_score"], "\n")

        # Save models as a pickle file
        models_to_save = {}
        for user, df in user_best_models.items():
            current_user_model = df["best_model"]
            models_to_save[user] = current_user_model

        # Save to the bucket
        write_pickle(models_to_save, "user_model_dicts", s3_client)
        write_pickle(user_model_metrics, "model_scores", s3_client)

    except Exception as e:
        err_msg = "Error running the request." + format(e)
        print(err_msg)
        ml.write_log(err_msg, 3, "Train.py", ml.lineno())
        sys.exit(3)
