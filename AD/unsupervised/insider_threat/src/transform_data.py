import pandas as pd
import logging
import glob
import os
import gc

# Ensure existing local direcory
os.makedirs("./insider_threat/data/", exist_ok=True)
os.makedirs("./insider_threat/temp/", exist_ok=True)

# Truncation Warning
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Read all Parquet format files and concatenate as one
def read_combine_file():
    try:
        # Collect data except categorical data
        file_path = glob.glob("./insider_threat/data/*.csv")
        file_path = [f for f in file_path if "cat_data.csv" not in f]
        logger.info(f"Found {len(file_path)} parquet files")

        files = []
        for i, file in enumerate(file_path):
            try:
                logger.info(f"Reading file {i+1}/{len(file_path)}: {file}")
                #current_file = pd.read_parquet(file)
                current_file = pd.read_csv(file)
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

    # Matched Session
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


def main():
    # Combine and write temporary files for preprocess
    combined_df = read_combine_file()
    session_df, _, _ = create_sessions(combined_df)

    # In case (over 1M rows)
    combined_df.to_parquet(
        "./insider_threat/temp/combined_data.parquet", index=False
    )
    combined_df.to_csv("./insider_threat/temp/combined_data.csv", index=False)

    print("Gather Data and Generate Session Data Done..")
    print("How Many Sessions Were Created? ", len(session_df))

    # Final return
    return "success"


main()