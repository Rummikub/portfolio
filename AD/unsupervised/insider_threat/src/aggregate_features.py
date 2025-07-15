import pandas as pd
import json
from datetime import datetime

OUTPUT_FILE = "csv"
CHUNK = 100000


# Function to read data to CSV or Parquet
def read_file():
    preprocessed_df = pd.read_csv("insider_threat/temp/combined_data.csv")
    preprocessed_df = preprocessed_df.sort_values(
        by="Timestamp", ascending=True
    ).reset_index(drop=True)

    # Create a mask for rows that match the deduplication criteria
    mask = preprocessed_df["Activity"].isin(
        ["Privileged Logon", "Logon", "Unlock", "UserLoggedIn", "Connected"]
    )

    # Split the dataframe
    rows_to_dedupe = preprocessed_df[mask].copy()
    rows_to_keep = preprocessed_df[~mask].copy()

    # Remove duplicates from the target rows, keeping only the first occurrence
    # based on Timestamp, Account_Name, and ComputerName
    deduplicated_rows = rows_to_dedupe.drop_duplicates(
        subset=["Timestamp", "Account_Name", "ComputerName"], keep="first"
    )

    # Combine the deduplicated rows with rows intended to keep
    result_df = pd.concat([deduplicated_rows, rows_to_keep], ignore_index=True)

    # Sort by timestamp to maintain chronological order
    result_df = result_df.sort_values("Timestamp").reset_index(drop=True)

    session_df = pd.read_csv("insider_threat/temp/session_data.csv")
    print(
        f"Dropped: {len(preprocessed_df)-len(result_df)} rows were dropped due to concurrent log-in logs on combined_df"
    )
    return result_df, session_df


# For Later Use -Function to write data to json or csv local file
def write_file(data, file_location):
    # write file to json
    if OUTPUT_FILE == "json":
        with open(file=file_location, mode="w", encoding="utf-8") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=4)
    # write file to csv
    elif OUTPUT_FILE == "csv":
        df = pd.DataFrame(data)
        df.to_csv(file_location, index=False)


# Function to count frequency to identify each user's primary computer
def preprocess_user_pc(upd):
    pc_counts = (
        upd.groupby(["Account_Name", "ComputerName"]).size().reset_index(name="count")
    )
    primary_pc = pc_counts.sort_values("count", ascending=False).drop_duplicates(
        "Account_Name"
    )[["Account_Name", "ComputerName"]]
    primary_pc = primary_pc.rename(columns={"ComputerName": "primary_pc"})

    upd = pd.merge(upd, primary_pc, on="Account_Name", how="left")
    upd["own_pc"] = (upd["ComputerName"] == upd["primary_pc"]).astype(
        int
    )  # 1 = primary,  0 = other
    upd.drop(columns=["primary_pc"], inplace=True)
    return upd


# Function to convert into readable time format
def format_time(x):
    if pd.notnull(x):
        return (pd.Timestamp("1970-01-01") + x).strftime("%H:%M:%S")
    return ""


# Function to add normal/abnormal working time range
def process_timerange(sdf):
    sdf["unlock_time"] = pd.to_datetime(sdf["unlock_time"])
    sdf["lock_time"] = pd.to_datetime(sdf["lock_time"])

    sdf["day_unlock"] = sdf["unlock_time"].dt.date
    sdf["day_lock"] = sdf["lock_time"].dt.date

    first_unlocks = (
        sdf.groupby(["Account_Name", "day_unlock"])["unlock_time"]
        .min()
        .reset_index(name="first_unlock")
    )
    last_locks = (
        sdf.groupby(["Account_Name", "day_lock"])["lock_time"]
        .max()
        .reset_index(name="last_lock")
    )

    # Edge Case
    first_unlocks["Seconds"] = (
        first_unlocks["first_unlock"].dt.hour * 3600
        + first_unlocks["first_unlock"].dt.minute * 60
        + first_unlocks["first_unlock"].dt.second
    )
    last_locks["Seconds"] = (
        last_locks["last_lock"].dt.hour * 3600
        + last_locks["last_lock"].dt.minute * 60
        + last_locks["last_lock"].dt.second
    )

    # Assign average time pre first lock and last lock as working time range
    start = (
        first_unlocks.groupby("Account_Name")["Seconds"]
        .mean()
        .reset_index(name="start_time")
    )
    end = (
        last_locks.groupby("Account_Name")["Seconds"]
        .mean()
        .reset_index(name="end_time")
    )

    # Merge and Convert to time delta
    sdf = sdf.merge(start, on="Account_Name", how="left")
    sdf = sdf.merge(end, on="Account_Name", how="left")

    sdf["start_time"] = pd.to_timedelta(sdf["start_time"], unit="s")
    sdf["end_time"] = pd.to_timedelta(sdf["end_time"], unit="s")

    # convert into certain time format
    sdf["start_time"] = sdf["start_time"].apply(format_time)
    sdf["end_time"] = sdf["end_time"].apply(format_time)

    return sdf


# Temporal Function to assign workingtime range across all users as 8AM - 7PM
def temp_timerange(sdf):
    if "Unnamed: 0" in sdf.columns:
        sdf = sdf.drop("Unnamed: 0", axis=1)

    sdf["unlock_time"] = pd.to_datetime(sdf["unlock_time"])
    sdf["lock_time"] = pd.to_datetime(sdf["lock_time"])

    sdf["start_time"] = pd.to_datetime(
        sdf["unlock_time"].dt.date.astype(str) + " 8:00:00"
    )
    sdf["end_time"] = pd.to_datetime(sdf["lock_time"].dt.date.astype(str) + " 19:00:00")

    sdf["start_time"] = sdf["start_time"].dt.time
    sdf["end_time"] = sdf["end_time"].dt.time

    sdf["start_time"] = sdf["start_time"].astype(str)
    sdf["end_time"] = sdf["end_time"].astype(str)

    return sdf


# Add or deduct features for preparing train data
def extract_features(df):
    # Convert string to datetime
    df["unlock_time"] = pd.to_datetime(df["unlock_time"])

    # Convert features
    df["logon_hour"] = df["unlock_time"].dt.hour
    df["day_of_a_week"] = df["unlock_time"].dt.dayofweek

    # Drop features
    df.drop(
        [
            "ComputerName",
            "unlock_time",
            "lock_time",
            "day_unlock",
            "day_lock",
            "start_time",
            "end_time",
        ],
        axis=1,
        inplace=True,
    )

    # Drop duplicates
    df.drop_duplicates(ignore_index=True, inplace=True)

    # Features in order
    final_df = df.reindex(
        columns=[
            "Account_Name",
            "duration_ts",
            "logon_hour",
            "day_of_a_week",
            "neutral_sites",
            "job_search",
            "hacking_sites",
            "neutral_sites_off_hour",
            "job_search_off_hour",
            "hacking_sites_off_hour",
            "logon_on_own_pc",
            "logon_on_other_pc",
            "logon_on_own_pc_off_hour",
            "logon_on_other_pc_off_hour",
            "device_connects_on_own_pc",
            "device_connects_on_own_pc_off_hour",
            "device_connects_on_other_pc",
            "device_connects_on_other_pc_off_hour",
            "documents_copy_own_pc",
            "documents_copy_own_pc_off_hour",
            "documents_copy_other_pc",
            "documents_copy_other_pc_off_hour",
            "exe_files_copy_own_pc",
            "exe_files_copy_own_pc_off_hour",
            "exe_files_copy_other_pc",
            "exe_files_copy_other_pc_off_hour",
        ]
    )

    return final_df


# Function to combine categorical data
def add_categorical_data(combined_df):
    # Get categorical data
    cat_df = pd.read_csv("insider_threat/data/cat_data.csv")
    # Groupby and extract first values
    grouped_cat = cat_df.groupby("Account_Name").agg("min").reset_index()
    # Left merge on Account_Name
    final_df = combined_df.merge(grouped_cat, on="Account_Name", how="left")
    # Append Data if Account_Name is matching
    return final_df


# Function to chunking while aggregation
def chunk_and_process(user_df, time_df, chunk_size, user_pc):
    http_results = []
    act_results = []

    total_rows = len(user_df)
    print(f"Processing {total_rows} rows in chunks of {chunk_size}")

    # Process in chunks by rows
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk_num = (start_idx // chunk_size) + 1
        total_chunks = (total_rows + chunk_size - 1) // chunk_size

        print(
            f"Processing chunk {chunk_num}/{total_chunks} (rows {start_idx}-{end_idx})"
        )

        # Get chunk of data
        user_chunk = user_df.iloc[start_idx:end_idx].copy()

        try:
            # Process HTTP web browsing for this chunk
            print(f"  Processing web browsing for chunk {chunk_num}")
            http_chunk = aggregate_web_browsing(user_chunk, time_df)  # Use full time_df
            print("Http Chunk Length:", len(http_chunk))
            if not http_chunk.empty:
                http_results.append(http_chunk)

            # Process file activities for this chunk
            print(f"  Processing file activities for chunk {chunk_num}")
            act_chunk = aggregate_activities(
                user_chunk, time_df, user_pc
            )  # Use full time_df
            print("Act Chunk Length:", len(act_chunk))
            if not act_chunk.empty:
                act_results.append(act_chunk)

        except Exception as e:
            print(f"  Error processing chunk {chunk_num}: {str(e)}")
            continue

        # Clear memory
        del user_chunk

    # Combine results and handle duplicates
    print("Combining results...")

    if http_results:
        http_df = pd.concat(http_results, ignore_index=True)
        print(f"Combined HTTP shape before dedup: {http_df.shape}")

        # Define keys for HTTP aggregation
        http_keys = [
            "Account_Name",
            "ComputerName",
            "unlock_time",
            "lock_time",
            "duration_ts",
            "day_unlock",
            "day_lock",
            "start_time",
            "end_time",
        ]
        http_count_cols = [col for col in http_df.columns if col not in http_keys]

        # Aggregate duplicates
        http_df = http_df.groupby(http_keys)[http_count_cols].sum().reset_index()
        print(f"Combined HTTP shape after dedup: {http_df.shape}")
    else:
        http_df = pd.DataFrame()

    if act_results:
        act_df = pd.concat(act_results, ignore_index=True)
        print(f"Combined ACT shape before dedup: {act_df.shape}")

        # Define keys for Activity aggregation
        act_keys = [
            "Account_Name",
            "ComputerName",
            "unlock_time",
            "lock_time",
            "duration_ts",
            "day_unlock",
            "day_lock",
            "start_time",
            "end_time",
        ]
        act_count_cols = [col for col in act_df.columns if col not in act_keys]

        # Aggregate duplicates
        act_df = act_df.groupby(act_keys)[act_count_cols].sum().reset_index()
        print(f"Combined ACT shape after dedup: {act_df.shape}")
    else:
        act_df = pd.DataFrame()

    return http_df, act_df


# Function for aggregation
def aggregate_web_browsing(user_df, time_df):
    # Ensure timestamp fields are datetime
    user_df["Timestamp"] = pd.to_datetime(user_df["Timestamp"])
    time_df["unlock_time"] = pd.to_datetime(time_df["unlock_time"])
    time_df["lock_time"] = pd.to_datetime(time_df["lock_time"])

    # Convert start_time and end_time to time if stored as strings
    time_df["start_time"] = time_df["start_time"].apply(
        lambda x: datetime.strptime(x, "%H:%M:%S").time() if isinstance(x, str) else x
    )
    time_df["end_time"] = time_df["end_time"].apply(
        lambda x: datetime.strptime(x, "%H:%M:%S").time() if isinstance(x, str) else x
    )

    # Merge user activity with session data (on Account_Name + ComputerName)
    merged = pd.merge(
        user_df,
        time_df,
        on=["Account_Name", "ComputerName"],
        how="left",
        suffixes=("", "_time"),
    )

    # Keep only events within each session
    merged = merged[
        (merged["Timestamp"] >= merged["unlock_time"])
        & (merged["Timestamp"] <= merged["lock_time"])
    ].copy()

    # Drop potential duplicate column
    if "own_pc_time" in merged.columns:
        merged.drop("own_pc_time", axis=1, inplace=True)

    # Extract time portion and compute off-hour flag
    merged["timestamp_time"] = merged["Timestamp"].dt.time
    merged["is_off_hour"] = ~merged.apply(
        lambda r: r["start_time"] <= r["timestamp_time"] <= r["end_time"], axis=1
    )

    # Group by session + off_hour flag
    key_cols = [
        "Account_Name",
        "ComputerName",
        "unlock_time",
        "lock_time",
        "is_off_hour",
    ]
    activity_counts = (
        merged.groupby(key_cols)
        .agg(
            neutral_sites=("Activity", lambda x: (x == "Browsing Neutral Sites").sum()),
            job_search=("Activity", lambda x: (x == "Browsing Job Sites").sum()),
            hacking_sites=("Activity", lambda x: (x == "Browsing Hacking Sites").sum()),
        )
        .reset_index()
    )

    # Separate off-hour into new columns
    for col in ["neutral_sites", "job_search", "hacking_sites"]:
        activity_counts[f"{col}_off_hour"] = (
            activity_counts[col] * activity_counts["is_off_hour"]
        )
        activity_counts[col] = activity_counts[col] * (~activity_counts["is_off_hour"])

    activity_counts.drop(columns=["is_off_hour"], inplace=True)

    # Final group to collapse into one row per session
    final_counts = (
        activity_counts.groupby(
            ["Account_Name", "ComputerName", "unlock_time", "lock_time"]
        )
        .sum()
        .reset_index()
    )

    # Merge counts back onto original session DataFrame
    session_with_counts = pd.merge(
        time_df,
        final_counts,
        on=["Account_Name", "ComputerName", "unlock_time", "lock_time"],
        how="left",
    )

    # Fill in missing values
    count_cols = [
        "neutral_sites",
        "job_search",
        "hacking_sites",
        "neutral_sites_off_hour",
        "job_search_off_hour",
        "hacking_sites_off_hour",
    ]
    for col in count_cols:
        if col not in session_with_counts.columns:
            session_with_counts[col] = 0
    session_with_counts[count_cols] = (
        session_with_counts[count_cols].fillna(0).astype(int)
    )

    return session_with_counts


def aggregate_activities(preprocessed_df, session_df, user_pc):
    # Ensure start_time and end_time are datetime.time
    session_df["start_time"] = pd.to_datetime(
        session_df["start_time"], format="%H:%M:%S", errors="coerce"
    ).dt.time
    session_df["end_time"] = pd.to_datetime(
        session_df["end_time"], format="%H:%M:%S", errors="coerce"
    ).dt.time

    merged_df = pd.merge(
        preprocessed_df,
        session_df,
        on=["Account_Name", "ComputerName"],
        how="left",
    )

    # Ensure datetime conversion
    merged_df["Timestamp"] = pd.to_datetime(merged_df["Timestamp"])
    merged_df["unlock_time"] = pd.to_datetime(merged_df["unlock_time"])
    merged_df["lock_time"] = pd.to_datetime(merged_df["lock_time"])

    # Filter within session, inclusive unlock_time
    merged_df = merged_df[
        (merged_df["Timestamp"] >= merged_df["unlock_time"])
        & (merged_df["Timestamp"] < merged_df["lock_time"])
    ]

    # Extract time from timestamp
    merged_df["timestamp_time"] = merged_df["Timestamp"].dt.time

    # Calculate off_hour
    merged_df["is_off_hour"] = ~merged_df.apply(
        lambda row: row["start_time"] <= row["timestamp_time"] <= row["end_time"],
        axis=1,
    )

    # Handle missing logon sessions
    logon_activities = ["Privileged Logon", "Logon", "UnLock"]
    group_keys = ["Account_Name", "ComputerName", "unlock_time", "lock_time"]

    sessions_with_logon = merged_df[
        merged_df["Activity"].isin(logon_activities)
    ].drop_duplicates(subset=group_keys)
    sessions_all = merged_df.drop_duplicates(subset=group_keys)

    missing_sessions = pd.merge(
        sessions_all,
        sessions_with_logon[group_keys],
        on=group_keys,
        how="left",
        indicator=True,
    ).query('_merge == "left_only"')

    # Use session_df to get own_pc/start/end time
    missing_sessions_full = pd.merge(
        missing_sessions[group_keys], session_df, on=group_keys, how="left"
    )

    # Inject default logon rows
    default_rows = []
    for _, row in missing_sessions_full.iterrows():
        ts_time = row["unlock_time"].time()
        is_off_hour = not (row["start_time"] <= ts_time <= row["end_time"])
        default_rows.append(
            {
                "Account_Name": row["Account_Name"],
                "ComputerName": row["ComputerName"],
                "unlock_time": row["unlock_time"],
                "lock_time": row["lock_time"],
                "Timestamp": row["unlock_time"],
                "Activity": "Logon",
                "Details": "",
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "timestamp_time": ts_time,
                "is_off_hour": is_off_hour,
            }
        )

    if default_rows:
        merged_df = pd.concat(
            [merged_df, pd.DataFrame(default_rows)], ignore_index=True
        )

    # Helper for file suffix match
    def endswith(series, suffixes):
        return series.str.lower().str.endswith(suffixes).fillna(False)

    # Classify Logon Activities
    merged_df["logon_on_own_pc"] = (
        merged_df["Activity"].isin(logon_activities)
        # & (merged_df["own_pc"] == 1)
        & (merged_df["ComputerName"] == merged_df["Account_Name"].map(user_pc))
        & (~merged_df["is_off_hour"])
    ).astype(int)

    merged_df["logon_on_other_pc"] = (
        merged_df["Activity"].isin(logon_activities)
        # & (merged_df["own_pc"] != 1)
        & (merged_df["ComputerName"] != merged_df["Account_Name"].map(user_pc))
        & (~merged_df["is_off_hour"])
    ).astype(int)

    merged_df["logon_on_own_pc_off_hour"] = (
        merged_df["Activity"].isin(logon_activities)
        # & (merged_df["own_pc"] == 1)
        & (merged_df["ComputerName"] == merged_df["Account_Name"].map(user_pc))
        & (merged_df["is_off_hour"])
    ).astype(int)

    merged_df["logon_on_other_pc_off_hour"] = (
        merged_df["Activity"].isin(logon_activities)
        # & (merged_df["own_pc"] != 1)
        & (merged_df["ComputerName"] != merged_df["Account_Name"].map(user_pc))
        & (merged_df["is_off_hour"])
    ).astype(int)

    # Classify Device Activitiess - will update for exact Activity name
    merged_df["device_connects_on_own_pc"] = (
        (merged_df["Activity"] == "PeFileWritten")
        # & (merged_df["own_pc"] == 1)
        & (merged_df["ComputerName"] == merged_df["Account_Name"].map(user_pc))
        & (~merged_df["is_off_hour"])
    ).astype(int)

    merged_df["device_connects_on_other_pc"] = (
        (merged_df["Activity"] == "PeFileWritten")
        # & (merged_df["own_pc"] != 1)
        & (merged_df["ComputerName"] != merged_df["Account_Name"].map(user_pc))
        & (~merged_df["is_off_hour"])
    ).astype(int)

    merged_df["device_connects_on_own_pc_off_hour"] = (
        (merged_df["Activity"] == "PeFileWritten")
        # & (merged_df["own_pc"] == 1)
        & (merged_df["ComputerName"] == merged_df["Account_Name"].map(user_pc))
        & (merged_df["is_off_hour"])
    ).astype(int)

    merged_df["device_connects_on_other_pc_off_hour"] = (
        (merged_df["Activity"] == "PeFileWritten")
        # & (merged_df["own_pc"] != 1)
        & (merged_df["ComputerName"] != merged_df["Account_Name"].map(user_pc))
        & (merged_df["is_off_hour"])
    ).astype(int)

    # Classify File Activities
    merged_df["documents_copy_own_pc"] = (
        endswith(merged_df["Details"], (".doc", ".docx"))
        # & (merged_df["own_pc"] == 1)
        & (merged_df["ComputerName"] == merged_df["Account_Name"].map(user_pc))
        & (~merged_df["is_off_hour"])
    ).astype(int)

    merged_df["documents_copy_other_pc"] = (
        endswith(merged_df["Details"], (".doc", ".docx"))
        # & (merged_df["own_pc"] != 1)
        & (merged_df["ComputerName"] != merged_df["Account_Name"].map(user_pc))
        & (~merged_df["is_off_hour"])
    ).astype(int)

    merged_df["documents_copy_own_pc_off_hour"] = (
        endswith(merged_df["Details"], (".doc", ".docx"))
        # & (merged_df["own_pc"] == 1)
        & (merged_df["ComputerName"] == merged_df["Account_Name"].map(user_pc))
        & (merged_df["is_off_hour"])
    ).astype(int)

    merged_df["documents_copy_other_pc_off_hour"] = (
        endswith(merged_df["Details"], (".doc", ".docx"))
        # & (merged_df["own_pc"] != 1)
        & (merged_df["ComputerName"] != merged_df["Account_Name"].map(user_pc))
        & (merged_df["is_off_hour"])
    ).astype(int)

    merged_df["exe_files_copy_own_pc"] = (
        endswith(merged_df["Details"], (".dll", ".mui", ".exe", ".tmp"))
        # & (merged_df["own_pc"] == 1)
        & (merged_df["ComputerName"] == merged_df["Account_Name"].map(user_pc))
        & (~merged_df["is_off_hour"])
    ).astype(int)

    merged_df["exe_files_copy_other_pc"] = (
        endswith(merged_df["Details"], (".dll", ".mui", ".exe", ".tmp"))
        # & (merged_df["own_pc"] != 1)
        & (merged_df["ComputerName"] != merged_df["Account_Name"].map(user_pc))
        & (~merged_df["is_off_hour"])
    ).astype(int)

    merged_df["exe_files_copy_own_pc_off_hour"] = (
        endswith(merged_df["Details"], (".dll", ".mui", ".exe", ".tmp"))
        # & (merged_df["own_pc"] == 1)
        & (merged_df["ComputerName"] == merged_df["Account_Name"].map(user_pc))
        & (merged_df["is_off_hour"])
    ).astype(int)

    merged_df["exe_files_copy_other_pc_off_hour"] = (
        endswith(merged_df["Details"], (".dll", ".mui", ".exe", ".tmp"))
        # & (merged_df["own_pc"] != 1)
        & (merged_df["ComputerName"] != merged_df["Account_Name"].map(user_pc))
        & (merged_df["is_off_hour"])
    ).astype(int)

    # Aggregate columns
    agg_columns = [
        "logon_on_own_pc",
        "logon_on_other_pc",
        "logon_on_own_pc_off_hour",
        "logon_on_other_pc_off_hour",
        "device_connects_on_own_pc",
        "device_connects_on_other_pc",
        "device_connects_on_own_pc_off_hour",
        "device_connects_on_other_pc_off_hour",
        "documents_copy_own_pc",
        "documents_copy_other_pc",
        "documents_copy_own_pc_off_hour",
        "documents_copy_other_pc_off_hour",
        "exe_files_copy_own_pc",
        "exe_files_copy_other_pc",
        "exe_files_copy_own_pc_off_hour",
        "exe_files_copy_other_pc_off_hour",
    ]

    activity_counts = merged_df.groupby(group_keys)[agg_columns].sum().reset_index()
    session_with_counts = pd.merge_ordered(
        session_df, activity_counts, on=group_keys, how="left"
    )
    session_with_counts.fillna(0, inplace=True)

    return session_with_counts


def main():
    combined_df, session_df = read_file()

    # Add user pc on consolidated data
    user_df = preprocess_user_pc(combined_df)
    time_df = temp_timerange(session_df)

    # Get Dictionary before aggregation
    user_dict = {}
    for user in user_df["Account_Name"].unique():
        user_data = user_df[
            (user_df["Account_Name"] == user) & (user_df["own_pc"] == 1)
        ]
        if not user_data.empty:
            computer_name = user_data["ComputerName"].iloc[0]
            user_dict[user] = computer_name.split("\n")[
                0
            ].strip()  # Take first if 1+ exists

    user_df = user_df.drop("own_pc", axis=1)
    http_df, act_df = chunk_and_process(
        user_df, time_df, chunk_size=100000, user_pc=user_dict
    )

    # Drop duplicates - ensure sanity before merge
    http_df_unique = http_df.drop_duplicates()
    act_df_unique = act_df.drop_duplicates()

    print("Checking for dups before merge:")

    keys = [
        "Account_Name",
        "ComputerName",
        "unlock_time",
        "lock_time",
        "duration_ts",
        "day_unlock",
        "day_lock",
        "start_time",
        "end_time",
    ]

    http_dups = http_df_unique[http_df_unique.duplicated(subset=keys, keep=False)]
    act_dups = act_df_unique[act_df_unique.duplicated(subset=keys, keep=False)]

    if not http_dups.empty:
        print(f"HTTP duplicates found: {len(http_dups)}")
        print(http_dups[keys].head())

    if not act_dups.empty:
        print(f"Activity duplicates found: {len(act_dups)}")
        print(act_dups[keys].head())

    # If duplicates exist, aggregate them
    if not http_dups.empty:
        count_cols_http = [c for c in http_df_unique.columns if c not in keys]
        http_df_unique = (
            http_df_unique.groupby(keys)[count_cols_http].sum().reset_index()
        )

    if not act_dups.empty:
        count_cols_act = [c for c in act_df_unique.columns if c not in keys]
        act_df_unique = act_df_unique.groupby(keys)[count_cols_act].sum().reset_index()

    # Debugging - Clean whitespace before merge
    for col in keys:
        if col in http_df_unique.columns:
            http_df_unique[col] = http_df_unique[col].astype(str).str.strip()
        if col in act_df_unique.columns:
            act_df_unique[col] = act_df_unique[col].astype(str).str.strip()

    # Create preprocessed data
    preprocessed_data = pd.merge(http_df_unique, act_df_unique, on=keys, how="left")

    # Debugging - Make sure count columns into integer and fill -1 if it was missing value
    count_cols = [c for c in preprocessed_data.columns if c not in keys]
    preprocessed_data[count_cols] = preprocessed_data[count_cols].fillna(-1).astype(int)

    # Sneak Peek
    http_df_unique.to_csv("insider_threat/temp/http_transformed_test.csv", index=False)
    act_df_unique.to_csv("insider_threat/temp/act_transformed_test.csv", index=False)
    preprocessed_data.to_csv(
        "insider_threat/temp/comb_preprocessed_test.csv", index=False
    )

    # Debugging - Final check to proceed
    if len(http_df_unique) == len(act_df_unique) == len(preprocessed_data):
        print("Success: total rows of web browsing, activities and session are matched")
    else:
        print(
            f"Fail: web browsing : {len(http_df_unique)} | activities : {len(act_df_unique)} | final : {len(preprocessed_data)} rows"
        )

    # Extract featurs from preprocessed data
    postprocessed_data = extract_features(preprocessed_data)

    # Add categorical data
    final_data = add_categorical_data(postprocessed_data)
    final_data.to_csv("insider_threat/data/train_data.csv", index=False)
    # In Case..
    final_data.to_parquet("insider_threat/data/train_data.parquet", index=False)


main()
