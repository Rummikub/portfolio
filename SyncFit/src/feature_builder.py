import pandas as pd

def build_user_features(path="data/synthetic_wearable_logs.csv"):
    df = pd.read_csv(path, parse_dates=['date'])

    # Aggregate features per user
    agg_funcs = {
        'steps': ['mean', 'std', 'min', 'max'],
        'heart_rate': ['mean', 'max'],
        'class_attended': 'sum',
        'churned': 'max'  # label
    }

    user_df = df.groupby('user_id').agg(agg_funcs)
    user_df.columns = ['_'.join(col) for col in user_df.columns]
    user_df = user_df.reset_index()

    # Drop both user_id and the target column
    X = user_df.drop(columns=['user_id', 'churned_max'])
    y = user_df['churned_max']

    return X, y

if __name__ == "__main__":
    build_user_features()
