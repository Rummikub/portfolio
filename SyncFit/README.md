SyncFit is a behavioral analytics simulation platform that models user engagement, class attendance, and churn risk based on wearable sensor data. This project was designed to mirror real-world health analytics platforms like Equinox+ or Apple Fitness.


##  Features

- Synthetic wearable data generator (steps, heart rate, attendance, churn)
- Session/user feature aggregation
- Churn prediction using XGBoost
- Streamlit dashboard for prediction & insight
- Airflow DAG for orchestration


## Project Structure

syncfit/
├── data/ # Contains synthetic_wearable_logs.csv
├── src/
│ ├── data_simulator.py # Simulate fitness logs
│ ├── feature_builder.py # Feature aggregation per user
│ ├── churn_model.py # XGBoost training
├── dashboard/app.py # Streamlit app
├── dags/syncfit_dag.py # Airflow DAG
└── models/ # Saved .pkl model