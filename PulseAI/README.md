PulseAI is a real-time user behavioral intelligence engine

It simulates, processes, and analyzes user session data. It's designed to replicate how major companies might track and respond to user behavior in real time.


This project demonstrates:

- data streaming simulation
- feature extraction from event logs
- real-time ML model integration
- data visualization
- ETL orchestration using Apache Airflow

> Project Goal
The goal of this project is to detect and describe patterns in user behavior as they haapen. 

Related Usecases:
- Real-time churn risk modeling
- Session-based recommendations
- Usage trend alerts
- Feature personalization

`data_streaming.py` simulates user behavior (clicks, searches, scrolls, etc) in real time. 

`feature_extraction.py` processes the raw event to extract signals

`model_realtime.py` baseline model as a simple PyTorch LSTM model that could be used to:
- Detect behavior drift
- Predict next action or engagement score

> Note: model training pipeline is not fully implemented yet but it's a good starter.

`pulseai_dag.py` for utilizing Apache AirFlow DAG that simulate event streaming, converts into feature vecotrs and a lot more.

`dashboard/app.py` creates dashboard that displays the latest processed user features, providing real-time chart and session activity