from airflow.decorators import dag, task
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow

default_args = {
    'owner': 'Ankit Kumar Sahoo',
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

@dag(
    dag_id='Airflow_Lab1',
    default_args=default_args,
    description='Dag example for Lab 1 of Airflow series',
    start_date=datetime(2026, 1, 15),
    schedule=None,
    catchup=False,
)
def airflow_lab1():

    @task
    def task_load_data():
        return load_data()

    @task
    def task_data_preprocessing(data):
        return data_preprocessing(data)

    @task
    def task_build_save_model(data):
        return build_save_model(data, "model.sav")

    @task
    def task_load_model_elbow(sse):
        return load_model_elbow("model.sav", sse)

    # Pipeline
    data         = task_load_data()
    preprocessed = task_data_preprocessing(data)
    sse          = task_build_save_model(preprocessed)
    task_load_model_elbow(sse)

dag_instance = airflow_lab1()

if __name__ == "__main__":
    dag_instance.test()