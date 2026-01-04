from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

from src.custom.credentials.factory import CredentialFactory
from src.custom.connectors.factory import ConnectorFactory

def credentials(**kwargs):
    """
    Action 1: Fetch raw credentials.
    """
    print("Fetching credentials for healthdb...")
    provider = CredentialFactory.get_provider(mode="airflow", conn_id="healthdb")
    config = provider.get_credentials()
    return config

def connection(ti, **kwargs):
    """
    Action 2: Pull credentials from XCom and use the callable connector.
    """
    # Pulling from the task_id 'credentials_task'
    config = ti.xcom_pull(task_ids='credentials_task')
    
    if not config:
        raise ValueError("No configuration found in XCom!")

    connector = ConnectorFactory.get_connector(
        connector_type="rdbms",
        config=config
    )

    # Since rdbms.py has __call__, we just 'call' the object
    # This runs connect() and returns the engine internally
    connector() 
    
    print("Motive achieved: healthdb connection successful!")

default_args = {
    'owner': 'data_team',
    'retries': 0,
    'retry_delay': timedelta(minutes=2)
}

with DAG(
    'health_data_pipeline',
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["structure", "health"]
) as dag:

    credentials_task = PythonOperator(
        task_id='credentials_task',
        python_callable=credentials,
    )

    connection_task = PythonOperator(
        task_id='connection_task',
        python_callable=connection,
    )

    credentials_task >> connection_task