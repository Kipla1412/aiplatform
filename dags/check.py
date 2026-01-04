import pendulum
from airflow.decorators import dag, task

# 1. Define the DAG using the @dag decorator
@dag(
    dag_id="hello_world_airflow_3",
    schedule="@daily",              # Runs once every day
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,                  # Don't run historical dates
    tags=["example", "learning"],
)
def my_first_dag():

    # 2. Define tasks as regular Python functions
    @task()
    def get_name():
        return "Umar"

    @task()
    def say_hello(name: str):
        print(f"Hello, {name}! Welcome to Airflow 3.1.5.")
        return f"Greetings sent to {name}"

    # 3. Set dependencies by simply calling the functions
    name_value = get_name()
    say_hello(name_value)

# 4. Instantiate the DAG
my_first_dag()