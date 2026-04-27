"""Airflow DAG for agricultural data ingestion."""

from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.sensors.http import HttpSensor

default_args = {
    "owner": "agri_pipeline",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


def fetch_weather_data(**context: Any) -> dict[str, Any]:
    """Fetch weather data from IMD/OpenWeather API."""
    # Placeholder for actual API integration
    return {"status": "success", "records_fetched": 100}


def fetch_satellite_data(**context: Any) -> dict[str, Any]:
    """Fetch satellite imagery metadata from Sentinel-2."""
    # Placeholder for actual API integration
    return {"status": "success", "scenes_available": 25}


def fetch_market_prices(**context: Any) -> dict[str, Any]:
    """Fetch market prices from e-NAM API."""
    # Placeholder for actual API integration
    return {"status": "success", "markets_updated": 15}


def validate_ingested_data(**context: Any) -> dict[str, Any]:
    """Run Great Expectations validation on ingested data."""
    # Placeholder for GE validation
    return {"validation_status": "passed", "expectations_run": 42}


def publish_to_kafka(**context: Any) -> dict[str, Any]:
    """Publish validated data to Kafka topics."""
    # Placeholder for Kafka producer
    return {"status": "published", "messages_sent": 150}


with DAG(
    dag_id="agri_ingest_weather",
    default_args=default_args,
    description="Ingest weather data and publish to Kafka",
    schedule_interval="0 */6 * * *",  # Every 6 hours
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion", "weather", "kafka"],
) as dag:

    wait_for_weather_api = HttpSensor(
        task_id="wait_for_weather_api",
        http_conn_id="imd_api",
        endpoint="/weather/latest",
        poke_interval=300,
        timeout=3600,
    )

    fetch_weather = PythonOperator(
        task_id="fetch_weather_data",
        python_callable=fetch_weather_data,
    )

    validate_weather = PythonOperator(
        task_id="validate_weather_data",
        python_callable=validate_ingested_data,
    )

    publish_weather = PythonOperator(
        task_id="publish_weather_to_kafka",
        python_callable=publish_to_kafka,
    )

    wait_for_weather_api >> fetch_weather >> validate_weather >> publish_weather


with DAG(
    dag_id="agri_ingest_satellite",
    default_args=default_args,
    description="Ingest satellite imagery metadata and publish to Kafka",
    schedule_interval="0 12 * * *",  # Daily at noon
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion", "satellite", "kafka"],
) as dag2:

    check_satellite_availability = HttpSensor(
        task_id="check_satellite_availability",
        http_conn_id="sentinel_api",
        endpoint="/products/search",
        poke_interval=600,
        timeout=7200,
    )

    fetch_satellite = PythonOperator(
        task_id="fetch_satellite_metadata",
        python_callable=fetch_satellite_data,
    )

    validate_satellite = PythonOperator(
        task_id="validate_satellite_data",
        python_callable=validate_ingested_data,
    )

    publish_satellite = PythonOperator(
        task_id="publish_satellite_to_kafka",
        python_callable=publish_to_kafka,
    )

    check_satellite_availability >> fetch_satellite >> validate_satellite >> publish_satellite


with DAG(
    dag_id="agri_ingest_market_prices",
    default_args=default_args,
    description="Ingest market prices from e-NAM and publish to Kafka",
    schedule_interval="0 14 * * *",  # Daily at 2 PM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ingestion", "market", "kafka"],
) as dag3:

    fetch_market = PythonOperator(
        task_id="fetch_market_prices",
        python_callable=fetch_market_prices,
    )

    validate_market = PythonOperator(
        task_id="validate_market_data",
        python_callable=validate_ingested_data,
    )

    publish_market = PythonOperator(
        task_id="publish_market_to_kafka",
        python_callable=publish_to_kafka,
    )

    fetch_market >> validate_market >> publish_market
