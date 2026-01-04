import asyncio
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.custom.credentials.factory import CredentialFactory
from src.custom.connectors.arxivconnector import ArxivConnector
from src.custom.downloader.arxivmetaextractor import ArxivMetaExtractor
from src.custom.downloader.arxivdownloader import ArxivPDFDownloader


def credentials(**kwargs):
    """
    Pull config from Airflow Connection using CredentialFactory
    """
    print("Fetching Arxiv credentials...")
    provider = CredentialFactory.get_provider(
        mode="airflow",
        conn_id="arxiv_api"
    )

    config = provider.get_credentials()
    print("Credentials loaded")
    return config

def fetch_arxiv(ti, **kwargs):
    """
     Get credentials from XCOM
     Call ArxivConnector
    """
    config = ti.xcom_pull(task_ids="credentials_task")

    if not config:
        raise ValueError("No config found in XCom!")

    connector = ArxivConnector(config)
    extractor = ArxivMetaExtractor(connector, config)

    async def run():
        papers = await extractor.fetch_papers(
            max_results=5,
            from_date=None,
            to_date=None
        )
        return papers

    papers = asyncio.run(run())
    print(f"Fetched {len(papers)} papers")

    return papers

def download_pdfs(ti, **kwargs):
    """
    Pull paper list
    Download PDFs
    """
    papers = ti.xcom_pull(task_ids="fetch_task")

    if not papers:
        print("No papers found")
        return

    downloader = ArxivPDFDownloader()

    async def run():
        for p in papers:
            print(f"Downloading: {p['title']}")
            path = await downloader.download(p)
            print(f"Saved: {path}")

    asyncio.run(run())


default_args = {
    "owner": "arxiv",
    "retries": 1,
    "retry_delay": timedelta(minutes=2)
}

with DAG(
    dag_id="arxiv_csai_ingestion",
    default_args=default_args,
    start_date=datetime(2025, 1, 2),
    schedule="@daily",
    catchup=False,
    description="Fetch CS.AI Papers + Download PDFs"
):

    credentials_task = PythonOperator(
        task_id="credentials_task",
        python_callable=credentials
    )

    fetch_task = PythonOperator(
        task_id="fetch_task",
        python_callable=fetch_arxiv
    )

    download_task = PythonOperator(
        task_id="download_task",
        python_callable=download_pdfs
    )

    credentials_task >> fetch_task >> download_task
