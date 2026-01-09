import asyncio
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from pathlib import Path
from src.custom.credentials.factory import CredentialFactory
from src.custom.connectors.factory import ConnectorFactory
from src.custom.downloader.arxivmetaextractor import ArxivMetaExtractor
from src.custom.downloader.arxivdownloader import ArxivPDFDownloader
from src.custom.extractors.pdfparserservice import PDFParserService
from src.custom.chunker.arxivchunker import TextChunker

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

    connector = ConnectorFactory.get_connector(
        connector_type="arxiv",
        config=config
    )
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
    config = ti.xcom_pull(task_ids="credentials_task")
    if not papers:
        print("No papers found")
        return

    downloader = ArxivPDFDownloader(config)

    async def run():
        downloaded = []
        for p in papers:
            print(f"Downloading: {p['title']}")
            path = await downloader.download(p)
            print(f"Saved: {path}")
            downloaded.append(str(path))  
        return downloaded

    return asyncio.run(run())
   

def parse_pdfs(ti, **kwargs):
    """
    Parse all downloaded PDFs in parallel
    """
    config = ti.xcom_pull(task_ids="credentials_task")
    pdf_paths = ti.xcom_pull(task_ids="download_task")

    if not pdf_paths:
        print("No PDF paths received from downloader")
        return []

    # parser_service= ExtractorFactory.get_extractor(
    #     extractor_type ="arxivparser",
    #     connection = None,
    #     config=config
    # )
    parser_service = PDFParserService(config)

    print("Parser reading dir:", parser_service.input_dir)
    print("Files to parse:", pdf_paths)

    async def run():
        results = []
        for path in pdf_paths:
            res = await parser_service.parse_pdf(Path(path))
            if res:
                results.append(res)
            return results

        parsed_docs = asyncio.run(run())

        print(f"Parsed {len(parsed_docs)} PDFs")
        return parsed_docs

        # results = await parser_service.parse_all_pdfs()
        # print(f"Parsed {len(results)} PDFs")

def chunks_pdfs(ti, **kwargs):

    """
    Chunk parsed PdfContent into TextChunks
    """
    parsed_docs = ti.xcom_pull(task_ids="parse_task")

    if not parsed_docs:
        print("No parsed documents received")
        return []

    chunker = TextChunker(
        chunk_size=500,
        overlap_size=100,
        min_chunk_size=120,
    )

    all_chunks = []

    for pdf in parsed_docs:
        chunks = chunker.chunk_pdf(pdf)
        print(f"PDF {pdf.metadata.get('arxiv_id')} -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks



default_args = {
    "owner": "arxiv",
    "retries": 1,
    "retry_delay": timedelta(minutes=2)
}

with DAG(
    dag_id="arxiv_csai_ingestion",
    default_args=default_args,
    start_date=datetime(2026, 1, 5),
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

    parse_task = PythonOperator(
        task_id="parse_task",
        python_callable=parse_pdfs
    )

    chunk_task = PythonOperator(
    task_id="chunk_task",
    python_callable=chunk_pdfs
    )

    credentials_task >> fetch_task >> download_task >> parse_task >> chunk_task
