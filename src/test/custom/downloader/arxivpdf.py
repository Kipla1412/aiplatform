import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from src.custom.downloader.arxivpdf import ArxivPDFDownloader
from src.custom.credentials.localsettings.pdfconfig import ArxivPDFConfig      

pytestmark = pytest.mark.asyncio


def build_downloader(tmp_path):
    cfg = ArxivPDFConfig(
        download_dir=str(tmp_path),
        timeout_seconds=5,
        rate_limit_delay=1,
        max_retries=3,
        retry_backoff=1,
    )
    return ArxivPDFDownloader(cfg)


SAMPLE_PAPER = {
    "arxiv_id": "1234.5678v1",
    "pdf_url": "http://arxiv.org/pdf/1234.5678v1.pdf"
}

@patch("httpx.AsyncClient")
async def test_download_success(mock_client, tmp_path):
    dl = build_downloader(tmp_path)

    class FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        async def aiter_bytes(self):
            yield b"chunk1"
            yield b"chunk2"

    client_instance = mock_client.return_value
    client_instance.__aenter__.return_value = client_instance
    client_instance.stream.return_value = FakeStream()

    path = await dl.download(SAMPLE_PAPER)

    assert path is not None
    assert path.exists()
    assert path.read_bytes() == b"chunk1chunk2"


@patch("httpx.AsyncClient")
async def test_download_uses_cache(mock_client, tmp_path):
    dl = build_downloader(tmp_path)

    file_path = tmp_path / "1234.5678v1.pdf"
    file_path.write_text("already here")

    path = await dl.download(SAMPLE_PAPER)

    # Should NOT call http
    mock_client.assert_not_called()

    assert path == file_path
    assert path.read_text() == "already here"

async def test_download_missing_fields(tmp_path):
    dl = build_downloader(tmp_path)

    result = await dl.download({"arxiv_id": "x"})
    assert result is None

@patch("httpx.AsyncClient")
async def test_download_retries_and_fails(mock_client, tmp_path):
    dl = build_downloader(tmp_path)

    # Always failing stream
    stream = AsyncMock()
    stream.__aenter__.side_effect = Exception("network boom")

    client_instance = mock_client.return_value
    client_instance.__aenter__.return_value = client_instance
    client_instance.stream.return_value = stream

    result = await dl.download(SAMPLE_PAPER)

    assert result is None
    assert mock_client.call_count >= 1

@patch("httpx.AsyncClient")
async def test_rate_limit_waits(mock_client, tmp_path):
    dl = build_downloader(tmp_path)
    dl.rate_limit_delay = 1

    # Fake http ok
    stream = AsyncMock()
    stream.__aenter__.return_value = stream
    stream.aiter_bytes.return_value = [b"x"]
    stream.raise_for_status = lambda: None

    client_instance = mock_client.return_value
    client_instance.__aenter__.return_value = client_instance
    client_instance.stream.return_value = stream

    # 1st call
    await dl.download(SAMPLE_PAPER, force=True)

    start = asyncio.get_event_loop().time()

    # 2nd call should wait ~1s
    await dl.download(SAMPLE_PAPER, force=True)

    elapsed = asyncio.get_event_loop().time() - start
    assert elapsed >= 1

#  pytest src/test/custom/downloader/arxivpdf.py -q