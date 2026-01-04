import pytest
import asyncio
from pathlib import Path

from src.custom.downloader.arxivdownloader import ArxivPDFDownloader
from src.custom.credentials.localsettings.pdfconfig import ArxivPDFConfig


pytestmark = pytest.mark.asyncio


# ---------------- Helper ----------------
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


# ---------------- TEST 1: SUCCESS ----------------
async def test_download_success(monkeypatch, tmp_path):
    dl = build_downloader(tmp_path)

    class FakeStreamResponse:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def raise_for_status(self):
            pass
        async def aiter_bytes(self):
            yield b"chunk1"
            yield b"chunk2"

    class FakeClient:
        def stream(self, *_args, **_kwargs):
            return FakeStreamResponse()

    async def fake_client():
        return FakeClient()

    monkeypatch.setattr(dl, "_get_client", fake_client)

    path = await dl.download(SAMPLE_PAPER)

    assert path is not None
    assert path.exists()
    assert path.read_bytes() == b"chunk1chunk2"

    await dl.close()


# ---------------- TEST 2: CACHE ----------------
async def test_download_uses_cache(monkeypatch, tmp_path):
    dl = build_downloader(tmp_path)

    file_path = tmp_path / "1234.5678v1.pdf"
    file_path.write_text("already here")

    async def fake_client():
        assert False, "HTTP should NOT be called when cache exists"

    monkeypatch.setattr(dl, "_get_client", fake_client)

    path = await dl.download(SAMPLE_PAPER)

    assert path == file_path
    assert path.read_text() == "already here"


# ---------------- TEST 3: MISSING FIELDS ----------------
async def test_download_missing_fields(tmp_path):
    dl = build_downloader(tmp_path)

    result = await dl.download({"arxiv_id": "x"})
    assert result is None


# ---------------- TEST 4: RETRY & FAIL ----------------
async def test_download_retries_and_fails(monkeypatch, tmp_path):
    dl = build_downloader(tmp_path)

    class BadStream:
        async def __aenter__(self):
            raise Exception("boom")
        async def __aexit__(self, *args):
            pass

    class FakeClient:
        def stream(self, *_args, **_kwargs):
            return BadStream()

    async def fake_client():
        return FakeClient()

    monkeypatch.setattr(dl, "_get_client", fake_client)

    result = await dl.download(SAMPLE_PAPER)

    assert result is None


# ---------------- TEST 5: RATE LIMIT ----------------
async def test_rate_limit_waits(monkeypatch, tmp_path):
    dl = build_downloader(tmp_path)
    dl.rate_limit_delay = 1

    class FakeStream:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def raise_for_status(self):
            pass
        async def aiter_bytes(self):
            yield b"x"

    class FakeClient:
        def stream(self, *_args, **_kwargs):
            return FakeStream()

    async def fake_client():
        return FakeClient()

    monkeypatch.setattr(dl, "_get_client", fake_client)

    await dl.download(SAMPLE_PAPER, force=True)

    start = asyncio.get_event_loop().time()

    await dl.download(SAMPLE_PAPER, force=True)

    elapsed = asyncio.get_event_loop().time() - start
    assert elapsed >= 1
