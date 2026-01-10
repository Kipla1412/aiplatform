import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock

from src.custom.embeddings.jarxivembeddings import JinaEmbeddingsService
from src.custom.connectors.apiconnector import JinaConnector




@pytest.mark.asyncio
async def test_embed_passages_success():
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    mock_response = {
        "model": "jina-embeddings-v3",
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ],
        "usage": {
            "prompt_tokens": 10,
            "total_tokens": 10,
        },
    }

    mock_client.post.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value=mock_response),
        raise_for_status=MagicMock(),
    )

    mock_connector = AsyncMock(spec=JinaConnector)
    mock_connector.connect.return_value = mock_client

    config = {
        "max_retries": 3,
        "base_backoff": 0.1,
        "model": "jina-embeddings-v3",
        "dimensions": 1024,
        "tasks": {
            "passage": "retrieval.passage",
            "query": "retrieval.query",
        },
    }

    service = JinaEmbeddingsService(mock_connector, config)

    embeddings = await service.embed_passages(
        ["hello world", "machine learning"],
        batch_size=2,
    )

    assert embeddings == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]



@pytest.mark.asyncio
async def test_embed_query_success():
    mock_client = AsyncMock(spec=httpx.AsyncClient)

    mock_response = {
        "model": "jina-embeddings-v3",
        "data": [
            {"embedding": [0.9, 0.8, 0.7]}
        ],
        "usage": {
            "prompt_tokens": 5,
            "total_tokens": 5,
        },
    }

    mock_client.post.return_value = MagicMock(
        status_code=200,
        json=MagicMock(return_value=mock_response),
        raise_for_status=MagicMock(),
    )

    mock_connector = AsyncMock(spec=JinaConnector)
    mock_connector.connect.return_value = mock_client

    config = {
        "max_retries": 3,
        "base_backoff": 0.1,
        "model": "jina-embeddings-v3",
        "dimensions": 1024,
        "tasks": {
            "passage": "retrieval.passage",
            "query": "retrieval.query",
        },
    }

    service = JinaEmbeddingsService(mock_connector, config)

    embedding = await service.embed_query("what is transformers?")

    assert embedding == [0.9, 0.8, 0.7]
