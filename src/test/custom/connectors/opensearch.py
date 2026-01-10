import pytest
from unittest.mock import patch, MagicMock

from src.custom.connectors.opensearch import OpenSearchConnector


@pytest.fixture
def config():
    return {
        "host": "localhost",
        "port": 9200,
        "use_ssl": False,
    }


@patch("src.custom.connectors.opensearch.OpenSearch")
def test_connector_initialization(mock_opensearch, config):
    """
    Test connector initializes without creating client eagerly.
    """
    connector = OpenSearchConnector(config)

    assert connector.host == "localhost"
    assert connector.port == 9200
    assert connector.use_ssl is False
    assert connector._client is None

    # OpenSearch client should NOT be created yet
    mock_opensearch.assert_not_called()


@patch("src.custom.connectors.opensearch.OpenSearch")
def test_connect_creates_client_once(mock_opensearch, config):
    """
    Test that connect() creates OpenSearch client lazily.
    """
    mock_client = MagicMock()
    mock_opensearch.return_value = mock_client

    connector = OpenSearchConnector(config)

    client = connector.connect()

    assert client == mock_client
    assert connector._client == mock_client

    mock_opensearch.assert_called_once_with(
        hosts=[{"host": "localhost", "port": 9200}],
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )


@patch("src.custom.connectors.opensearch.OpenSearch")
def test_connect_reuses_existing_client(mock_opensearch, config):
    """
    Test that connect() reuses the same OpenSearch client.
    """
    mock_client = MagicMock()
    mock_opensearch.return_value = mock_client

    connector = OpenSearchConnector(config)

    client1 = connector.connect()
    client2 = connector.connect()

    assert client1 is client2
    assert mock_opensearch.call_count == 1  # only created once


@patch("src.custom.connectors.opensearch.OpenSearch")
def test_close_resets_client(mock_opensearch, config):
    """
    Test that close() clears the client reference.
    """
    mock_client = MagicMock()
    mock_opensearch.return_value = mock_client

    connector = OpenSearchConnector(config)
    connector.connect()

    assert connector._client is not None

    connector.close()

    assert connector._client is None
