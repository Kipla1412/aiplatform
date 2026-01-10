import pytest
from opensearchpy import OpenSearch

from src.custom.connectors.opensearch import OpenSearchConnector


@pytest.fixture(scope="module")
def config():
    return {
        "host": "localhost",
        "port": 9200,
        "use_ssl": False,
    }


@pytest.fixture(scope="module")
def connector(config):
    connector = OpenSearchConnector(config)
    yield connector
    connector.close()


def test_real_connection_created(connector):
    """
    Test that OpenSearchConnector creates a real client.
    """
    client = connector.connect()

    assert isinstance(client, OpenSearch)


def test_real_cluster_health(connector):
    """
    Test that OpenSearch cluster is reachable and healthy.
    """
    client = connector.connect()

    health = client.cluster.health()

    assert "status" in health
    assert health["status"] in ["green", "yellow"]


def test_client_reuse(connector):
    """
    Test that connect() reuses the same client.
    """
    client1 = connector.connect()
    client2 = connector.connect()

    assert client1 is client2


def test_real_index_creation_and_deletion(connector):
    """
    Test basic index lifecycle using real OpenSearch.
    """
    client = connector.connect()
    index_name = "test_connector_index"

    # Cleanup before test
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)

    # Create index
    response = client.indices.create(index=index_name)
    assert response["acknowledged"] is True

    # Verify index exists
    assert client.indices.exists(index=index_name)

    # Delete index
    delete_response = client.indices.delete(index=index_name)
    assert delete_response["acknowledged"] is True


def test_close_resets_client(connector):
    """
    Test that close() resets internal client reference.
    """
    client = connector.connect()
    assert connector._client is not None

    connector.close()
    assert connector._client is None
