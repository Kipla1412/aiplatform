import asyncio
from src.custom.connectors.apiconnector import JinaConnector

config = {
    "api_key": "",
    "base_url": "https://api.jina.ai/v1",
    "timeout_seconds": 30,
}

async def test():
    connector = JinaConnector(config)
    client = await connector.connect()

    r = await client.post(
        "/embeddings",
        json={
            "model": "jina-embeddings-v3",
            "input": ["hello world"]
        }
    )
    print("Status:", r.status_code)
    print("Response:", r.json())

    await connector.close()

asyncio.run(test())
