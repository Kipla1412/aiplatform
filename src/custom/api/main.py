# from fastapi import FastAPI
# from pydantic import BaseModel
# from src.custom.credentials.factory import CredentialFactory
# from src.custom.connectors.factory import ConnectorFactory
# from src.custom.loaders.opensearch import OpenSearchService
# from src.custom.embeddings.jarxivembeddings import JinaEmbeddingsService
# import asyncio

# app = FastAPI(title="ArXiv Hybrid Search API")

# # ---------- Request Model ----------
# class SearchRequest(BaseModel):
#     query: str
#     size: int = 5

# # ---------- Helper: Get OpenSearch Service ----------
# def get_opensearch_service():
#     provider = CredentialFactory.get_provider(
#         mode="airflow",   # or "airflow" if you prefer
#         conn_id="opensearch_api"
#     )
#     config = provider.get_credentials()

#     connector = ConnectorFactory.get_connector(
#         connector_type="opensearch",
#         config=config
#     )

#     return OpenSearchService(connector, config)

# # ---------- Helper: Get Embeddings ----------
# async def get_embedding(text: str):
#     provider = CredentialFactory.get_provider(
#         mode="airflow",
#         conn_id="jina_api"
#     )
#     config = provider.get_credentials()

#     connector = ConnectorFactory.get_connector(
#         connector_type="jina",
#         config=config
#     )

#     service = JinaEmbeddingsService(connector, config)
#     vec = await service.embed_passages([text])
#     return vec[0]

# # ---------- FASTAPI ENDPOINT ----------
# @app.post("/search/hybrid")
# async def hybrid_search(req: SearchRequest):

#     # 1) Create embedding for user query
#     query_embedding = await get_embedding(req.query)

#     # 2) Get OpenSearch service (same as Airflow)
#     service = get_opensearch_service()

#     # 3) Call your existing hybrid search
#     results = service.search_unified(
#         query=req.query,
#         query_embedding=query_embedding,
#         size=req.size,
#         use_hybrid=True,
#         min_score=0.0
#     )

#     return {
#         "query": req.query,
#         "total_results": results["total"],
#         "hits": results["hits"]
#     }

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from src.custom.connectors.factory import ConnectorFactory
from src.custom.loaders.opensearch import OpenSearchService
from src.custom.credentials.factory import CredentialFactory
from src.custom.embeddings.jarxivembeddings import JinaEmbeddingsService
from src.custom.connectors.ollama import OllamaConnector
from src.custom.llm.ollama import OllamaClient
from src.custom.llm.cache.redis import CacheClient
from src.custom.llm.schemas.redisschema import AskRequest, AskResponse

# Load env
load_dotenv()


class SearchServices:

    def __init__(self):
        self.embeddings_service: JinaEmbeddingsService | None = None
        self.opensearch_service: OpenSearchService | None = None
        self.ollama_client: OllamaClient | None = None
        self.cache_client: CacheClient | None = None  

    def init_jina(self):
        """Initialize Jina Embeddings Service (matches your config)"""
        jina_config = {
            "base_url": os.getenv("JINA_BASE_URL"),
            "api_key": os.getenv("JINA_API_KEY"),
            "timeout_seconds": int(os.getenv("JINA_TIMEOUT", 30)),
            "max_retries": int(os.getenv("JINA_MAX_RETRIES", 5)),
            "base_backoff": float(os.getenv("JINA_BASE_BACKOFF", 1)),
            "model": os.getenv("JINA_MODEL", "jina-embeddings-v3"),
            "dimensions": int(os.getenv("JINA_DIMENSIONS", 1024)),
            "tasks": {
                "passage": os.getenv("JINA_TASK_PASSAGE", "retrieval.passage"),
                "query": os.getenv("JINA_TASK_QUERY", "retrieval.query"),
            },
            "batch_size": int(os.getenv("JINA_BATCH_SIZE", 50)),
        }

        jina_connector = ConnectorFactory.get_connector(
            connector_type="jina",
            config=jina_config,
        )

        self.embeddings_service = JinaEmbeddingsService(jina_connector, jina_config)

    def init_opensearch(self):

        """Initialize OpenSearch Service (matches your Airflow extra)"""
        opensearch_config = {
            "host": os.getenv("OPENSEARCH_HOST"),
            "username": os.getenv("OPENSEARCH_USER"),
            "password": os.getenv("OPENSEARCH_PASSWORD"),
            "port": int(os.getenv("OPENSEARCH_PORT", 9200)),
            "index_name": os.getenv("OPENSEARCH_INDEX_NAME", "arxiv-papers"),
            "chunk_index_suffix": os.getenv("OPENSEARCH_CHUNK_SUFFIX", "chunks"),
            "rrf_pipeline_id": os.getenv(
                "OPENSEARCH_RRF_PIPELINE", "hybrid-rrf-pipeline"
            ),
            # extra (not strictly required by your service but kept consistent)
            "vector_dimension": int(
                os.getenv("OPENSEARCH_VECTOR_DIMENSION", 1024)
            ),
            "vector_space_type": os.getenv(
                "OPENSEARCH_VECTOR_SPACE", "cosinesimil"
            ),
            "use_ssl": os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true",
            "verify_certs": os.getenv(
                "OPENSEARCH_VERIFY_CERTS", "false"
            ).lower()
            == "true",
            "ssl_show_warn": os.getenv("OPENSEARCH_SSL_WARN", "false").lower()
            == "true",
            "hybrid_search_size_multiplier": int(
                os.getenv("OPENSEARCH_HYBRID_MULTIPLIER", 2)
            ),
        }

        os_connector = ConnectorFactory.get_connector(
            connector_type="opensearch",
            config=opensearch_config,
        )

        self.opensearch_service = OpenSearchService(
            os_connector, opensearch_config
        )

    def init_ollama(self):

        provider = CredentialFactory.get_provider(
            mode="ollama",
            conn_id=None
        )

        config = provider.get_credentials()
        connector = ConnectorFactory.get_connector(
            connector_type="ollama",
            config=config
        )
        self.ollama_client = OllamaClient(connector, config)
    

    async def embed_query(self, text: str):
        vec = await self.embeddings_service.embed_passages([text])
        return vec[0]

    def init_redis(self):
        """Initialize Redis cache client"""


        provider = CredentialFactory.get_provider(
            mode="redis",
            conn_id=None
        )
        
        cache_config = provider.get_credentials()
        cache_connector = ConnectorFactory.get_connector(
            connector_type="redis",
            config=cache_config
        )

        self.cache_client = CacheClient(cache_connector, cache_config)

services = SearchServices()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI starting up...")

    services.init_jina()
    services.init_opensearch()
    services.init_ollama()
    services.init_redis()

    print("Services initialized")
    yield
    print("FastAPI shutting down...")

app = FastAPI(title="ArXiv Hybrid Search API", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str
    size: int = 5
    categories: list[str] | None = None
    use_hybrid: bool = True
    min_score: float = 0.0

@app.post("/search/hybrid")
async def hybrid_search(req: SearchRequest):
    query_embedding = await services.embed_query(req.query)

    results = services.opensearch_service.search_unified(
        query=req.query,
        query_embedding=query_embedding,
        size=req.size,
        categories=req.categories,
        use_hybrid=req.use_hybrid,
        min_score=req.min_score,
    )

    return {
        "query": req.query,
        "total_results": results["total"],
        "hits": results["hits"],
    }


@app.post("/rag/answer", response_model=AskResponse)
async def rag_answer(req: AskRequest):

    # 0️⃣ Check Redis cache (exact match)
    cached = await services.cache_client.find_cached_response(req)
    if cached:
        return cached

    # 1️⃣ Generate embedding only if hybrid search is enabled
    query_embedding = None
    if req.use_hybrid:
        query_embedding = await services.embed_query(req.query)

    # 2️⃣ Retrieve chunks from OpenSearch
    search_results = services.opensearch_service.search_unified(
        query=req.query,
        query_embedding=query_embedding,
        size=req.top_k,                 # 🔥 AskRequest → top_k
        categories=req.categories,
        use_hybrid=req.use_hybrid,
        min_score=0.0,
    )

    chunks = search_results.get("hits", [])

    if not chunks:
        response = AskResponse(
            query=req.query,
            answer="No relevant documents found to answer this question.",
            sources=[],
            chunks_used=0,
            search_mode="bm25" if not req.use_hybrid else "hybrid",
        )
        return response

    # 3️⃣ Generate RAG answer using Ollama
    rag_output = await services.ollama_client.generate_rag_answer(
        query=req.query,
        chunks=chunks,
        model=req.model,               # 🔥 AskRequest → model
    )

    response = AskResponse(
        query=req.query,
        answer=rag_output["answer"],
        sources=rag_output.get("sources", []),
        chunks_used=len(chunks),
        search_mode="bm25" if not req.use_hybrid else "hybrid",
    )

    # 4️⃣ Store in Redis cache (exact match)
    await services.cache_client.store_response(req, response)

    return response


# @app.post("/rag/answer")
# async def rag_answer(req: SearchRequest):


#     cached = await services.cache_client.find_cached_response(req)
#     if cached:
#         return cached

#     # 1️ Embed query
#     query_embedding = await services.embed_query(req.query)

#     # 2️ Retrieve chunks from OpenSearch
#     search_results = services.opensearch_service.search_unified(
#         query=req.query,
#         query_embedding=query_embedding,
#         size=req.size,
#         categories=req.categories,
#         use_hybrid=req.use_hybrid,
#         min_score=req.min_score,
#     )

#     chunks = search_results["hits"]

#     if not chunks:
#         return {
#             "answer": "No relevant documents found to answer this question.",
#             "sources": [],
#             "confidence": "low",
#             "citations": [],
#         }

#     # 3️ Generate RAG answer using Ollama
#     answer = await services.ollama_client.generate_rag_answer(
#         query=req.query,
#         chunks=chunks,
#     )

#     # 4️  Store in Redis cache
#     await services.cache_client.store_response(req, answer)

#     return answer

#Compositional Zero-Shot Learning