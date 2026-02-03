import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import time
from src.custom.connectors.factory import ConnectorFactory
from src.custom.loaders.opensearch import OpenSearchService
from src.custom.credentials.factory import CredentialFactory
from src.custom.embeddings.jarxivembeddings import JinaEmbeddingsService
from src.custom.connectors.ollama import OllamaConnector
from src.custom.llm.ollama import OllamaClient
from src.custom.llm.cache.redis import CacheClient
from src.custom.llm.schemas.redisschema import AskRequest, AskResponse
from src.custom.tracing.tracker import RAGTracer
from src.custom.tracing.langfuseengine import LangfuseTracer
# Load env
load_dotenv()


class SearchServices:

    def __init__(self):
        self.embeddings_service: JinaEmbeddingsService | None = None
        self.opensearch_service: OpenSearchService | None = None
        self.ollama_client: OllamaClient | None = None
        self.cache_client: CacheClient | None = None  
        self.langfuse_tracer: RAGTracer | None = None

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

    def init_langfuse(self):
        """Initialize Langfuse tracing connector"""

        provider = CredentialFactory.get_provider(
            mode="langfuse",
            conn_id=None  # or your airflow connection id if used
        )

        langfuse_config = provider.get_credentials()

        langfuse_connector = ConnectorFactory.get_connector(
            connector_type="langfuse",
            config=langfuse_config
        )

        langfuse_tracer = LangfuseTracer(langfuse_connector)
        self.langfuse_tracer = RAGTracer(langfuse_tracer)


services = SearchServices()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI starting up...")

    services.init_jina()
    services.init_opensearch()
    services.init_ollama()
    services.init_redis()
    services.init_langfuse()

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
# @app.post("/rag/answer", response_model=AskResponse)
# async def rag_answer(req: AskRequest):

#     tracer = services.langfuse_tracer

#     with tracer.trace_request(
#         name="rag_answer",
#         input_data={"query": req.query, "model": req.model},
#         user_id="anonymous",
#         session_id="session_anonymous",
#     ) as trace:

#         # 0️⃣ Cache check
#         cached = await services.cache_client.find_cached_response(req)
#         if cached:
#             return cached

#         # 1️⃣ Embedding
#         query_embedding = None
#         if req.use_hybrid:
#             with tracer.span(trace, "query_embedding", {"query": req.query}):
#                 query_embedding = await services.embed_query(req.query)

#         # 2️⃣ Retrieval
#         with tracer.span(trace, "search_retrieval", {"top_k": req.top_k}):
#             search_results = services.opensearch_service.search_unified(
#                 query=req.query,
#                 query_embedding=query_embedding,
#                 size=req.top_k,
#                 categories=req.categories,
#                 use_hybrid=req.use_hybrid,
#                 min_score=0.0,
#             )

#         chunks = search_results.get("hits", [])

#         if not chunks:
#             return AskResponse(
#                 query=req.query,
#                 answer="No relevant documents found to answer this question.",
#                 sources=[],
#                 chunks_used=0,
#                 search_mode="bm25" if not req.use_hybrid else "hybrid",
#             )

#         # 3️⃣ Prompt construction
#         with tracer.span(trace, "prompt_construction", {"chunks": len(chunks)}):
#             prompt = "RAG Prompt"  # replace with real builder

#         # 4️⃣ LLM Generation
#         with tracer.generation(trace, req.model, prompt) as gen:
#             rag_output = await services.ollama_client.generate_rag_answer(
#                 query=req.query,
#                 chunks=chunks,
#                 model=req.model,
#             )
#             if gen:
#                 gen.update(output={"answer": rag_output["answer"]})

#         response = AskResponse(
#             query=req.query,
#             answer=rag_output["answer"],
#             sources=rag_output.get("sources", []),
#             chunks_used=len(chunks),
#             search_mode="bm25" if not req.use_hybrid else "hybrid",
#         )

#         # 5️⃣ Cache store
#         await services.cache_client.store_response(req, response)

#         # 6️⃣ Score trace
#         tracer.score(trace, name="answer_quality", value=0.95)

#         return response


@app.post("/rag/answer", response_model=AskResponse)
async def rag_answer(req: AskRequest):

    tracer = services.langfuse_tracer
    start_time = time.time()

    with tracer.trace_request(user_id="anonymous", query=req.query) as trace:

        # 0️⃣ Cache check
        cached = await services.cache_client.find_cached_response(req)
        if cached:
            return cached

        # 1️⃣ Embedding
        query_embedding = None
        if req.use_hybrid:
            with tracer.trace_embedding(trace, req.query):
                query_embedding = await services.embed_query(req.query)

        # 2️⃣ Retrieval
        with tracer.trace_search(trace, req.query, req.top_k) as search_span:
            search_results = services.opensearch_service.search_unified(
                query=req.query,
                query_embedding=query_embedding,
                size=req.top_k,
                categories=req.categories,
                use_hybrid=req.use_hybrid,
                min_score=0.0,
            )

            chunks = search_results.get("hits", [])
            arxiv_ids = [c.get("arxiv_id") for c in chunks if c.get("arxiv_id")]
            # tracer.end_search(search_span, chunks, arxiv_ids, search_results.get("total", 0))

        if not chunks:
            response = AskResponse(
                query=req.query,
                answer="No relevant documents found to answer this question.",
                sources=[],
                chunks_used=0,
                search_mode="bm25" if not req.use_hybrid else "hybrid",
            )
            # tracer.end_request(trace, response.answer, time.time() - start_time)
            return response
        
        with tracer.trace_prompt_construction(trace, chunks):
            prompt = "RAG Prompt"  # or however you build it

        # 3️⃣ LLM Generation
        with tracer.trace_generation(trace, req.model, "RAG Prompt") as gen_span:
            rag_output = await services.ollama_client.generate_rag_answer(
                query=req.query,
                chunks=chunks,
                model=req.model,
            )
            if gen_span:
                gen_span.update(output={"answer": rag_output["answer"]})
            # tracer.end_generation(gen_span, rag_output["answer"], req.model)

        response = AskResponse(
            query=req.query,
            answer=rag_output["answer"],
            sources=rag_output.get("sources", []),
            chunks_used=len(chunks),
            search_mode="bm25" if not req.use_hybrid else "hybrid",
        )

        # 4️⃣ Cache store
        await services.cache_client.store_response(req, response)

        # tracer.end_request(trace, response.answer, time.time() - start_time)
        tracer.score_answer(trace, 0.95)

        return response

