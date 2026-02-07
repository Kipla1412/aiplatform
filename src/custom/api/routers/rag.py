import time
from contextlib import nullcontext
from fastapi import APIRouter, HTTPException

from src.custom.api.core.service import services
from src.custom.llm.schemas.redisschema import AskRequest, AskResponse

router = APIRouter()


@router.post("/answer", response_model=AskResponse)
async def rag_answer(req: AskRequest):
    if not services.llm or not services.search:
        raise HTTPException(500, "RAG services unavailable")

    tracer = services.tracer
    start_time = time.time()

    with tracer.trace_request("anonymous", req.query) if tracer else nullcontext() as trace:

        # Cache
        if services.cache:
            cached = await services.cache.find_cached_response(req)
            if cached:
                return cached

        # Embedding
        query_embedding = None
        if req.use_hybrid and services.embeddings:
            query_embedding = (await services.embeddings.embed_passages([req.query]))[0]

        # Retrieval
        results = services.search.search_unified(
            query=req.query,
            query_embedding=query_embedding,
            size=req.top_k,
            categories=req.categories,
            use_hybrid=req.use_hybrid,
            min_score=0.0,
        )

        chunks = results.get("hits", [])
        if not chunks:
            return AskResponse(
                query=req.query,
                answer="No relevant documents found.",
                sources=[],
                chunks_used=0,
                search_mode="hybrid" if req.use_hybrid else "bm25",
            )

        # Generation
        rag_output = await services.llm.generate_rag_answer(
            query=req.query,
            chunks=chunks,
            model=req.model,
        )

        response = AskResponse(
            query=req.query,
            answer=rag_output["answer"],
            sources=rag_output.get("sources", []),
            chunks_used=len(chunks),
            search_mode="hybrid" if req.use_hybrid else "bm25",
        )

        if services.cache:
            await services.cache.store_response(req, response)

        if tracer:
            tracer.score_answer(trace, 0.95)

        return response
