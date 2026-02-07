from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from src.custom.api.core.service import services

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    size: int = 5
    categories: Optional[List[str]] = None
    use_hybrid: bool = True
    min_score: float = 0.0

@router.post("/hybrid")
async def hybrid_search(req: SearchRequest):
    if not services.embeddings or not services.search:
        raise HTTPException(500, "Search services unavailable")

    embedding = await services.embeddings.embed_passages([req.query])

    results = services.search.search_unified(
        query=req.query,
        query_embedding=embedding[0],
        size=req.size,
        categories=req.categories,
        use_hybrid=req.use_hybrid,
        min_score=req.min_score,
    )

    return {
        "query": req.query,
        "total_results": results.get("total", 0),
        "hits": results.get("hits", []),
    }