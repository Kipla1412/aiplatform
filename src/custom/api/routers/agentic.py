from fastapi import APIRouter, HTTPException
from src.custom.api.core.service import services

router = APIRouter()

@router.post("/ask")
async def ask_rag(query: str, user_id: str = "api_user"):
    if not services.agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")

    try:
        result = await services.agent.ask(query=query, user_id=user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
