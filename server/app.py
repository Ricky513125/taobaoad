from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
from core.model import load_models
from core.recall import RecallService
import numpy as np

app = FastAPI(title="Taobao Recall API")

# 初始化服务
try:
    user_tower, _ = load_models("models")
    recall_service = RecallService("data/vectors", user_tower)
except Exception as e:
    logging.error(f"Service init failed: {str(e)}")
    raise


class UserQuery(BaseModel):
    features: Dict[str, Any]
    top_k: int = 1000


@app.post("/recall")
async def recall_endpoint(query: UserQuery):
    """召回接口

    Request Example:
    {
        "features": {
            "user_cms_segid": 123,
            "user_gender": 1,
            ...
        },
        "top_k": 1000
    }
    """
    try:
        item_ids, scores = recall_service.recall(query.features, query.top_k)
        return {
            "status": "success",
            "data": {
                "item_ids": item_ids.tolist(),
                "scores": scores.tolist()
            }
        }
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)