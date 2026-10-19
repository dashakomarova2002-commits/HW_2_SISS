import uuid
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException, Query, status

from schemas import (
    FarmRequest, PredictionResponse, BatchRequest, BatchPredictionResponse,
    PaginatedResponse, QueueStatusResponse, LLMChatRequest, LLMChatResponse,
    PredictWithLLMResponse
)
from services import build_prediction, is_valid_region
from database import (
    init_db, save_one_prediction, get_predictions_by_req_id,
    get_predictions_list, get_predictions_by_weekday_paginated
)
from queue_manager import (
    queue, worker, QUEUE_MAX_SIZE, MAX_BATCH_SIZE, MAX_BATCH_WAIT_MS,
    INFERENCE_TIMEOUT_SECONDS
)
from llm_service import call_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    task = asyncio.create_task(worker())
    yield
    task.cancel()


app = FastAPI(
    title="Agro Scoring API",
    description="API for agricultural risk scoring (Lab 2)",
    version="2.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.perf_counter()
    logger.info(f"[{request_id}] START {request.method} {request.url.path}")

    try:
        response: Response = await call_next(request)
    except Exception as e:
        logger.error(f"[{request_id}] ERROR {str(e)}")
        response = Response(
            content=f'{{"detail": "Internal server error", "request_id": "{request_id}"}}',
            status_code=500,
            media_type="application/json"
        )

    process_time = time.perf_counter() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    logger.info(f"[{request_id}] END {request.method} {request.url.path} - {response.status_code} in {process_time:.4f}s")
    return response


@app.get("/health")
async def health(request: Request):
    return {"status": "ok", "request_id": request.state.request_id}


@app.get("/model-info")
async def model_info():
    return {
        "model_name": "agro-risk-model",
        "model_version": "1.0",
        "model_type": "risk-scoring",
        "status": "ready"
    }


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(request_body: FarmRequest, request: Request):
    request_id = request.state.request_id
    if not is_valid_region(request_body.region):
        raise HTTPException(status_code=400, detail="Unknown region")

    result = build_prediction(request_body, request_id)
    await save_one_prediction(result, source="single")
    return result


@app.get("/predictions", response_model=list[PredictionResponse])
async def get_predictions_list_endpoint(
    limit: int = Query(default=10, ge=1, le=100),
    risk_level: str | None = None
):
    return await get_predictions_list(limit=limit, risk_level=risk_level)




@app.get("/requests/{request_id}/predictions", response_model=list[PredictionResponse])
async def get_predictions_by_request(request_id: str):
    rows = await get_predictions_by_req_id(request_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Predictions not found")
    return rows


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(request_body: BatchRequest, request: Request):
    if len(request_body.items) > 8:
        raise HTTPException(status_code=422, detail="Batch size exceeds MAX_BATCH_SIZE (8)")

    batch_req_id = request.state.request_id
    results = []
    for item in request_body.items:
        if not is_valid_region(item.region):
            raise HTTPException(status_code=400, detail=f"Unknown region: {item.region}")
        res = build_prediction(item, batch_req_id)
        results.append(res)
        await save_one_prediction(res, source="batch")
    return {"items": results}


@app.get("/predictions/by-weekday", response_model=PaginatedResponse)
async def get_by_weekday(
    weekday: str,
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=5, ge=1, le=50)
):
    try:
        return await get_predictions_by_weekday_paginated(weekday, page, per_page)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid weekday name")

@app.get("/predictions/{request_id}", response_model=PredictionResponse)
async def get_prediction_endpoint(request_id: str):
    rows = await get_predictions_by_req_id(request_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Prediction not found")
    if len(rows) > 1:
        raise HTTPException(status_code=409, detail="Multiple predictions found. Use /requests/{request_id}/predictions")
    return rows[0]


@app.post("/predict-queued", response_model=PredictionResponse)
async def predict_queued(request_body: FarmRequest, request: Request):
    if not is_valid_region(request_body.region):
        raise HTTPException(status_code=400, detail="Unknown region")
    if queue.full():
        raise HTTPException(status_code=429, detail="Queue is full. Try again later.")

    req_id = request.state.request_id
    future = asyncio.get_running_loop().create_future()
    await queue.put(({"request_id": req_id, "future": future}, request_body))

    try:
        result = await asyncio.wait_for(future, timeout=INFERENCE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")
    return result


@app.get("/queue-status", response_model=QueueStatusResponse)
async def get_queue_status():
    return {
        "queue_size": queue.qsize(),
        "queue_max_size": QUEUE_MAX_SIZE,
        "max_batch_size": MAX_BATCH_SIZE,
        "max_batch_wait_ms": MAX_BATCH_WAIT_MS,
        "inference_timeout_seconds": INFERENCE_TIMEOUT_SECONDS
    }


@app.post("/llm/chat", response_model=LLMChatResponse)
async def llm_chat(request_body: LLMChatRequest, request: Request):
    req_id = request.state.request_id
    try:
        llm_result = await call_llm(
            prompt=request_body.prompt,
            model=request_body.model,
            temperature=request_body.temperature,
            max_tokens=request_body.max_tokens
        )
    except Exception as e:
        if str(e) == "timeout":
            raise HTTPException(status_code=504, detail="LLM timeout")
        elif str(e) == "provider_error":
            raise HTTPException(status_code=502, detail="LLM provider error")
        else:
            raise HTTPException(status_code=503, detail="LLM service unavailable")

    return {
        "request_id": req_id,
        "requested_model": request_body.model,
        "returned_model": llm_result["returned_model"],
        "answer": llm_result["answer"],
        "finish_reason": llm_result["finish_reason"],
        "latency_ms": llm_result["latency_ms"]
    }


@app.post("/predict-with-llm", response_model=PredictWithLLMResponse)
async def predict_with_llm(request_body: FarmRequest, request: Request):
    req_id = request.state.request_id
    if not is_valid_region(request_body.region):
        raise HTTPException(status_code=400, detail="Unknown region")

    prediction = build_prediction(request_body, req_id)
    prompt = f"Объясни результат агроскоринга: риск {prediction['risk_score']} ({prediction['risk_level']}). Рекомендация: {prediction['recommendation']}. Ответь кратко."

    try:
        llm_result = await call_llm(prompt)
        explanation = llm_result["answer"]
    except Exception:
        explanation = "LLM explanation unavailable."

    await save_one_prediction(prediction, source="with_llm")
    return {**prediction, "explanation": explanation}
