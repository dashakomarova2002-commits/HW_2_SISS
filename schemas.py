from pydantic import BaseModel, Field
from typing import List, Optional


class FarmRequest(BaseModel):
    farm_id: str
    region: str
    crop_type: str
    area_ha: float = Field(gt=0)
    temperature_avg: float
    precipitation_mm: float = Field(ge=0)
    payment_delay_days: int = Field(ge=0)
    previous_defaults: int = Field(ge=0)
    debt: float = Field(ge=0)


class PredictionResponse(BaseModel):
    request_id: str
    farm_id: str
    risk_score: float
    risk_level: str
    recommendation: str
    model_version: str


class BatchRequest(BaseModel):
    items: List[FarmRequest]


class BatchPredictionResponse(BaseModel):
    items: List[PredictionResponse]


class PaginationMeta(BaseModel):
    weekday: str
    current_page: int
    per_page: int
    total_pages: int
    total_items: int
    next_page: Optional[int] = None
    prev_page: Optional[int] = None


class PaginatedResponse(BaseModel):
    data: List[dict]
    meta: PaginationMeta


class QueueStatusResponse(BaseModel):
    queue_size: int
    queue_max_size: int
    max_batch_size: int
    max_batch_wait_ms: int
    inference_timeout_seconds: float


class LLMChatRequest(BaseModel):
    prompt: str
    model: str = "auto"
    temperature: float = 0.3
    max_tokens: int = 300


class LLMChatResponse(BaseModel):
    request_id: str
    requested_model: str
    returned_model: str
    answer: str
    finish_reason: str
    latency_ms: float


class PredictWithLLMResponse(PredictionResponse):
    explanation: str
