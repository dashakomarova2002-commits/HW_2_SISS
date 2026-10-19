from schemas import FarmRequest
from model import calculate_risk

ALLOWED_REGIONS = {"Krasnodar", "Rostov", "Stavropol"}


def risk_level(score: float) -> str:
    if score < 0.3:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def recommendation(level: str) -> str:
    if level == "low":
        return "Standard review"
    if level == "medium":
        return "Additional check required"
    return "High risk. Manual review required"


def is_valid_region(region: str) -> bool:
    return region in ALLOWED_REGIONS


def build_prediction(request: FarmRequest, request_id: str) -> dict:
    score = calculate_risk(request)
    level = risk_level(score)
    return {
        "request_id": request_id,
        "farm_id": request.farm_id,
        "risk_score": score,
        "risk_level": level,
        "recommendation": recommendation(level),
        "model_version": "1.0"
    }