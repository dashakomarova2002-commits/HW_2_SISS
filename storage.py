predictions: dict = {}


def save_prediction(request_id: str, result: dict) -> None:
    predictions[request_id] = result


def get_prediction(request_id: str) -> dict | None:
    return predictions.get(request_id)


def list_predictions(limit: int = 10, risk_level: str | None = None) -> list:
    values = list(predictions.values())
    if risk_level is not None:
        values = [item for item in values if item["risk_level"] == risk_level]
    return values[:limit]