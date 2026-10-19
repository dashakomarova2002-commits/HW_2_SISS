import httpx
import time
import logging

logger = logging.getLogger(__name__)

LLM_BASE_URL = "https://vireonix.ai/v1"
LLM_MODEL = "auto"
LLM_TIMEOUT_SECONDS = 60.0

async def call_llm(prompt: str, model: str = "auto", temperature: float = 0.3, max_tokens: int = 300) -> dict:
    start_time = time.perf_counter()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Ты — полезный ассистент."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    async with httpx.AsyncClient(timeout=LLM_TIMEOUT_SECONDS) as client:
        try:
            resp = await client.post(f"{LLM_BASE_URL}/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            logger.error("LLM Timeout")
            raise Exception("timeout")
        except httpx.HTTPStatusError as e:
            logger.error(f"LLM HTTP error: {e.response.status_code}")
            raise Exception("provider_error")
        except Exception as e:
            logger.error(f"LLM error: {str(e)}")
            raise Exception("unavailable")

    latency_ms = (time.perf_counter() - start_time) * 1000
    return {
        "returned_model": data.get("model", "unknown"),
        "answer": data["choices"][0]["message"]["content"],
        "finish_reason": data["choices"][0].get("finish_reason", "stop"),
        "latency_ms": latency_ms
    }
