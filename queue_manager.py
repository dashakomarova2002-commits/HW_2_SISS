import asyncio
import time
import logging
from services import build_prediction
from database import save_one_prediction

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 8
MAX_BATCH_WAIT_MS = 50
QUEUE_MAX_SIZE = 100
INFERENCE_TIMEOUT_SECONDS = 2.0

queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)

async def worker():
    while True:
        first_item = await queue.get()
        batch = [first_item]
        start_wait = time.perf_counter()

        while len(batch) < MAX_BATCH_SIZE:
            time_left = (MAX_BATCH_WAIT_MS / 1000) - (time.perf_counter() - start_wait)
            if time_left <= 0:
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout=time_left)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        logger.info(f"Formed batch of size {len(batch)}")

        try:
            results = []
            for req_item, farm_data in batch:
                res = build_prediction(farm_data, req_item["request_id"])
                results.append(res)
                await save_one_prediction(res, source="queued")

            for (req_item, _), res in zip(batch, results):
                if not req_item["future"].done():
                    req_item["future"].set_result(res)
        except Exception as e:
            for req_item, _ in batch:
                if not req_item["future"].done():
                    req_item["future"].set_exception(e)

        for _ in batch:
            queue.task_done()
