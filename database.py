import aiosqlite
import logging

DB_PATH = "agro_scoring.db"
logger = logging.getLogger(__name__)

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                farm_id TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                model_version TEXT NOT NULL,
                source TEXT DEFAULT 'single',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()
        logger.info("Database initialized")

async def save_one_prediction(result: dict, source: str = "single"):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO predictions (request_id, farm_id, risk_score, risk_level, recommendation, model_version, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            result["request_id"], result["farm_id"], result["risk_score"],
            result["risk_level"], result["recommendation"], result["model_version"], source
        ))
        await db.commit()

async def get_predictions_by_req_id(request_id: str) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM predictions WHERE request_id = ?", (request_id,)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

async def get_predictions_list(limit: int = 10, risk_level: str = None) -> list:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        query = "SELECT * FROM predictions"
        params = []
        if risk_level:
            query += " WHERE risk_level = ?"
            params.append(risk_level)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        async with db.execute(query, tuple(params)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

async def get_predictions_by_weekday_paginated(weekday: str, page: int, per_page: int) -> dict:
    weekday_map = {
        "sunday": "0", "monday": "1", "tuesday": "2", "wednesday": "3",
        "thursday": "4", "friday": "5", "saturday": "6"
    }
    wd_num = weekday_map.get(weekday.lower())
    if wd_num is None:
        raise ValueError("Invalid weekday")

    offset = (page - 1) * per_page
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT COUNT(*) FROM predictions WHERE strftime('%w', created_at) = ?", (wd_num,)
        ) as cursor:
            total_items = (await cursor.fetchone())[0]
        async with db.execute(
            "SELECT * FROM predictions WHERE strftime('%w', created_at) = ? ORDER BY id DESC LIMIT ? OFFSET ?",
            (wd_num, per_page, offset)
        ) as cursor:
            rows = await cursor.fetchall()
            data = [dict(row) for row in rows]

    total_pages = (total_items + per_page - 1) // per_page if total_items > 0 else 1
    return {
        "data": data,
        "meta": {
            "weekday": weekday,
            "current_page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "total_items": total_items,
            "next_page": page + 1 if page < total_pages else None,
            "prev_page": page - 1 if page > 1 else None
        }
    }
