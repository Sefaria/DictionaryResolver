"""
Mongo persistence for the batch driver.

Collections (in the same `Lexicon` DB the cache and log already use):
  batch_tasks  - one doc per unit of LLM work (phrase extraction / vet / resolve)
  batch_rounds - one doc per submitted provider batch, for restart-safe polling

Task lifecycle:
  pending -> in_batch -> (applied, back to pending for another turn | done | failed)
"""
from __future__ import annotations
import datetime
from typing import List, Optional

from bson import ObjectId
from sefaria.system.database import client

db = client["Lexicon"]
tasks = db["batch_tasks"]
rounds = db["batch_rounds"]

tasks.create_index([("run_id", 1), ("status", 1)])
rounds.create_index([("run_id", 1), ("status", 1)])


def now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def create_task(run_id: str, kind: str, ref: str, segment: str, word: Optional[str] = None,
                params: Optional[dict] = None, extra: Optional[dict] = None) -> ObjectId:
    doc = {
        "run_id": run_id,
        "kind": kind,              # "phrases" | "vet" | "resolve"
        "ref": ref,
        "segment": segment,
        "word": word,
        "params": params,          # anthropic request params for the *next* LLM call
        "status": "pending",
        "turn": 0,
        "attempts": 0,
        "created_at": now(),
        "updated_at": now(),
    }
    if extra:
        doc.update(extra)
    return tasks.insert_one(doc).inserted_id


def pending_tasks(run_id: str, limit: int) -> List[dict]:
    return list(tasks.find({"run_id": run_id, "status": "pending"}).limit(limit))


def mark_in_batch(task_ids: List[ObjectId], batch_id: str) -> None:
    tasks.update_many({"_id": {"$in": task_ids}},
                      {"$set": {"status": "in_batch", "batch_id": batch_id, "updated_at": now()}})


def advance_task(task_id: ObjectId, expected_turn: int, new_params: dict) -> bool:
    """Append-another-turn transition; idempotent via the turn guard."""
    res = tasks.update_one(
        {"_id": task_id, "turn": expected_turn, "status": "in_batch"},
        {"$set": {"params": new_params, "status": "pending", "updated_at": now()},
         "$inc": {"turn": 1}})
    return res.modified_count == 1


def complete_task(task_id: ObjectId, expected_turn: int, result: dict) -> bool:
    res = tasks.update_one(
        {"_id": task_id, "turn": expected_turn, "status": "in_batch"},
        {"$set": {"status": "done", "result": result, "params": None, "updated_at": now()}})
    return res.modified_count == 1


def fail_task(task_id: ObjectId, reason: str) -> None:
    tasks.update_one({"_id": task_id},
                     {"$set": {"status": "failed", "error": reason, "updated_at": now()}})


def requeue_task(task_id: ObjectId, max_attempts: int) -> None:
    """Return a task to pending after a batch-level error, up to max_attempts."""
    doc = tasks.find_one({"_id": task_id})
    if doc is None:
        return
    if doc["attempts"] + 1 >= max_attempts:
        fail_task(task_id, "exceeded max batch attempts")
    else:
        tasks.update_one({"_id": task_id},
                         {"$set": {"status": "pending", "updated_at": now()}, "$inc": {"attempts": 1}})


def create_round(run_id: str, batch_id: str, task_ids: List[ObjectId]) -> ObjectId:
    return rounds.insert_one({
        "run_id": run_id,
        "batch_id": batch_id,
        "task_ids": task_ids,
        "status": "submitted",
        "created_at": now(),
        "updated_at": now(),
    }).inserted_id


def open_rounds(run_id: str) -> List[dict]:
    return list(rounds.find({"run_id": run_id, "status": "submitted"}))


def close_round(round_id: ObjectId) -> None:
    rounds.update_one({"_id": round_id}, {"$set": {"status": "ended", "updated_at": now()}})


def run_status(run_id: str) -> dict:
    pipeline = [{"$match": {"run_id": run_id}},
                {"$group": {"_id": {"kind": "$kind", "status": "$status"}, "n": {"$sum": 1}}}]
    out = {}
    for row in tasks.aggregate(pipeline):
        out[f"{row['_id']['kind']}/{row['_id']['status']}"] = row["n"]
    return dict(sorted(out.items()))


def has_work(run_id: str) -> bool:
    return tasks.count_documents({"run_id": run_id, "status": {"$in": ["pending", "in_batch"]}}, limit=1) > 0


def clear_run(run_id: str) -> None:
    tasks.delete_many({"run_id": run_id})
    rounds.delete_many({"run_id": run_id})
