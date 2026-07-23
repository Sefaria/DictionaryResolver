from __future__ import annotations
import datetime
from sefaria.system.database import client  # a pymongo client

db = client["Lexicon"]
log_collection = db["log"]  # stores logs of operations.  {ref, word, action, ...}


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, (str, int, float, bool, type(None), datetime.datetime)):
        return value
    return str(value)


def log(action: str, state):
    log_state = _jsonable(dict(state))
    log_state["action"] = action
    log_state["logged_at"] = datetime.datetime.now(datetime.timezone.utc)
    log_collection.insert_one(log_state)


def clear_log():
    log_collection.drop()
