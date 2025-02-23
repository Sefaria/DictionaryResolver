from __future__ import annotations
from sefaria.system.database import client  # a pymongo client
from langchain_core.load import dumpd

db = client["Lexicon"]
log_collection = db["log"]  # stores logs of operations.  {ref, word, action, state}

def log(action: str, state):
    # messages = [m.to_json() for m in state["messages"]] if state.get("messages") else None
    # cached_associations: # list[LexiconAssociations]
    # selected_association: # list[LexRef]
    # determination = state["determination"].model_dump()  # WordDetermination
    log_state = dumpd(state)
    log_collection.insert_one(log_state | {"action": action})

def clear_log():
    log_collection.drop()