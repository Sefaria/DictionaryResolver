"""
Batch-round driver for the Dictionary Resolver.

Every pending LLM call across every word/segment in a run is submitted as one
Anthropic Message Batch (50% of standard token price). Tool calls are executed
locally between rounds. All conversation state lives in Mongo (`Lexicon.batch_tasks`),
so the process can be killed and restarted at any point: in-flight batches are
re-polled by batch_id on startup.

Usage (env needs DJANGO_SETTINGS_MODULE, PYTHONPATH to Sefaria-Project, ANTHROPIC_API_KEY):
    python resolver.py process "Sanhedrin 63a"      # seed + run to completion
    python resolver.py seed "Sanhedrin 63a:4"
    python resolver.py run --run-id "Sanhedrin 63a:4"
    python resolver.py status --run-id "Sanhedrin 63a"
    python resolver.py clear --run-id "Sanhedrin 63a"
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import time

import anthropic

import django
django.setup()
from sefaria.model import Ref, TextChunk

import config
import store
import agent_core
from agent_core import (
    phrase_extraction_params, interpret_phrase_response, words_for_segment,
    build_vetting_candidates, vetting_params, interpret_vetting_response,
    determination_initial_params, interpret_determination_response, tool_result_block,
)
from cache import get_cached_associations, add_segment_to_cache, add_empty_association_to_cache
from db import record_determination, record_empty_determination
from models import LexRef, WordDetermination
from tools import words_api, LOCAL_TOOL_FUNCTIONS
from log import log

# force=True: sefaria's Django settings configure the root logger during django.setup(),
# which would otherwise make this a silent no-op.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger("resolver")

_client = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client

VTITLE = "William Davidson Edition - Vocalized Aramaic"


# --- Seeding -----------------------------------------------------------------

def seed(run_id: str, ref_str: str, vtitle: str = VTITLE) -> int:
    ref = Ref(ref_str)
    segments = ref.all_segment_refs() if not ref.is_segment_level() else [ref]
    n = 0
    for seg in segments:
        text = TextChunk.remove_html_and_make_presentable(seg.text('he', vtitle=vtitle).text)
        if not text or not text.strip():
            logger.warning("No text for %s (vtitle=%s); skipping", seg.normal(), vtitle)
            continue
        store.create_task(run_id, "phrases", seg.normal(), text,
                          params=phrase_extraction_params(text))
        n += 1
    logger.info("Seeded %d segments for run %s", n, run_id)
    return n


# --- Content sanitizing ------------------------------------------------------

_KEEP_KEYS = {
    "text": ["type", "text"],
    "tool_use": ["type", "id", "name", "input"],
}


def sanitize_content(content_blocks) -> list[dict]:
    """Reduce SDK content blocks to the minimal wire form we replay to the API."""
    out = []
    for block in content_blocks:
        d = block if isinstance(block, dict) else block.model_dump()
        keep = _KEEP_KEYS.get(d.get("type"))
        if keep:
            out.append({k: d[k] for k in keep})
    return out


# --- Word-task creation ------------------------------------------------------

def create_word_task(run_id: str, ref: str, segment: str, word: str) -> None:
    """Create a vet task if the cache has candidates, else an uninitialized resolve task."""
    cached = get_cached_associations(word)
    candidates = build_vetting_candidates(cached) if cached else []
    if candidates:
        store.create_task(run_id, "vet", ref, segment, word,
                          params=vetting_params(word, segment, candidates),
                          extra={"candidates": candidates})
    else:
        store.create_task(run_id, "resolve", ref, segment, word, params=None)


async def init_resolve_task(task: dict) -> None:
    """Fill in initial params for a resolve task (requires a words API call).
    If cached associations have appeared since the task was created (word resolved
    in another segment this run), convert to a vet task instead."""
    word, ref, segment = task["word"], task["ref"], task["segment"]

    # A word may have been resolved in another segment since this task was created;
    # vet those fresh cache candidates instead of running a full determination.
    # (Skipped if this task already went through vetting and rejected them.)
    cached = [] if task.get("vetted") else get_cached_associations(word)
    candidates = build_vetting_candidates(cached) if cached else []
    if candidates:
        store.tasks.update_one({"_id": task["_id"]},
                               {"$set": {"kind": "vet",
                                         "candidates": candidates,
                                         "params": vetting_params(word, segment, candidates),
                                         "updated_at": store.now()}})
        return

    possible, associated = await words_api(word, ref)
    params = determination_initial_params(ref, word, segment, possible, associated)
    store.tasks.update_one({"_id": task["_id"]},
                           {"$set": {"params": params, "updated_at": store.now()}})


# --- Recording ---------------------------------------------------------------

def record_resolution(task: dict, selected_association: list[LexRef],
                      determination: WordDetermination | None) -> None:
    state = {
        "ref": task["ref"],
        "word": task["word"],
        "segment": task["segment"],
        "selected_association": selected_association,
        "determination": determination,
    }
    if not selected_association:
        log("No Association Found", state)
        record_empty_determination(state)
        add_empty_association_to_cache(state)
    else:
        add_segment_to_cache(state)
        record_determination(state)
        log("Recorded Determination", state)


# --- Applying batch results --------------------------------------------------

def apply_phrases_result(run_id: str, task: dict, blocks: list[dict]) -> None:
    phrases = interpret_phrase_response(blocks)
    words = words_for_segment(task["segment"], phrases)
    for word in words:
        create_word_task(run_id, task["ref"], task["segment"], word)
    store.complete_task(task["_id"], task["turn"], {"phrases": phrases, "words": words})
    logger.info("%s: %d words/phrases", task["ref"], len(words))


def apply_vet_result(run_id: str, task: dict, blocks: list[dict]) -> None:
    candidates = task["candidates"]
    idx = interpret_vetting_response(blocks, len(candidates))
    if idx is None:
        # No cached candidate held up; fall through to a fresh determination.
        store.tasks.update_one({"_id": task["_id"], "turn": task["turn"], "status": "in_batch"},
                               {"$set": {"kind": "resolve", "params": None, "status": "pending",
                                         "vetted": True, "updated_at": store.now()},
                                "$inc": {"turn": 1}})
        return
    cand = candidates[idx]
    lexrefs = [LexRef(**lr) for lr in cand["lexrefs"]]
    determination = None
    if not lexrefs:
        determination = WordDetermination(word=task["word"], reasoning=cand.get("reasoning", ""),
                                          entries_to_keep=[], entries_to_remove=[], entries_to_add=[])
    record_resolution(task, lexrefs, determination)
    store.complete_task(task["_id"], task["turn"],
                        {"via": "vetting", "selected_index": idx,
                         "selected_association": [lr.model_dump() for lr in lexrefs]})


async def apply_resolve_result(run_id: str, task: dict, blocks: list[dict]) -> None:
    kind, payload = interpret_determination_response(blocks)

    if kind == "final":
        determination: WordDetermination = payload
        selected = determination.entries_to_keep + determination.entries_to_add
        record_resolution(task, selected, determination)
        store.complete_task(task["_id"], task["turn"],
                            {"via": "determination", "determination": determination.model_dump()})
        return

    if kind == "invalid":
        store.fail_task(task["_id"], f"invalid agent response: {payload}")
        return

    if task["turn"] + 1 >= config.MAX_AGENT_TURNS:
        store.fail_task(task["_id"], "exceeded max agent turns")
        logger.warning("%s / %s: exceeded max agent turns", task["ref"], task["word"])
        return

    if kind == "tool_results":
        tool_results = payload
    else:  # kind == "lookups": execute the lookup tools locally
        tool_results = []
        for call in payload:
            fn = LOCAL_TOOL_FUNCTIONS.get(call["name"])
            if fn is None:
                tool_results.append(tool_result_block(call["id"], f"Unknown tool: {call['name']}", is_error=True))
                continue
            try:
                result = await fn(**call["input"])
                tool_results.append(tool_result_block(call["id"], result))
            except Exception as e:
                tool_results.append(tool_result_block(call["id"], f"Tool error: {e}", is_error=True))

    # Near the turn cap, tell the model to wrap up rather than letting it
    # research its way into a hard failure.
    if task["turn"] + 3 >= config.MAX_AGENT_TURNS:
        tool_results = tool_results + [{
            "type": "text",
            "text": "You have used most of your lookup budget. Please conclude now: call WordDetermination with the best entries you have verified so far, or an empty determination if none are appropriate.",
        }]

    params = task["params"]
    params["messages"] = params["messages"] + [
        {"role": "assistant", "content": blocks},
        {"role": "user", "content": tool_results},
    ]
    store.advance_task(task["_id"], task["turn"], params)


async def apply_result(run_id: str, task: dict, blocks: list[dict]) -> None:
    if task["kind"] == "phrases":
        apply_phrases_result(run_id, task, blocks)
    elif task["kind"] == "vet":
        apply_vet_result(run_id, task, blocks)
    elif task["kind"] == "resolve":
        await apply_resolve_result(run_id, task, blocks)


# --- The driver loop ---------------------------------------------------------

async def poll_and_apply_round(run_id: str, round_doc: dict) -> None:
    batch_id = round_doc["batch_id"]
    while True:
        batch = client().messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        counts = batch.request_counts
        logger.info("batch %s: processing=%d succeeded=%d errored=%d",
                    batch_id, counts.processing, counts.succeeded, counts.errored)
        await asyncio.sleep(config.POLL_INTERVAL_SECONDS)

    for result in client().messages.batches.results(batch_id):
        task_id_str, turn_str = result.custom_id.rsplit("_", 1)
        task = store.tasks.find_one({"_id": store.ObjectId(task_id_str)})
        if task is None or task["status"] != "in_batch" or task["turn"] != int(turn_str):
            continue  # already applied (restart replay) or stale
        if result.result.type == "succeeded":
            blocks = sanitize_content(result.result.message.content)
            try:
                await apply_result(run_id, task, blocks)
            except Exception:
                logger.exception("Failed applying result for %s / %s", task["ref"], task.get("word"))
                store.fail_task(task["_id"], "exception while applying result")
        elif result.result.type == "errored":
            err = result.result.error
            inner_type = str(getattr(getattr(err, "error", None), "type", "") or "")
            if "invalid_request" in inner_type:
                store.fail_task(task["_id"], f"invalid request: {err}")
            else:  # rate limit / server error -> retryable
                store.requeue_task(task["_id"], config.MAX_TASK_ATTEMPTS)
        else:  # canceled / expired -> retryable
            store.requeue_task(task["_id"], config.MAX_TASK_ATTEMPTS)

    store.close_round(round_doc["_id"])


async def submit_round(run_id: str) -> bool:
    """Initialize any uninitialized resolve tasks, then submit all pending work
    as one batch. Returns True if a batch was submitted."""
    uninitialized = list(store.tasks.find({"run_id": run_id, "status": "pending",
                                           "kind": "resolve", "params": None}))
    if uninitialized:
        logger.info("Initializing %d resolve tasks", len(uninitialized))
        await asyncio.gather(*[init_resolve_task(t) for t in uninitialized])

    pending = store.pending_tasks(run_id, config.MAX_REQUESTS_PER_BATCH)
    pending = [t for t in pending if t.get("params")]
    if not pending:
        return False

    requests = []
    for t in pending:
        params = t["params"]
        if t["kind"] == "resolve":  # cache the replayed agent prefix; single-shot tasks gain nothing
            params = agent_core.add_prompt_caching(params)
        requests.append({"custom_id": f"{t['_id']}_{t['turn']}", "params": params})
    batch = client().messages.batches.create(requests=requests)
    task_ids = [t["_id"] for t in pending]
    store.mark_in_batch(task_ids, batch.id)
    store.create_round(run_id, batch.id, task_ids)
    logger.info("Submitted batch %s with %d requests", batch.id, len(requests))
    return True


async def run(run_id: str) -> None:
    start = time.time()
    while True:
        # First, resume any in-flight batches (restart safety).
        for round_doc in store.open_rounds(run_id):
            logger.info("Resuming open batch %s", round_doc["batch_id"])
            await poll_and_apply_round(run_id, round_doc)

        submitted = await submit_round(run_id)
        if not submitted:
            if not store.has_work(run_id):
                break
            # tasks stuck without params or in_batch without an open round shouldn't happen;
            # avoid a hot loop if they do
            logger.warning("Work remains but nothing submittable; status: %s", store.run_status(run_id))
            await asyncio.sleep(config.POLL_INTERVAL_SECONDS)

    logger.info("Run %s complete in %.0fs. Status: %s", run_id, time.time() - start, store.run_status(run_id))


# --- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch dictionary resolver")
    parser.add_argument("command", choices=["seed", "run", "process", "status", "clear", "retry-failed"])
    parser.add_argument("ref", nargs="?", help="Sefaria ref (for seed/process)")
    parser.add_argument("--run-id", help="Run identifier (defaults to the ref)")
    parser.add_argument("--vtitle", default=VTITLE)
    args = parser.parse_args()

    run_id = args.run_id or args.ref
    if run_id is None:
        parser.error("need a ref or --run-id")

    if args.command == "seed":
        seed(run_id, args.ref, args.vtitle)
    elif args.command == "run":
        asyncio.run(run(run_id))
    elif args.command == "process":
        if store.tasks.count_documents({"run_id": run_id}, limit=1) == 0:
            seed(run_id, args.ref, args.vtitle)
        else:
            logger.info("Run %s already seeded; resuming", run_id)
        asyncio.run(run(run_id))
    elif args.command == "status":
        print(store.run_status(run_id))
    elif args.command == "clear":
        store.clear_run(run_id)
        print(f"Cleared run {run_id}")
    elif args.command == "retry-failed":
        # Restart failed resolve tasks from scratch (fresh prompt, turn 0).
        res = store.tasks.update_many(
            {"run_id": run_id, "status": "failed", "kind": "resolve"},
            {"$set": {"status": "pending", "params": None, "turn": 0, "attempts": 0,
                      "error": None, "updated_at": store.now()}})
        logger.info("Requeued %d failed tasks", res.modified_count)
        asyncio.run(run(run_id))


if __name__ == "__main__":
    main()
