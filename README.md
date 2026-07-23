# Dictionary Resolver

Dictionary Resolver processes segments of Jewish texts and determines precise dictionary
associations for words and phrases, writing the results to Sefaria `WordForm` objects.
It is built for offline, corpus-scale runs: every LLM call goes through the
[Anthropic Message Batches API](https://platform.claude.com/docs/en/build-with-claude/batch-processing)
at 50% of standard token pricing, and all state is checkpointed in Mongo so a run can be
killed and resumed at any point.

## Architecture

The pipeline is a **batch-round driver** (`resolver.py`). Each *round* gathers every
pending LLM call across the whole run — phrase extraction, cached-association vetting,
and determination-agent turns for every word — and submits them as one provider batch.
When the batch ends, results are applied: tool calls (dictionary lookups) are executed
locally against the Sefaria API, appended to each word's conversation, and the next
round is submitted. Words resolve independently; a word that needs four agent turns
just participates in four rounds.

```
seed(ref) ──► phrases task per segment
                  │  (batch round)
                  ▼
             word list per segment ──► cached associations? ──► vet task   (haiku)
                  │                              │no                 │rejected
                  ▼                              ▼                   ▼
             resolve task (sonnet agent loop: lookups ⇆ model, batched per turn)
                  │
                  ▼
             WordForm writes + Lexicon.assocs cache + Lexicon.log
```

State machine per task (`Lexicon.batch_tasks`): `pending → in_batch → pending (next turn) | done | failed`.
Submitted batches are recorded in `Lexicon.batch_rounds`; on restart, open batches are
re-polled by `batch_id` (batch results are retrievable for 29 days), and result
application is idempotent via a per-task turn guard.

### Models

| Role | Model | Why |
|---|---|---|
| Determination agent | `claude-sonnet-5` | scholarly judgment + tool use; near-Opus quality at 1/3 the price |
| Vetting cached candidates | `claude-haiku-4-5` | narrow selection task |
| Phrase extraction | `claude-haiku-4-5` | narrow extraction task |

Override with `DICTRES_DETERMINATION_MODEL` / `DICTRES_VETTING_MODEL` / `DICTRES_PHRASE_MODEL`.

### Modules

```
resolver.py    # driver + CLI: seeding, batch submit/poll, result application, restarts
agent_core.py  # request builders + response interpreters (raw Anthropic wire format)
store.py       # Mongo task/round persistence (Lexicon.batch_tasks, Lexicon.batch_rounds)
tools.py       # Sefaria dictionary lookups (words API, ES search) + local entry validation
db.py          # WordForm writes (record/remove refs, create wordforms)
cache.py       # Lexicon.assocs word→associations cache
models.py      # Pydantic models (LexRef, WordDetermination, ...)
config.py      # models, batch tuning, Sefaria API base
log.py         # execution log in Lexicon.log
```

## Requirements

- Python 3.12 with `anthropic`, `aiohttp`, `bs4`, `pydantic`, and the Sefaria-Project
  dependency set (the `s6` conda env has all of this).
- A Mongo instance with Sefaria data (lexicon_entry, word_form, texts).
- `DJANGO_SETTINGS_MODULE=sefaria.settings` and `PYTHONPATH` including Sefaria-Project.
- `ANTHROPIC_API_KEY` in the environment.

## Usage

```bash
./run.sh process "Sanhedrin 63a"        # seed + run to completion (resumable)
./run.sh status --run-id "Sanhedrin 63a"
./run.sh run --run-id "Sanhedrin 63a"   # resume after a restart
./run.sh clear --run-id "Sanhedrin 63a" # drop run state (not results)
```

`process` is idempotent: if the run is already seeded it resumes rather than reseeding.
Batch rounds typically land in minutes for small runs; the Batches API guarantees
completion within 24 hours, so a full-tractate run is an overnight job.

The default text version is the William Davidson Vocalized Aramaic edition
(`--vtitle` to override).

## History

An earlier iteration ran on LangGraph with per-request rate limiting; it bogged down on
cost and job management. A LangGraph batching extension
(`~/sefaria/batch_bridge`, published concept: `langgraph-batch`) was explored and works,
but its durable-execution story requires PostgreSQL plus durable LangGraph checkpointers,
which this project does not otherwise need. The batch-round driver above gets the same
50% price with strictly less machinery. The LangGraph implementation was removed in the
same commit that added this driver; see git history.
