import os

# Models. Determinations need scholarly judgment across Hebrew/Aramaic; Sonnet 5
# is near-Opus on this class of task at 1/3 the price. Vetting and phrase
# extraction are narrow classification tasks; Haiku 4.5 suffices.
DETERMINATION_MODEL = os.environ.get("DICTRES_DETERMINATION_MODEL", "claude-sonnet-5")
VETTING_MODEL = os.environ.get("DICTRES_VETTING_MODEL", "claude-haiku-4-5")
PHRASE_MODEL = os.environ.get("DICTRES_PHRASE_MODEL", "claude-haiku-4-5")

DETERMINATION_MAX_TOKENS = 4096
VETTING_MAX_TOKENS = 1024
PHRASE_MAX_TOKENS = 1024

# Sefaria API host for dictionary lookups (words API + search).
# The DB writes always go to the local Mongo that sefaria.model is configured for.
SEFARIA_API_BASE = os.environ.get("DICTRES_SEFARIA_API_BASE", "https://www.sefaria.org")

# Prompt-cache TTL for the determination agent's replayed prefix: "5m" or "1h".
# Tradeoff measured on Sanhedrin 63b: 1h costs a 2x write premium, 5m only 1.25x.
# Because each turn appends full dictionary entries, the per-turn delta (and so the
# write) is large, and the 1h premium ate most of the read saving (10% net). 5m should
# do much better provided rounds stay inside the window; a miss just costs full price.
CACHE_TTL = os.environ.get("DICTRES_CACHE_TTL", "5m")

# Batch driver tuning
POLL_INTERVAL_SECONDS = int(os.environ.get("DICTRES_POLL_INTERVAL", "10"))
# How many results to apply concurrently. The work is dominated by dictionary lookups
# over HTTP, so concurrency collapses the dead time between rounds; keep it modest so
# we don't hammer the Sefaria API.
APPLY_CONCURRENCY = int(os.environ.get("DICTRES_APPLY_CONCURRENCY", "16"))
MAX_REQUESTS_PER_BATCH = 10_000
MAX_AGENT_TURNS = 10         # LLM turns per word before giving up
MAX_TASK_ATTEMPTS = 3        # resubmissions after batch-level errors
