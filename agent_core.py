"""
Pure request-builders and response-interpreters for the three LLM roles:
phrase extraction, cached-association vetting, and the determination agent.

Everything speaks the raw Anthropic Messages wire format (plain dicts), so
conversation state can be persisted in Mongo between batch rounds.
"""
from __future__ import annotations
import json
import re
import unicodedata
from typing import List, Optional, Tuple

from models import LexRef, WordDetermination
from tools import SEARCH_WORD_FORMS_TOOL, SEARCH_DICTIONARIES_TOOL, get_entry
from util import prune_lexicon_entry, split_hebrew_text
import config

# NB: the LexRef schema is inlined (no JSON-schema $ref/$defs) because these params
# are persisted in Mongo, which rejects keys beginning with `$`.
_LEXREF_SCHEMA = {
    "type": "object",
    "properties": {
        "headword": {
            "type": "string",
            "description": "The exact headword of the dictionary entry as recorded in the headword field, including exact vowels, numerals, and spacing.",
        },
        "lexicon_name": {
            "type": "string",
            "description": "The name of the lexicon, as recorded in the parent_lexicon field",
        },
    },
    "required": ["headword", "lexicon_name"],
}

WORD_DETERMINATION_TOOL = {
    "name": "WordDetermination",
    "description": (
        "Record a determination of dictionary entries to keep, add and remove. "
        "This can only be used after all lookups have completed and determinations have been made. "
        "It can not be called with other tools. It is a terminal tool."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "word": {"type": "string", "description": "The word being determined"},
            "reasoning": {"type": "string", "description": "The reasoning behind the determination"},
            "entries_to_keep": {"type": "array", "items": _LEXREF_SCHEMA},
            "entries_to_remove": {"type": "array", "items": _LEXREF_SCHEMA},
            "entries_to_add": {"type": "array", "items": _LEXREF_SCHEMA},
        },
        "required": ["word", "reasoning", "entries_to_keep", "entries_to_remove", "entries_to_add"],
    },
}

DETERMINATION_TOOLS = [SEARCH_WORD_FORMS_TOOL, SEARCH_DICTIONARIES_TOOL, WORD_DETERMINATION_TOOL]


# --- Phrase extraction -------------------------------------------------------

PHRASES_TOOL = {
    "name": "PhrasesInSegment",
    "description": "Record the multi-word phrases found in the segment",
    "input_schema": {
        "type": "object",
        "properties": {
            "phrases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The multi-word phrases in the segment, exactly as written",
            }
        },
        "required": ["phrases"],
    },
}


def phrase_extraction_params(segment: str) -> dict:
    prompt = (
        "In the segment of text below, what multi-word phrases or names are present that might be "
        "found in a dictionary?  Please return the phrases exactly as written, do not translate or "
        f"transliterate.\n\n{segment}"
    )
    return {
        "model": config.PHRASE_MODEL,
        "max_tokens": config.PHRASE_MAX_TOKENS,
        "tools": [PHRASES_TOOL],
        "tool_choice": {"type": "tool", "name": "PhrasesInSegment"},
        "messages": [{"role": "user", "content": prompt}],
    }


def interpret_phrase_response(content_blocks: List[dict]) -> List[str]:
    for block in content_blocks:
        if block.get("type") == "tool_use" and block.get("name") == "PhrasesInSegment":
            return [p for p in block["input"].get("phrases", []) if isinstance(p, str) and p.strip()]
    return []


def words_for_segment(segment: str, phrases: List[str]) -> List[str]:
    words = split_hebrew_text(segment)
    words += [p for p in phrases if re.search(r"\s", p)]  # only true multi-word phrases
    words = [unicodedata.normalize("NFC", w) for w in words]
    return list(dict.fromkeys(words))


# --- Vetting of cached association candidates --------------------------------

SELECT_CANDIDATE_TOOL = {
    "name": "SelectCandidate",
    "description": "Record which candidate association, if any, correctly defines the word as used in this specific passage",
    "input_schema": {
        "type": "object",
        "properties": {
            "selected_index": {
                "type": ["integer", "null"],
                "description": "The 0-based index of the first candidate whose entries define the word's meaning in THIS passage, or null if none clearly fit (which triggers a fresh determination). Prefer null over an uncertain match.",
            },
            "reasoning": {"type": "string", "description": "Brief reasoning for the selection"},
        },
        "required": ["selected_index", "reasoning"],
    },
}


def build_vetting_candidates(cached_associations) -> List[dict]:
    """
    Convert cached LexiconAssociations into vet-able candidate dicts.
    Returns a list of {"lexrefs": [...] , "contents": [...] } or
    {"lexrefs": [], "reasoning": str} for cached empty determinations.
    Candidates whose entries no longer exist in the DB are dropped.
    """
    candidates = []
    for assoc in cached_associations:
        if assoc.lexrefs:
            entries = [get_entry(lexref) for lexref in assoc.lexrefs]
            if not all(entries):
                continue
            contents = [prune_lexicon_entry(entry.contents()) for entry in entries]
            candidates.append({
                "lexrefs": [lr.model_dump() for lr in assoc.lexrefs],
                "contents": contents,
            })
        else:
            candidates.append({"lexrefs": [], "reasoning": assoc.reasoning or ""})
    return candidates


def vetting_params(word: str, segment: str, candidates: List[dict]) -> dict:
    described = []
    for i, cand in enumerate(candidates):
        if cand["lexrefs"]:
            described.append(f"Candidate {i}:\n{cand['contents']}")
        else:
            described.append(
                f"Candidate {i}: A previous search determined that there are no valid dictionary "
                f"entries for this word, with this reasoning: {cand.get('reasoning', '')}"
            )
    candidate_text = "\n---\n".join(described)
    prompt = f"""You are a scholar of Jewish texts.  Your task is to check whether a previously-determined dictionary association fits this word AS IT IS USED IN THIS SPECIFIC PASSAGE.

The same word form can carry different meanings in different passages, and the Talmud sometimes deliberately distinguishes senses of a lexically identical word.  A candidate is valid only if its entries define the meaning the word actually bears here - not merely that they are plausible entries for the word in general.

Word: {word}
Text: {segment}

Candidate associations (previously determined for this word form in other contexts), listed most-recently-determined first:
{candidate_text}

Select the first candidate whose entries correctly and sufficiently define the word as used in THIS passage.
A candidate declaring that no entries exist is valid only if that reasoning genuinely applies here.
If you have any doubt that a candidate's sense fits this occurrence - or if the passage uses the word in a sense none of the candidates capture - return null.  Returning null triggers a fresh, careful determination, so prefer null over a loose or uncertain match."""
    return {
        "model": config.VETTING_MODEL,
        "max_tokens": config.VETTING_MAX_TOKENS,
        "tools": [SELECT_CANDIDATE_TOOL],
        "tool_choice": {"type": "tool", "name": "SelectCandidate"},
        "messages": [{"role": "user", "content": prompt}],
    }


def interpret_vetting_response(content_blocks: List[dict], num_candidates: int) -> Optional[int]:
    for block in content_blocks:
        if block.get("type") == "tool_use" and block.get("name") == "SelectCandidate":
            idx = block["input"].get("selected_index")
            if isinstance(idx, int) and 0 <= idx < num_candidates:
                return idx
    return None


# --- Determination agent ------------------------------------------------------

def determination_system_prompt(is_phrase: bool) -> str:
    unit = "phrase" if is_phrase else "word"
    return f"""You are a scholar of Jewish texts.
    You will be given a segment of text, a {unit} from within that segment, a list of dictionary entries currently associated with that {unit}, and sometimes some potential entries that may or may not be correct.
    Your job is to find the dictionary entries that best defines the {unit} given.  If the given entries are not accurate or sufficient, you will need to find the best entries to replace or augment them.
    You will be able to search for structured dictionary entries with headword searches. {"Please do not search for or return the individual words within the phrase, only entries relating to the whole phrase itself." if is_phrase else ""}
    The dictionaries at your disposal include the Jastrow Aramaic Dictionary, the Klein Dictionary of Hebrew, the BDB dictionaries of biblical Hebrew and Aramaic, the Ben Yehuda Dictionary of Hebrew, and an encyclopedia of talmudic concepts and idioms called Kovetz Yesodot VaChakirot.  Each of those is preferable in its domain - Jastrow for Aramaic, Klein and Ben Yehuda for Hebrew, and BDB for Biblical language.
    When you are satisfied that you have found the best dictionary entries, return a short explanation of your work, an array of currently associated dictionary entries that should be kept, those that should be removed, and an array of dictionary entries to add. ONLY return entries that you have reviewed and are entirely sure are present in the dictionary.
    Please return headwords exactly as you have seen them, with the same vowels and any superscript characters or numerals.  If no entries are appropriate, please respond with a determination empty of dictionary entries.
    Stay focused on the given {unit} only - do not research the other words of the segment; they are handled separately.
    Be decisive: when the entries already provided (or your first searches) contain a suitable entry, conclude immediately.  If two or three searches have returned nothing relevant, further rephrasing is unlikely to help - conclude with an empty determination rather than continuing to search."""


def determination_initial_params(ref: str, word: str, segment: str,
                                 possible_entries: List[dict], associated_entries: List[dict]) -> dict:
    is_phrase = bool(re.search(r"\s", word))
    associated_clause = ("There are no entries currently associated with this word."
                         if not associated_entries else "Associated Entries:\n" + str(associated_entries))
    possible_clause = "---\nPossible Entries:\n" + str(possible_entries) if possible_entries else ""

    human = f"""
    From: {ref}
    Text:  {segment}
    {"Phrase" if is_phrase else "Word"} to define: {word}
    ---
    {associated_clause}
    {possible_clause}
    """
    return {
        "model": config.DETERMINATION_MODEL,
        "max_tokens": config.DETERMINATION_MAX_TOKENS,
        "system": determination_system_prompt(is_phrase),
        "thinking": {"type": "disabled"},
        "tools": DETERMINATION_TOOLS,
        "tool_choice": {"type": "any"},
        "messages": [{"role": "user", "content": human}],
    }


def interpret_determination_response(content_blocks: List[dict]) -> Tuple[str, object]:
    """
    Interpret one assistant turn of the determination agent.

    Returns one of:
      ("final", WordDetermination)           - a valid, verified determination
      ("tool_results", [tool_result blocks]) - error results to append; ask the model again
      ("lookups", [tool_use blocks])         - lookup tools for the caller to execute
      ("invalid", reason)                    - unusable response (no tool calls at all)
    """
    tool_calls = [b for b in content_blocks if b.get("type") == "tool_use"]
    if not tool_calls:
        return "invalid", "no tool calls in response"

    determination_calls = [c for c in tool_calls if c["name"] == "WordDetermination"]

    if determination_calls and len(tool_calls) > 1:
        # WordDetermination is terminal and may not be combined with other calls.
        results = []
        for c in determination_calls:
            results.append({
                "type": "tool_result",
                "tool_use_id": c["id"],
                "content": "WordDetermination can not be called with other tools.",
                "is_error": True,
            })
        # The non-determination calls still expect results; refuse those too so the
        # turn is well-formed, and let the model redo its lookups cleanly.
        for c in tool_calls:
            if c["name"] != "WordDetermination":
                results.append({
                    "type": "tool_result",
                    "tool_use_id": c["id"],
                    "content": "Not executed: WordDetermination was called in the same turn. Please repeat this lookup without WordDetermination, or call WordDetermination alone.",
                    "is_error": True,
                })
        return "tool_results", results

    if determination_calls:
        call = determination_calls[0]
        try:
            determination = WordDetermination(**call["input"])
        except Exception as e:
            return "tool_results", [{
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": f"Invalid WordDetermination arguments: {e}",
                "is_error": True,
            }]

        selected = determination.entries_to_keep + determination.entries_to_add
        mistaken = [entry for entry in selected if not get_entry(entry)]
        if mistaken:
            message = "WordDetermination can not be called with invalid entries.\nThe following entries are not valid dictionary entries:\n"
            for entry in mistaken:
                message += f"{entry.lexicon_name} {entry.headword}\n"
            return "tool_results", [{
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": message,
                "is_error": True,
            }]
        return "final", determination

    # Only lookup tools were called; the caller executes them asynchronously.
    return "lookups", tool_calls


def tool_result_block(tool_use_id: str, result, is_error: bool = False) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": json.dumps(result, ensure_ascii=False) if not isinstance(result, str) else result,
        "is_error": is_error,
    }


# --- Payload instrumentation --------------------------------------------------

def measure_payload(params: dict) -> dict:
    """
    Character-level breakdown of a request, for cost attribution: how much of what we
    send is dictionary content (tool_result) versus prompt scaffolding (system, tools)
    versus the model's own output being replayed (tool_use, text).

    Characters rather than tokens so this is free - no API calls in the hot path.
    Measured ratio for this content is ~2.6 chars/token.
    """
    out = {"system": 0, "tools": 0, "text": 0, "tool_use": 0, "tool_result": 0}
    out["system"] = len(params.get("system") or "")
    out["tools"] = len(json.dumps(params.get("tools") or [], ensure_ascii=False))
    for message in params.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            out["text"] += len(content)
            continue
        for block in content or []:
            if not isinstance(block, dict):
                continue
            size = len(json.dumps(block, ensure_ascii=False))
            out[block.get("type") if block.get("type") in out else "text"] += size
    return out


# --- Prompt caching for the multi-turn determination agent -------------------

# "5m" is the API default and is expressed by omitting ttl entirely.
CACHE_CONTROL = ({"type": "ephemeral"} if config.CACHE_TTL == "5m"
                 else {"type": "ephemeral", "ttl": config.CACHE_TTL})


def add_prompt_caching(params: dict) -> dict:
    """
    Return a copy of a determination request with 1h-TTL cache breakpoints so each
    agent turn reads the replayed conversation prefix (system + tools + earlier turns)
    from cache at ~0.1x instead of reprocessing it at full price.

    Breakpoints go on the last content block of the final message (writes the current
    prefix) and of the message two turns back (the read point matching the previous
    round's write). A messages breakpoint caches everything rendered before it, so the
    shared system prompt and tool definitions are covered without separate breakpoints.

    TTL is config.CACHE_TTL. Only the determination agent replays a growing prefix; the
    single-shot vet and phrase requests gain nothing, so this is not applied to them.

    The transform is applied at submission time and not persisted, so stored params
    stay clean and breakpoints are recomputed fresh each round.
    """
    msgs = params.get("messages")
    if not msgs:
        return params
    out = {**params, "messages": [dict(m) for m in msgs]}
    msgs = out["messages"]

    break_idxs = {len(msgs) - 1}
    if len(msgs) >= 3:
        break_idxs.add(len(msgs) - 3)  # end of the previous round's submission

    for i in break_idxs:
        m = msgs[i]
        content = m["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        else:
            content = [dict(b) if isinstance(b, dict) else b for b in content]
        if content and isinstance(content[-1], dict):
            content[-1] = {**content[-1], "cache_control": CACHE_CONTROL}
        m["content"] = content
    return out
