from __future__ import annotations
from llm import model
from models import BoolOutput
from util import prune_lexicon_entry
from typing import Optional
from langsmith import traceable
from tools import get_entry
from log import log

@traceable(
    run_type="chain",
    name="Vet Association Candidates",
    project_name="Dictionary Resolver"
)
async def vet_association_candidates(state: dict) -> dict:
    """
    initial state includes:
        - word: str
        - segment: str
        - cached_associations: list[list[LexRef]]
    we add:
        - selected_association: list[LexRef]

    For a given word and segment, vet a list of association candidates, each with multiple lexicon entries.
    We presume that these are listed most recently matched first, and thus most likely first.
    Returns original state object, mutated. .
    """
    log("Begin Vet Association Candidates", state)
    for association_candidate in state["cached_associations"]:
        entries = [get_entry(lexref) for lexref in association_candidate]
        if not all(entries):
            continue
        contents = [prune_lexicon_entry(entry.contents()) for entry in entries]

        if await is_validate_association(state["word"], state["segment"], contents):
            state["selected_association"] = association_candidate
            log("Matched Association Candidate", state)
            return state
    log("No Association Candidates Matched", state)
    return state

async def is_validate_association(word: str, segment: str, entries: list[dict]) -> bool:
    prompt = f"""You are a scholar of Jewish texts.  Your task is to validate associations of words to dictionary entries.
    For the word {word} in the text {segment}, the following entries are proposed. 
    
    {entries}
    
    Please return True if these are valid and sufficient dictionary entries.  False, otherwise."""
    result = await model.with_structured_output(BoolOutput).ainvoke(prompt)
    return result.value

