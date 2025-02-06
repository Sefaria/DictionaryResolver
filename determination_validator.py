from __future__ import annotations
from llm import model
from models import LexRef
from util import prune_lexicon_entry
from sefaria.model import LexiconEntry
from typing import Optional


def vet_association_candidates(state: dict) -> dict:
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
    for association_candidate in state["cached_associations"]:
        entries = [get_entry(lexref) for lexref in association_candidate]
        if not all(entries):
            continue
        contents = [prune_lexicon_entry(entry.contents()) for entry in entries]

        if is_validate_association(state["word"], state["segment"], contents):
            state["determination"] = association_candidate
            return state
    return state

def is_validate_association(word: str, segment: str, entries: list[dict]) -> bool:
    prompt = f"""You are a scholar of Jewish texts.  Your task is to validate associations of words to dictionary entries.
    For the word {word} in the text {segment}, the following entries are proposed. 
    
    {entries}
    
    Please return True if these are valid and sufficient dictionary entries.  False, otherwise."""
    return model.with_structured_output(bool).invoke(prompt)

def get_entry(lexref: LexRef) -> Optional[LexiconEntry]:
    """
    This uses the Sefaria code to directly connect to the DB.
    Ideally this would be an API, to match dependencies for the rest of this system, but no existing API fills this need.
    :param lexref:
    :return:
    """
    return LexiconEntry().load({"headword": lexref.headword, "parent_lexicon": lexref.lexicon_name})
