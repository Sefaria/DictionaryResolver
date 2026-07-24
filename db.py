import django
django.setup()
from sefaria.model import WordForm, WordFormSet
from models import LexRef
from typing import Optional
from sefaria.utils.hebrew import strip_nikkud
from log import log

LLM = "LLM Dictionary Resolver"


def superseded_wordforms(word: str, ref: str, keep: Optional[WordForm] = None):
    """
    Every wordform (any source) that claims this word at this ref, matched by
    CONSONANTAL form rather than exact vocalization. The LLM determination is
    authoritative for the segment, so pre-existing associations under a different
    vowel-pattern or prefix-split (e.g. prefix_adder_1's כריסו) - which an exact
    form match would miss - are superseded here. `keep` is the wordform carrying
    our own determination, which is not scrubbed.
    """
    cf = strip_nikkud(word)
    return [wf for wf in WordFormSet({"c_form": cf, "refs": ref})
            if keep is None or wf != keep]


def record_determination(state: dict) -> None:
    # As currently used, we shouldn't trip this, but defensive programming
    if not state["selected_association"]:
        return

    # Our determination lives in the LLM layer; find or create its wordform there.
    matching_wordform = get_matching_wordform(state["word"], state["selected_association"])
    superseded = superseded_wordforms(state["word"], state["ref"], keep=matching_wordform)

    if matching_wordform:
        add_ref_to_wordform(matching_wordform, state["ref"])
    else:
        create_wordform(state["word"], state["selected_association"], state["ref"])

    for wordform in superseded:
        remove_ref_from_wordform(wordform, state["ref"])

def remove_ref_from_wordform(wordform: WordForm, ref: str):
    """
    Remove a reference from a wordform.
    :param word:
    :param ref:
    :return:
    """
    if getattr(wordform, "refs", None) is None:
        return
    elif ref not in wordform.refs:
        return
    else:
        wordform.refs.remove(ref)
    wordform.save()

def get_matching_wordform(word: str, associations: list[LexRef]) -> Optional[WordForm]:
    """
    Query the database for a wordform that matches exactly the given word and associations.
    :param word:
    :param associations:
    :return:
    """
    query = {
        "form": word,
        "generated_by": LLM,   # our determinations live in the LLM layer; never attach to a legacy wordform
        "lookups": {
            "$all": [ { "headword": x.headword, "parent_lexicon": x.lexicon_name } for x in associations]
        },
        "$expr": {"$eq": [{"$size": "$lookups"}, len(associations)]}
    }
    wordform = WordForm().load(query)
    return wordform

def add_ref_to_wordform(wordform: WordForm, ref: str):
    """
    Add a reference to a wordform.
    :param word:
    :param ref:
    :return:
    """
    if getattr(wordform, "refs", None) is None:
        wordform.refs = [ref]
    elif ref in wordform.refs:
        return
    else:
        wordform.refs += [ref]
    wordform.save()

def create_wordform(word: str, associations: list[LexRef], ref: str):
    """
    Create and save a new WordForm record with the given word and associations.
    :param word:
    :param associations:
    :param ref:
    :return:
    """
    wordform = WordForm({
        "form": word,
        "lookups": [ { "headword": x.headword, "parent_lexicon": x.lexicon_name } for x in associations ],
        "refs": [ref],
        "c_form": strip_nikkud(word),
        "generated_by": "LLM Dictionary Resolver"
    })

    wordform.save()

def clear_wordforms():
    WordFormSet({"generated_by": "LLM Dictionary Resolver"}).delete()

def record_empty_determination(state: dict) -> None:
    """
    We determined this word has no valid dictionary entry here, so no wordform should
    claim it at this ref - scrub every existing association (any source, matched by
    consonantal form) for this word at this ref.
    """
    for wordform in superseded_wordforms(state["word"], state["ref"]):
        remove_ref_from_wordform(wordform, state["ref"])