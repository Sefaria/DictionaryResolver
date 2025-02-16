import django
django.setup()
from sefaria.model import WordForm, WordFormSet
from models import LexRef
from typing import Optional
from sefaria.utils.hebrew import strip_nikkud

def record_determination(state: dict) -> None:
    # As currently used, we shouldn't trip this, but defensive programming
    if not state["selected_association"]:
        return

    existing_wordform = get_existing_wordform(state["word"], state["selected_association"])
    if existing_wordform:
        add_ref_to_wordform(existing_wordform, state["ref"])
    else:
        create_wordform(state["word"], state["selected_association"], state["ref"])

def get_existing_wordform(word: str, associations: list[LexRef]) -> Optional[WordForm]:
    """
    Query the database for a wordform that matches exactly the given word and associations.
    :param word:
    :param associations:
    :return:
    """
    query = {
        "form": word,
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