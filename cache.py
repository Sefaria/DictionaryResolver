from sefaria.system.database import client  # a pymongo client
from models import LexRef, WordFormAssociations, SegmentsAndLexRefs

db = client["Lexicon"]
cache_collection = db["assocs"] # stores instances of WordFormAssociations

def clear_cache() -> None:
    """
    Clear the cache of all entries.
    :return:
    """
    cache_collection.drop()

def get_cached_associations(wordform: str) -> list[list[LexRef]]:
    """
    Given a wordform, return the lexicon entries sets that have been previously determined to be associated with it.
    :param wordform:
    :return:
    """
    entry = cache_collection.find_one({"word": wordform})

    if entry:
        wfa = WordFormAssociations(**entry)
        return [x.lexrefs for x in wfa.associations]
    else:
        return []

def add_segment_to_cache(state: dict) -> None:
    """
    For the given wordform -
    If the wordform doesn't exist, add it, with the segment and lexicon entries.
    If the wordform and set of lexicon entries already exists in the database, add this segment to the list of segment associated.
    If the set of lexicon entries does not already exist, log that list of lexicon entries with this segment listed as the only segment.
    :param wordform:
    :param segment:
    :param lexrefs:
    :return:
    """
    # As currently used, we shouldn't trip this, but defensive programming
    if not state["selected_association"]:
        # log
        return

    entry = cache_collection.find_one({"word": state["word"]})
    if entry:
        wfa = WordFormAssociations(**entry)

        # Update the entry with the new segment
        for assoc in wfa.associations:
            if set(assoc.lexrefs) == set(state["selected_association"]):
                if state["ref"] not in assoc.refs:
                    assoc.refs.append(state["ref"])
                break
        else:
            wfa.associations.append(SegmentsAndLexRefs(lexrefs=state["selected_association"], refs=[state["ref"]]))

        cache_collection.update_one({"word": state["word"]}, {"$set": wfa.model_dump()})

    else:
        new_wfa = WordFormAssociations(
            word=state["word"],
            associations=[SegmentsAndLexRefs(lexrefs=state["selected_association"], refs=[state["ref"]])]
        )
        cache_collection.insert_one(new_wfa.model_dump())

