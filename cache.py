from sefaria.system.database import client  # a pymongo client
from models import LexRef, WordFormAssociations, SegmentsAndLexRefs

db = client["Lexicon"]
collection = db["assocs"] # stores instances of WordFormAssociations

def find_associations(wordform: str) -> list[list[LexRef]]:
    """
    Given a wordform, return the lexicon entries sets that have been previously determined to be associated with it.
    :param wordform:
    :return:
    """
    entry = collection.find_one({"word": wordform})

    if entry:
        wfa = WordFormAssociations(**entry)
        return [x.lexrefs for x in wfa.associations]
    else:
        return []

def add_segment(wordform: str, segment: str, lexrefs: list[LexRef]):
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
    entry = collection.find_one({"word": wordform})
    if entry:
        wfa = WordFormAssociations(**entry)

        # Update the entry with the new segment
        for assoc in wfa.associations:
            if set(assoc.lexrefs) == set(lexrefs):
                if segment not in assoc.segments:
                    assoc.segments.append(segment)
                break
        else:
            wfa.associations.append(SegmentsAndLexRefs(lexrefs=lexrefs, segments=[segment]))

        collection.update_one({"word": wordform}, {"$set": wfa.model_dump()})

    else:
        new_wfa = WordFormAssociations(
            word=wordform,
            associations=[
                SegmentsAndLexRefs(lexrefs=lexrefs, segments=[segment])
            ]
        )
        collection.insert_one(new_wfa.model_dump())

