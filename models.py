from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List


class LexRef(BaseModel):
    headword: str = Field(description="The exact headword of the dictionary entry as recorded in the headword field")
    lexicon_name: str = Field(description="The name of the lexicon, as recorded in the parent_lexicon field")

    def __eq__(self, other):
        if not isinstance(other, LexRef):
            return NotImplemented
        return self.headword == other.headword and self.lexicon_name == other.lexicon_name

    def __hash__(self):
        # Return a hash based on the same fields used in __eq__
        return hash((self.headword, self.lexicon_name))

class SegmentsAndLexRefs(BaseModel):
    lexrefs: List[LexRef] = Field(description="The dictionary entries associated with the word form")
    refs: List[str] = Field(description="The refs of the segments of text in which the word form appears")

class WordFormAssociations(BaseModel):
    word: str = Field(description="The word form being associated with the dictionary entries")
    associations: List[SegmentsAndLexRefs] = Field(description="The associations of the word form with dictionary entries")

class WordDetermination(BaseModel):
    """Record a determination of dictionary entries to keep, add and remove.  This can only be used after all lookups have completed and determinations have been made. It can not be called with other tools. It is a terminal tool."""
    word: str = Field(description="The word being determined")
    reasoning: str = Field(description="The reasoning behind the determination")
    entries_to_keep: List[LexRef] = Field(description="The dictionary entries to keep")
    entries_to_remove: List[LexRef] = Field(description="The dictionary entries to remove")
    entries_to_add: List[LexRef] = Field(description="The dictionary entries to add")

class PhrasesInSegment(BaseModel):
    phrases: List[str] = Field(description="The phrases in the segment")

class BoolOutput(BaseModel):
    value: bool = Field(description="The boolean result of the operation")

