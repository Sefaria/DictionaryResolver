from __future__ import annotations
from typing import List
from util import split_hebrew_text
from llm import model
from models import PhrasesInSegment
from langsmith import traceable
import unicodedata

@traceable(
    run_type="chain",
    name="Split Segment",
    project_name="Dictionary Resolver"
)
def split_segment(input:str) -> List[str]:
    """
    Split a segment of text into words and phrases that may be found in a dictionary.
    :param input:
    :return:
    """
    # Check for multi-word phrases that may be in the dictionary
    prompt = f"In the segment of text below, what multi-word phrases or names are present that might be found in a dictionary?  Please return the phrases exactly as written, do not translate or transliterate.\n\n{input}"
    phrases = model.with_structured_output(PhrasesInSegment).invoke(prompt)

    words = split_hebrew_text(input)
    words += phrases.phrases
    words = [ unicodedata.normalize("NFC", w) for w in words ]
    return words
