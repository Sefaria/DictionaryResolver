from __future__ import annotations
from pydantic import BaseModel, Field
from llm import model

class Validation(BaseModel):
    val: bool = Field(description="Whether the dictionary entry is valid for this word")


def check_validate_association(word: str, segment: str, entry) -> bool:

    prompt = f"""You are a scholar of Jewish texts.  Your task is to validate associations of words to dictionary entries.
    For the word {word} in the text {segment}, is {entry} a valid and sufficient dictionary entry?"""
    answer = model.with_structured_output(Validation).invoke(prompt)
    return answer.val