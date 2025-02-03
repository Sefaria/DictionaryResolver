from __future__ import annotations
from typing import Optional, List, Union, Dict, Any
from bs4 import BeautifulSoup


def strip_html(text: str, tags: Optional[List[str]] = None) -> str:
    """
    Strips HTML from the given text:
      - If `tags` is None, removes all HTML tags and returns plain text.
      - If `tags` is provided, removes only those specific tags from the HTML,
        unwrapping their contents, and leaves other tags intact.

    :param text: The input text (HTML).
    :param tags: A list of tag names to specifically remove (unwrap).
                 If None, all tags are stripped.
    :return: The processed text after the desired HTML tags have been stripped.
    """
    soup = BeautifulSoup(text, "html.parser")

    if tags is None:
        # Remove all HTML tags, leaving only text content
        return soup.get_text()
    else:
        # Remove only specified tags, keeping their inner text
        for tag in tags:
            for match in soup.find_all(tag):
                match.unwrap()
        return str(soup)


def clean_nested_html(data: Union[Dict[str, Any], List[Any], str],
                      tags: List[str] | None = None
                      ) -> Union[Dict[str, Any], List[Any], str]:
    """
    Recursively traverse dictionaries and lists, stripping HTML from all strings.
    """
    if isinstance(data, dict):
        # If it’s a dictionary, recurse on each key/value
        for key, value in data.items():
            data[key] = clean_nested_html(value, tags)
        return data

    elif isinstance(data, list):
        # If it’s a list, recurse on each element
        return [clean_nested_html(item, tags) for item in data]

    elif isinstance(data, str):
        # If it’s a string, strip HTML
        return strip_html(data, tags)

    # If it’s neither dict, list, nor string, just return as-is
    return data


def split_hebrew_text(text: str) -> List[str]:
    """
    Split a Hebrew text into words, removing unwanted punctuation.
    Return a list of unique words.
    """
    cleaned_words = []

    for word in text.split():
        # Strip punctuation from start and end, internal " will be preserved.  ' is not stripped for fear of losing abbreviations.
        word = word.strip('?!,.;:"')
        if word:
            cleaned_words.append(word)

    # Eliminate duplicates.  Todo: Hint to the LLM to look at all meanings when it show more than once.
    cleaned_words = list(dict.fromkeys(cleaned_words))
    return cleaned_words
