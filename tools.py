from __future__ import annotations
import requests
from typing import Tuple, List
from langchain_core.tools import tool
from util import clean_nested_html

LEXICON_CONTENT_KEYS = ["headword", "parent_lexicon", "content"]

lexicon_map = {
    "Reference/Dictionary/Jastrow" : 'Jastrow Dictionary',
    "Reference/Dictionary/Klein Dictionary" : 'Klein Dictionary',
    "Reference/Dictionary/BDB" : 'BDB Dictionary',
    "Reference/Dictionary/BDB Aramaic" : 'BDB Aramaic Dictionary',
    "Reference/Encyclopedic Works/Kovetz Yesodot VaChakirot" : 'Kovetz Yesodot VaChakirot'
    # Krupnik
}
lexicon_names = list(lexicon_map.values())
lexicon_search_filters = list(lexicon_map.keys())


def words_api(query: str, ref:str=None)->Tuple[List[dict], List[dict]]:
    """
    :param query:
    :param ref:
    :return: Tuple - list of possible entries, list of associated entries
    """
    if ref:
        response = requests.get(
            f"https://www.sefaria.org/api/words/{query}?always_consonants=1&never_split=1&lookup_ref={ref}")
    else:
        response = requests.get(f"https://www.sefaria.org/api/words/{query}?always_consonants=1&never_split=1")
    response.raise_for_status()

    candidates = [clean_nested_html(d) for d in response.json() if d["parent_lexicon"] in lexicon_names]

    if ref:
        possible_entries = [{k: v for k, v in d.items() if k in LEXICON_CONTENT_KEYS} for d in candidates if ref not in d.get("refs", [])]
        associated_entries = [{k: v for k, v in d.items() if k in LEXICON_CONTENT_KEYS} for d in candidates if ref in d.get("refs", [])]
        return possible_entries, associated_entries
    else:
        return [{k: v for k, v in d.items() if k in LEXICON_CONTENT_KEYS} for d in candidates], []


@tool
def search_word_forms(query: str):
    """Given a word form as written, returns structured dictionary entries that match the word form"""
    possible, associated = words_api(query)     # Since there is no ref passed, associated will be []
    return possible


def _search(query, filters=None):
    url = "https://www.sefaria.org/api/search-wrapper/es8"

    # If filters is a list, use it as is. If it's not a list, make it a list.
    filter_list = filters if isinstance(filters, list) else [filters] if filters else []
    filter_fields = [None] * len(filter_list)

    payload =    {
      "aggs": [],
      "field": "naive_lemmatizer",
      "filter_fields": filter_fields,
      "filters": filter_list,
      "query": query,
      "size": 8,
      "slop": 10,
      "sort_fields": [
        "pagesheetrank"
      ],
      "sort_method": "score",
      "sort_reverse": False,
      "sort_score_missing": 0.04,
      "source_proj": True,
      "type": "text"
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()


@tool
def search_dictionaries(query: str):
    """Given a text query, returns textual content of dictionary entries that match the query in any part of their entry"""
    response = _search(query, filters=lexicon_search_filters)

    return [
        {
            "ref": hit["_source"]["ref"],
            "headword": hit["_source"]["titleVariants"][0],
            "lexicon_name": lexicon_map[hit["_source"]["path"]],
            "text": hit["_source"]["exact"],
        }
        for hit in response["hits"]["hits"]
    ]
