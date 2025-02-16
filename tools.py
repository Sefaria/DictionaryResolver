from __future__ import annotations
from langchain_core.tools import tool
from util import prune_lexicon_entry
import aiohttp
from typing import Tuple, List, Optional
from models import LexRef
import django
django.setup()
from sefaria.model import LexiconEntry

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


async def words_api(query: str, ref: str = None) -> Tuple[List[dict], List[dict]]:
    """
    Fetch dictionary entries for a given query using aiohttp.

    :param query: The word to search for.
    :param ref: Optional reference to filter the results.
    :return: A tuple containing:
             - a list of possible entries (those not already associated with the ref)
             - a list of associated entries (those that are already associated with the ref)
    """
    base_url = f"https://www.sefaria.org/api/words/{query}?always_consonants=1&never_split=1"
    url = f"{base_url}&lookup_ref={ref}" if ref else base_url

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            response.raise_for_status()
            json_data = await response.json()

    # Assume lexicon_names and prune_lexicon_entry are defined elsewhere in your module.
    candidates = [d for d in json_data if d["parent_lexicon"] in lexicon_names]

    if ref:
        possible_entries = [prune_lexicon_entry(d) for d in candidates if ref not in d.get("refs", [])]
        associated_entries = [prune_lexicon_entry(d) for d in candidates if ref in d.get("refs", [])]
        return possible_entries, associated_entries
    else:
        return [prune_lexicon_entry(d) for d in candidates], []


@tool
async def search_word_forms(query: str):
    """Given a word form as written, returns structured dictionary entries that match the word form"""
    possible, associated = await words_api(query)     # Since there is no ref passed, associated will be []
    return possible


async def _search(query, filters=None):
    url = "https://www.sefaria.org/api/search-wrapper/es8"

    # If filters is a list, use it as is. If it's not a list, make it a list.
    filter_list = filters if isinstance(filters, list) else [filters] if filters else []
    filter_fields = [None] * len(filter_list)

    payload = {
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

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            json_data = await response.json()
    return json_data


@tool
async def search_dictionaries(query: str):
    """Given a text query, returns textual content of dictionary entries that match the query in any part of their entry"""
    response = await _search(query, filters=lexicon_search_filters)

    return [
        {
            "ref": hit["_source"]["ref"],
            "headword": hit["_source"]["titleVariants"][0],
            "lexicon_name": lexicon_map[hit["_source"]["path"]],
            "text": hit["_source"]["exact"],
        }
        for hit in response["hits"]["hits"]
    ]

def get_entry(lexref: LexRef) -> Optional[LexiconEntry]:
    """
    This uses the Sefaria code to directly connect to the DB.
    Ideally this would be an API, to match dependencies for the rest of this system, but no existing API fills this need.
    :param lexref:
    :return:
    """
    return LexiconEntry().load({"headword": lexref.headword, "parent_lexicon": lexref.lexicon_name})

