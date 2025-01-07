import json
import re
import asyncio
import logging

from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Any, Dict, Union
import requests
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_anthropic import ChatAnthropic
from langsmith import traceable
from bs4 import BeautifulSoup


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = ChatAnthropic(model="claude-3-5-sonnet-20241022")


### Lexicon Scope ###
lexicon_map = {
    "Jastrow" : 'Jastrow Dictionary',
    "Klein Dictionary" : 'Klein Dictionary',
    "BDB" : 'BDB Dictionary',
    "BDB Aramaic" : 'BDB Aramaic Dictionary',
    "Kovetz Yesodot VaChakirot" : 'Kovetz Yesodot VaChakirot'
}

lexicon_names = list(lexicon_map.values())



###  Models  ###
class PhrasesInSegment(BaseModel):
    phrases: List[str] = Field(description="The phrases in the segment")

class DictionaryEntryReference(BaseModel):
    headword: str = Field(description="The exact headword of the dictionary entry as recorded in the headword field")
    lexicon_name: str = Field(description="The name of the lexicon, as recorded in the parent_lexicon field")

class WordDetermination(BaseModel):
    """Record a determination of dictionary entries to keep, add and remove.  This can only be used after all lookups have completed and determinations have been made. It can not be called with other tools. It is a terminal tool."""
    word: str = Field(description="The word being determined")
    reasoning: str = Field(description="The reasoning behind the determination")
    dictionary_entries_to_keep: List[DictionaryEntryReference] = Field(description="The dictionary entries to keep")
    dictionary_entries_to_remove: List[DictionaryEntryReference] = Field(description="The dictionary entries to remove")
    dictionary_entries_to_add: List[DictionaryEntryReference] = Field(description="The dictionary entries to add")



### Tools ###
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
        possible_entries = [{k: v for k, v in d.items() if k in ["headword", "parent_lexicon", "content"]} for d in candidates if ref not in d.get("refs", [])]
        associated_entries = [{k: v for k, v in d.items() if k in ["headword", "parent_lexicon", "content"]} for d in candidates if ref in d.get("refs", [])]
        return possible_entries, associated_entries
    else:
        return [{k: v for k, v in d.items() if k in ["headword", "parent_lexicon", "content"]} for d in candidates], []

@tool
def search_word_forms(query: str):
    """Given a word form as written, returns structured dictionary entries that match the word form"""
    possible, associated = words_api(query)
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
    response = _search(query, filters=[
        "Reference/Dictionary/Jastrow",
        "Reference/Dictionary/Klein Dictionary",
        "Reference/Dictionary/BDB",
        "Reference/Dictionary/BDB Aramaic",
        "Reference/Encyclopedic Works/Kovetz Yesodot VaChakirot"
    ])

    # split by / and get the last entry after the last /
    last_part = lambda x: x

    return [
        {
            "ref": hit["_source"]["ref"],
            "headword": hit["_source"]["titleVariants"][0],
            "lexicon_name": lexicon_map[hit["_source"]["path"].split("/")[-1]],
            "text": hit["_source"]["exact"],
        }
        for hit in response["hits"]["hits"]
    ]


tools = [search_word_forms, search_dictionaries, WordDetermination]
model_with_tools = model.bind_tools(tools, tool_choice="any")




### Graph State ###

class WordState(MessagesState):
    word: str
    segment: str
    ref: str
    determination: WordDetermination



### Graph Nodes ###

def initiate_determination(state: WordState):
    """
    Fan out for each word and phrase:
    #   Get the currently associated dictionary entries
    #   Make an LLM tool call with dictionary lookup to determine the best dictionary entries
    #   Return dictionary entries to add and dictionary entries to remove
    """
    is_phrase = re.search(r"\s", state["word"])
    if is_phrase:
        system_message = SystemMessage(content="""You are a scholar of Jewish texts. 
        You will be given a segment of text, a phrase from within that segment, a list of dictionary entries currently associated with that phrase, and sometimes some potential entries that may or may not be correct. 
        Your job is to find the dictionary entries that best defines the phrase given.  If the given entries are not accurate or sufficient, you will need to find the best entries to replace or augment them.
        You will be able to search for structured dictionary entries with headword searches.  Please do not search for or return the individual words within the phrase, only entries relating to the whole phrase itself. 
        The dictionaries at your disposal include the Jastrow Aramaic Dictionary, the Klein Dictionary of Hebrew, the BDB dictionaries of biblical Hebrew and Aramaic, and an encyclopedia of talmudic concepts and idioms called Kovetz Yesodot VaChakirot.  Each of those is preferable in its domain - Jastrow for Aramaic, Klein for Hebrew, and BDB for Biblical language. 
        When you are satisfied that you have found the best dictionary entries, return a short explanation of your work, an array of currently associated dictionary entries that should be kept, those that should be removed, and an array of dictionary entries to add.""")
    else:
        system_message = SystemMessage(content="""You are a scholar of Jewish texts. 
        You will be given a segment of text, a word from within that segment, a list of dictionary entries currently associated with that word, and sometimes some potential entries that may or may not be correct.
        Your job is to find the dictionary entries that best defines the word given.  If the given entries are not accurate or sufficient, you will need to find the best entries to replace or augment them.
        You will be able to search for structured dictionary entries with plain-text searches across dictionaries, and with specific headword searches. 
        The dictionaries at your disposal include the Jastrow Aramaic Dictionary, the Klein Dictionary of Hebrew, the BDB dictionaries of biblical Hebrew and Aramaic, and an encyclopedia of talmudic concepts and idioms called Kovetz Yesodot VaChakirot.  Each of those is preferable in its domain - Jastrow for Aramaic, Klein for Hebrew, and BDB for Biblical language. 
        When you are satisfied that you have found the best dictionary entries, return a short explanation of your work, an array of currently associated dictionary entries that should be kept, those that should be removed, and an array of dictionary entries to add.""")


    possible_entries, associated_entries = words_api(state["word"], state["ref"])

    associated_clause = "There are no entries currently associated with this word." if not associated_entries else "Associated Entries:\n" + str(associated_entries)
    possible_clause = "---\nPossible Entries:\n" + str(possible_entries) if possible_entries else ""

    human_message = HumanMessage(content=f"""
    From: {state["ref"]}
    Text:  {state["segment"]}
    {"Phrase" if is_phrase else "Word"} to define: {state["word"]}
    ---
    {associated_clause}
    {possible_clause}
    """)

    return {"messages": [system_message, human_message]}

# Define the function that calls the model
def call_determination_model(state: WordState):
    response = model_with_tools.invoke(state["messages"])
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function that responds to the user
def respond_with_determination(state: WordState):
    # Construct the final answer from the arguments of the last tool call
    tool_calls = state["messages"][-1].tool_calls
    tool_call = next(call for call in tool_calls if call["name"] == "WordDetermination")
    response = WordDetermination(**tool_call["args"])
    # Since we're using tool calling to return structured output,
    # we need to add  a tool message corresponding to the WordDetermination tool call,
    # This is due to LLM providers' requirement that AI messages with tool calls
    # need to be followed by a tool message for each tool call
    tool_message = {
        "type": "tool",
        "content": "Here is your structured response",
        "tool_call_id": tool_call["id"],
    }
    return {"determination": response, "messages": [tool_message]}

def refuse_determination(state: WordState):
    tool_calls = state["messages"][-1].tool_calls
    calls = [c for c in state["messages"][-1].tool_calls if c["name"] == "search_word_forms"]
    messages = []
    for c in calls:
        tool_message = {
            "type": "tool",
            "content": "WordDetermination can not be called with other tools.",
            "tool_call_id": c["id"],
            "status": "error"
        }
        messages.append(tool_message)
    return {"messages": messages}


def should_continue_determination(state: WordState):
    last_message = state["messages"][-1]
    multiple = len(last_message.tool_calls) > 1
    calls = []
    for call in last_message.tool_calls:
        name = call["name"]
        if name == "WordDetermination":
            if multiple:
                calls += ["refuse_determination"]
            else:
                calls += ["respond_with_determination"]
        elif name == "search_word_forms":
            calls += ["call_search_word_forms"]
        elif name == "search_dictionaries":
            calls += ["call_search_dictionaries"]
        else:
            logging.error("Unknown tool call: " + last_message.tool_calls[0]["name"])
    return calls


def call_search_dictionaries(state: WordState):
    calls = [c for c in state["messages"][-1].tool_calls if c["name"] == "search_dictionaries"]
    messages = []

    for c in calls:
        query = c["args"]["query"]
        result = search_dictionaries.invoke(query)
        tool_message = {
            "type": "tool",
            "content": json.dumps(result),
            "tool_call_id": c["id"],
        }
        messages.append(tool_message)

    return {"messages": messages}


def call_search_word_forms(state: WordState):
    # Gather only the calls for the "search_word_forms" tool
    calls = [c for c in state["messages"][-1].tool_calls if c["name"] == "search_word_forms"]
    messages = []

    for c in calls:
        query = c["args"]["query"]
        # Actually invoke the tool function
        result = search_word_forms.invoke(query)
        # Build a single tool_result for this call
        tool_message = {
            "type": "tool",
            "content": json.dumps(result),
            "tool_call_id": c["id"],  # must match the tool_use block’s ID
        }
        messages.append(tool_message)

    # Return all results as additional messages
    return {"messages": messages}



### Graph Definition ###

graph = StateGraph(WordState)
graph.add_node("initiate_determination", initiate_determination)
graph.add_node("call_determination_model", call_determination_model)
graph.add_node("respond_with_determination", respond_with_determination)
graph.add_node("refuse_determination", refuse_determination)
graph.add_node("call_search_word_forms", call_search_word_forms)
graph.add_node("call_search_dictionaries", call_search_dictionaries)

graph.add_edge(START,"initiate_determination")
graph.add_edge("initiate_determination", "call_determination_model")
graph.add_conditional_edges("call_determination_model", should_continue_determination)
graph.add_edge("call_search_word_forms", "call_determination_model")
graph.add_edge("call_search_dictionaries", "call_determination_model")
graph.add_edge("refuse_determination",  "call_determination_model")
graph.add_edge("respond_with_determination",  END)

agent = graph.compile()


### Utilities ###

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


### Main Functions ###
def split_segment(input:str) -> List[str]:
    """
    Split a segment of text into words and phrases that may be found in a dictionary.
    :param input:
    :return:
    """
    words = split_hebrew_text(input)
    # Check for multi-word phrases that may be in the dictionary
    prompt = f"In the segment of text below, what multi-word phrases are present that might be found in a dictionary?\n\n{input}"
    phrases = model.with_structured_output(PhrasesInSegment).invoke(prompt)
    words += phrases.phrases
    return words

@traceable(
    run_type="chain",
    name="Correct Words in Segment",
    project_name="Correct Words in Segment"
)
async def correct_words_in_segment(ref: str, segment: str) -> List[WordDetermination]:
    """

    :param ref: Sefaria Ref in normal form
    :param segment: The full text of the segment
    :return:
    """
    words = split_segment(segment)

    async def get_determination(word: str) -> WordDetermination:
        inputs = {"ref": ref, "segment": segment, "word": word}
        result = await agent.ainvoke(inputs, stream_mode="values")
        # Assuming `result` is a dict with a "determination" key
        return result["determination"]

    # Schedule all tasks to run in parallel
    tasks = [asyncio.create_task(get_determination(word)) for word in words]

    # Wait for all tasks to complete and gather their results
    determinations = await asyncio.gather(*tasks)

    return list(determinations)

# inputs = {"ref": "Chullin 60a:2", "segment": '''אמר ליה קיסר לרבי יהושע בן חנניה בעינא דאיצבית ליה נהמא לאלהיכו אמר ליה לא מצית אמאי נפישי חילוותיה א"ל איברא אמר ליה פוק צבית לגידא דרביתא דרויחא עלמא טרח שיתא ירחי קייטא אתא זיקא כנשיה לימא'''}


if __name__ == "__main__":
    # ref="Chullin 60a:2"
    # segment = '''אמר ליה קיסר לרבי יהושע בן חנניה בעינא דאיצבית ליה נהמא לאלהיכו אמר ליה לא מצית אמאי נפישי חילוותיה א"ל איברא אמר ליה פוק צבית לגידא דרביתא דרויחא עלמא טרח שיתא ירחי קייטא אתא זיקא כנשיה לימא'''

    #ref = "Jerusalem Talmud Demai 2.3.3"
    #segment = '''הָא מִכְּלָל דּוּ מוֹדֵי עַל קַמָּייָתָא. לֵית הָדָא פְלִיגָא עַל דְּרִבִּי יוֹנָה דְּרִבִּי יוֹנָה אָמַר חֲבֵירִין אֵינָן חֲשׁוּדִין לֹא לֶאֱכֹל וְלֹא לְהַאֲכִיל שֶׁלֹּא יֵלֵךְ וִיטַּמֵּא גוּפוֹ וְיָבֹא וִיטַמֵּא טָהֳרוֹת. וַאֲפִילוּ עַל דְּרִבִּי יוֹסֵי לֵית הָדָא פְלִיגָא. דְּרִבִּי יוֹסֵי אָמַר חֲבֵירִין חֲשׁוּדִין לוֹכַל וְאֵין חֲשׁוּדִין לְהַאֲכִיל. תַּמָּן לְטָהֳרוֹת אֲבָל הָכָא לְמַעְשְׂרוֹת. הַנֶּאֱמָן עַל הַטָּהֳרוֹת נֶאֱמָן עַל הַמַּעְשְׂרוֹת.'''

    segment = ''' חַד בַּר נַשׁ הֲוָה קָאִים רְדִי. פָּֽסְקָת תּוֹרָתֵיהּ קוֹמוֹי. הֲװָת פָֽרְיָא וְהוּא פָרֵי פָֽרְיָא וְהוּא פָרֵי עַד דְּאִשְׁתְּכָח יְהִיב בְּבָבֶל. אָֽמְרוּ לֵיהּ אֵימַת נָֽפְקָת. אָמַר לוֹן יוֹמָא דֵין. אָֽמְרִין בְּהֵיידָא אָתִיתָא. אָמַר לוֹן בְּדָא. אָמַר לוֹן אִיתָא חָמֵי לֹן. נְפַק בְּעֵי מֵיחְמַייָא לוֹן וְלָא חַכִּים בְּהֵיידָא'''
    ref = "Jerusalem Talmud Maaser Sheni 5:2:6"

    determinations = asyncio.run(correct_words_in_segment(ref, segment))
    print(determinations)

