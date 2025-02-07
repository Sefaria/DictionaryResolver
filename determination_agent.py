from __future__ import annotations
import json
import logging
import re

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from llm import model
from models import WordDetermination, LexRef
from tools import search_word_forms, search_dictionaries, words_api

tools = [search_word_forms, search_dictionaries, WordDetermination]
model_with_tools = model.bind_tools(tools, tool_choice="any")


async def get_determination(state: dict) -> dict:
    """
    Get the determination for a single word or phrase in a segment of text
    initial state includes:
        - word: str
        - segment: str
        - cached_associations: list[list[LexRef]]
    we add:
        - selected_association: list[LexRef]
        - determination: WordDetermination

    :param word: The word or phrase being determined
    :param ref: Normal Sefaria Ref
    :param segment: The full text of the segment
    :return: WordDetermination
    """
    result = await word_determination_agent.ainvoke(state, stream_mode="values")
    state.update(result)


class WordState(MessagesState):
    """ State of the overall graph """
    word: str
    segment: str
    ref: str
    cached_associations: list[list[LexRef]]
    selected_association: list[LexRef]
    determination: WordDetermination

def determination_agent() -> CompiledStateGraph:
    """
    Create a state graph for the determination agent.
    This agent will determine the best dictionary entries for a given word or phrase within a larger section of text.
    The agent will be able to search for dictionary entries, and will be able to call a model to determine the best entries.
    """
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

    return graph.compile()
word_determination_agent = determination_agent()

async def initiate_determination(state: WordState):
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


    possible_entries, associated_entries = await words_api(state["word"], state["ref"])

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


def call_determination_model(state: WordState):
    response = model_with_tools.invoke(state["messages"])
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def respond_with_determination(state: WordState):
    # Construct the final answer from the arguments of the last tool call
    tool_calls = state["messages"][-1].tool_calls
    tool_call = next(call for call in tool_calls if call["name"] == "WordDetermination")
    determination = WordDetermination(**tool_call["args"])
    selected_association = determination.entries_to_keep + determination.entries_to_add
    # Since we're using tool calling to return structured output,
    # we need to add  a tool message corresponding to the WordDetermination tool call,
    # This is due to LLM providers' requirement that AI messages with tool calls
    # need to be followed by a tool message for each tool call
    tool_message = {
        "type": "tool",
        "content": "Here is your structured response",
        "tool_call_id": tool_call["id"],
    }
    return {"determination": determination, "selected_association": selected_association, "messages": [tool_message]}


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


