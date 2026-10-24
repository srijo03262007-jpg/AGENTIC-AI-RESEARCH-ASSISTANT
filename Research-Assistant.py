import json
import os
import re
from typing_extensions import TypedDict

# 1. SET THE API KEY FIRST (Consider using st.secrets or os.environ securely)
os.environ["TAVILY_API_KEY"] = (
    "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"
)

import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

summery_template = """ Summarize the following content into a concise paragraph that directly addresses the query. Ensure the summary highlights the key points relevant to the query while maintaining clarity and completeness. Query: {query} Content: {content} """

generate_response_template = """ Given the following user query and content, generate a response that directly answers the query using relevant information from the content. Ensure that the response is clear, concise, and well-structured. Additionally, provide a brief summary of the key points from the response. Question: {question} Context: {context} Answer: """


class ResearchState(TypedDict):
    query: str
    sources: list[str]
    web_results: list[str]
    summarized_results: list[str]
    response: str


class ResearchStateInput(TypedDict):
    query: str


class ResearchStateOutput(TypedDict):
    sources: list[str]
    response: str


def clean_text(text: str):
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()


def search_web(state: ResearchState):
    # FIX: Native LangChain Tavily tool accepts output_format="results"
    search = TavilySearchResults(max_results=3, output_format="results")
    raw_results = search.invoke(state["query"])
    
    # FIX: If the tool returns a JSON string instead of a list, safely parse it
    if isinstance(raw_results, str):
        try:
            search_results = json.loads(raw_results)
        except json.JSONDecodeError:
            # Fallback if parsing fails
            search_results = [{"url": "N/A", "content": raw_results}]
    else:
        search_results = raw_results
    
    # Safely handle extracting values from dictionaries
    sources = []
    web_results = []
    for result in search_results:
        if isinstance(result, dict):
            sources.append(result.get("url", "N/A"))
            web_results.append(result.get("content", ""))
        else:
            # Fallback if a single item is somehow a string
            web_results.append(str(result))

    return {
        "sources": sources,
        "web_results": web_results,
    }


def summarize_results(state: ResearchState):
    model = ChatOllama(model="deepseek-r1:8b")
    prompt = ChatPromptTemplate.from_template(summery_template)
    chain = prompt | model

    summarized_list = []
    for content in state["web_results"]:
        summary = chain.invoke({"query": state["query"], "content": content})
        clean_content = clean_text(summary.content)
        summarized_list.append(clean_content)

    return {"summarized_results": summarized_list}


def generate_results(state: ResearchState):
    model = ChatOllama(model="deepseek-r1:8b")
    prompt = ChatPromptTemplate.from_template(generate_response_template)
    chain = prompt | model
    content = "\n\n".join([summary for summary in state["summarized_results"]])
    response_obj = chain.invoke(
        {"question": state["query"], "context": content}
    )
    return {"response": response_obj.content}


builder = StateGraph(
    ResearchState, input=ResearchStateInput, output=ResearchStateOutput
)
builder.add_node("search_web", search_web)
builder.add_node("summarize_results", summarize_results)
builder.add_node("generate_response", generate_results)

builder.add_edge(START, "search_web")
builder.add_edge("search_web", "summarize_results")
builder.add_edge("summarize_results", "generate_response")
builder.add_edge("generate_response", END)

graph = builder.compile()

st.title("AI Researcher")
query = st.text_input("Enter your research query:")

if query:
    response_state = graph.invoke({"query": query})
    st.write(clean_text(response_state["response"]))
    st.subheader("Sources:")
    for source in response_state["sources"]:
        st.write(source)
