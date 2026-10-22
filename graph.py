import time
from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, END
from agents import (
    run_market_agent,
    run_website_agent,
    fetch_site_images,
    run_customer_agent,
    run_content_agent,
)


def take_last(a, b):
    return b


class GraphState(TypedDict):
    idea: str
    founder_context: Optional[dict]
    market_summary: Optional[dict]
    website_content: Optional[dict]
    site_images: Optional[list]
    customers: Optional[dict]
    content: Optional[dict]
    # Two parallel nodes both write status, so it needs a reducer.
    status: Annotated[str, take_last]


def market_node(state: GraphState) -> dict:
    result = run_market_agent(state["idea"], state.get("founder_context"))
    time.sleep(3)  # pace Groq calls (8K tokens per minute limit)
    return {"market_summary": result, "status": "market_done"}


def website_node(state: GraphState) -> dict:
    result = run_website_agent(state["idea"], state.get("market_summary"), state.get("founder_context"))
    images = fetch_site_images(result, state["idea"])
    time.sleep(3)
    return {"website_content": result, "site_images": images, "status": "website_done"}


def customers_node(state: GraphState) -> dict:
    result = run_customer_agent(state["idea"], state.get("market_summary"), state.get("founder_context"))
    time.sleep(3)
    return {"customers": result, "status": "customers_done"}


def content_node(state: GraphState) -> dict:
    result = run_content_agent(
        state["idea"],
        state.get("market_summary"),
        state.get("website_content"),
        state.get("customers"),
        state.get("founder_context"),
    )
    return {"content": result, "status": "content_done"}


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("market", market_node)
    workflow.add_node("website", website_node)
    workflow.add_node("customers", customers_node)
    workflow.add_node("content", content_node)

    workflow.set_entry_point("market")

    # Website and Customer Finder run in parallel after Market.
    workflow.add_edge("market", "website")
    workflow.add_edge("market", "customers")

    # The list form is a JOIN: Content starts only after BOTH have finished.
    # (Two separate add_edge calls would start Content as soon as either finished.)
    workflow.add_edge(["website", "customers"], "content")

    workflow.add_edge("content", END)

    return workflow.compile()


graph = build_graph()

# Used by the server to show live progress.
AGENT_ORDER = ["market", "website", "customers", "content"]
DEPENDENCIES = {
    "market": [],
    "website": ["market"],
    "customers": ["market"],
    "content": ["website", "customers"],
}
