from langgraph.graph import StateGraph, START, END
from src.state import RAGState
from src import nodes


def build_graph():
    wf = StateGraph(RAGState)

    wf.add_node("classify", nodes.classify_query)
    wf.add_node("translate_query", nodes.translate_query)
    wf.add_node("retrieve", nodes.retrieve)
    wf.add_node("grade", nodes.grade_documents)
    wf.add_node("web_search", nodes.web_search_node)
    wf.add_node("generate", nodes.generate)

    wf.add_edge(START, "classify")
    wf.add_edge("classify", "translate_query")
    wf.add_edge("translate_query", "retrieve")
    wf.add_edge("retrieve", "grade")
    wf.add_conditional_edges(
        "grade",
        nodes.should_retry,
        {"retry": "web_search", "generate": "generate"},
    )
    wf.add_edge("web_search", "generate")
    wf.add_edge("generate", END)

    return wf.compile()

def should_retry(state: RAGState) -> str:
    grade = state.get("grade")
    n_rel = len(state.get("documents", []))
    if (grade == "irrelevant" or n_rel < 5) and state.get("retry_count", 0) < 1:
        return "retry"
    return "generate"