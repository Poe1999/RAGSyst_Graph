from langgraph.graph import StateGraph, START, END
from src.state import RAGState
from src import nodes

def route_after_classify(state: RAGState) -> str:
    """
    Роутер после classify. Определяет, какая ветка графа дальше:
      • chat      → сразу generate (без поиска)
      • search    → обычный путь (translate → retrieve → grade → generate)
      • composite → plan → synthesize
    """
    intent = state.get("query_intent", "search")
    if intent == "chat":
        return "chat"
    if intent == "composite":
        return "composite"
    return "search"


def build_graph():
    wf = StateGraph(RAGState)

    # узлы
    wf.add_node("spellcheck", nodes.spellcheck_query)
    wf.add_node("classify", nodes.classify_query)
    wf.add_node("translate_query", nodes.translate_query)
    wf.add_node("retrieve", nodes.retrieve)
    wf.add_node("grade", nodes.grade_documents)
    wf.add_node("refine_query", nodes.refine_query)      # ← НОВЫЙ
    wf.add_node("web_search", nodes.web_search_node)
    wf.add_node("generate", nodes.generate)
    wf.add_node("plan", nodes.plan)
    wf.add_node("synthesize", nodes.synthesize)

    # начало
    wf.add_edge(START, "spellcheck")
    wf.add_edge("spellcheck", "classify")

    wf.add_conditional_edges(
        "classify",
        route_after_classify,
        {"chat": "generate", "search": "translate_query", "composite": "plan"},
    )

    # search-ветка
    wf.add_edge("translate_query", "retrieve")
    wf.add_edge("retrieve", "grade")
    wf.add_conditional_edges(
        "grade",
        nodes.should_retry,
        {
            "refine": "refine_query",          # ← НОВАЯ ветка
            "web_search": "web_search",
            "generate": "generate",
        },
    )

    # LOOP: refine → retrieve (второй проход)
    wf.add_edge("refine_query", "retrieve")    # ← НОВОЕ ребро (обратно в retrieve)

    # web_search → generate
    wf.add_edge("web_search", "generate")

    # composite-ветка
    wf.add_edge("plan", "synthesize")

    # финалы
    wf.add_edge("generate", END)
    wf.add_edge("synthesize", END)

    return wf.compile()
