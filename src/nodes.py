"""
Узлы LangGraph для RAG-системы.

Граф:
    START → classify → translate_query → retrieve → grade
                                                      ↓
                                             [generate | web_search → generate] → END

Особенности:
  • translate_query активируется только для arXiv (англоязычный корпус).
  • grade_documents реализует CRAG: оценка релевантности через GigaChat LLM.
  • generate всегда отвечает на русском, независимо от языка контекста.
"""
from src.llm import get_llm
from src.retrievers import get_hybrid_retriever
from src.web_search import tavily_search
from src.state import RAGState


def _log(state: RAGState, msg: str) -> list[str]:
    return state.get("steps_log", []) + [msg]


def classify_query(state: RAGState) -> dict:
    """
    Определяет, к какому источнику относится запрос.
    Ключевой признак: ЖАНР, а не тема.
      habr  — статьи, туториалы, обзоры, "как сделать X", "что такое X"
      lenta — новости, события, "что произошло", "когда случилось"
      arxiv — научные статьи, физика, математика, "исследования"
    """
    llm = get_llm()
    prompt = (
        "Ты классификатор запросов RAG-системы. Определи, к какому источнику "
        "относится запрос. Смотри на ЖАНР запроса, а не только на тему.\n\n"
        "Источники:\n"
        "  habr  — технические СТАТЬИ, туториалы, обзоры, разборы, "
        "          'как устроен X', 'что такое X', 'как сделать X', "
        "          сравнения инструментов, опыт разработки\n"
        "  lenta — НОВОСТИ, события, происшествия, 'что произошло', "
        "          'что случилось', 'новости про X', упоминание года/даты, "
        "          политика, экономика, культура как события\n"
        "  arxiv — НАУЧНЫЕ статьи: физика, математика, квантовая теория, "
        "          теория струн, космология, 'исследования', 'статьи про'\n"
        "  general — small talk, арифметика, творческие задачи, "
        "            запросы не по теме источников\n\n"
        "Правила:\n"
        "  • Если в запросе есть 'новости', 'события', 'что произошло', "
        "    конкретный год/дата — это LENTA, даже если тема техническая.\n"
        "  • Если запрос про 'как работает', 'как сделать', 'что такое' — это HABR.\n"
        "  • Если запрос про физику/математику/исследования — ARXIV.\n\n"
        "Ответь РОВНО одним словом: habr, lenta, arxiv или general.\n\n"
        f"Запрос: {state['query']}"
    )
    raw = llm.invoke(prompt).content.strip().lower()
    qtype = next(
        (t for t in ("habr", "lenta", "arxiv", "general") if t in raw),
        "general",
    )
    return {
        "query_type": qtype,
        "steps_log": _log(state, f"Классификация → {qtype}"),
    }


def translate_query(state: RAGState) -> dict:
    """
    Для arXiv-запросов переводим на английский, т.к. корпус англоязычный.
    Для habr/lenta оставляем как есть.
    """
    if state.get("query_type") != "arxiv":
        return {"query_en": state["query"]}

    llm = get_llm()
    prompt = (
        "Переведи запрос с русского на английский для поиска в научных статьях arXiv.\n"
        "Правила:\n"
        "  • специальные термины, аббревиатуры, имена, формулы, LaTeX — не переводи;\n"
        "  • сохраняй регистр и научный стиль;\n"
        "  • верни ТОЛЬКО перевод, без пояснений и кавычек.\n\n"
        f"Запрос: {state['query']}"
    )
    query_en = llm.invoke(prompt).content.strip()

    if not query_en or len(query_en) > 500:
        query_en = state["query"]

    return {
        "query_en": query_en,
        "steps_log": _log(state, f"Перевод RU→EN: «{query_en[:100]}»"),
    }


def retrieve(state: RAGState) -> dict:
    """
    Ищет релевантные документы в выбранном источнике.
    Для arXiv использует query_en (переведённый запрос).
    Дедуплицирует результаты BM25 + FAISS по началу текста.
    """
    if state["query_type"] == "general":
        return {
            "documents": [],
            "steps_log": _log(state, "general → пропуск поиска"),
        }

    retriever = get_hybrid_retriever(state["query_type"])
    search_query = state.get("query_en") or state["query"]
    docs = retriever.invoke(search_query)

    seen = set()
    unique = []
    for d in docs:
        key = d.page_content[:200]
        if key not in seen:
            seen.add(key)
            unique.append(d)

    unique = unique[:20]

    return {
        "documents": unique,
        "steps_log": _log(
            state,
            f"[{state['query_type']}] {len(unique)} уникальных из {len(docs)}",
        ),
    }


def grade_documents(state: RAGState) -> dict:
    """
    Каждый документ оценивается LLM: релевантен ли он запросу.
    Итоговый grade:
      relevant   — все документы релевантны
      ambiguous  — часть релевантна
      irrelevant — ни один не релевантен
    """
    docs = state.get("documents", [])
    if not docs:
        return {
            "grade": "irrelevant",
            "steps_log": _log(state, "Оценка: нет документов"),
        }

    llm = get_llm()
    relevant = []
    for d in docs:
        prompt = (
            "Оцени, релевантен ли фрагмент документа вопросу. "
            "Ответь одним словом: 'yes' или 'no'.\n\n"
            f"Вопрос: {state['query']}\n\n"
            f"Фрагмент:\n{d.page_content[:1200]}"
        )
        verdict = llm.invoke(prompt).content.strip().lower()
        if "yes" in verdict:
            relevant.append(d)

    if not relevant:
        grade = "irrelevant"
    elif len(relevant) == len(docs):
        grade = "relevant"
    else:
        grade = "ambiguous"

    return {
        "documents": relevant,
        "grade": grade,
        "steps_log": _log(
            state,
            f"Оценка: {grade} ({len(relevant)}/{len(docs)} релевантных)",
        ),
    }


def web_search_node(state: RAGState) -> dict:
    """
    Ищет в интернете через Tavily, если локальный retrieval провалился.
    Увеличивает retry_count, чтобы не зациклиться.
    """
    results = tavily_search(state["query"])
    retry = state.get("retry_count", 0) + 1
    return {
        "web_results": results,
        "retry_count": retry,
        "steps_log": _log(
            state,
            f"Web-поиск: {len(results)} результатов (retry #{retry})",
        ),
    }


def generate(state: RAGState) -> dict:
    """
    RAG-генерация, если контекст есть.
    Free-generation (из знаний LLM), если контекста нет — по ТЗ.
    Возвращает ответ, список источников и использованные чанки.
    """
    llm = get_llm()
    docs = state.get("documents", [])
    web_docs = state.get("web_results", [])
    all_docs = docs + web_docs

    sources = []
    for i, d in enumerate(all_docs, 1):
        sources.append({
            "n": i,
            "title": d.metadata.get("title", "(без названия)"),
            "url": d.metadata.get("url", ""),
            "source": d.metadata.get("source", "?"),
            "date": d.metadata.get("date", ""),
            "content": d.page_content[:600],
        })

    if all_docs:
        parts = []
        for i, d in enumerate(all_docs, 1):
            src = d.metadata.get("source", "?")
            url = d.metadata.get("url", "")
            title = d.metadata.get("title", "")
            header = f"[{i}] source={src}"
            if title:
                header += f" | {title}"
            if url:
                header += f" | {url}"
            parts.append(f"{header}\n{d.page_content}")
        context = "\n\n---\n\n".join(parts)

        prompt = (
            "Ты — ассистент RAG-системы. Ответь на вопрос, опираясь на контекст.\n\n"
            "Правила:\n"
            "  1. Используй информацию из контекста, где это уместно.\n"
            "  2. Если в контексте нет прямого ответа, но есть близкая информация — "
            "     используй её и **дополни своими знаниями**, явно помечая: "
            "     «(на основе общих знаний)».\n"
            "  3. Отвечай на русском языке, даже если контекст на английском. "
            "     Научные термины можно оставить в оригинале.\n"
            "  4. Цитируй источники номерами: [1], [2].\n\n"
            f"Вопрос: {state['query']}\n\n"
            f"Контекст:\n{context}\n\n"
            "Ответ:"
        )
        mode = "rag"
    else:
        prompt = (
            "Ты — ассистент. В локальной базе не нашлось информации по запросу. "
            "Ответь из собственных знаний, чётко указав в начале: "
            "«В базе данных нет информации, отвечаю из общих знаний.»\n"
            "Отвечай на русском языке.\n\n"
            f"Вопрос: {state['query']}\n\n"
            "Ответ:"
        )
        mode = "free"

    answer = llm.invoke(prompt).content.strip()

    return {
        "answer": answer,
        "sources": sources,
        "generation_mode": mode,
        "steps_log": _log(
            state,
            f"Ответ сгенерирован ({'RAG' if mode == 'rag' else 'из знаний LLM'})"
        ),
    }

def should_retry(state: RAGState) -> str:
    """
    После grade_documents:
      • если документы нерелевантны и retry_count < 1 → идём в web_search;
      • иначе → сразу в generate.
    """
    if state.get("grade") == "irrelevant" and state.get("retry_count", 0) < 1:
        return "retry"
    return "generate"