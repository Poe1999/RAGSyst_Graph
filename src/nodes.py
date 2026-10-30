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
import json
import re
from pathlib import Path
from src.llm import get_llm
from src.retrievers import get_hybrid_retriever
from src.web_search import tavily_search
from src.state import RAGState
from spellchecker import SpellChecker
import pymorphy3
from pydantic import BaseModel, Field
from typing import Literal
from collections import defaultdict

_spell_ru = SpellChecker(language="ru")
_spell_en = SpellChecker(language="en")
_morph = pymorphy3.MorphAnalyzer()


def _log(state: RAGState, msg: str) -> list[str]:
    return state.get("steps_log", []) + [msg]


_VOCAB_PATH = Path(__file__).parent.parent / "domain_vocab.json"
_PROTECTED_EN: set[str] = set()

if _VOCAB_PATH.exists():
    _vocab = json.loads(_VOCAB_PATH.read_text(encoding="utf-8"))
    _spell_ru.word_frequency.load_words(_vocab.get("ru", []))
    _spell_en.word_frequency.load_words(_vocab.get("en", []))
    print(f"[spellcheck] доменный словарь загружен: RU={len(_vocab.get('ru', []))}, EN={len(_vocab.get('en', []))}")
else:
    print("[spellcheck] domain_vocab.json не найден, используем встроенные словари")

_EXTRA_RU = {
    "гигачат", "гигачата", "гигачатом",
    "нейронки", "нейронок",
    "коворкинг", "коворкинга",
    "криптовалюта", "криптовалюты",
    "блокчейн", "блокчейна",
    "промптинг", "промпта", "промптов",
    "эмбеддинга", "токенизация", "токенизации",
    "нейросеть", "нейронка", "нейронный", "трансформер", "трансформеры",
    "сбер", "эмбеддинг", "эмбеддинги", "промпт", "промптинг",
    "векторизация", "токенизация", "лемматизация", "кластеризация",
    "агент", "агенты", "агентный", "фреймворк", "фреймворки",
    "инференс", "ранжирование", "реранкер", "реранкинг",
}


_CYR = re.compile(r"^[а-яё]+$", re.IGNORECASE)
_LAT = re.compile(r"^[a-z]+$", re.IGNORECASE)
_TOKEN_SPLIT = re.compile(r"([^\w]+)")



def _is_known_russian(word: str) -> bool:
    """Слово — реальная форма из словаря pymorphy3?"""
    if word in _EXTRA_RU:
        return True
    parsed = _morph.parse(word)
    # is_known = слово есть в морфологическом словаре (не гипотеза)
    return bool(parsed) and parsed[0].is_known


def _is_morph_variant(original: str, correction: str) -> bool:
    """Проверяет, что изменение — только в окончании (не опечатка)."""
    if not correction:
        return False
    common = 0
    for a, b in zip(original, correction):
        if a == b:
            common += 1
        else:
            break
    # если общий префикс ≥ 75% длины — это форма слова, не опечатка
    return common / max(len(original), len(correction)) >= 0.75

def _correct_token(token: str) -> str:
    if not token or not token.isalpha():
        return token

    lower = token.lower()

    # EN-защита: научные термины и бренды
    if lower in _PROTECTED_EN:
        return token
    # короткие / аббревиатуры / CamelCase
    if len(token) < 4:
        return token
    if token.isupper():
        return token
    if any(c.isupper() for c in token[1:]):
        return token
    # ---- АНГЛИЙСКИЙ ----
    if _LAT.match(token):
        if lower in _spell_en:
            return token
        fix = _spell_en.correction(lower)
        if fix and fix != lower and len(fix) >= 3:
            return fix.capitalize() if token[0].isupper() else fix
        return token

    # ---- РУССКИЙ ----
    if _CYR.match(token):
        # 1) pymorphy3 знает слово → не трогаем (струн, нейросеть, если)
        if _is_known_russian(lower):
            return token
        # 2) не знает → возможно опечатка → пробуем исправить
        fix = _spell_ru.correction(lower)
        if fix and fix != lower and len(fix) >= 3:
            # 2а) но только если исправление — тоже валидное слово
            if _is_known_russian(fix):
                return fix.capitalize() if token[0].isupper() else fix
        return token

    return token

def spellcheck_query(state: RAGState) -> dict:
    original = state["query"]
    parts = _TOKEN_SPLIT.split(original)
    corrected = [_correct_token(p) if i % 2 == 0 else p for i, p in enumerate(parts)]
    result = "".join(corrected)

    changed = result != original
    return {
        "query_corrected": result,
        "steps_log": _log(
            state,
            f"Опечатки: «{original}» → «{result}»" if changed else "Опечаток нет",
        ),
    }


class SubQuestion(BaseModel):
    """Один подвопрос составного запроса с указанием источника для поиска."""
    question: str = Field(description="Самодостаточный подвопрос")
    source: Literal["habr", "lenta", "arxiv", "general"] = Field(
        description="Источник для этого подвопроса"
    )


class QueryClassification(BaseModel):
    """
      Классификация запроса пользователя RAG-системы.
      Определяет намерение (chat/search/composite), источник для поиска
      и, для составных запросов, список подвопросов.
      """
    intent: Literal["chat", "search", "composite"] = Field(
        description="chat — small talk; search — простой поиск; composite — 2-3 подвопроса"
    )
    source: Literal["habr", "lenta", "arxiv", "general"] = Field(
        description="Источник для search. Для composite и chat — general."
    )
    sub_questions: list[SubQuestion] = Field(
        default_factory=list,
        description="Для composite: 2-3 подвопроса с источниками"
    )
    reason: str = Field(description="Краткое обоснование")



def classify_query(state: RAGState) -> dict:
    """
    Один LLM-вызов определяет:
      • intent: chat / search / composite
      • source: habr / lenta / arxiv / general
      • sub_questions: для composite

    Использует with_structured_output — LLM возвращает JSON по схеме Pydantic,
    никакого парсинга строк.
    """
    llm = get_llm()
    structured_llm = llm.with_structured_output(QueryClassification)

    query = state.get("query_corrected") or state["query"]

    prompt = (
    "Ты классификатор запросов RAG-системы. Верни JSON по схеме.\n\n"

    "## Источники\n"
    "  • habr — IT-статьи, туториалы, 'как устроен X', 'что такое X',\n"
    "    сравнения инструментов, опыт разработки\n"
    "  • lenta — новости, события, 'что произошло', 'новости про X',\n"
    "    конкретный год (2009), политика, экономика\n"
    "  • arxiv — научные статьи: физика, математика, квантовая теория,\n"
    "    теория струн, AdS/CFT, космология\n"
    "  • general — общие знания / small talk (для intent=chat)\n\n"

    "## 1. Определи intent\n\n"

    "### chat — ответ из ОБЩИХ ЗНАНИЙ LLM\n"
    "    Включает:\n"
    "      – приветствия, small talk, шутки, арифметика\n"
    "      – исторические факты, биографии, даты ('когда умер Ленин')\n"
    "      – общие определения ('что такое любовь', 'что такое гравитация')\n"
    "      – география ('столица Франции')\n"
    "      – СОСТАВНЫЕ вопросы из общих знаний через 'и':\n"
    "        'когда умер Пушкин и что такое ряд Тейлора' → chat\n"
    "        'что такое любовь и что такое дружба' → chat\n"
    "        (одна генерация ответит на оба — поиск в базе не нужен)\n"
    "    Критерий: 'ответ есть в Википедии и не меняется со временем?' → chat\n\n"
    "      – НЕ temporal: 'когда умер Пушкин' (один факт) → chat\n"

    "### search — ОДНА тема, ОДИН поиск в базе\n"
    "      – 'как работает LangGraph?' → search/habr\n"
    "      – 'новости Apple 2009' → search/lenta\n"
    "      – 'что такое AdS/CFT?' → search/arxiv\n"
    "    Критерий: 'нужна конкретная статья/новость/исследование из базы?' → search\n\n"

    "### composite — ТОЛЬКО при ОБОИХ условиях:\n"
    "      1) нужно искать в базе по 2–3 РАЗНЫМ темам\n"
    "      2) результаты надо СИНТЕЗИРОВАТЬ (сравнить / связать / объяснить причину)\n\n"
    "    Типы composite:\n"
    "      – сравнение: 'сравни X и Y', 'чем отличается X от Y'\n"
    "      – связь: 'как X связано с Y', 'что общего'\n"
    "      – причинность: 'почему X привёл к Y'\n"
    "      – многочастность: 'найди X и объясни Y'\n"
    "      – ВРЕМЕННАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ (важно!):\n"
    "          'что раньше X или Y', 'что произошло раньше',\n"
    "          'расположи по времени X, Y, Z',\n"
    "          'когда появился X и когда Y', 'что было до/после X'\n"
    "        → composite, потому что нужно найти ДВЕ+ даты и сравнить\n\n"
    "    Примеры temporal:\n"
    "      'Что раньше: Hubble или iPhone?' → composite\n"
    "      'Когда появился AdS/CFT и M-теория?' → composite\n"
    "      'Расположи по времени X, Y, Z' → composite\n\n"
    "    Примеры:\n"
    "      'Что раньше: запуск Hubble или первый iPhone?' → composite\n"
    "      'Когда появился термин AdS/CFT и когда — M-теория?' → composite\n"
    "    НЕ composite (важно!):\n"
    "      – 'когда умер Пушкин и как разложить в ряд Тейлора' → chat\n"
    "        (оба факта из общих знаний, поиск не нужен)\n"
    "      – 'что такое любовь и что такое дружба' → chat\n"
    "        (философия, не из базы)\n\n"
    
    "## 2. Определи source (только для search; для composite — general)\n"
    "  source выбирается из: habr / lenta / arxiv / general\n\n"

    "## 3. Для composite — разбей на 2–3 подвопроса\n"
    "  • Каждый подвопрос — самодостаточный поисковый запрос.\n"
    "  • КАЖДЫЙ получает СВОЙ source, независимо от других.\n\n"
    "  Примеры:\n"
    "    'сравни RAG и Graph RAG'\n"
    "      → [{'question': 'что такое RAG', 'source': 'habr'},\n"
    "         {'question': 'что такое Graph RAG', 'source': 'habr'}]\n\n"
    "    'сравни аварию 2010 и 2011'\n"
    "      → [{'question': 'авария 2010', 'source': 'lenta'},\n"
    "         {'question': 'авария 2011', 'source': 'lenta'}]\n\n"
    "    'сравни ChatGPT и AdS/CFT'\n"
    "      → [{'question': 'что такое ChatGPT', 'source': 'habr'},\n"
    "         {'question': 'что такое AdS/CFT', 'source': 'arxiv'}]\n\n"
    "    'что общего между трансформерами и OpenAI'\n"
    "      → [{'question': 'трансформеры', 'source': 'habr'},\n"
    "         {'question': 'новости OpenAI', 'source': 'lenta'}]\n\n"

    "## Правила классификации\n"
    "  • 'новости', 'что произошло', год/дата → lenta\n"
    "  • 'как работает', 'как сделать', 'что такое X' (X — техническая тема) → habr\n"
    "  • физика/математика/исследования → arxiv\n"
    "  • small talk, общие факты, составные из общих знаний → chat/general\n\n"

    "## Ключевое различие (запомни!)\n"
    "  chat + chat = chat       (одна генерация отвечает на оба)\n"
    "  search + search = search (обе темы в одном источнике, один поиск)\n"
    "  search + search + синтез = composite\n"
    "  chat + search = composite (найти в базе и дополнить)\n\n"

    "## Примеры (полный набор)\n"
    "  'Привет' → chat/general\n"
    "  'Когда умер Ленин?' → chat/general\n"
    "  'Что такое любовь?' → chat/general\n"
    "  'Когда умер Пушкин и как разложить в ряд Тейлора' → chat/general\n"
    "  'Столица Франции' → chat/general\n\n"
    "  'Как работает LangGraph?' → search/habr\n"
    "  'Что такое FAISS?' → search/habr\n"
    "  'Новости Apple 2009' → search/lenta\n"
    "  'Что такое AdS/CFT?' → search/arxiv\n\n"
    "  'Сравни RAG и Graph RAG' → composite/habr\n"
    "  'Что общего между ChatGPT и GigaChat' → composite/habr\n"
    "  'Почему трансформеры вытеснили RNN' → composite/habr\n"
    "  'Найди статью про FAISS и объясни IVFPQ' → composite/habr\n"
    "  'Сравни аварию 2010 и 2011' → composite/lenta\n\n"

    f"Запрос: {query}"
)

    try:
        result = structured_llm.invoke(prompt)
    except Exception as e:
        # fallback: если structured output не сработал
        return {
            "query_intent": "search",
            "query_type": "general",
            "sub_questions": [],
            "steps_log": _log(state, f"Классификация → fallback (search/general) [{e}]"),
        }

    source = result.source if result.intent == "search" else "general"
    subs = [{"question": sq.question, "source": sq.source} for sq in result.sub_questions]
    return {
        "query_intent": result.intent,
        "query_type": source,
        "sub_questions": subs,
        "steps_log": _log(
            state,
            f"Классификация → {result.intent}/{source}"
            + (f", подвопросы: {[(sq['question'], sq['source']) for sq in subs]}"
               if subs else "")
            + f" ({result.reason})",
        ),
    }

def plan(state: RAGState) -> dict:
    """
    Для composite: ищет по каждому подвопросу в его источнике.
    Каждый документ помечается всеми подвопросами, к которым он релевантен.
    Не мутирует исходные Document-объекты из docstore.
    """
    import hashlib
    from langchain_core.documents import Document

    subs = state.get("sub_questions", [])
    if not subs:
        return {"steps_log": _log(state, "plan: нет подвопросов")}

    # key (hash) → {"doc": Document-копия, "origins": [q1, q2, ...]}
    doc_map: dict[str, dict] = {}

    for sq in subs:
        source = sq["source"]
        q = sq["question"]
        if source == "general":
            continue

        retriever = get_hybrid_retriever(source)
        docs = retriever.invoke(q)
        print(f"[plan] '{q}' → {len(docs)} документов")

        for d in docs[:10]:
            # ХЭШ полного текста, а не первые 200 символов
            key = hashlib.md5(d.page_content.encode("utf-8", errors="ignore")).hexdigest()

            if key in doc_map:
                if q not in doc_map[key]["origins"]:
                    doc_map[key]["origins"].append(q)
            else:
                # ВАЖНО: создаём НОВЫЙ Document, не мутируем исходный
                new_meta = dict(d.metadata)
                doc_map[key] = {
                    "doc": Document(page_content=d.page_content, metadata=new_meta),
                    "origins": [q],
                }

    # проставляем metadata со всеми origin
    unique = []
    for item in doc_map.values():
        d = item["doc"]
        d.metadata["origin_questions"] = item["origins"]
        d.metadata["origin_question"] = item["origins"][0]
        unique.append(d)

    print(f"[plan] итого уникальных: {len(unique)}")

    return {
        "documents": unique[:30],
        "steps_log": _log(
            state,
            f"plan: {len(subs)} подвопросов, {len(unique)} уникальных документов",
        ),
    }
def synthesize(state: RAGState) -> dict:
    """Генерация ответа по composite: группируем документы по подвопросам."""
    llm = get_llm()
    docs = state.get("documents", [])
    query = state.get("query_corrected") or state["query"]


    if not docs:
        # composite без документов — free-generation прямо здесь
        prompt = (
            "Ты — ассистент. В локальной базе не нашлось информации по запросу. "
            "Ответь из собственных знаний, чётко указав в начале: "
            "«В базе данных нет информации, отвечаю из общих знаний.»\n"
            "Отвечай на русском языке.\n\n"
            f"Вопрос: {query}\n\n"
            "Ответ:"
        )
        answer = llm.invoke(prompt).content.strip()
        return {
            "answer": answer,
            "sources": [],
            "generation_mode": "composite_free",
            "steps_log": _log(state, "Синтез: нет документов, ответ из знаний LLM"),
        }

    grouped = defaultdict(list)
    for d in docs:
        origins = d.metadata.get("origin_questions")
        if not origins:
            origins = [d.metadata.get("origin_question", "—")]
        for q in origins:
            grouped[q].append(d)

    print(f"[synthesize] групп: {len(grouped)}, документов: {len(docs)}")

    # формируем промпт с секциями
    parts = []
    idx = 1
    for q, q_docs in grouped.items():
        parts.append(f"\n### Подвопрос: {q}")
        for d in q_docs:
            src = d.metadata.get("source", "?")
            url = d.metadata.get("url", "")
            title = d.metadata.get("title", "")
            header = f"[{idx}] source={src}"
            if title:
                header += f" | {title}"
            if url:
                header += f" | {url}"
            parts.append(f"{header}\n{d.page_content}")
            idx += 1

    context = "\n\n".join(parts)

    prompt = (
        "Ты — ассистент RAG-системы. Пользователь задал СОСТАВНОЙ вопрос "
        "(сравнение, связь, многочастность).\n\n"
        "В контексте ниже документы сгруппированы по подвопросам. "
        "Используй все группы, чтобы дать полный структурированный ответ.\n\n"
        "Правила:\n"
        "  1. Отвечай на русском языке.\n"
        "  2. Структурируй ответ: сначала по каждому пункту, потом синтез.\n"
        "  3. Цитируй источники номерами: [1], [2].\n"
        "  4. Если по какому-то подвопросу нет данных — честно скажи.\n\n"
        f"Вопрос: {query}\n\n"
        f"Контекст (сгруппирован по подвопросам):\n{context}\n\n"
        "Ответ:"
    )

    answer = llm.invoke(prompt).content.strip()

    # собираем sources для UI
    sources = []
    for i, d in enumerate(docs, 1):
        sources.append({
            "n": i,
            "title": d.metadata.get("title", "(без названия)"),
            "url": d.metadata.get("url", ""),
            "source": d.metadata.get("source", "?"),
            "date": d.metadata.get("date", ""),
            "origin_question": d.metadata.get("origin_question", ""),
            "content": d.page_content[:600],
        })

    return {
        "answer": answer,
        "sources": sources,
        "generation_mode": "composite",
        "steps_log": _log(state, f"Синтез ответа по {len(grouped)} подвопросам"),
    }


def translate_query(state: RAGState) -> dict:
    if state.get("query_type") != "arxiv":
        return {"query_en": state["query"]}

    llm = get_llm()
    query = state.get("query_corrected") or state["query"]
    prompt = (
        "Переведи запрос с русского на английский для поиска в научных статьях arXiv.\n"
        "Правила:\n"
        "  • специальные термины, аббревиатуры, имена, формулы, LaTeX — не переводи;\n"
        "  • сохраняй регистр и научный стиль;\n"
        "  • верни ТОЛЬКО перевод, без пояснений и кавычек.\n\n"
        f"Запрос: {state['query']}"
    )
    try:
        query_en = llm.invoke(prompt).content.strip()
    except Exception as e:
        query_en = query   # fallback
    if not query_en or len(query_en) > 500:
        query_en = query
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
    search_query = (
            state.get("query_refined")  # ← приоритет 1: после refine
            or state.get("query_en")  # ← приоритет 2: перевод
            or state.get("query_corrected")  # ← приоритет 3: после spellcheck
            or state["query"]  # ← fallback: оригинал
    )
    docs = retriever.invoke(search_query)

    seen = set()
    unique = []
    for d in docs:
        key = d.page_content[:200]
        if key not in seen:
            seen.add(key)
            unique.append(d)

    limit = state.get("settings_top_k", 15)
    unique = unique[:limit]

    return {
        "documents": unique,
        "steps_log": _log(
            state,
            f"[{state['query_type']}] {len(unique)} уникальных из {len(docs)}",
        ),
    }


def grade_documents(state: RAGState) -> dict:
    """
    CRAG-оценка релевантности документов.
    ВАЖНО: батч-режим — один LLM-вызов на все документы вместо N отдельных.
    Экономия: ~60% input-токенов (4000 → 1500 для 20 документов).

    Возвращает:
      • documents — только релевантные
      • grade — relevant / ambiguous / irrelevant
    """
    docs = state.get("documents", [])
    if not docs:
        return {
            "grade": "irrelevant",
            "steps_log": _log(state, "Оценка: нет документов"),
        }

    llm = get_llm()

    docs_block = "\n\n".join(
        f"[{i}] {d.page_content[:500]}"
        for i, d in enumerate(docs, 1)  # нумерация с 1
    )

    prompt = (
        "Ты — оценщик релевантности документов в RAG-системе.\n"
        "Для каждого документа определи, ПОМОГАЕТ ли он ответить на вопрос.\n\n"
        "Правила:\n"
        "  • 'relevant' — содержит прямой ответ или ключевые факты по теме\n"
        "  • 'irrelevant' — не по теме, общие слова, другая тематика\n"
        "  • сомневаешься → 'irrelevant' (лучше отсеять, чем пустить мусор)\n\n"
        "Верни ТОЛЬКО валидный JSON без markdown-обёрток:\n"
        '{"relevant": [1, 3, 7], "irrelevant": [2, 4, 5, 6, 8]}\n'
        "Где числа — номера документов в квадратных скобках.\n\n"
        f"Вопрос: {state['query']}\n\n"
        f"Документы:\n{docs_block}\n\n"
        "JSON:"
    )

    # --- Вызов LLM с fallback ---
    try:
        raw = llm.invoke(prompt).content.strip()
    except Exception as e:
        # LLM упал — не фильтруем, пропускаем всё в generate
        return {
            "grade": "ambiguous",
            "steps_log": _log(state, f"Оценка: LLM упал ({e}), пропускаем все"),
        }

    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    relevant_ids = set()
    parse_ok = False
    try:
        # Ищем первый { ... } — на случай, если LLM добавил текст до/после
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            relevant_ids = set(int(i) for i in data.get("relevant", []))
            parse_ok = True
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"[grade] JSON parse error: {e}, raw={raw[:200]}")

    # --- Fallback: если JSON сломан, используем эвристику ---
    if not parse_ok:
        # Простая эвристика: считаем релевантными документы,
        # где есть хотя бы одно слово из запроса длиннее 4 символов
        query_words = {
            w.lower() for w in re.findall(r"\w{5,}", state["query"])
        }
        for i, d in enumerate(docs, 1):
            doc_lower = d.page_content.lower()
            if any(w in doc_lower for w in query_words):
                relevant_ids.add(i)
        # если эвристика ничего не нашла — берём топ-5 (они ближе всего по эмбеддингу)
        if not relevant_ids:
            relevant_ids = set(range(1, min(6, len(docs) + 1)))

    # --- Формируем итоговый список ---
    relevant = [d for i, d in enumerate(docs, 1) if i in relevant_ids]
    n_rel = len(relevant)
    n_total = len(docs)

    if n_rel == 0:
        grade = "irrelevant"
    elif n_rel == n_total:
        grade = "relevant"
    else:
        grade = "ambiguous"

    return {
        "documents": relevant,
        "grade": grade,
        "steps_log": _log(
            state,
            f"Оценка: {grade} ({n_rel}/{n_total} релевантных, "
            f"{'batch' if parse_ok else 'fallback'})",
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

def refine_query(state: RAGState) -> dict:
    """
    Переформулирует запрос, если grade признал результаты нерелевантными.
    Вызывается из should_retry при retry_count=0 и плохом grade.
    Возвращает новый запрос → граф идёт обратно в retrieve.
    """
    llm = get_llm()
    query_type = state.get("query_type", "general")

    # Для arxiv работаем с английским, для остальных — с русским
    if query_type == "arxiv":
        working_query = state.get("query_en") or state["query"]
    else:
        working_query = state.get("query_corrected") or state["query"]

    grade = state.get("grade", "?")
    n_docs = len(state.get("documents", []))

    prompt = (
        "Ты — переформулировщик поисковых запросов RAG-системы.\n\n"
        f"Источник: {query_type}\n"
        f"Исходный запрос: {working_query}\n"
        f"Предыдущая попытка: grade={grade}, релевантных={n_docs}\n\n"
        "Переформулируй запрос так, чтобы получить больше релевантных результатов:\n"
        "  1. Используй синонимы и альтернативные термины.\n"
        "  2. Добавь конкретики (уточни технологию, аспект, контекст).\n"
        "  3. Убери общие слова.\n"
        "  4. Сохрани исходный смысл.\n"
        "  5. Для arxiv — сохрани английский язык.\n\n"
        "Верни ТОЛЬКО новый запрос — без пояснений, без кавычек, одной строкой."
    )

    try:
        refined = llm.invoke(prompt).content.strip()
    except Exception:
        refined = working_query

    # Очистка и защита
    refined = refined.strip("\"'`«» ")
    if not refined or len(refined) > 500 or refined.lower() == working_query.lower():
        # Fallback: расширим запрос контекстными словами
        refined = f"{working_query} подробно объяснение примеры"

    retry = state.get("retry_count", 0) + 1

    return {
        "query_refined": refined,
        "retry_count": retry,
        "steps_log": _log(
            state,
            f"Переформулировка #{retry}: «{working_query}» → «{refined}»",
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

    query = state.get("query_corrected") or state["query"]

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
            f"Вопрос: {query}\n\n"
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
            f"Вопрос: {query}\n\n"
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
    После grade:
      • хороший grade → generate
      • плохой + retry_count=0 → refine (переформулируем → ещё раз retrieve)
      • плохой + retry_count=1 → web_search
      • плохой + retry_count>=2 → generate (сдаёмся)
    """
    grade = state.get("grade")
    n_rel = len(state.get("documents", []))
    retry_count = state.get("retry_count", 0)
    is_bad = grade == "irrelevant" or n_rel < 5

    if not is_bad:
        return "generate"
    if retry_count == 0:
        return "refine"
    if retry_count == 1:
        return "web_search"
    return "generate"