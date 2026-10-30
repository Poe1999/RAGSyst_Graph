"""
RAG-система на LangGraph — современный UI.
Запуск: streamlit run ui/app.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd

from src.graph import build_graph
from src.cost_tracker import CostTracker


# ---------------------------------------------------------------
# Страница
# ---------------------------------------------------------------
st.set_page_config(
    page_title="RAG · LangGraph",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------
# Кастомный CSS: бейджи, карточки источников
# ---------------------------------------------------------------
st.markdown("""
<style>
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 4px;
}
.badge-chat      { background: #d4edda; color: #155724; }
.badge-search    { background: #fff3cd; color: #856404; }
.badge-composite { background: #ffe5b4; color: #8a4b00; }
.badge-general   { background: #e2e3e5; color: #383d41; }
.badge-habr      { background: #e8f0fe; color: #1967d2; }
.badge-lenta     { background: #fce8e6; color: #c5221f; }
.badge-arxiv     { background: #f3e8fd; color: #7b1fa2; }
.badge-mode-rag  { background: #cfe2ff; color: #084298; }
.badge-mode-free { background: #f8d7da; color: #721c24; }
.badge-mode-composite { background: #d1e7dd; color: #0f5132; }
.cost-line {
    font-family: ui-monospace, monospace;
    font-size: 0.8rem;
    color: #6c757d;
    margin: 4px 0 12px 0;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Session state
# ---------------------------------------------------------------
def init_session():
    defaults = {
        "graph": None,
        "history": [],
        "session_cost": 0.0,
        "session_in": 0,
        "session_out": 0,
        "session_calls": 0,
        "settings": {
            "top_k_grade": 15,
            "show_steps": True,
            "show_sources": True,
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.graph is None:
        with st.spinner("Инициализация графа..."):
            st.session_state.graph = build_graph()


init_session()


# ---------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------
def badge(text, kind):
    return f'<span class="badge badge-{kind}">{text}</span>'


def intent_badge(intent):
    emoji = {"chat": "🧠", "search": "📚", "composite": "🔀"}.get(intent, "❓")
    kind = intent if intent in ("chat", "search", "composite") else "general"
    return badge(f"{emoji} {intent}", kind)


def source_badge(source):
    kind = source if source in ("habr", "lenta", "arxiv") else "general"
    return badge(source, kind)


def mode_badge(mode):
    labels = {"rag": "📚 RAG", "free": "🧠 LLM", "composite": "🔀 Composite"}
    return badge(labels.get(mode, mode or "?"), f"mode-{mode or 'free'}")


def fmt_rub(x):
    return f"{x:.4f} ₽"


def fmt_tok(n):
    return f"{n:,}".replace(",", " ")


# ---------------------------------------------------------------
# Вкладки
# ---------------------------------------------------------------
tab_chat, tab_dash, tab_settings, tab_about = st.tabs([
    "💬 Чат", "📊 Cost Dashboard", "⚙️ Настройки", "ℹ️ О системе",
])


# ===============================================================
# ВКЛАДКА 1: ЧАТ
# ===============================================================
with tab_chat:
    st.markdown("### 💬 Задайте вопрос")
    st.caption(
        "Система определит тип запроса (chat / search / composite), "
        "исправит опечатки, найдёт документы и сгенерирует ответ"
    )

    query = st.text_input(
        "Запрос:",
        placeholder="Например: Как работает LangGraph?",
        label_visibility="collapsed",
        key="query_input",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        send = st.button("🚀 Отправить", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑 Очистить историю", use_container_width=False):
            st.session_state.history = []
            st.session_state.session_cost = 0
            st.session_state.session_in = 0
            st.session_state.session_out = 0
            st.session_state.session_calls = 0
            st.rerun()

    if send and query.strip():
        init_state = {
            "query": query, "query_corrected": "", "query_refined": "",
            "query_intent": "", "sub_questions": [], "query_en": "",
            "query_type": "", "documents": [], "web_results": [], "answer": "",
            "sources": [], "generation_mode": "", "grade": "", "retry_count": 0,
            "steps_log": [],
            "settings_top_k": st.session_state.settings["top_k_grade"],
        }

        tracker = CostTracker()
        final = dict(init_state)
        steps_ph = st.empty()
        t0 = time.time()

        with st.spinner("Обработка запроса..."):
            try:
                for event in st.session_state.graph.stream(
                    init_state, config={"callbacks": [tracker]}
                ):
                    node = list(event.keys())[0]
                    if isinstance(event[node], dict):
                        final.update(event[node])
                    if st.session_state.settings["show_steps"]:
                        with steps_ph.container():
                            st.markdown("**🔄 Выполняется:**")
                            for s in final.get("steps_log", []):
                                st.caption(f"• {s}")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")
                st.stop()

        elapsed = time.time() - t0
        steps_ph.empty()

        run = {
            "query": query,
            "answer": final.get("answer", ""),
            "intent": final.get("query_intent", "?"),
            "source": final.get("query_type", "?"),
            "mode": final.get("generation_mode", "?"),
            "grade": final.get("grade", ""),
            "sources": final.get("sources", []),
            "steps_log": final.get("steps_log", []),
            "sub_questions": final.get("sub_questions", []),
            "cost": tracker.total_cost,
            "tokens_in": tracker.total_input,
            "tokens_out": tracker.total_output,
            "calls": tracker.total_calls,
            "time": elapsed,
            "nodes": dict(tracker.node_stats),
        }
        st.session_state.history.append(run)
        st.session_state.session_cost += tracker.total_cost
        st.session_state.session_in += tracker.total_input
        st.session_state.session_out += tracker.total_output
        st.session_state.session_calls += tracker.total_calls

    # --- История ---
    if st.session_state.history:
        st.markdown("---")
        st.markdown(f"### 📜 История ({len(st.session_state.history)})")

        for idx, run in enumerate(reversed(st.session_state.history)):
            real_idx = len(st.session_state.history) - idx - 1

            st.markdown(f"**❓ {run['query']}**")
            st.markdown(
                intent_badge(run["intent"])
                + source_badge(run["source"])
                + mode_badge(run["mode"]),
                unsafe_allow_html=True,
            )

            st.markdown(run["answer"])

            st.markdown(
                f'<div class="cost-line">'
                f'💰 <b>{fmt_rub(run["cost"])}</b> &nbsp;·&nbsp; '
                f'in: {fmt_tok(run["tokens_in"])} · out: {fmt_tok(run["tokens_out"])} '
                f'&nbsp;·&nbsp; calls: {run["calls"]} &nbsp;·&nbsp; ⏱ {run["time"]:.1f} с'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Composite sub_questions
            if run["sub_questions"] and run["intent"] == "composite":
                with st.expander(f"🔀 Подвопросы ({len(run['sub_questions'])})"):
                    for sq in run["sub_questions"]:
                        if isinstance(sq, dict):
                            st.caption(f"• {sq.get('question', '?')} → {sq.get('source', '?')}")
                        else:
                            st.caption(f"• {sq}")

            # Steps
            if run["steps_log"] and st.session_state.settings["show_steps"]:
                with st.expander(f"🔄 Шаги графа ({len(run['steps_log'])})"):
                    for s in run["steps_log"]:
                        st.caption(f"• {s}")

            # Sources
            if run["sources"] and st.session_state.settings["show_sources"]:
                with st.expander(f"📎 Источники ({len(run['sources'])})"):
                    for src in run["sources"]:
                        title = src.get("title") or "—"
                        st.markdown(f"**[{src.get('n', '?')}]** {title}")
                        if src.get("origin_question"):
                            st.caption(f"↳ подвопрос: {src['origin_question'][:100]}")
                        if src.get("url"):
                            st.markdown(f"🔗 [{src['url']}]({src['url']})")
                        if src.get("date"):
                            st.caption(f"📅 {src['date']}")
                        st.code((src.get("content") or "")[:500], language=None)
                        st.markdown("---")

            st.markdown("---")


# ===============================================================
# ВКЛАДКА 2: COST DASHBOARD
# ===============================================================
with tab_dash:
    st.markdown("### 📊 Статистика сессии")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Запросов", len(st.session_state.history))
    c2.metric("Стоимость", fmt_rub(st.session_state.session_cost))
    c3.metric("Токенов (in / out)",
              f"{fmt_tok(st.session_state.session_in)} / {fmt_tok(st.session_state.session_out)}")
    c4.metric("LLM-вызовов", st.session_state.session_calls)

    st.markdown("---")

    if not st.session_state.history:
        st.info("📭 Пока нет запросов. Задайте первый в вкладке «Чат».")
    else:
        # --- Разбивка по узлам (последний запрос) ---
        st.markdown("### 🔍 Разбивка по узлам (последний запрос)")
        last = st.session_state.history[-1]

        if last["nodes"]:
            rows = []
            for name, s in last["nodes"].items():
                cost = (s["input"] + s["output"]) * 0.000065
                rows.append({
                    "Узел": name,
                    "Вызовов": s["calls"],
                    "Input": s["input"],
                    "Output": s["output"],
                    "Стоимость ₽": round(cost, 6),
                    "Время (с)": round(s["time"], 2),
                })
            df_nodes = pd.DataFrame(rows).sort_values("Стоимость ₽", ascending=False)
            st.dataframe(df_nodes, use_container_width=True, hide_index=True)
            if (df_nodes["Стоимость ₽"] > 0).any():
                st.bar_chart(df_nodes.set_index("Узел")["Стоимость ₽"], use_container_width=True)
        else:
            st.caption("(в последнем запросе не было LLM-вызовов)")

        st.markdown("---")

        # --- История ---
        st.markdown("### 📈 История запросов")
        df_hist = pd.DataFrame([
            {
                "#": i + 1,
                "Запрос": r["query"][:60] + ("…" if len(r["query"]) > 60 else ""),
                "Intent": r["intent"],
                "Source": r["source"],
                "Стоимость ₽": round(r["cost"], 4),
                "In": r["tokens_in"],
                "Out": r["tokens_out"],
                "Calls": r["calls"],
                "Время (с)": round(r["time"], 1),
            }
            for i, r in enumerate(st.session_state.history)
        ])
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        if len(df_hist) > 1:
            st.line_chart(df_hist.set_index("#")["Стоимость ₽"], use_container_width=True)

        # --- Средняя стоимость по типу ---
        st.markdown("### 💰 Средняя стоимость по типу")
        stats = {}
        for r in st.session_state.history:
            it = r["intent"]
            if it not in stats:
                stats[it] = {"n": 0, "cost": 0.0}
            stats[it]["n"] += 1
            stats[it]["cost"] += r["cost"]

        cols = st.columns(max(1, len(stats)))
        for col, (it, s) in zip(cols, stats.items()):
            col.metric(it, fmt_rub(s["cost"] / s["n"]), f"{s['n']} запр.")


# ===============================================================
# ВКЛАДКА 3: НАСТРОЙКИ
# ===============================================================
with tab_settings:
    st.markdown("### ⚙️ Настройки системы")
    st.caption("Часть параметров применяется сразу, часть — только после перезапуска Streamlit")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 🤖 Модель LLM")
        st.code("GigaChat-2 (Lite)", language=None)
        st.caption(
            "Смена модели — через `.env`: `GIGACHAT_MODEL=GigaChat-2-Max`. "
            "После смены перезапустите Streamlit."
        )

        st.markdown("#### 💰 Тарифы")
        st.caption("• GigaChat-2 Lite: **65 ₽** / 1M токенов")
        st.caption("• GigaChat-2 Max: ~**600 ₽** / 1M токенов")

        st.markdown("#### 🗂 Источники")
        st.caption("Habr · Lenta · arXiv — индексируются локально (FAISS + BM25)")

    with col_r:
        st.markdown("#### 🎛 Параметры retrieval")
        st.session_state.settings["top_k_grade"] = st.slider(
            "Документов на grade",
            min_value=5, max_value=20,
            value=st.session_state.settings["top_k_grade"],
            help="Больше — точнее, но дороже (растёт input на grade)",
        )

        st.markdown("#### 👁 Отображение")
        st.session_state.settings["show_steps"] = st.checkbox(
            "Показывать шаги графа",
            value=st.session_state.settings["show_steps"],
        )
        st.session_state.settings["show_sources"] = st.checkbox(
            "Показывать источники и чанки",
            value=st.session_state.settings["show_sources"],
        )

    st.markdown("---")
    st.markdown("#### 🔄 Действия")

    a, b, c = st.columns(3)
    with a:
        if st.button("🔁 Пересоздать граф", use_container_width=True):
            st.session_state.graph = None
            st.rerun()
    with b:
        if st.button("🗑 Сбросить статистику", use_container_width=True):
            st.session_state.history = []
            st.session_state.session_cost = 0
            st.session_state.session_in = 0
            st.session_state.session_out = 0
            st.session_state.session_calls = 0
            st.rerun()
    with c:
        if st.button("🗑 Сбросить всё", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ===============================================================
# ВКЛАДКА 4: О СИСТЕМЕ
# ===============================================================
with tab_about:
    st.markdown("### ℹ️ RAG-система на LangGraph")

    st.markdown("""
**Гибридный RAG по трём источникам** — Habr, Lenta, arXiv.

#### 🎯 Возможности
| | |
|---|---|
| **Классификация** | chat / search / composite |
| **Опечатки** | pymorphy3 + pyspellchecker + доменный словарь |
| **Гибридный поиск** | BM25 + FAISS (IVFPQ) через RRF |
| **Самокоррекция** | CRAG + refine_query → web_search (до 2 retry) |
| **Кросс-язычность** | RU → arXiv EN через GigaChat-перевод |
| **Multi-hop** | 2–3 подвопроса для composite |
| **Учёт стоимости** | токены + ₽ по каждому узлу |

#### 🗂 Источники
- **Habr** — IT-статьи (~200 000 чанков)
- **Lenta** — новости 2000–2019 (~35 000 чанков)
- **arXiv** — hep-th, gr-qc, cs.LG (~184 000 чанков)

#### 🧠 Стек
- **LangGraph 1.0** — оркестрация (граф с состоянием)
- **GigaChat-2** — LLM (classify, grade, generate)
- **multilingual-e5-small** — эмбеддинги (ONNX int8, CPU)
- **FAISS IVFPQ** — векторный поиск (сжатие 24×)
- **BM25** — точные ключевые слова
- **Streamlit** — UI
""")

    # --- Граф ---
    graph_png = Path("graph.png")
    if graph_png.exists():
        st.markdown("#### 🔀 Граф системы")
        st.image(str(graph_png), use_container_width=True)
    else:
        st.info(
            "Граф не найден. Сгенерируйте: "
            "`python scripts/visualize_graph.py`"
        )