import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.graph import build_graph

st.set_page_config(page_title="LangGraph RAG", layout="wide")
st.title("🔎 RAG-система на LangGraph")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

query = st.text_input("Введите запрос:")
show_steps = st.checkbox("Показывать шаги графа", value=True)
show_sources = st.checkbox("Показывать источники и чанки", value=True)

if st.button("Найти") and query:
    init_state = {
        "query": query, "query_en": "", "query_type": "",
        "documents": [], "web_results": [], "answer": "",
        "sources": [], "generation_mode": "",
        "grade": "", "retry_count": 0, "steps_log": [],
    }

    placeholder_steps = st.empty()
    final_state = init_state

    for event in st.session_state.graph.stream(init_state):
        node = list(event.keys())[0]
        final_state = event[node]
        if show_steps and "steps_log" in final_state:
            with placeholder_steps.container():
                st.markdown("**Промежуточные шаги:**")
                for s in final_state["steps_log"]:
                    st.write(f"- {s}")

    mode = final_state.get("generation_mode", "?")
    badge = "📚 RAG" if mode == "rag" else "🧠 LLM (свободная генерация)"
    st.markdown(f"### 💬 Ответ  ·  _{badge}_")
    st.write(final_state.get("answer", "—"))

    sources = final_state.get("sources", [])
    if show_sources and sources:
        st.markdown(f"### 📎 Источники ({len(sources)})")
        for s in sources:
            with st.expander(f"[{s['n']}] {s['title']}  —  {s['source']}"):
                if s["url"]:
                    st.markdown(f"**URL:** [{s['url']}]({s['url']})")
                if s["date"]:
                    st.write(f"**Дата:** {s['date']}")
                st.markdown("**Чанк:**")
                st.code(s["content"], language=None)
    elif show_sources:
        st.info("Источники отсутствуют — ответ сгенерирован из знаний LLM.")