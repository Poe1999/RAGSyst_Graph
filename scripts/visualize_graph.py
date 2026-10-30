import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph import build_graph

graph = build_graph().get_graph()

# 1) PNG через mermaid.ink
try:
    png = graph.draw_mermaid_png()
    Path("graph.png").write_bytes(png)
    print("OK: graph.png")
except Exception as e:
    print(f"mermaid.ink недоступен ({e})")

    # 2) Локальный graphviz
    try:
        png = graph.draw_png()
        Path("graph.png").write_bytes(png)
        print("OK: graph.png (graphviz)")
    except Exception as e2:
        print(f"graphviz не работает ({e2})")

        # 3) Mermaid для ручной конвертации
        Path("graph.mmd").write_text(graph.draw_mermaid(), encoding="utf-8")
        print("Сохранён graph.mmd → https://mermaid.live")