import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph import build_graph

g = build_graph()
png = g.get_graph().draw_mermaid_png()
out = Path(__file__).parent.parent / "graph.png"
out.write_bytes(png)
print("Сохранено:", out)