# scripts/run_test_suite.py
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph import build_graph
from src.cost_tracker import CostTracker

TEST_CASES = json.loads(Path("test_cases.json").read_text(encoding="utf-8"))


def run_one(graph, case):
    init_state = {
        "query": case["query"], "query_corrected": "", "query_refined": "",
        "query_intent": "", "sub_questions": [], "query_en": "",
        "query_type": "", "documents": [], "web_results": [], "answer": "",
        "sources": [], "generation_mode": "", "grade": "", "retry_count": 0,
        "steps_log": [],
    }
    tracker = CostTracker()   # ← новый трекер на каждый запрос
    t0 = time.time()
    visited = []
    final = dict(init_state)

    try:
        # передаём callback в граф
        for event in graph.stream(init_state, config={"callbacks": [tracker]}):
            node = list(event.keys())[0]
            visited.append(node)
            if isinstance(event[node], dict):
                final.update(event[node])
    except Exception as e:
        return {"error": str(e), "elapsed": time.time() - t0, "visited": visited}

    return {
        "intent": final.get("query_intent", "?"),
        "source": final.get("query_type", "?"),
        "subs_count": len(final.get("sub_questions", [])),
        "corrected": final.get("query_corrected", ""),
        "grade": final.get("grade", ""),
        "mode": final.get("generation_mode", ""),
        "n_docs": len(final.get("documents", [])),
        "n_web": len(final.get("web_results", [])),
        "visited": visited,
        "elapsed": time.time() - t0,
        "answer": final.get("answer", "")[:150],
        # --- метрики стоимости ---
        "tokens_in": tracker.total_input,
        "tokens_out": tracker.total_output,
        "cost": tracker.total_cost,
        "calls": tracker.total_calls,
        "nodes_cost": tracker.node_stats,   # dict: узел → {input, output, calls, time}
    }


def print_node_breakdown(nodes_cost: dict, indent: str = "         "):
    """Печатает разбивку по узлам, отсортированную по стоимости."""
    if not nodes_cost:
        return
    items = sorted(
        nodes_cost.items(),
        key=lambda x: -(x[1]["input"] + x[1]["output"]),
        reverse=False,
    )
    # показываем только узлы, где были LLM-вызовы
    items = [(n, s) for n, s in items if s["calls"] > 0]
    if not items:
        return
    print(f"{indent}по узлам:")
    for name, s in items:
        t_in, t_out = s["input"], s["output"]
        c = s["calls"]
        print(f"{indent}  {name:<14} {c:>2} вызов(ов)  "
              f"in={t_in:>6}  out={t_out:>5}")


def check_answer_quality(r, case):
    """Дополнительные проверки кроме intent/source."""
    checks = []
    answer = r.get("answer", "").lower()
    query_en = r.get("query_en", "").lower()

    # 1. multi_hop: subs_count >= 3
    if case.get("expected_subs_min"):
        ok = r["subs_count"] >= case["expected_subs_min"]
        checks.append(("subs_count", ok, f"{r['subs_count']}/{case['expected_subs_min']}"))

    # 2. expected_answer_contains
    if case.get("expected_answer_contains"):
        missing = [kw for kw in case["expected_answer_contains"] if kw.lower() not in answer]
        ok = not missing
        checks.append(("answer_contains", ok, f"missing={missing}"))

    # 3. expected_refutation (adversarial)
    if case.get("expected_refutation"):
        refutation_words = ["нет,", "неверно", "это не", "не является", "не соответствует", "неправда"]
        ok = any(w in answer for w in refutation_words)
        checks.append(("refutation", ok, "опровержение найдено" if ok else "опровержения НЕТ"))

    # 4. cross_lang: query_en отличается от query
    if case["complexity"].startswith("cross_lang"):
        ok = query_en and query_en != case["query"].lower()
        checks.append(("translated", ok, query_en[:50]))

    return checks

def main():
    print("=" * 90)
    print(f"ТЕСТОВЫЙ НАБОР: {len(TEST_CASES)} вопросов")
    print("=" * 90)

    graph = build_graph()
    passed = 0
    failed = []

    # агрегаты по категориям intent
    stats_by_intent = {}   # intent → {"count", "cost", "t_in", "t_out", "time"}
    total_cost = 0.0
    total_in = 0
    total_out = 0

    for case in TEST_CASES:
        r = run_one(graph, case)
        if "error" in r:
            print(f"❌ #{case['id']} ERROR: {r['error'][:60]}")
            failed.append(case["id"])
            continue

        expected_intent = case["intent"]
        expected_source = case.get("source", "any")

        intent_ok = r["intent"] == expected_intent
        source_ok = (expected_source == "any") or (r["source"] == expected_source)
        ok = intent_ok and source_ok

        # агрегируем по ожидаемому intent (для групповой статистики)
        key = expected_intent
        s = stats_by_intent.setdefault(
            key, {"count": 0, "cost": 0.0, "in": 0, "out": 0, "time": 0.0}
        )
        s["count"] += 1
        s["cost"] += r["cost"]
        s["in"] += r["tokens_in"]
        s["out"] += r["tokens_out"]
        s["time"] += r["elapsed"]

        total_cost += r["cost"]
        total_in += r["tokens_in"]
        total_out += r["tokens_out"]

        if ok:
            passed += 1
            print(f"✅ #{case['id']} [{r['intent']}/{r['source']}] {case['query'][:50]}")
        else:
            print(f"❌ #{case['id']} ожидалось [{expected_intent}/{expected_source}], "
                  f"получено [{r['intent']}/{r['source']}]")
            print(f"         {case['query'][:70]}")
            if r.get("corrected") and r["corrected"] != case["query"]:
                print(f"         исправлено: {r['corrected'][:70]}")
            failed.append(case["id"])

        print(f"         путь: {' → '.join(r['visited'])}")
        print(f"         docs={r['n_docs']}, web={r['n_web']}, "
              f"grade={r['grade']}, mode={r['mode']}")
        print(f"         💰 in={r['tokens_in']:>6}  out={r['tokens_out']:>5}  "
              f"calls={r['calls']:>2}  cost={r['cost']:.4f} ₽  "
              f"time={r['elapsed']:.1f} с")
        print_node_breakdown(r["nodes_cost"], indent="         ")
        print()

    print("=" * 90)
    print(f"ИТОГО: {passed}/{len(TEST_CASES)} пройдено")
    if failed:
        print(f"Провалы: {failed}")

    print("=" * 90)
    print("СТАТИСТИКА ПО ТИПАМ ЗАПРОСОВ")
    print("=" * 90)
    print(f"{'intent':<12} {'n':>3} {'avg cost':>10} {'avg in':>8} "
          f"{'avg out':>8} {'avg time':>10}")
    print("-" * 70)
    for intent, s in stats_by_intent.items():
        n = s["count"]
        print(f"{intent:<12} {n:>3} "
              f"{s['cost']/n:>9.4f}₽ "
              f"{s['in']/n:>8.0f} "
              f"{s['out']/n:>8.0f} "
              f"{s['time']/n:>8.1f}с")
    print("-" * 70)
    print(f"{'ВСЕГО':<12} {len(TEST_CASES):>3} "
          f"{total_cost:>9.4f}₽ "
          f"{total_in:>8} "
          f"{total_out:>8} "
          f"{sum(s['time'] for s in stats_by_intent.values()):>8.0f}с")
    print("=" * 90)


if __name__ == "__main__":
    main()