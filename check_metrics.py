import json
import re
import statistics
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEV_PATH = PROJECT_DIR / "dev.jsonl"
BENCHMARK_PATH = PROJECT_DIR / "benchmark.jsonl"
OUTPUT_DIR = PROJECT_DIR / "outputs"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.jsonl"
BENCHMARK_RESULTS_PATH = OUTPUT_DIR / "benchmark_results.json"
GRID_PATH = OUTPUT_DIR / "parameter_grid.json"
REPORT_PATH = OUTPUT_DIR / "report.json"

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
VM_RATE_PER_HOUR = 65.0
DEFAULT_PARAMS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 20,
    "repetition_penalty": 1.1,
    "max_tokens": 350,
}

EXPECTED_FIELDS = {"product_id", "description", "pros", "cons", "tags"}
MIN_DESCRIPTION_LENGTH = 120
MAX_DESCRIPTION_LENGTH = 700
MIN_DESCRIPTION_SENTENCES = 2
MAX_DESCRIPTION_SENTENCES = 5
MIN_PROS, MAX_PROS = 2, 5
MIN_CONS, MAX_CONS = 1, 3
MIN_TAGS, MAX_TAGS = 3, 8


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def count_sentences(text: str) -> int:
    return len([
        s for s in re.split(r"[.!?]+(?!\d)", text)
        if s.strip()
    ])


def validate_card(card: dict, expected_product_id: str) -> list[str]:
    errors = []
    if not isinstance(card, dict):
        return ["не JSON-объект"]

    received = set(card)
    if EXPECTED_FIELDS - received:
        errors.append(f"нет полей: {sorted(EXPECTED_FIELDS - received)}")
    if received - EXPECTED_FIELDS:
        errors.append(f"лишние поля: {sorted(received - EXPECTED_FIELDS)}")

    if card.get("product_id") != expected_product_id:
        errors.append("product_id не совпал")

    desc = card.get("description")
    if not isinstance(desc, str) or not desc.strip():
        errors.append("пустой description")
    else:
        desc = desc.strip()
        if not (MIN_DESCRIPTION_LENGTH <= len(desc) <= MAX_DESCRIPTION_LENGTH):
            errors.append("длина description")
        n = count_sentences(desc)
        if not (MIN_DESCRIPTION_SENTENCES <= n <= MAX_DESCRIPTION_SENTENCES):
            errors.append("предложений")

    for name, arr, lo, hi in (
        ("pros", card.get("pros"), MIN_PROS, MAX_PROS),
        ("cons", card.get("cons"), MIN_CONS, MAX_CONS),
        ("tags", card.get("tags"), MIN_TAGS, MAX_TAGS),
    ):
        if not isinstance(arr, list):
            errors.append(f"тип {name}")
        elif not (lo <= len(arr) <= hi):
            errors.append(f"кол-во {name}")
        elif not all(isinstance(i, str) and i.strip() for i in arr):
            errors.append(f"пустые {name}")

    return errors


def percentile(v, q):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(int(q * len(s)), len(s) - 1)]


def main():
    products = load_jsonl(BENCHMARK_PATH)
    predictions = load_jsonl(PREDICTIONS_PATH)

    n = len(predictions)
    print(f"predictions: {n}, benchmark: {len(products)}")
    if n != len(products):
        print("⚠ количество не совпадает")

    # --- валидация каждой карточки ---
    valid = 0
    reasons = Counter()
    for i, pred in enumerate(predictions):
        expected_id = products[i]["product_id"] if i < len(products) else None
        card = pred.get("card", pred)
        errs = validate_card(card, expected_id)
        if errs:
            for e in errs:
                reasons[e.split(":")[0].strip()] += 1
        else:
            valid += 1

    # --- метрики ---
    lat = [p["elapsed_sec"] for p in predictions]
    in_tok = sum(p["input_tokens"] for p in predictions)
    out_tok = sum(p["output_tokens"] for p in predictions)

    # --- время финального прогона из benchmark_results.json ---
    elapsed_total = round(sum(lat), 1)
    concurrency = 8
    params = DEFAULT_PARAMS
    bench_rows = []

    if BENCHMARK_RESULTS_PATH.exists():
        payload = json.loads(BENCHMARK_RESULTS_PATH.read_text(encoding="utf-8"))

        # поддержка и нового формата (dict), и старого (list)
        if isinstance(payload, dict):
            bench_rows = payload.get("rows", [])
            final_run = payload.get("final_run", {})
            if final_run:
                elapsed_total = final_run["wall_sec"]     # ← время на все 300
                concurrency = final_run["concurrency"]
                params = final_run.get("params", DEFAULT_PARAMS)
            elif bench_rows:
                base = bench_rows[0]["valid_rate"]
                ok = [r for r in bench_rows if r["valid_rate"] >= 0.98 * base]
                wp = max(ok, key=lambda r: r["throughput_rps"]) if ok else bench_rows[0]
                elapsed_total = wp["wall_sec"]
                concurrency = wp["concurrency"]
        else:
            # старый формат — список
            bench_rows = payload
            if bench_rows:
                base = bench_rows[0]["valid_rate"]
                ok = [r for r in bench_rows if r["valid_rate"] >= 0.98 * base]
                wp = max(ok, key=lambda r: r["throughput_rps"]) if ok else bench_rows[0]
                elapsed_total = wp["wall_sec"]
                concurrency = wp["concurrency"]

    cost = (elapsed_total / 3600 * VM_RATE_PER_HOUR) / n * 1000

    # --- report.json ---
    report = {}
    if REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    report["run"] = {
        "model": MODEL,
        "concurrency": concurrency,
        "params": params,
        "elapsed_sec": round(elapsed_total, 1),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "n": n,
        "avg_latency": round(statistics.mean(lat), 2) if lat else 0.0,
        "avg_attempts": round(
            statistics.mean(p["attempts"] for p in predictions), 2
        ) if predictions else 0.0,
    }
    report["metrics"] = {
        "valid": valid,
        "valid_rate": round(valid / n, 3) if n else 0.0,
        "throughput": round(n / elapsed_total, 2) if elapsed_total else 0.0,
        "time_for_300_sec": round(elapsed_total * 300 / n, 1) if n else 0.0,
        "cost_vllm_per_1000": round(cost, 2),
        "errors_by_reason": dict(reasons.most_common()),
        "latency_p50": round(percentile(lat, 0.50), 3),
        "latency_p95": round(percentile(lat, 0.95), 3),
        "latency_p99": round(percentile(lat, 0.99), 3),
    }

    # таблицы из run_batch
    if bench_rows:
        report["benchmark_table"] = bench_rows
    if GRID_PATH.exists():
        report["param_grid"] = json.loads(GRID_PATH.read_text(encoding="utf-8"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- вывод ---
    print("\n" + "=" * 50)
    print("METRICS")
    print("=" * 50)
    print(f"Total:          {n}")
    print(f"Valid:          {valid}")
    print(f"Invalid:        {n - valid}")
    print(f"Valid rate:     {report['metrics']['valid_rate']:.2%}")
    print(f"Throughput:     {report['metrics']['throughput']} rps")
    print(f"Latency p95:    {report['metrics']['latency_p95']} s")
    print(f"Cost:           {report['metrics']['cost_vllm_per_1000']} ₽ / 1000")
    print(f"Errors:         {report['metrics']['errors_by_reason']}")
    print("=" * 50)

    if report["metrics"]["valid_rate"] >= 0.80:
        print("✅ Требование качества >= 80% выполнено")
    else:
        print("❌ Требование качества >= 80% НЕ выполнено")

    print(f"\nreport: {REPORT_PATH}")


if __name__ == "__main__":
    main()