import asyncio
import json
import re
import time
from pathlib import Path
import random
import copy
import statistics
import httpx
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    BadRequestError
)
from collections import Counter

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

LOCAL_BASE_URL = "http://localhost:8000/v1"

INPUT_TOKEN_BUDGET = 1500

VM_RATE_PER_HOUR = 65.0

MAX_GENERATION_ATTEMPTS = 3

PROJECT_DIR = Path(__file__).resolve().parent
DEV_PATH = PROJECT_DIR / "dev.jsonl"
BENCHMARK_PATH = PROJECT_DIR / "benchmark.jsonl"
OUTPUT_DIR = PROJECT_DIR / "outputs"
BENCHMARK_RESULTS_PATH = OUTPUT_DIR / "benchmark_results.json"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.jsonl"
REPORT_PATH = OUTPUT_DIR / "report.json"
GRID_PATH = OUTPUT_DIR / "parameter_grid.json"
PARAM_GRID_PATH = OUTPUT_DIR / "parameter_grid.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

local_client = AsyncOpenAI(
    base_url=LOCAL_BASE_URL,
    api_key="EMPTY",
    http_client=httpx.AsyncClient(
        trust_env=False,
        timeout=600,
    ),
)

EXPECTED_FIELDS = {
    "product_id",
    "description",
    "pros",
    "cons",
    "tags",
}

MIN_DESCRIPTION_LENGTH = 120
MAX_DESCRIPTION_LENGTH = 700

MIN_DESCRIPTION_SENTENCES = 2
MAX_DESCRIPTION_SENTENCES = 5

MIN_PROS = 2
MAX_PROS = 5

MIN_CONS = 1
MAX_CONS = 3

MIN_TAGS = 3
MAX_TAGS = 8

SCHEMA = {
    "type": "object",
    "properties": {
        "product_id": {
            "type": "string",
            "minLength": 1
        },
        "description": {
            "type": "string",
            "minLength": 120,
            "maxLength": 700
        },
        "pros": {
            "type": "array",
            "minItems": 2,
            "maxItems": 5,
            "items": {
                "type": "string",
                "minLength": 1
            }
        },
        "cons": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "string",
                "minLength": 1
            }
        },
        "tags": {
            "type": "array",
            "minItems": 3,
            "maxItems": 8,
            "items": {
                "type": "string",
                "minLength": 1
            }
        }
    },
    "required": [
        "product_id",
        "description",
        "pros",
        "cons",
        "tags"
    ],
    "additionalProperties": False
}

DEFAULT_PARAMS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 350,
    "top_k": 20,
    "repetition_penalty": 1.1,
}

SYSTEM_PROMPT = """
Ты создаёшь карточку товара для маркетплейса. Отвечай на русском языке.

Правила по смыслу:

- description — связное описание товара обычным текстом, 2–5 предложений.
  Без списков. Пиши по делу, без воды и рекламных штампов.

- pros — реальные плюсы товара.
  Бери только то, что есть в данных товара и отзывах. Не выдумывай.

- cons — реальные минусы товара.
  Бери только то, что есть в данных товара и отзывах. Не выдумывай.
  Если минусов мало — укажи самый заметный.

- tags — короткие ключевые слова товара.
  Одно-два слова на тег. Без повторов. Без общих слов вроде «товар», «качество».

- product_id — точно скопируй из входного товара.
"""


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


def render_product(product: dict) -> str:
    parts = []

    for key, value in product.items():
        parts.append(f"{key}: {value}")

    return "\n".join(parts)


def count_messages_tokens(messages: list[dict]) -> int:
    """
    Считает токены именно токенайзером используемой модели.
    """

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    token_ids = tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]

    return len(token_ids)


class ResponseValidationError(ValueError):
    """Ошибка парсинга или проверки ответа модели."""

    def __init__(self, message: str, reason: str = "прочее"):
        super().__init__(message)
        self.reason = reason


def build_messages(product: dict):
    product = copy.deepcopy(product)
    while True:
        user_content = render_product(product)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        token_count = count_messages_tokens(messages)

        if token_count <= INPUT_TOKEN_BUDGET:
            return messages, token_count

        product["reviews"].pop()


def count_sentences(text):
    return len(
        [
            s
            for s in re.split(
            r"[.!?]+(?!\d)",
            text,
        )
            if s.strip()
        ]
    )


def parse_and_validate(
        content: str,
        expected_product_id: str,
) -> dict:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise ResponseValidationError(
            f"Ответ не является корректным JSON: {error.msg}",
            reason="json"
        ) from error

    if not isinstance(data, dict):
        raise ResponseValidationError(
            "На верхнем уровне должен находиться JSON-объект",
            reason="структура"
        )

    received_fields = set(data)

    missing_fields = (
            EXPECTED_FIELDS - received_fields
    )

    unexpected_fields = (
            received_fields - EXPECTED_FIELDS
    )

    if missing_fields:
        raise ResponseValidationError(
            "Отсутствуют обязательные поля: "
            f"{sorted(missing_fields)}", reason="структура",
        )

    if unexpected_fields:
        raise ResponseValidationError(
            "Получены лишние поля: "
            f"{sorted(unexpected_fields)}",
            reason="структура",
        )

    if expected_product_id != data["product_id"]:
        raise ResponseValidationError(
            f"Неверный product_id. Ожидали {expected_product_id}. Получили {data['product_id']}",
            reason="product_id",
        )

    description = data["description"]

    if not isinstance(description, str):
        raise ResponseValidationError(
            "Поле description должно быть строкой",
            reason="тип description",
        )

    description = description.strip()

    if not description:
        raise ResponseValidationError(
            "Поле description не должно быть пустым",
            reason="пустой description",
        )

    if not (
            MIN_DESCRIPTION_LENGTH
            <= len(description)
            <= MAX_DESCRIPTION_LENGTH
    ):
        raise ResponseValidationError(
            "Поле description должно содержать "
            f"от {MIN_DESCRIPTION_LENGTH} до "
            f"{MAX_DESCRIPTION_LENGTH} символов",
            reason="длина description",
        )

    sentence_count = count_sentences(description)

    if not (
            MIN_DESCRIPTION_SENTENCES
            <= sentence_count
            <= MAX_DESCRIPTION_SENTENCES
    ):
        raise ResponseValidationError(
            "Поле description должно содержать "
            f"от {MIN_DESCRIPTION_SENTENCES} до "
            f"{MAX_DESCRIPTION_SENTENCES} предложений",
            reason="предложений",
        )

    pros = data["pros"]

    if not isinstance(pros, list):
        raise ResponseValidationError(
            "Поле pros должно быть списком",
            reason="тип pros",
        )

    if not MIN_PROS <= len(pros) <= MAX_PROS:
        raise ResponseValidationError(
            "Поле pros должно содержать "
            f"от {MIN_PROS} до {MAX_PROS} элементов",
            reason="кол-во pros",
        )

    if not all(
            isinstance(item, str) and item.strip()
            for item in pros
    ):
        raise ResponseValidationError(
            "Все элементы pros должны быть "
            "непустыми строками",
            reason="пустые pros",
        )

    cons = data["cons"]

    if not isinstance(cons, list):
        raise ResponseValidationError(
            "Поле cons должно быть списком",
            reason="тип cons",
        )

    if not MIN_CONS <= len(cons) <= MAX_CONS:
        raise ResponseValidationError(
            "Поле cons должно содержать "
            f"от {MIN_CONS} до {MAX_CONS} элементов",
            reason="кол-во cons",
        )

    if not all(
            isinstance(item, str) and item.strip()
            for item in cons
    ):
        raise ResponseValidationError(
            "Все элементы cons должны быть "
            "непустыми строками", reason="пустые cons"
        )

    tags = data["tags"]

    if not isinstance(tags, list):
        raise ResponseValidationError(
            "Поле tags должно быть списком", reason="тип tags",
        )

    if not MIN_TAGS <= len(tags) <= MAX_TAGS:
        raise ResponseValidationError(
            "Поле tags должно содержать "
            f"от {MIN_TAGS} до {MAX_TAGS} элементов",
            reason="кол-во tags",
        )

    if not all(
            isinstance(item, str) and item.strip()
            for item in tags
    ):
        raise ResponseValidationError(
            "Все элементы tags должны быть "
            "непустыми строками",
            reason="пустые tags",
        )

    return {
        "product_id": data["product_id"],
        "description": description,
        "pros": [item.strip() for item in pros],
        "cons": [item.strip() for item in cons],
        "tags": [item.strip() for item in tags],
    }


def calculate_retry_delay(
        attempt: int,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
) -> float:
    """Рассчитать задержку перед следующей попыткой."""

    exponential_delay = base_delay * 2 ** (attempt - 1)
    jitter = random.uniform(0, base_delay)

    return min(
        exponential_delay + jitter,
        max_delay,
    )


async def request_json(
        messages: list[dict],
        params: dict,
):
    extra_body = {
        "repetition_penalty": params["repetition_penalty"],
        "guided_json": SCHEMA,
        "guided_decoding_backend": "xgrammar",
    }

    if params["top_k"] is not None:
        extra_body["top_k"] = params["top_k"]

    response = await local_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=params["temperature"],
        top_p=params["top_p"],
        max_tokens=params["max_tokens"],
        extra_body=extra_body,
    )

    content = response.choices[0].message.content

    if not content:
        raise ResponseValidationError(
            "Модель вернула пустой ответ"
        )

    return content, response.usage


async def generate_one(
        product: dict,
        params: dict | None = None,
):
    started = time.perf_counter()

    messages, input_tokens = build_messages(product)

    total_input_tokens = 0
    total_output_tokens = 0
    attempts = 0
    last_error = None
    last_reason = None
    errors: list[dict] = []
    retryable_errors = (
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        InternalServerError,
    )

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        content = None
        attempts += 1

        try:
            content, usage = await request_json(
                messages=messages,
                params=params,
            )

            total_input_tokens += getattr(
                usage,
                "prompt_tokens",
                input_tokens,
            )

            total_output_tokens += getattr(
                usage,
                "completion_tokens",
                0,
            )

            card = parse_and_validate(
                content,
                expected_product_id=product["product_id"],
            )

            elapsed = time.perf_counter() - started

            return {
                "card": card,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "attempts": attempts,
                "elapsed_sec": elapsed,
                "valid": True,
                "errors": errors,
                "error_reason": None,
            }
        except retryable_errors as error:
            errors.append({
                "attempt": attempt,
                "kind": "retryable",
                "type": type(error).__name__,
                "message": str(error),
            })
            last_error = str(error)
            if attempt == MAX_GENERATION_ATTEMPTS:
                break
            delay = calculate_retry_delay(attempt=attempt)
            await asyncio.sleep(delay)

        except ResponseValidationError as error:
            errors.append({
                "attempt": attempt,
                "kind": "validation",
                "reason": error.reason,
                "type": type(error).__name__,
                "message": str(error),
            })
            print(
                f"\nINVALID product={product['product_id']} "
                f"[{error.reason}]: {error}"
            )
            last_error = f"{error.reason}: {error}"
            last_reason = error.reason

            if content is not None:
                messages.append({
                    "role": "assistant",
                    "content": content,
                })

                messages.append({
                    "role": "user",
                    "content": (
                        "Предыдущий ответ не прошёл проверку. "
                        f"Причина: {last_error}. "
                        "Исправь ответ и верни только JSON."
                    ),
                })

    elapsed = time.perf_counter() - started

    fallback_card = {
        "product_id": product["product_id"],
        "description": "",
        "pros": [],
        "cons": [],
        "tags": [],
    }

    return {
        "card": fallback_card,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "attempts": attempts,
        "elapsed_sec": elapsed,
        "valid": False,
        "errors": errors,
        "last_error": last_error,
        "error_reason": last_reason
    }


async def run_concurrent(
        items,
        concurrency,
        generate_one_func,
        params

):
    sem = asyncio.Semaphore(concurrency)

    async def wrapped(item):
        async with sem:
            return await generate_one_func(item, params)

    started = time.perf_counter()

    results = await asyncio.gather(
        *[
            wrapped(item)
            for item in items
        ]
    )

    elapsed = time.perf_counter() - started

    return results, elapsed


def percentile(v, q):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(int(q * len(s)), len(s) - 1)]


def summarize(results, wall, conc):
    lat = [r["elapsed_sec"] for r in results]
    valid = sum(1 for r in results if r["valid"])

    reasons = Counter(
        e.get("reason", e.get("kind", "?"))
        for r in results
        for e in r.get("errors", [])
    )

    return {
        "concurrency": conc,
        "n": len(results),
        "wall_sec": round(wall, 2),
        "throughput_rps": round(len(results) / wall, 2),
        "latency_mean": round(statistics.mean(lat), 3),
        "latency_p50": round(percentile(lat, 0.50), 3),
        "latency_p95": round(percentile(lat, 0.95), 3),
        "latency_p99": round(percentile(lat, 0.99), 3),
        "input_tokens": sum(r["input_tokens"] for r in results),
        "output_tokens": sum(r["output_tokens"] for r in results),
        "valid": valid,
        "valid_rate": round(valid / len(results), 3),
        "attempts_mean": round(
            statistics.mean(r["attempts"] for r in results), 2
        ),
        "errors_by_reason": dict(reasons.most_common()),
    }


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = load_jsonl(DEV_PATH)
    bench = load_jsonl(BENCHMARK_PATH)
    print(f"dev: {len(dev)}, benchmark: {len(bench)}\n")

    # --- Сетка параметров ---
    print("=== Сетка параметров ===")
    base = dict(DEFAULT_PARAMS)
    grid = [
        base,
        {**base, "temperature": 0.0},
        {**base, "temperature": 0.7},
        {**base, "top_p": 1.0},
        {**base, "top_k": 50},
        {**base, "top_k": None},
        {**base, "repetition_penalty": 1.0},
        {**base, "repetition_penalty": 1.1},
        {**base, "max_tokens": 250},
        {**base, "max_tokens": 500},
    ]
    grid_rows = []
    for i, params in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}] {params}")
        results, wall = await run_concurrent(dev[:20], 4, generate_one, params)
        s = summarize(results, wall, 4)
        descs = [r["card"]["description"] for r in results if r["valid"]]
        grid_rows.append({
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "top_k": params["top_k"],
            "repetition_penalty": params["repetition_penalty"],
            "max_tokens": params["max_tokens"],
            "valid_rate": s["valid_rate"],
            "avg_description_len": round(statistics.mean(len(d) for d in descs)) if descs else 0,
            "avg_output_tokens": round(statistics.mean(r["output_tokens"] for r in results), 1),
        })
    best = max(grid_rows, key=lambda r: (r["valid_rate"], -r["avg_output_tokens"]))
    tuned = {k: best[k] for k in DEFAULT_PARAMS}
    print(f"Лучшие: {tuned}\n")
    GRID_PATH.write_text(
        json.dumps(grid_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- Бенчмарк по concurrency ---
    print("=== Бенчмарк ===")
    await run_concurrent(bench[:4], 4, generate_one, tuned)   # прогрев
    bench_rows = []
    for conc in (4, 8, 16):
        print(f"conc={conc}...", end=" ", flush=True)
        results, wall = await run_concurrent(bench[:100], conc, generate_one, tuned)
        s = summarize(results, wall, conc)
        bench_rows.append(s)
        print(f"rps={s['throughput_rps']} p95={s['latency_p95']}s valid={s['valid_rate']}")

    # --- Рабочая точка ---
    base_valid = bench_rows[0]["valid_rate"]
    ok = [r for r in bench_rows if r["valid_rate"] >= 0.98 * base_valid]
    ok = [r for r in ok if r["latency_p95"] <= 15.0] or [bench_rows[0]]
    wp = max(ok, key=lambda r: r["throughput_rps"])
    print(f"\nРабочая точка: conc={wp['concurrency']}, rps={wp['throughput_rps']}, "
          f"p95={wp['latency_p95']}s, valid={wp['valid_rate']}")

    # --- Финальный прогон на всех 300 → predictions.jsonl ---
    print(f"\nФинальный прогон на {len(bench)}...")
    final_results, final_wall = await run_concurrent(
        bench, wp["concurrency"], generate_one, tuned,
    )

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_PATH.open("w", encoding="utf-8") as f:
        for r in final_results:
            f.write(json.dumps({
                "card": r["card"],
                "valid": r["valid"],
                "attempts": r["attempts"],
                "elapsed_sec": round(r["elapsed_sec"], 3),
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "error_reason": r["error_reason"],
            }, ensure_ascii=False) + "\n")
    print(f"predictions: {PREDICTIONS_PATH} ({len(final_results)})")

    # --- benchmark_results.json: rows + final_run (без отдельного run_meta) ---
    payload = {
        "rows": bench_rows,
        "final_run": {
            "wall_sec": round(final_wall, 2),
            "n": len(final_results),
            "concurrency": wp["concurrency"],
            "params": tuned,
        },
    }
    BENCHMARK_RESULTS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"benchmark: {BENCHMARK_RESULTS_PATH} "
          f"(final_wall={final_wall:.1f}s, n={len(final_results)})")

    await local_client.close()


if __name__ == "__main__":
    asyncio.run(main())
