import asyncio
import json
import re
import time
from pathlib import Path
import random
import httpx
from openai import AsyncOpenAI
from transformers import AutoTokenizer


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

LOCAL_BASE_URL = "http://localhost:8000/v1"

INPUT_TOKEN_BUDGET = 1500

VM_RATE_PER_HOUR = 65.0


MAX_GENERATION_ATTEMPTS = 3



PROJECT_DIR = Path(__file__).resolve().parent
DEV_PATH = PROJECT_DIR / "dev.jsonl"
BENCHMARK_PATH = PROJECT_DIR / "benchmark.jsonl"
OUTPUT_DIR = PROJECT_DIR / "outputs"

PREDICTIONS_PATH = OUTPUT_DIR / "predictions.jsonl"
REPORT_PATH = OUTPUT_DIR / "report.json"
GRID_PATH = OUTPUT_DIR / "parameter_grid.json"

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

DEFAULT_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 350,
    "top_k": 50,
    "repetition_penalty": 1.2,
}

SYSTEM_PROMPT = """
Ты создаёшь карточку товара для маркетплейса.

Верни только JSON-объект.
Никакого markdown.
Не используй ```json.
Не добавляй текст до или после JSON.

Формат:

{
  "product_id": "string",
  "description": "string",
  "pros": ["string"],
  "cons": ["string"],
  "tags": ["string"]
}

СТРОГИЕ ОГРАНИЧЕНИЯ:

1. product_id:
- должен точно совпадать с product_id входного товара.

2. description:
- от 120 до 700 символов;
- ровно от 2 до 5 предложений;
- обычный связный текст;
- без списков;
- без markdown.

3. pros:
- ровно 3 элемента;
- каждый элемент — непустая строка.

4. cons:
- ровно 2 элемента;
- каждый элемент — непустая строка;
- используй только реальные недостатки из товара и отзывов.

5. tags:
- РОВНО 5 элементов;
- каждый элемент — короткая непустая строка;
- не повторяй теги;
- не создавай длинные фразы;
- не создавай новые теги после пятого;
- после 5 тегов массив tags ОБЯЗАТЕЛЬНО заканчивается.

ВАЖНО:
Сначала сформируй все поля.
Затем проверь:
- description: 120-700 символов;
- description: 2-5 предложений;
- pros: ровно 3;
- cons: ровно 2;
- tags: ровно 5.

После проверки верни только полностью закрытый JSON.
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


def build_messages(product: dict):

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
    expected_product_id:str,
) -> dict:

    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise ResponseValidationError(
            f"Ответ не является корректным JSON: {error.msg}"
        ) from error

    if not isinstance(data, dict):
        raise ResponseValidationError(
            "На верхнем уровне должен находиться JSON-объект"
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
            f"{sorted(missing_fields)}"
        )

    if unexpected_fields:
        raise ResponseValidationError(
            "Получены лишние поля: "
            f"{sorted(unexpected_fields)}"
        )

    if expected_product_id != data["product_id"] :
        raise ResponseValidationError(
            f"Неверный product_id. Ожидали {expected_product_id}. Получили {data['product_id']}"
        )

    description = data["description"]

    if not isinstance(description, str):
        raise ResponseValidationError(
            "Поле description должно быть строкой"
        )

    description = description.strip()

    if not description:
        raise ResponseValidationError(
            "Поле description не должно быть пустым"
        )

    if not (
        MIN_DESCRIPTION_LENGTH
        <= len(description)
        <= MAX_DESCRIPTION_LENGTH
    ):
        raise ResponseValidationError(
            "Поле description должно содержать "
            f"от {MIN_DESCRIPTION_LENGTH} до "
            f"{MAX_DESCRIPTION_LENGTH} символов"
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
            f"{MAX_DESCRIPTION_SENTENCES} предложений"
        )

    pros = data["pros"]

    if not isinstance(pros, list):
        raise ResponseValidationError(
            "Поле pros должно быть списком"
        )

    if not MIN_PROS <= len(pros) <= MAX_PROS:
        raise ResponseValidationError(
            "Поле pros должно содержать "
            f"от {MIN_PROS} до {MAX_PROS} элементов"
        )

    if not all(
        isinstance(item, str) and item.strip()
        for item in pros
    ):
        raise ResponseValidationError(
            "Все элементы pros должны быть "
            "непустыми строками"
        )

    cons = data["cons"]

    if not isinstance(cons, list):
        raise ResponseValidationError(
            "Поле cons должно быть списком"
        )

    if not MIN_CONS <= len(cons) <= MAX_CONS:
        raise ResponseValidationError(
            "Поле cons должно содержать "
            f"от {MIN_CONS} до {MAX_CONS} элементов"
        )

    if not all(
        isinstance(item, str) and item.strip()
        for item in cons
    ):
        raise ResponseValidationError(
            "Все элементы cons должны быть "
            "непустыми строками"
        )

    tags = data["tags"]

    if not isinstance(tags, list):
        raise ResponseValidationError(
            "Поле tags должно быть списком"
        )

    if not MIN_TAGS <= len(tags) <= MAX_TAGS:
        raise ResponseValidationError(
            "Поле tags должно содержать "
            f"от {MIN_TAGS} до {MAX_TAGS} элементов"
        )

    if not all(
        isinstance(item, str) and item.strip()
        for item in tags
    ):
        raise ResponseValidationError(
            "Все элементы tags должны быть "
            "непустыми строками"
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
    }

    
    if params["top_k"] is not None:
        extra_body["top_k"] = params["top_k"]

    response = await local_client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=params["temperature"],
    top_p=params["top_p"],
    max_tokens=params["max_tokens"],
    response_format={"type": "json_object"},
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
    params = params or DEFAULT_PARAMS

    started = time.perf_counter()

    messages, input_tokens = build_messages(product)

    last_error = None

    for attempt in range(1, MAX_GENERATION_ATTEMPTS+1):
        content=None
        try:
            content, usage = await request_json(
                messages=messages,
                params=params,
            )
            card = parse_and_validate(content,expected_product_id=product["product_id"])

            elapsed = time.perf_counter() - started

            return {
                    "card": card,
                    "input_tokens": getattr(
                        usage,
                        "prompt_tokens",
                        input_tokens,
                    ),
                    "output_tokens": getattr(
                        usage,
                        "completion_tokens",
                        0,
                    ),
                    "elapsed_sec": elapsed,
                    "valid": True,
                }
        except ResponseValidationError as error:
            print(f"\nINVALID product={product['product_id']}")
            print(f"Reason: {error}")
            print(f"Model response:\n{content}")
            last_error = str(error)

            if attempt == MAX_GENERATION_ATTEMPTS:
                break
            

            delay = calculate_retry_delay(
                    attempt=attempt,
                )
            await asyncio.sleep(delay)

            if content is not None:
                messages.append({
                    "role": "assistant",
                    "content": content,
                })

                messages.append({
                    "role":"user",
                    "content":(
                        "Предыдущий ответ не прошёл проверку. "
                        f"Причина: {last_error}. "
                        "Исправь ответ и верни только JSON."
                    )
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
        "input_tokens": input_tokens,
        "output_tokens": 0,
        "elapsed_sec": elapsed,
        "valid": False,
        "error": last_error,
    }




async def run_concurrent(
    items,
    concurrency,
    generate_one_func,
):
    sem = asyncio.Semaphore(concurrency)

    async def wrapped(item):
        async with sem:
            return await generate_one_func(item)

    started = time.perf_counter()

    results = await asyncio.gather(
        *[
            wrapped(item)
            for item in items
        ]
    )

    elapsed = time.perf_counter() - started

    return results, elapsed




# async def run_concurrency_test(
#     products: list[dict],
# ):
#     """
#     Замеряем несколько уровней concurrency.

#     Для теста используем первые 64 товара.
#     """

#     test_products = products[:64]

#     concurrency_levels = [
#         1,
#         4,
#         8,
#         16,
#         32,
#     ]

#     rows = []

#     for concurrency in concurrency_levels:

#         async def generate(item):
#             return await generate_one(
#                 item,
#                 DEFAULT_PARAMS,
#             )

#         results, elapsed = await run_concurrent(
#             test_products,
#             concurrency,
#             generate,
#         )

#         valid_count = sum(
#             result["valid"]
#             for result in results
#         )

#         n = len(results)

#         valid_rate = valid_count / n
#         throughput = n / elapsed

#         rows.append(
#             {
#                 "concurrency": concurrency,
#                 "n": n,
#                 "elapsed_sec": elapsed,
#                 "throughput_cards_per_sec": throughput,
#                 "valid_rate": valid_rate,
#             }
#         )

#     return rows



# async def run_parameter_grid(
#     products: list[dict],
# ):
#     """
#     Небольшой one-factor-at-a-time эксперимент.
#     """

#     test_products = products[:20]

#     experiments = []

#     parameter_values = {
#         "temperature": [
#             0.0,
#             0.3,
#             0.7,
#         ],
#         "top_p": [
#             0.9,
#             1.0,
#         ],
#         "top_k": [
#             20,
#             50,
#         ],
#         "repetition_penalty": [
#             1.0,
#             1.1,
#             1.2,
#         ],
#         "max_tokens": [
#             250,
#             350,
#             500,
#         ],
#     }

#     for parameter_name, values in parameter_values.items():

#         for value in values:

#             params = DEFAULT_PARAMS.copy()
#             params[parameter_name] = value

#             async def generate(item):
#                 return await generate_one(
#                     item,
#                     params,
#                 )

#             started = time.perf_counter()

#             results = await asyncio.gather(
#                 *[
#                     generate(item)
#                     for item in test_products
#                 ]
#             )

#             elapsed = time.perf_counter() - started

#             valid_count = sum(
#                 result["valid"]
#                 for result in results
#             )

#             experiments.append(
#                 {
#                     "parameter": parameter_name,
#                     "value": value,
#                     "n": len(results),
#                     "elapsed_sec": elapsed,
#                     "valid_rate": (
#                         valid_count / len(results)
#                     ),
#                 }
#             )

#     return experiments



# async def run_full_benchmark(
#     products: list[dict],
#     concurrency: int,
# ):
#     async def generate(item):
#         return await generate_one(
#             item,
#             DEFAULT_PARAMS,
#         )

#     results, elapsed = await run_concurrent(
#         products,
#         concurrency,
#         generate,
#     )

#     predictions = [
#         result["card"]
#         for result in results
#     ]

#     total_input_tokens = sum(
#         result["input_tokens"]
#         for result in results
#     )

#     total_output_tokens = sum(
#         result["output_tokens"]
#         for result in results
#     )

#     valid_count = sum(
#         result["valid"]
#         for result in results
#     )

#     n = len(results)

#     cost_run = (
#         elapsed
#         / 3600
#         * VM_RATE_PER_HOUR
#     )

#     cost_per_1000 = (
#         cost_run
#         / n
#         * 1000
#     )

#     return {
#         "predictions": predictions,
#         "run": {
#             "model": MODEL,
#             "n": n,
#             "elapsed_sec": elapsed,
#             "throughput_cards_per_sec": (
#                 n / elapsed
#             ),
#             "input_tokens": total_input_tokens,
#             "output_tokens": total_output_tokens,
#             "valid_rate": valid_count / n,
#             "vm_rate_per_hour": VM_RATE_PER_HOUR,
#             "cost_run_rub": cost_run,
#             "cost_per_1000_cards_rub": cost_per_1000,
#             "concurrency": concurrency,
#             "generation_params": DEFAULT_PARAMS,
#         },
#     }



# def save_report(
#     run_data: dict,
#     concurrency_table: list[dict],
#     params_grid: list[dict],
# ):
#     OUTPUT_DIR.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     report = {
#         "run": run_data,
#         "metrics": {},
#         "api_comparison": {},
#     }

#     report["run"]["concurrency_table"] = (
#         concurrency_table
#     )

#     report["run"]["parameter_grid"] = (
#         params_grid
#     )

#     with REPORT_PATH.open(
#         "w",
#         encoding="utf-8",
#     ) as file:
#         json.dump(
#             report,
#             file,
#             ensure_ascii=False,
#             indent=2,
#         )


# async def main():
#     OUTPUT_DIR.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     dev_products = load_jsonl(
#         DEV_PATH
#     )

#     benchmark_products = load_jsonl(
#         BENCHMARK_PATH
#     )

#     print(
#         f"dev products: {len(dev_products)}"
#     )

#     print(
#         f"benchmark products: "
#         f"{len(benchmark_products)}"
#     )


#     print("Running parameter grid...")

#     params_grid = await run_parameter_grid(
#         dev_products
#     )


#     print("Running concurrency test...")

#     concurrency_table = await run_concurrency_test(
#         dev_products
#     )

#     suitable = [
#         row
#         for row in concurrency_table
#         if row["valid_rate"] >= 0.80
#     ]

#     if not suitable:
#         raise RuntimeError(
#             "Ни один уровень concurrency "
#             "не достиг valid_rate >= 0.80"
#         )

#     best = max(
#         suitable,
#         key=lambda row: row[
#             "throughput_cards_per_sec"
#         ],
#     )

#     best_concurrency = best["concurrency"]

#     print(
#         f"Selected concurrency: "
#         f"{best_concurrency}"
#     )


#     print("Running full benchmark...")

#     benchmark_result = await run_full_benchmark(
#         benchmark_products,
#         best_concurrency,
#     )

#     predictions = benchmark_result[
#         "predictions"
#     ]

#     save_jsonl(
#         PREDICTIONS_PATH,
#         predictions,
#     )


#     save_report(
#         run_data=benchmark_result["run"],
#         concurrency_table=concurrency_table,
#         params_grid=params_grid,
#     )

#     print()
#     print("Done.")
#     print(
#         f"Predictions: {PREDICTIONS_PATH}"
#     )
#     print(
#         f"Report: {REPORT_PATH}"
#     )
#     print(
#         f"Elapsed: "
#         f"{benchmark_result['run']['elapsed_sec']:.2f} sec"
#     )
#     print(
#         f"Throughput: "
#         f"{benchmark_result['run']['throughput_cards_per_sec']:.3f} cards/sec"
#     )
#     print(
#         f"Valid rate: "
#         f"{benchmark_result['run']['valid_rate']:.2%}"
#     )
#     print(
#         f"Cost / 1000: "
#         f"{benchmark_result['run']['cost_per_1000_cards_rub']:.2f} ₽"
#     )

#     await local_client.close()
async def run_full_benchmark(
    products: list[dict],
    concurrency: int,
):
    async def generate(item):
        return await generate_one(
            item,
            DEFAULT_PARAMS,
        )

    results, elapsed = await run_concurrent(
        products,
        concurrency,
        generate,
    )

    predictions = [
        result["card"]
        for result in results
    ]

    valid_count = sum(
        result["valid"]
        for result in results
    )

    n = len(results)

    return {
        "predictions": predictions,
        "elapsed_sec": elapsed,
        "valid_rate": valid_count / n,
        "throughput": n / elapsed,
    }

# async def main():
#     OUTPUT_DIR.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     dev_products = load_jsonl(
#         DEV_PATH
#     )

#     print(
#         f"dev products: {len(dev_products)}"
#     )

#     print("\nRunning generation...")

#     # Просто генерируем карточки для dev.jsonl
#     concurrency = 4

#     async def generate(item):
#         return await generate_one(
#             item,
#             DEFAULT_PARAMS,
#         )

#     results, elapsed = await run_concurrent(
#         dev_products,
#         concurrency,
#         generate,
#     )

#     # ---------------------------------
#     # Сохраняем карточки
#     # ---------------------------------

#     predictions = [
#         result["card"]
#         for result in results
#     ]

#     save_jsonl(
#         PREDICTIONS_PATH,
#         predictions,
#     )

#     # ---------------------------------
#     # Статистика
#     # ---------------------------------

#     valid_count = sum(
#         result["valid"]
#         for result in results
#     )

#     n = len(results)

#     valid_rate = valid_count / n
#     throughput = n / elapsed

#     print("\nDone.")

#     print(
#         f"Predictions: "
#         f"{PREDICTIONS_PATH}"
#     )

#     print(
#         f"Elapsed: "
#         f"{elapsed:.2f} sec"
#     )

#     print(
#         f"Throughput: "
#         f"{throughput:.3f} cards/sec"
#     )

#     print(
#         f"Valid rate: "
#         f"{valid_rate:.2%}"
#     )

#     await local_client.close()


# async def run_parameter_grid(
#     products: list[dict],
# ):
#     # По заданию grid запускаем на 20 товарах
#     test_products = products[:20]

#     experiments = []

#     parameter_values = {
#         "temperature": [
#             0.0,
#             0.3,
#             0.7,
#         ],
#         "top_p": [
#             0.9,
#             1.0,
#         ],
#         "top_k": [
#             20,
#             50,
#             None,  # unlimited
#         ],
#         "repetition_penalty": [
#             1.0,
#             1.1,
#             1.2,
#         ],
#         "max_tokens": [
#             250,
#             350,
#             500,
#         ],
#     }

#     total_experiments = sum(
#         len(values)
#         for values in parameter_values.values()
#     )

#     experiment_number = 0

#     for parameter_name, values in parameter_values.items():

#         for value in values:

#             experiment_number += 1

#             params = DEFAULT_PARAMS.copy()
#             params[parameter_name] = value

#             print()
#             print("=" * 70)
#             print(
#                 f"Experiment "
#                 f"{experiment_number}/{total_experiments}"
#             )
#             print(
#                 f"{parameter_name} = {value}"
#             )
#             print("=" * 70)

#             started = time.perf_counter()

#             results = await asyncio.gather(
#                 *[
#                     generate_one(
#                         product,
#                         params,
#                     )
#                     for product in test_products
#                 ]
#             )

#             elapsed = (
#                 time.perf_counter()
#                 - started
#             )

#             valid_count = sum(
#                 result["valid"]
#                 for result in results
#             )

#             valid_rate = (
#                 valid_count
#                 / len(results)
#             )

#             valid_descriptions = [
#                 result["card"]["description"]
#                 for result in results
#                 if result["valid"]
#             ]

#             avg_description_length = (
#                 sum(
#                     len(description)
#                     for description
#                     in valid_descriptions
#                 )
#                 / len(valid_descriptions)
#                 if valid_descriptions
#                 else 0
#             )

#             total_output_tokens = sum(
#                 result["output_tokens"]
#                 for result in results
#             )

#             avg_output_tokens = (
#                 total_output_tokens
#                 / len(results)
#             )

#             throughput = (
#                 len(results)
#                 / elapsed
#                 if elapsed > 0
#                 else 0
#             )

#             row = {
#                 "parameter": parameter_name,
#                 "value": value,
#                 "n": len(results),
#                 "elapsed_sec": round(
#                     elapsed,
#                     3,
#                 ),
#                 "throughput": round(
#                     throughput,
#                     4,
#                 ),
#                 "valid_count": valid_count,
#                 "valid_rate": round(
#                     valid_rate,
#                     4,
#                 ),
#                 "avg_description_length": round(
#                     avg_description_length,
#                     2,
#                 ),
#                 "output_tokens": total_output_tokens,
#                 "avg_output_tokens": round(
#                     avg_output_tokens,
#                     2,
#                 ),
#             }

#             experiments.append(row)

#             print(
#                 f"valid: "
#                 f"{valid_count}/{len(results)} "
#                 f"({valid_rate:.2%})"
#             )

#             print(
#                 f"avg description: "
#                 f"{avg_description_length:.1f}"
#             )

#             print(
#                 f"avg output tokens: "
#                 f"{avg_output_tokens:.1f}"
#             )

#             print(
#                 f"elapsed: "
#                 f"{elapsed:.2f} sec"
#             )

#             print(
#                 f"throughput: "
#                 f"{throughput:.3f} cards/sec"
#             )

#     return experiments


# ============================================================
# MAIN
# ============================================================

# async def main():

#     OUTPUT_DIR.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     products = load_jsonl(
#         DEV_PATH,
#     )

#     print(
#         f"Loaded products: {len(products)}"
#     )

#     if len(products) < 20:
#         raise RuntimeError(
#             "Для parameter grid нужно "
#             "минимум 20 товаров"
#         )

#     print()
#     print(
#         "Running parameter grid "
#         "on first 20 products..."
#     )

#     started = time.perf_counter()

#     experiments = await run_parameter_grid(
#         products,
#     )

#     total_elapsed = (
#         time.perf_counter()
#         - started
#     )

#     with GRID_PATH.open(
#         "w",
#         encoding="utf-8",
#     ) as file:
#         json.dump(
#             experiments,
#             file,
#             ensure_ascii=False,
#             indent=2,
#         )

#     print()
#     print("=" * 70)
#     print("GRID FINISHED")
#     print("=" * 70)

#     print(
#         f"Total time: "
#         f"{total_elapsed:.2f} sec"
#     )

#     print(
#         f"Results saved to: "
#         f"{GRID_PATH}"
#     )

#     print()
#     print("Results:")

#     for row in experiments:
#         print(
#             f"{row['parameter']:25} "
#             f"{str(row['value']):10} "
#             f"valid={row['valid_rate']:.2%} "
#             f"avg_desc={row['avg_description_length']:.1f} "
#             f"avg_tokens={row['avg_output_tokens']:.1f}"
#         )

#     await local_client.close()


# if __name__ == "__main__":
#     asyncio.run(main())


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dev_products = load_jsonl(DEV_PATH)

    print(f"dev products: {len(dev_products)}")
    print("\nRunning generation...")

    concurrency = 4

    async def generate(item):
        return await generate_one(item, DEFAULT_PARAMS)

    results, elapsed = await run_concurrent(
        dev_products,
        concurrency,
        generate,
    )

    predictions = [result["card"] for result in results]
    save_jsonl(PREDICTIONS_PATH, predictions)

    valid_count = sum(
        result["valid"]
        for result in results
    )

    n = len(results)
    valid_rate = valid_count / n
    throughput = n / elapsed

    print("\nInvalid results:")

    for result in results:
        if not result["valid"]:
            print(
                f"{result['card']['product_id']}: "
                f"{result.get('error')}"
            )

    print("\nDone.")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Elapsed: {elapsed:.2f} sec")
    print(f"Throughput: {throughput:.3f} cards/sec")
    print(f"Valid rate: {valid_rate:.2%}")

    await local_client.close()


if __name__ == "__main__":
    asyncio.run(main())