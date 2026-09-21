import asyncio
import copy
import time

import environs
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from transformers import AutoTokenizer

from run_batch import DEFAULT_PARAMS
from run_batch import DEV_PATH
from run_batch import PREDICTIONS_PATH
from run_batch import ResponseValidationError
from run_batch import SYSTEM_PROMPT
from run_batch import calculate_retry_delay
from run_batch import load_jsonl, save_jsonl
from run_batch import parse_and_validate
from run_batch import render_product

env = environs.Env()
env.read_env()

openrouter_api_key = env.str("OPENROUTER_API_KEY")

EXPECTED_FIELDS = {
    "product_id",
    "description",
    "pros",
    "cons",
    "tags",
}

INPUT_TOKEN_BUDGET = 1500

VM_RATE_PER_HOUR = 65.0

MAX_GENERATION_ATTEMPTS = 3
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
MODEL_NAME = 'nex-agi/nex-n2.5-mini:free'
remote_tokenizer = AutoTokenizer.from_pretrained('nex-agi/nex-n2.5-mini')

remote_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    timeout=120,
)


def count_messages_tokens(messages: list[dict]) -> int:
    """
    Считает токены именно токенайзером используемой модели.
    """

    text = remote_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    token_ids = remote_tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]

    return len(token_ids)


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


async def request_json(
        messages: list[dict],
        params: dict,
):
    response = await remote_client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=params["temperature"],
        top_p=params["top_p"],
        max_tokens=params['max_tokens'],
        response_format={
            "type": "json_object",
        },
        extra_body={
            "top_k": params["top_k"],
            "repetition_penalty": params["repetition_penalty"],
        },

    )
    print("finish_reason:", response.choices[0].finish_reason)
    content = response.choices[0].message.content

    if not content:
        raise ResponseValidationError(
            "Модель вернула пустой ответ"
        )

    return content, response.usage


async def generate_one(
        product: dict,
        params: dict | None = DEFAULT_PARAMS,
):
    started = time.perf_counter()

    messages, input_tokens = build_messages(product)

    total_input_tokens = 0
    total_output_tokens = 0
    attempts = 0
    last_error = None
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
            }
        except retryable_errors as error:
            last_error = str(error)
            if attempt == MAX_GENERATION_ATTEMPTS:
                break
            delay = calculate_retry_delay(attempt=attempt)
            await asyncio.sleep(delay)

        except ResponseValidationError as error:
            print(
                f"\nINVALID product={product['product_id']}"
            )
            print(f"Reason: {error}")
            print(f"Model response:\n{content}")

            last_error = str(error)

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


async def main():
    dev_products = load_jsonl(DEV_PATH)

    # Для задания достаточно 20–30 товаров.
    products = dev_products[:5]

    print(f"dev products: {len(dev_products)}")
    print(f"OpenRouter sample: {len(products)}")
    print("\nRunning OpenRouter generation...")

    concurrency = 2

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

    save_jsonl(
        PREDICTIONS_PATH,
        predictions,
    )

    valid_count = sum(
        result["valid"]
        for result in results
    )

    n = len(results)

    valid_rate = (
        valid_count / n
        if n
        else 0
    )

    throughput = (
        n / elapsed
        if elapsed > 0
        else 0
    )

    total_input_tokens = sum(
        result["input_tokens"]
        for result in results
    )

    total_output_tokens = sum(
        result["output_tokens"]
        for result in results
    )

    avg_input_tokens = (
        total_input_tokens / n
        if n
        else 0
    )

    avg_output_tokens = (
        total_output_tokens / n
        if n
        else 0
    )

    print("\nInvalid results:")

    for result in results:
        if not result["valid"]:
            print(
                f"{result['card']['product_id']}: "
                f"{result.get('error')}"
            )

    print("\n" + "=" * 60)
    print("OPENROUTER FINISHED")
    print("=" * 60)

    print(f"Model: {MODEL_NAME}")
    print(f"Products: {n}")
    print(f"Valid: {valid_count}/{n}")
    print(f"Valid rate: {valid_rate:.2%}")

    print(f"Elapsed: {elapsed:.2f} sec")
    print(f"Throughput: {throughput:.3f} cards/sec")

    print(
        f"Total prompt tokens: "
        f"{total_input_tokens}"
    )

    print(
        f"Average prompt tokens: "
        f"{avg_input_tokens:.1f}"
    )

    print(
        f"Total completion tokens: "
        f"{total_output_tokens}"
    )

    print(
        f"Average completion tokens: "
        f"{avg_output_tokens:.1f}"
    )

    print(
        f"Predictions: "
        f"{PREDICTIONS_PATH}"
    )

    await remote_client.close()


if __name__ == "__main__":
    asyncio.run(main())
