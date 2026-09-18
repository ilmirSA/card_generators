import json
import re
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

DEV_PATH = PROJECT_DIR / "dev.jsonl"
PREDICTIONS_PATH = PROJECT_DIR / "outputs" / "predictions.jsonl"


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


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def count_sentences(text: str) -> int:
    return len(
        [
            sentence
            for sentence in re.split(
                r"[.!?]+(?!\d)",
                text,
            )
            if sentence.strip()
        ]
    )


def validate_card(
    card: dict,
    expected_product_id: str,
) -> list[str]:

    errors = []

    if not isinstance(card, dict):
        return ["Ответ не является JSON-объектом"]

    received_fields = set(card)

    missing_fields = EXPECTED_FIELDS - received_fields
    unexpected_fields = received_fields - EXPECTED_FIELDS

    if missing_fields:
        errors.append(
            f"Отсутствуют поля: {sorted(missing_fields)}"
        )

    if unexpected_fields:
        errors.append(
            f"Лишние поля: {sorted(unexpected_fields)}"
        )

    if "product_id" not in card:
        errors.append("Отсутствует product_id")
    elif card["product_id"] != expected_product_id:
        errors.append(
            f"Неверный product_id: "
            f"{card['product_id']} != {expected_product_id}"
        )

    if "description" not in card:
        errors.append("Отсутствует description")
    else:
        description = card["description"]

        if not isinstance(description, str):
            errors.append("description должен быть строкой")
        else:
            description = description.strip()

            if not (
                MIN_DESCRIPTION_LENGTH
                <= len(description)
                <= MAX_DESCRIPTION_LENGTH
            ):
                errors.append(
                    f"description: {len(description)} символов, "
                    f"ожидалось {MIN_DESCRIPTION_LENGTH}-"
                    f"{MAX_DESCRIPTION_LENGTH}"
                )

            sentence_count = count_sentences(description)

            if not (
                MIN_DESCRIPTION_SENTENCES
                <= sentence_count
                <= MAX_DESCRIPTION_SENTENCES
            ):
                errors.append(
                    f"description: {sentence_count} предложений, "
                    f"ожидалось {MIN_DESCRIPTION_SENTENCES}-"
                    f"{MAX_DESCRIPTION_SENTENCES}"
                )

    if "pros" not in card:
        errors.append("Отсутствует pros")
    else:
        pros = card["pros"]

        if not isinstance(pros, list):
            errors.append("pros должен быть списком")
        else:
            if not MIN_PROS <= len(pros) <= MAX_PROS:
                errors.append(
                    f"pros: {len(pros)} элементов, "
                    f"ожидалось {MIN_PROS}-{MAX_PROS}"
                )

            if not all(
                isinstance(item, str) and item.strip()
                for item in pros
            ):
                errors.append(
                    "pros содержит пустые или нестроковые элементы"
                )

    if "cons" not in card:
        errors.append("Отсутствует cons")
    else:
        cons = card["cons"]

        if not isinstance(cons, list):
            errors.append("cons должен быть списком")
        else:
            if not MIN_CONS <= len(cons) <= MAX_CONS:
                errors.append(
                    f"cons: {len(cons)} элементов, "
                    f"ожидалось {MIN_CONS}-{MAX_CONS}"
                )

            if not all(
                isinstance(item, str) and item.strip()
                for item in cons
            ):
                errors.append(
                    "cons содержит пустые или нестроковые элементы"
                )

    if "tags" not in card:
        errors.append("Отсутствует tags")
    else:
        tags = card["tags"]

        if not isinstance(tags, list):
            errors.append("tags должен быть списком")
        else:
            if not MIN_TAGS <= len(tags) <= MAX_TAGS:
                errors.append(
                    f"tags: {len(tags)} элементов, "
                    f"ожидалось {MIN_TAGS}-{MAX_TAGS}"
                )

            if not all(
                isinstance(item, str) and item.strip()
                for item in tags
            ):
                errors.append(
                    "tags содержит пустые или нестроковые элементы"
                )

    return errors


def main():
    products = load_jsonl(DEV_PATH)
    predictions = load_jsonl(PREDICTIONS_PATH)

    if len(products) != len(predictions):
        print(
            f"WARNING: количество товаров не совпадает: "
            f"products={len(products)}, "
            f"predictions={len(predictions)}"
        )

    total = min(len(products), len(predictions))

    valid_count = 0
    invalid_count = 0

    for index in range(total):
        product = products[index]
        prediction = predictions[index]

        product_id = product["product_id"]

        errors = validate_card(
            prediction,
            expected_product_id=product_id,
        )

        if errors:
            invalid_count += 1

            print(
                f"\n❌ [{index + 1}] product_id={product_id}"
            )

            for error in errors:
                print(f"   - {error}")
        else:
            valid_count += 1

    if total == 0:
        print("Нет данных для проверки")
        return

    valid_rate = valid_count / total

    print("\n" + "=" * 50)
    print("METRICS")
    print("=" * 50)

    print(f"Total:       {total}")
    print(f"Valid:       {valid_count}")
    print(f"Invalid:     {invalid_count}")
    print(f"Valid rate:  {valid_rate:.2%}")

    print("=" * 50)

    if valid_rate >= 0.80:
        print("Требование качества >= 80% выполнено")
    else:
        print("Требование качества >= 80% НЕ выполнено")


if __name__ == "__main__":
    main()