"""Stream batch results in completion order with iter_extract_many_async."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from openextract import ExtractionInput, extract_many, iter_extract_many_async

class Contact(BaseModel):
    name: str
    email: str


MODEL = TestModel(custom_output_args={"name": "Ada Lovelace", "email": "ada@example.com"})

INPUTS = [
    ExtractionInput(b"Ada Lovelace <ada@example.com>", media_type="text/plain", name="ada.txt"),
    ExtractionInput(b"missing media type", name="broken.bin"),
    ExtractionInput(b"Grace Hopper <grace@example.com>", media_type="text/plain", name="grace.txt"),
]


def _print_item(index: int, result: Contact | Exception) -> None:
    name = INPUTS[index].name
    if isinstance(result, Exception):
        print(f"  input[{index}] {name}: {type(result).__name__}: {result}")
        return
    print(f"  input[{index}] {name}: {result.email}")


def list_api() -> None:
    print("extract_many — input order, waits for the full batch:")
    results = extract_many(
        schema=Contact,
        model=MODEL,
        input_files=INPUTS,
        return_exceptions=True,
        max_concurrency=2,
    )
    for index, result in enumerate(results):
        _print_item(index, result)


async def stream_api() -> None:
    print("iter_extract_many_async — completion order, yields as each item finishes:")
    async for index, result in iter_extract_many_async(
        schema=Contact,
        model=MODEL,
        input_files=INPUTS,
        return_exceptions=True,
        max_concurrency=2,
    ):
        _print_item(index, result)


def main() -> None:
    list_api()
    print()
    asyncio.run(stream_api())


if __name__ == "__main__":
    main()
