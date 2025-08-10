from __future__ import annotations

from typing import Any

import open_xtract.main as ox_module
from pydantic import BaseModel

from cookbook.legal_extraction import ContractSummary, extract_contract_summary


class _FakeStructured:
    def __init__(self, schema: type[BaseModel]) -> None:
        self._schema = schema

    def invoke(self, _messages: list[Any]) -> BaseModel:
        # Populate fields with type-safe dummy values
        values: dict[str, Any] = {}
        for name, field in self._schema.model_fields.items():
            anno = field.annotation
            if anno in (int, float):
                values[name] = 0
            elif anno is bool:
                values[name] = False
            else:
                values[name] = "ok"
        return self._schema.model_validate(values)


class _FakeChatOpenAI:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        pass

    def with_structured_output(self, schema: type[BaseModel]) -> _FakeStructured:
        return _FakeStructured(schema)


def test_extract_contract_summary_monkeypatched(monkeypatch: Any) -> None:
    # Swap out the real ChatOpenAI with our fake structured LLM
    monkeypatch.setattr(ox_module, "ChatOpenAI", _FakeChatOpenAI)

    text = (
        "This Agreement is between ACME and BetaCorp effective Jan 1, 2024 under Delaware law. "
        "The term is 12 months and includes a termination for cause clause."
    )

    result = extract_contract_summary(text)

    assert isinstance(result, ContractSummary)
    # Spot-check types/values from our fake
    assert isinstance(result.party_a, str)
    assert isinstance(result.term_months, int)