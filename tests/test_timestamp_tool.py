from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import pytest

from agents import timestamp_tool


@pytest.mark.asyncio
async def test_get_parakeet_model_caches_by_model_name(monkeypatch) -> None:
    timestamp_tool._PARAKEET_MODEL_CACHE.clear()
    load_calls: list[str] = []

    class FakeASRModel:
        @staticmethod
        def from_pretrained(model_name: str):
            load_calls.append(model_name)
            return {"model_name": model_name}

    nemo_module = ModuleType("nemo")
    collections_module = ModuleType("nemo.collections")
    asr_module = ModuleType("nemo.collections.asr")
    asr_module.models = SimpleNamespace(ASRModel=FakeASRModel)

    monkeypatch.setitem(sys.modules, "nemo", nemo_module)
    monkeypatch.setitem(sys.modules, "nemo.collections", collections_module)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", asr_module)

    model_a_first = await timestamp_tool._get_parakeet_model("model-a")
    model_a_second = await timestamp_tool._get_parakeet_model("model-a")
    model_b = await timestamp_tool._get_parakeet_model("model-b")

    assert model_a_first is model_a_second
    assert model_a_first["model_name"] == "model-a"
    assert model_b["model_name"] == "model-b"
    assert load_calls == ["model-a", "model-b"]

    timestamp_tool._PARAKEET_MODEL_CACHE.clear()
