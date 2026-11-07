"""Проверки клиента модели.

Главное здесь — thinking-режим. Модели семейства Qwen3 и DeepSeek-R1 по
умолчанию рассуждают вслух и могут вернуть рассуждение вместо JSON.
Агент разбирает строгий JSON, поэтому режим выключается явно, а не
«как получится».
"""

from __future__ import annotations

import httpx
import pytest

from pf.config import Settings
from pf.llm import LLMError, OllamaClient, OpenAICompatibleClient, extract_json


def _response(payload: dict[str, object]) -> httpx.Response:
    """Собрать ответ, пригодный для ``raise_for_status``.

    Args:
        payload: Тело ответа.

    Returns:
        Ответ с привязанным запросом.
    """
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    return httpx.Response(200, json=payload, request=request)


def test_think_is_disabled_by_default() -> None:
    """Thinking-режим выключен по умолчанию."""
    assert Settings().ollama_think is False


def test_ollama_payload_has_think_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """В запрос к Ollama уходит явный флаг thinking-режима."""
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        captured.update(kwargs.get("json", {}) or {})  # type: ignore[arg-type]
        return _response(
            {
                "message": {"content": '{"ok": true}'},
                "prompt_eval_count": 10,
                "eval_count": 5,
                "total_duration": 1_000_000,
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OllamaClient(Settings(ollama_think=False))
    response = client.complete("система", "задача")

    assert captured["think"] is False
    assert response.content == '{"ok": true}'


def test_ollama_usage_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Расход и модель попадают в ответ — по ним считается стоимость."""
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        captured.update(kwargs.get("json", {}) or {})  # type: ignore[arg-type]
        return _response(
            {
                "message": {"content": "{}"},
                "prompt_eval_count": 100,
                "eval_count": 40,
                # Ollama отдаёт total_duration в наносекундах
                "total_duration": 2_500_000_000,
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OllamaClient(Settings(ollama_model="qwen3:8b"))
    response = client.complete("система", "задача")

    assert captured["model"] == "qwen3:8b"
    assert response.usage.total_tokens == 140
    assert response.usage.model == "qwen3:8b"
    assert response.usage.duration_ms == 2500


def test_truncated_json_is_reported() -> None:
    """Обрезанный ответ — понятная ошибка, а не «не удалось разобрать»."""
    with pytest.raises(LLMError, match="обрезан"):
        extract_json('{"facts": [{"text": "начало без конца')


def test_plain_json_is_parsed() -> None:
    """Обычный JSON разбирается."""
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json_is_parsed() -> None:
    """JSON в блоке кода разбирается."""
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_cloud_max_tokens_is_raised() -> None:
    """Облачному провайдеру лимит ответа поднимается.

    На локальном лимите (``PF_MAX_TOKENS=2400``) JSON стратегии обрывался,
    и прогон на облаке падал на «JSON не закрыт». Локальная модель
    остаётся на значении из настроек — ей этого хватает.
    """
    local = Settings(llm_provider="ollama", max_tokens=2400)
    assert local.effective_max_tokens == 2400

    cloud = Settings(
        llm_provider="openai",
        max_tokens=2400,
        llm_model="deepseek-flash",
        llm_api_key="test-key",
    )
    assert cloud.effective_max_tokens >= 8000


def test_openai_payload_uses_effective_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """В запрос к облаку уходит поднятый лимит, а не локальный."""
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        captured.update(kwargs.get("json", {}) or {})  # type: ignore[arg-type]
        return _response(
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OpenAICompatibleClient(
        Settings(
            llm_provider="openai",
            max_tokens=2400,
            llm_model="deepseek-flash",
            llm_api_key="test-key",
            llm_base_url="https://api.deepseek.com",
        )
    )
    client.complete("система", "задача")

    assert captured["max_tokens"] >= 8000
