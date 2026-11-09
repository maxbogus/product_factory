"""Модель по роли: дешёвая на извлечение, сильная на синтез.

Бриф называет «специализированные модели»: извлечение фактов не требует
сильной модели, а синтез стратегии — требует. Здесь проверяется механизм:
роль получает свою модель, если человек её задал, а без настройки поведение
не меняется — все роли идут на модели по умолчанию.

Экономия видна по расходу: журнал уже считает стоимость каждого шага
(``LLMUsage.cost_units``), поэтому дешёвая модель на извлечении даёт
меньшую цену того же шага. Задача — не замерить экономию на живой модели
(это ``make bench``), а доказать, что модель роли доезжает до провайдера.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from pf.config import Settings
from pf.llm import LLMResponse, LLMUsage, OllamaClient
from pf.roles import model_for
from pf.workflow import Workflow


class _RecordingClient:
    """Модель, которая запоминает, с какой моделью её позвали."""

    def __init__(self) -> None:
        """Начать пустой список моделей."""
        self.models: list[str] = []

    def complete(self, system: str, user: str, *, model: str = "") -> LLMResponse:
        """Ответить заготовкой интика и записать модель.

        Args:
            system: Системная инструкция.
            user: Текст задачи.
            model: Модель, которую выбрал рабочий процесс.

        Returns:
            Ответ с валидной постановкой задачи.
        """
        self.models.append(model)
        answer: dict[str, Any] = {
            "question": "Почему падает удержание?",
            "goal": "увеличить удержание мобильного тарифа",
            "audience": "команда по управлению тарифом",
            "constraints": [],
            "available_data": [],
            "open_questions": [],
        }
        return LLMResponse(content=json.dumps(answer, ensure_ascii=False), usage=LLMUsage())


def _local(**values: Any) -> Settings:
    """Настройки локального провайдера — без зависимости от окружения.

    Явный провайдер нужен для повторяемости: ``PF_LLM_PROVIDER=openai``,
    оставшийся в окружении, иначе переключил бы ``effective_model``.

    Args:
        **values: Дополнительные поля настроек.

    Returns:
        Настройки с локальной моделью по умолчанию.
    """
    return Settings(llm_provider="ollama", ollama_model="qwen3:8b", **values)


def test_role_model_defaults_to_single_model() -> None:
    """Без настройки все роли идут на модели по умолчанию.

    Иначе демонстрация требовала бы трёх скачанных моделей, а ``make doctor``
    и живые тесты — новой настройки. Поведение не меняется, пока человек
    не задал имена моделей.
    """
    config = _local(role_model_extraction="", role_model_planning="", role_model_synthesis="")
    assert model_for("intake", config) == "qwen3:8b"
    assert model_for("strategy", config) == "qwen3:8b"


def test_role_model_tiers_are_distinct_when_configured() -> None:
    """Дешёвая на извлечении, средняя на плане и противоречиях, сильная на синтезе."""
    config = _local(
        role_model_extraction="qwen3:1.7b",
        role_model_planning="qwen3:4b",
        role_model_synthesis="qwen3:14b",
    )
    assert model_for("collector", config) == "qwen3:1.7b"
    assert model_for("analyst", config) == "qwen3:1.7b"
    assert model_for("planner", config) == "qwen3:4b"
    assert model_for("redteam", config) == "qwen3:4b"
    assert model_for("options", config) == "qwen3:14b"
    assert model_for("strategy", config) == "qwen3:14b"


def test_role_model_unknown_role_falls_back_to_default() -> None:
    """Неизвестная роль не роняет работу: она идёт на модели по умолчанию."""
    config = _local(role_model_synthesis="qwen3:14b")
    assert model_for("незнакомая роль", config) == "qwen3:8b"


def test_role_model_is_passed_to_client() -> None:
    """Рабочий процесс передаёт клиенту модель роли, а не общую."""
    client = _RecordingClient()
    config = _local(role_model_extraction="qwen3:1.7b")
    work = Workflow(client=client, config=config)
    work._call_role("intake", {"вопрос": "почему падает удержание?"})
    assert client.models == ["qwen3:1.7b"]


def test_role_model_reaches_provider_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Имя модели доезжает до запроса провайдера и до расхода.

    Без этого «модель по роли» — только настройка: провайдер всё равно
    получил бы модель по умолчанию, и экономии не было бы.
    """
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        captured.update(kwargs.get("json", {}) or {})  # type: ignore[arg-type]
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={
                "message": {"content": "{}"},
                "prompt_eval_count": 10,
                "eval_count": 5,
                "total_duration": 1_000_000,
            },
            request=request,
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OllamaClient(_local())
    response = client.complete("система", "задача", model="qwen3:1.7b")

    assert captured["model"] == "qwen3:1.7b"
    assert response.usage.model == "qwen3:1.7b"
