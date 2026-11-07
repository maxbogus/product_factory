"""Проверки сравнения агента с обычным чатом.

Смысл модуля — измерить разницу между агентом и «просто чатом». Здесь
проверяется, что разница считается честно и что baseline не умеет того,
чего не умеет модель сама по себе.
"""

from __future__ import annotations

from pf.bench import (
    BaselineResult,
    baseline_metrics,
    compare,
    run_baseline,
    settings_for_model,
)
from pf.config import Settings
from pf.llm import LLMResponse, LLMUsage
from pf.tracking import RunMetrics


class FakeClient:
    """Модель, которая отвечает заранее заданным текстом."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> LLMResponse:
        self.calls.append((system, user))
        return LLMResponse(
            content=self.text,
            usage=LLMUsage(prompt_tokens=100, completion_tokens=200),
        )


def test_baseline_asks_same_question() -> None:
    """Baseline получает тот же вопрос, что и агент."""
    client = FakeClient("Ответ")
    run_baseline("Почему падает удержание?", client=client)
    _system, user = client.calls[0]
    assert "Почему падает удержание?" in user


def test_baseline_counts_markers() -> None:
    """В ответе обычного чата считаются факты и варианты."""
    client = FakeClient(
        "Известно, что тарифы выросли. Вариант первый — скидка. "
        "Вариант второй — удержание. По данным оператора."
    )
    result = run_baseline("задача", client=client)
    assert result.facts > 0
    assert result.variants > 0
    assert result.tokens == 300


def test_baseline_error_is_captured() -> None:
    """Ошибка baseline не роняет замер."""

    class Failing:
        def complete(self, system: str, user: str) -> LLMResponse:
            raise RuntimeError("модель недоступна")

    result = run_baseline("задача", client=Failing())  # type: ignore[arg-type]
    assert result.error
    assert "модель недоступна" in result.error


def test_baseline_has_no_confirmed_citations() -> None:
    """У обычного чата не бывает подтверждённых оснований.

    Их проверяет код агента дословным поиском: у модели такого
    механизма нет, поэтому подтвердить основание она не может.
    """
    result = BaselineResult(text="Ответ")
    metrics = baseline_metrics("задача", result)
    assert metrics.citations == 0
    assert metrics.mode == "baseline"


def test_compare_reports_ratios() -> None:
    """Сравнение даёт отношения токенов и времени."""
    agent = RunMetrics(mode="agent", total_tokens=21683, duration_sec=108.0, citations=4)
    baseline = RunMetrics(mode="baseline", total_tokens=3000, duration_sec=10.0, citations=0)
    result = compare(agent, baseline)

    assert result["token_ratio"] == 7.23
    assert result["time_ratio"] == 10.8
    assert result["citations_delta"] == 4


def test_compare_survives_zero_baseline() -> None:
    """Сравнение не делит на ноль, если baseline ничего не вернул."""
    agent = RunMetrics(mode="agent", total_tokens=100, duration_sec=1.0)
    baseline = RunMetrics(mode="baseline", total_tokens=0, duration_sec=0.0)
    result = compare(agent, baseline)
    assert result["token_ratio"] == 0.0
    assert result["time_ratio"] == 0.0


def test_model_name_goes_to_provider_field() -> None:
    """Имя из ``--models`` попадает в поле модели выбранного провайдера.

    На облаке прогон шёл не на той модели, что просили: имя писалось в
    ``ollama_model``, а клиент берёт ``llm_model``. Теперь поле зависит
    от провайдера.
    """
    local = Settings(llm_provider="ollama", ollama_model="qwen3:8b")
    picked = settings_for_model("qwen2.5:7b", local)
    assert picked.ollama_model == "qwen2.5:7b"
    assert picked.effective_model == "qwen2.5:7b"

    cloud = Settings(
        llm_provider="openai",
        llm_model="deepseek-flash",
        llm_base_url="https://api.deepseek.com",
        llm_api_key="test-key",
    )
    picked_cloud = settings_for_model("deepseek-v4-pro", cloud)
    assert picked_cloud.llm_model == "deepseek-v4-pro"
    assert picked_cloud.effective_model == "deepseek-v4-pro"
