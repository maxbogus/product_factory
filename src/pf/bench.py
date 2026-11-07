"""Замеры: агент против обычного чата с ИИ.

Бриф (дополнительный уровень) требует: «Сравнение с исходным способом.
Измерено, насколько агент быстрее, дешевле или точнее, чем аналитик
вручную или обычный чат с ИИ, и какой ценой».

Модуль делает ровно это и ничего сверх:

- **агент** — полный рабочий процесс с проверкой оснований;
- **baseline** — тот же вопрос тому же модели одним промптом, без
  проверки, без журнала, без восстановления после сбоя.

Разница между ними и есть измеряемая величина. Отдельно видно, за что
платит агент: подтверждённые основания, найденные пробелы, явные
варианты с возражениями — всё то, чего у обычного чата нет по
определению.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pf.config import Settings, settings
from pf.llm import LLMClient, build_client
from pf.tracking import RunMetrics

#: Вопрос по умолчанию для замеров — тот же, что в демонстрации.
DEFAULT_TASK = "Почему падает удержание у мобильного тарифа и что с этим делать?"

#: Инструкция для baseline: обычный чат с той же моделью.
BASELINE_SYSTEM = "Ты — руководитель цифрового продукта. Отвечай по существу, по-русски, без воды."

#: Что спрашиваем у baseline. Вопрос тот же, что получает агент, но без
#: материалов: у обычного чата их нет, и в этом суть сравнения.
BASELINE_PROMPT = """Задача руководителя цифрового продукта:
{question}

Дай развёрнутый ответ: что известно, что предположительно, какие есть
варианты действий, что и как проверить."""


def settings_for_model(model: str, config: Settings | None = None) -> Settings:
    """Настройки прогона для конкретной модели выбранного провайдера.

    Провайдер берётся из переданных настроек (или из окружения): у
    локального Ollama имя модели — ``ollama_model``, у облачного API —
    ``llm_model``. Раньше замер всегда писал имя в ``ollama_model``, и на
    облаке прогон шёл не на той модели, что просили.

    Args:
        model: Имя модели из аргумента ``--models``.
        config: Настройки, из которых берётся провайдер. По умолчанию —
            глобальные настройки.

    Returns:
        Копия настроек с подставленным именем модели.
    """
    used = config or settings
    field = "ollama_model" if used.llm_provider == "ollama" else "llm_model"
    return used.model_copy(update={field: model})


@dataclass(slots=True)
class BaselineResult:
    """Результат baseline-прогона.

    Attributes:
        text: Ответ модели.
        tokens: Потрачено токенов.
        duration_sec: Длительность ответа.
        facts: Сколько фактов насчитано в ответе.
        variants: Сколько вариантов упомянуто.
        error: Ошибка, если ответ не получен.
    """

    text: str = ""
    tokens: int = 0
    duration_sec: float = 0.0
    facts: int = 0
    variants: int = 0
    error: str = ""


def _count_markers(text: str, markers: tuple[str, ...]) -> int:
    """Посчитать, сколько раз встретились маркеры.

    Грубая, но честная мера: у обычного чата нет схемы, поэтому
    считаем словами. Сравнение идёт с агентом, где те же сущности
    посчитаны точно.

    Args:
        text: Текст ответа.
        markers: Слова-маркеры.

    Returns:
        Число совпадений.
    """
    lowered = text.lower()
    return sum(lowered.count(marker) for marker in markers)


def run_baseline(
    task: str,
    client: LLMClient | None = None,
) -> BaselineResult:
    """Прогнать «обычный чат» на той же модели.

    Args:
        task: Вопрос руководителя.
        client: Клиент модели. По умолчанию — из настроек.

    Returns:
        Измерение baseline-прогона.
    """
    llm = client or build_client()
    prompt = BASELINE_PROMPT.format(question=task)
    started = time.perf_counter()
    try:
        response = llm.complete(BASELINE_SYSTEM, prompt)
    except Exception as exc:  # noqa: BLE001 — baseline не должен ронять замер
        return BaselineResult(
            duration_sec=round(time.perf_counter() - started, 3),
            error=f"{type(exc).__name__}: {exc}",
        )
    duration = round(time.perf_counter() - started, 3)
    text = response.content
    return BaselineResult(
        text=text,
        tokens=response.usage.total_tokens,
        duration_sec=duration,
        facts=_count_markers(text, ("факт", "известно", "данные", "согласно")),
        variants=_count_markers(text, ("вариант", "варианты", "альтернатив", "сценарий")),
    )


def run_agent(
    task: str,
    session_id: str,
    client: LLMClient | None = None,
    config: Settings | None = None,
) -> RunMetrics:
    """Прогнать агента и измерить результат.

    Args:
        task: Вопрос руководителя.
        session_id: Идентификатор сессии.
        client: Клиент модели. По умолчанию — из настроек.
        config: Настройки агента.

    Returns:
        Измерение прогона агента.
    """
    from pf.journal import Journal, journal_path
    from pf.state import SessionState, load
    from pf.workflow import run_full_workflow

    settings_used = config or settings
    started = time.perf_counter()
    error = ""
    state: SessionState | None = None
    try:
        state = run_full_workflow(task, session_id, client=client, config=settings_used)
    except Exception as exc:  # noqa: BLE001 — замер не должен роняться
        error = f"{type(exc).__name__}: {exc}"
        state = load(session_id)
    duration = round(time.perf_counter() - started, 3)

    journal = Journal(journal_path(session_id))
    records = journal.read()
    citations = sum(len(record.citations) for record in records)
    rejected = sum(len(record.rejected_details) for record in records)
    facts = 0
    variants = 0
    if state is not None:
        analysis = state.results.get("analyzing", {})
        if isinstance(analysis, dict):
            facts = len(analysis.get("facts", []) or [])
        options = state.results.get("options", {})
        if isinstance(options, dict):
            variants = len(options.get("variants", []) or [])

    return RunMetrics(
        task=task,
        model=settings_used.effective_model,
        provider=settings_used.llm_provider,
        mode="agent",
        steps=len(records),
        citations=citations,
        rejected=rejected,
        facts=facts,
        variants=variants,
        total_tokens=journal.total_tokens,
        duration_sec=duration,
        cost_units=journal.total_cost_units,
        errors=error,
    )


def baseline_metrics(
    task: str, result: BaselineResult, config: Settings | None = None
) -> RunMetrics:
    """Оформить baseline как измерение.

    Args:
        task: Вопрос руководителя.
        result: Результат baseline-прогона.
        config: Настройки агента.

    Returns:
        Измерение baseline-прогона.
    """
    used = config or settings
    return RunMetrics(
        task=task,
        model=used.effective_model,
        provider=used.llm_provider,
        mode="baseline",
        steps=1,
        citations=0,
        facts=result.facts,
        variants=result.variants,
        total_tokens=result.tokens,
        duration_sec=result.duration_sec,
        errors=result.error,
    )


def compare(agent: RunMetrics, baseline: RunMetrics) -> dict[str, Any]:
    """Сравнить агента и обычный чат.

    Args:
        agent: Измерение агента.
        baseline: Измерение обычного чата.

    Returns:
        Словарь с абсолютными и относительными величинами.
    """
    token_ratio = (
        round(agent.total_tokens / baseline.total_tokens, 2) if baseline.total_tokens else 0.0
    )
    time_ratio = (
        round(agent.duration_sec / baseline.duration_sec, 2) if baseline.duration_sec else 0.0
    )
    return {
        "agent": agent.as_metrics(),
        "baseline": baseline.as_metrics(),
        "token_ratio": token_ratio,
        "time_ratio": time_ratio,
        "citations_delta": agent.citations - baseline.citations,
        "notes": [
            "Подтверждённых оснований у обычного чата быть не может: "
            "их проверяет код агента, а не модель.",
            "Токены агента больше: это цена проверки оснований, "
            "фиксации обязательств и журнала шагов.",
        ],
    }
