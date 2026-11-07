"""Проверки политики расширения: когда прекращать добор оснований.

Расширение — не бесплатное действие: сетевая ошибка, пустой результат,
дубль и исчерпанный бюджет одинаково означают, что новый раунд не даст
знания. Без явной политики агент ходит по кругу: в прогоне `demo-v5`
шесть отказов подряд выглядели как «проверка работает», хотя два из них
были одним и тем же источником.

Здесь проверяется только политика — без модели и без сети: код ответа
превращается в исход, исход — в решение «повторить / закрыть / не
ходить больше в этот класс источников».
"""

from __future__ import annotations

import pytest

from pf.policy import (
    MAX_TRANSIENT_ATTEMPTS,
    NO_PROGRESS_LIMIT,
    AttemptOutcome,
    NodeProgress,
    SourceHealth,
    classify_code,
    classify_content,
    classify_http,
    content_key,
)


def test_not_found_is_permanent_and_not_retried() -> None:
    """404 — постоянный отказ: повторять бессмысленно."""
    assert classify_http(404) is AttemptOutcome.PERMANENT
    health = SourceHealth()
    health.record(AttemptOutcome.PERMANENT)
    assert health.unavailable
    assert not health.should_retry()


def test_robots_is_permanent() -> None:
    """Запрет robots.txt не лечится повтором."""
    assert classify_http(200, robots=True) is AttemptOutcome.PERMANENT


def test_server_error_is_transient_then_unavailable() -> None:
    """5xx — временный отказ: повторяем, но не бесконечно."""
    assert classify_http(500) is AttemptOutcome.TRANSIENT
    health = SourceHealth()
    for _ in range(MAX_TRANSIENT_ATTEMPTS):
        assert health.should_retry()
        health.record(AttemptOutcome.TRANSIENT)
    assert health.unavailable
    assert not health.should_retry(), "класс источника исчерпан — больше не ходим"


def test_timeout_and_rate_limit_are_transient() -> None:
    """Таймаут и 429 — тоже временные: сеть подводит, а не источник."""
    assert classify_http(None) is AttemptOutcome.TRANSIENT
    assert classify_http(429) is AttemptOutcome.TRANSIENT


def test_success_is_ok() -> None:
    """2xx — транспорт в порядке; что нашли — решает содержание."""
    assert classify_http(200) is AttemptOutcome.OK


def test_duplicate_by_text_hash_is_not_progress() -> None:
    """Разные адреса с одним текстом — один и тот же абзац."""
    first = content_key("  Абонентская   база МТС — 87 млн ")
    second = content_key("абонентская база мтс — 87 млн")
    assert first == second

    outcome = classify_content(new=0, duplicate=2)
    assert outcome is AttemptOutcome.DUPLICATE
    assert not outcome.is_progress


def test_empty_and_duplicate_results_are_not_progress() -> None:
    """Пустой результат — тоже отсутствие знания, а не «почти получилось»."""
    assert classify_content(new=0, duplicate=0) is AttemptOutcome.EMPTY
    assert classify_content(new=1, duplicate=0) is AttemptOutcome.NEW
    assert AttemptOutcome.NEW.is_progress


def test_node_closes_after_repeated_empty_results() -> None:
    """Два пустых раунда подряд — узел закрыт, возвращаться некуда."""
    node = NodeProgress()
    for _ in range(NO_PROGRESS_LIMIT):
        assert not node.is_closed
        node.record(AttemptOutcome.EMPTY)
    assert node.is_closed
    assert node.closed_reason == "no_progress"


def test_new_ground_reopens_progress_counter() -> None:
    """Найденное основание сбрасывает счётчик неудач узла."""
    node = NodeProgress()
    node.record(AttemptOutcome.EMPTY)
    node.record(AttemptOutcome.NEW)
    node.record(AttemptOutcome.EMPTY)
    assert not node.is_closed, "после находки счётчик начинается заново"


def test_unavailable_source_closes_node_immediately() -> None:
    """Если класс источника исчерпан, узел закрывается без ожидания."""
    node = NodeProgress()
    node.record(AttemptOutcome.UNAVAILABLE)
    assert node.is_closed
    assert node.closed_reason == "unavailable"
    assert not node.is_closed_with_reason("no_progress")


def test_budget_outcome_stops_without_retry() -> None:
    """Исчерпанный бюджет — стоп расширения, а не повод повторить."""
    health = SourceHealth()
    health.record(AttemptOutcome.BUDGET)
    assert not health.should_retry()
    assert AttemptOutcome.BUDGET.is_stop


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("ok", AttemptOutcome.OK),
        ("robots", AttemptOutcome.PERMANENT),
        ("http_404", AttemptOutcome.PERMANENT),
        ("http_403", AttemptOutcome.PERMANENT),
        ("http_429", AttemptOutcome.TRANSIENT),
        ("http_503", AttemptOutcome.TRANSIENT),
        ("timeout", AttemptOutcome.TRANSIENT),
        ("network", AttemptOutcome.TRANSIENT),
        ("empty", AttemptOutcome.EMPTY),
    ],
)
def test_page_code_is_classified(code: str, expected: AttemptOutcome) -> None:
    """Машинный код страницы превращается в исход без разбора текста.

    Текст статуса («страница недоступна: HTTPStatusError») не отличает
    404 от 500, а политике нужно именно это различие: постоянный отказ
    не повторяют, временный — повторяют.
    """
    assert classify_code(code) is expected


@pytest.mark.parametrize("status", [400, 401, 403, 451])
def test_client_errors_are_permanent(status: int) -> None:
    """Ошибки клиента не лечатся повтором: запрос просто не тот."""
    assert classify_http(status) is AttemptOutcome.PERMANENT
