"""Проверки вывода в отчёте по прогону.

Отчёт отвечал на вопрос «сколько оснований подтверждено» и не отвечал на
вопрос, ради которого запускался прогон: что решили и на чём это держится.
Здесь проверяется, что вывод попадает в отчёт вместе со статусом своих
оснований — и что отсутствие вывода названо словами, а не пропущено.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pf.journal import Journal, StepRecord, journal_path
from pf.report import build_session_report


def _strategy_record(
    *, status: str = "подкреплено", confirmed: int = 2, rejected: int = 1
) -> StepRecord:
    """Запись шага стратегии.

    Args:
        status: Статус оснований вывода.
        confirmed: Сколько оснований вывода подтверждено.
        rejected: Сколько отклонено.

    Returns:
        Запись журнала.
    """
    return StepRecord(
        ordinal=7,
        role="strategy",
        phase="strategy",
        summary="стратегия: продуктовая + коммуникационная + план проверки",
        citations=["project/data/telecom_open_metrics.csv#row0"],
        rejected_details=["project/data/cbr_rates.csv#row1: источник не относится к задаче"],
        detail={
            "product": {
                "audience": "клиенты мобильного тарифа",
                "alternatives": ["Оставить как есть", "Закрыть направление"],
                "chosen_direction": "Внедрить гибкие тарифные планы",
                "deliberate_omissions": ["Неизвестны точные причины падения удержания"],
                "value_to_cost": "Риски закрытия направления выше затрат на адаптацию",
            },
            "communication": {
                "audience": "клиенты мобильного тарифа",
                "barrier": "Цена выше, чем у альтернатив",
                "key_message": "Мы держим цену под контролем",
                "proof": "Данные об абонентской базе",
                "channels": "приложение, сайт",
            },
            "verification": {
                "expected_change": "Удержание вырастет на 2 п.п.",
                "compare_against": "уровень прошлого квартала",
                "observation_period": "3 месяца",
                "adverse_effects": ["Маржа снизится"],
                "success_criteria": ["Удержание клиентов вырастет"],
            },
            "conclusion_check": {
                "checked": confirmed + rejected,
                "confirmed": confirmed,
                "rejected": rejected,
                "status": status,
            },
        },
    )


def _keeper_record() -> StepRecord:
    """Запись шага фиксации решения.

    Returns:
        Запись журнала.
    """
    return StepRecord(
        ordinal=8,
        role="keeper",
        phase="keeper",
        summary="решение: 1 обязательство",
        citations=["project/data/telecom_open_metrics.csv#row1"],
        detail={
            "decision": "Внедрить гибкие тарифные планы",
            "commitments": [
                {
                    "statement": "Удержание вырастет на 2 п.п.",
                    "owner": "команда тарифа",
                    "metric": "удержание клиентов",
                    "review_condition": "если через 3 месяца прироста нет",
                    "review_by": "2024-06-30",
                }
            ],
        },
    )


def _session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, records: list[StepRecord]) -> None:
    """Записать синтетическую сессию.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
        records: Записи журнала.
    """
    monkeypatch.setattr("pf.config.settings.journal_dir", tmp_path)
    journal = Journal(journal_path("session-conclusion"))
    for record in records:
        journal.append(record)


def test_report_shows_what_to_do(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Вывод попадает в отчёт: направление, решение и обязательства.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    _session(tmp_path, monkeypatch, [_strategy_record(), _keeper_record()])

    report = build_session_report("session-conclusion")
    text = report.as_markdown()

    assert report.conclusion is not None
    assert report.conclusion.direction == "Внедрить гибкие тарифные планы"
    assert report.conclusion.decision == "Внедрить гибкие тарифные планы"
    assert "## Вывод" in text
    assert "Внедрить гибкие тарифные планы" in text
    assert "если через 3 месяца прироста нет" in text


def test_report_names_ungrounded_conclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Вывод без подтверждённого основания назван прямо.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    _session(
        tmp_path,
        monkeypatch,
        [_strategy_record(status="неподкреплено", confirmed=0, rejected=2), _keeper_record()],
    )

    text = build_session_report("session-conclusion").as_markdown()

    assert "не подкреплён" in text or "неподкреплено" in text
    assert "cbr_rates.csv" in text


def test_report_says_when_there_is_no_conclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Сессия без вывода говорит об этом, а не молчит.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    _session(
        tmp_path,
        monkeypatch,
        [
            StepRecord(
                ordinal=0,
                role="collector",
                phase="collecting",
                summary="сбор",
                citations=["project/data/numbers.csv#row0"],
            )
        ],
    )

    report = build_session_report("session-conclusion")
    text = report.as_markdown()

    assert report.conclusion is None
    assert "Вывод не получен" in text
