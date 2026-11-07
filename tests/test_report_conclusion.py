"""Проверки вывода в отчёте по прогону.

Отчёт отвечал на вопрос «сколько оснований подтверждено» и не отвечал на
вопрос, ради которого запускался прогон: что решили и на чём это держится.
Здесь проверяется, что вывод попадает в отчёт вместе со статусом своих
оснований — и что отсутствие вывода названо словами, а не пропущено.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pf.journal import Journal, StepRecord, journal_path
from pf.report import build_session_report


def _strategy_record(
    *,
    status: str = "подкреплено",
    confirmed: int = 2,
    rejected: int = 1,
    checked: int | None = None,
    grounds: dict[str, list[str]] | None = None,
    evidence: list[str] | None = None,
) -> StepRecord:
    """Запись шага стратегии.

    Args:
        status: Статус оснований вывода.
        confirmed: Сколько оснований вывода подтверждено.
        rejected: Сколько отклонено.
        checked: Сколько проверено. По умолчанию — подтверждённые и отклонённые.
        grounds: Ключи оснований по вердиктам — так их называет новый журнал.
        evidence: Ссылки оснований вывода; повторы сохраняются, как в ответе модели.

    Returns:
        Запись журнала.
    """
    check: dict[str, Any] = {
        "checked": confirmed + rejected if checked is None else checked,
        "confirmed": confirmed,
        "rejected": rejected,
        "status": status,
    }
    if grounds is not None:
        check["grounds"] = grounds
    product: dict[str, Any] = {
        "audience": "клиенты мобильного тарифа",
        "alternatives": ["Оставить как есть", "Закрыть направление"],
        "chosen_direction": "Внедрить гибкие тарифные планы",
        "deliberate_omissions": ["Неизвестны точные причины падения удержания"],
        "value_to_cost": "Риски закрытия направления выше затрат на адаптацию",
    }
    if evidence is not None:
        product["evidence"] = [
            {"source_id": item, "paragraph": None, "quote": "цитата"} for item in evidence
        ]
    return StepRecord(
        ordinal=7,
        role="strategy",
        phase="strategy",
        summary="стратегия: продуктовая + коммуникационная + план проверки",
        citations=["project/data/telecom_open_metrics.csv#row0"],
        rejected_details=["project/data/cbr_rates.csv#row1: источник не относится к задаче"],
        detail={
            "product": product,
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
            "conclusion_check": check,
        },
    )


def _keeper_record(check: dict[str, Any] | None = None) -> StepRecord:
    """Запись шага фиксации решения.

    Args:
        check: Статус оснований вывода на шаге фиксации, если он есть.

    Returns:
        Запись журнала.
    """
    detail: dict[str, Any] = {
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
    }
    if check is not None:
        detail["conclusion_check"] = check
    return StepRecord(
        ordinal=8,
        role="keeper",
        phase="keeper",
        summary="решение: 1 обязательство",
        citations=["project/data/telecom_open_metrics.csv#row1"],
        detail=detail,
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


def test_report_counts_unique_grounds_across_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Два шага, ссылающиеся на одни строки, не удваивают основания вывода.

    В прогоне `demo-v9` стратегия и фиксация решения объявляли по четыре
    основания из двух строк таблицы. Отчёт складывал записи: «проверено 8,
    подтверждено 8» — рядом с «уникальных оснований 2» в том же документе.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    row0 = "project/data/telecom_open_metrics.csv#row0"
    row1 = "project/data/telecom_open_metrics.csv#row1"
    grounds = {"confirmed": [row0, row1], "rejected": [], "needs_review": []}
    check: dict[str, Any] = {
        "checked": 2,
        "confirmed": 2,
        "rejected": 0,
        "status": "подкреплено",
        "grounds": grounds,
    }
    _session(
        tmp_path,
        monkeypatch,
        [
            _strategy_record(confirmed=2, rejected=0, grounds=grounds),
            _keeper_record(check),
        ],
    )

    report = build_session_report("session-conclusion")
    text = report.as_markdown()

    assert report.conclusion is not None
    assert report.conclusion.checked == 2
    assert report.conclusion.confirmed == 2
    assert report.conclusion.unique is True
    assert "уникальных оснований" in text
    assert "проверено 2" in text


def test_report_old_journal_without_keys_is_honest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Журнал без ключей: при отсутствии отказов число уникальных известно.

    Старые журналы (до правила уникальных оснований) называют только
    счётчики. Но если ни одно основание вывода не отклонено, уникальных
    ровно столько, сколько объявлено в выводе: это следует из записанных
    вердиктов, а не из догадки.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    row0 = "project/data/telecom_open_metrics.csv#row0"
    row1 = "project/data/telecom_open_metrics.csv#row1"
    _session(
        tmp_path,
        monkeypatch,
        [
            _strategy_record(
                confirmed=4,
                rejected=0,
                evidence=[row0, row1, row0, row1],
            )
        ],
    )

    conclusion = build_session_report("session-conclusion").conclusion

    assert conclusion is not None
    assert conclusion.checked == 2
    assert conclusion.confirmed == 2
    assert conclusion.unique is True


def test_report_old_journal_with_rejections_counts_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Журнал без ключей и с отказами говорит «записей», а не «оснований».

    Распределить отказы по уникальным ключам старый журнал не позволяет:
    неизвестно, какая из повторных ссылок отклонена. Выдать записи за
    уникальные основания значило бы соврать в удобную сторону.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    row0 = "project/data/telecom_open_metrics.csv#row0"
    row1 = "project/data/telecom_open_metrics.csv#row1"
    _session(
        tmp_path,
        monkeypatch,
        [
            _strategy_record(
                status="подкреплено частично",
                confirmed=1,
                rejected=1,
                evidence=[row0, row1],
            )
        ],
    )

    report = build_session_report("session-conclusion")
    text = report.as_markdown()

    assert report.conclusion is not None
    assert report.conclusion.unique is False
    assert "записей" in text
    assert "подкреплено частично" in text


def test_report_marks_conclusion_awaiting_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Вывод, у которого основания ждут человека, не назван подкреплённым.

    Раньше отчёт считал статус по трём числам и не знал про «ждёт
    проверки»: внешний источник подтвердить нечем, а в отчёте вывод
    выглядел подкреплённым.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    _session(
        tmp_path,
        monkeypatch,
        [
            _strategy_record(
                status="ждёт проверки",
                confirmed=0,
                rejected=0,
                grounds={"confirmed": [], "rejected": [], "needs_review": ["web/Новости"]},
            )
        ],
    )

    report = build_session_report("session-conclusion")
    text = report.as_markdown()

    assert report.conclusion is not None
    assert report.conclusion.status == "ждёт проверки"
    assert "ждут проверки" in text
    assert "Основания вывода: подкреплено**" not in text
