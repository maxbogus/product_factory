"""Проверки вывода в отчёте по прогону.

Отчёт отвечал на вопрос «сколько оснований подтверждено» и не отвечал на
вопрос, ради которого запускался прогон: что решили и на чём это держится.
Здесь проверяется, что вывод попадает в отчёт вместе со статусом своих
оснований — и что отсутствие вывода названо словами, а не пропущено.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from pf.ground import CONCLUSION_RULE
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
    off_topic: dict[str, list[str]] | None = None,
    rule: str | None = None,
    rejected_details: list[str] | None = None,
    assumptions: list[str] | None = None,
) -> StepRecord:
    """Запись шага стратегии.

    Args:
        status: Статус оснований вывода.
        confirmed: Сколько оснований вывода подтверждено.
        rejected: Сколько отклонено.
        checked: Сколько проверено. По умолчанию — подтверждённые и отклонённые.
        grounds: Ключи оснований по вердиктам — так их называет новый журнал.
        evidence: Ссылки оснований вывода; повторы сохраняются, как в ответе модели.
        off_topic: Показатели вывода и оснований, когда они не совпали.
        rule: Имя правила счёта. Не задано — как в унаследованном журнале.
        rejected_details: Строки отказов шага; по ним отчёт видит противоречие.
        assumptions: Что должно быть верно, чтобы вывод работал.

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
    if off_topic is not None:
        check["off_topic"] = off_topic
    if rule is not None:
        check["rule"] = rule
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
    if assumptions is not None:
        product["assumptions"] = assumptions
    return StepRecord(
        ordinal=7,
        role="strategy",
        phase="strategy",
        summary="стратегия: продуктовая + коммуникационная + план проверки",
        citations=["project/data/telecom_open_metrics.csv#row0"],
        rejected_details=(
            ["project/data/cbr_rates.csv#row1: источник не относится к задаче"]
            if rejected_details is None
            else rejected_details
        ),
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


def test_report_names_off_topic_grounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Отчёт называет, что основания не про тот показатель.

    Дословная цитата про абонентскую базу не подкрепляет вывод про удержание.
    Без этой строки читатель видит только «подкреплено частично» и не знает
    почему — статус выглядит случайной строгостью.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    _session(
        tmp_path,
        monkeypatch,
        [
            _strategy_record(
                status="подкреплено частично",
                confirmed=1,
                rejected=0,
                off_topic={
                    "grounds": ["абонентская база"],
                    "conclusion": ["удержание клиентов"],
                },
            ),
            _keeper_record(),
        ],
    )

    report = build_session_report("session-conclusion")

    assert report.conclusion is not None
    assert report.conclusion.off_topic_grounds == ["абонентская база"]
    text = report.as_markdown()
    assert "не про тот показатель" in text
    assert "абонентская база" in text
    assert "удержание клиентов" in text


def test_off_topic_grounds_cap_status_at_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Без отказов, но с чужим показателем вывод не бывает «подкреплено».

    Счётчики говорят «1 из 1 подтверждено», и прежнее правило объявило бы
    вывод подкреплённым. Показатель у основания другой — значит потолок
    «подкреплено частично», и это решает одно правило на весь репозиторий
    (`pf.ground.conclusion_status`), а не два разных расчёта.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    _session(
        tmp_path,
        monkeypatch,
        [
            _strategy_record(
                status="подкреплено",
                confirmed=1,
                rejected=0,
                off_topic={"grounds": ["абонентская база"], "conclusion": ["удержание"]},
            ),
            _keeper_record(),
        ],
    )

    report = build_session_report("session-conclusion")

    assert report.conclusion is not None
    assert report.conclusion.status == "подкреплено частично"
    assert "Основания вывода: подкреплено частично" in report.as_markdown()


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


def test_report_groups_repeated_rejections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Одно и то же отклонение не повторяется в списке по числу шагов.

    В прогоне `cloud-deepseek-2` строка «источник не назван — вместо ссылки
    указано поле „Ответ человека"» встречалась трижды, а выгрузка `.docx` —
    тоже трижды. Читается это как шесть разных отказов, хотя отказ один.
    Число повторов сохраняется: оно показывает, на скольких шагах модель
    повторила ошибку.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    repeated = "project/data/numbers.csv#row0: цитата не найдена в источнике дословно"
    other = "project/data/cbr_rates.csv#row1: источник не относится к задаче"
    records = [
        StepRecord(
            ordinal=1,
            role="collector",
            phase="collecting",
            summary="сбор",
            rejected_details=[repeated, other],
        ),
        StepRecord(
            ordinal=2,
            role="analyst",
            phase="analyzing",
            summary="разбор",
            rejected_details=[repeated],
        ),
    ]
    _session(tmp_path, monkeypatch, records)

    text = build_session_report("session-conclusion").as_markdown()

    assert text.count(f"- {repeated}") == 1, "повтор остался отдельной строкой"
    assert "(×2)" in text
    assert f"- {other}" in text


def test_report_marks_run_unfit_for_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Прогон без вывода назван непригодным, а не равноправным участником.

    `demo-final` и `demo-v6` читались как обычные отчёты: числа, расход,
    отклонения — и только разобравшись, можно было заметить, что ответа на
    вопрос в них нет.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    _session(
        tmp_path,
        monkeypatch,
        [
            StepRecord(
                ordinal=1,
                role="collector",
                phase="collecting",
                summary="сбор",
                citations=["project/data/numbers.csv#row0"],
            )
        ],
    )

    text = build_session_report("session-conclusion").as_markdown()

    assert "не даёт решения" in text


def test_report_names_share_as_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Доля подписана честно: она про записи, а не про основания.

    В `cloud-deepseek-baseline-1` и `cloud-deepseek-2` «Доля подтверждённых»
    считалась по записям, а читалась как доля уникальных оснований: 83 % и
    73 % при 9 и 7 уникальных основаниях.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    _session(tmp_path, monkeypatch, [_strategy_record(), _keeper_record()])

    text = build_session_report("session-conclusion").as_markdown()

    assert "Доля подтверждённых записей" in text


def test_report_counts_commitments_without_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Обязательства без владельца названы числом, а не спрятаны в таблице.

    В `cloud-deepseek-2` четыре обязательства из шести имели владельца
    «не назначен». Это видно в таблице, но не видно как риск: обязательство
    без владельца некому исполнить.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    keeper = _keeper_record()
    detail = dict(keeper.detail or {})
    detail["commitments"] = [
        {
            "statement": "Внедрить уведомление о сумме продления",
            "owner": "руководитель продукта",
            "metric": "доля уведомлений до списания",
        },
        {
            "statement": "Сверить текст во всех каналах",
            "owner": "не назначен",
            "metric": "совпадение суммы между каналами",
        },
    ]
    keeper = StepRecord(
        ordinal=8,
        role="keeper",
        phase="keeper",
        summary="решение: 2 обязательства",
        detail=detail,
    )
    _session(tmp_path, monkeypatch, [_strategy_record(), keeper])

    text = build_session_report("session-conclusion").as_markdown()

    assert "Без владельца: 1 из 2" in text


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


# ── Правило счёта ─────────────────────────────────────────────────────────────
#
# `demo-v9` печатал «Основания вывода: подкреплено» рядом с таблицей, где те же
# две строки помечены «отклонено»: статус пришёл из журнала, записанного
# правилом, которое считало посчитанное значение в строке расчёта цитатой
# (`D-36` это правило сняло). Журнал своего правила не называл, и отчёт
# принимал запись за сегодняшнюю оценку.

ROW0 = "project/data/telecom_open_metrics.csv#row0"
ROW1 = "project/data/telecom_open_metrics.csv#row1"


def test_report_stale_rule_status_is_labelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Журнал без имени правила не выдаёт свой статус за текущий.

    Унаследованный журнал записан правилом, которого уже нет: оно считало
    посчитанное значение в строке расчёта цитатой. Отчёт обязан назвать это
    записью прогона и сказать, каким правилом считают сейчас, — иначе рядом
    стоят «подкреплено» и «отклонено», и читатель выбирает удобное.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    _session(
        tmp_path,
        monkeypatch,
        [
            _strategy_record(
                confirmed=2,
                rejected=0,
                evidence=[ROW0, ROW1],
                rejected_details=[f"{ROW0}: значение посчитано, а не процитировано"],
            )
        ],
    )

    report = build_session_report("session-conclusion")
    text = report.as_markdown()

    assert report.conclusion is not None
    assert report.conclusion.rule_is_current is False
    assert "по правилу прогона" in text
    assert "не записано" in text
    assert CONCLUSION_RULE in text
    assert "Основания вывода: подкреплено**" not in text
    assert "Журнал себе противоречит" in text


def test_report_stale_rule_marker_absent_for_current_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Журнал, записанный текущим правилом, печатает статус как есть.

    Подпись нужна только унаследованным журналам: если правило записано и
    совпадает с текущим, оговорка превратилась бы в шум и перестала читаться.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    _session(
        tmp_path,
        monkeypatch,
        [
            _strategy_record(
                confirmed=2,
                rejected=0,
                evidence=[ROW0, ROW1],
                rule=CONCLUSION_RULE,
            )
        ],
    )

    report = build_session_report("session-conclusion")
    text = report.as_markdown()

    assert report.conclusion is not None
    assert report.conclusion.rule_is_current is True
    assert "Основания вывода: подкреплено**" in text
    assert "по правилу прогона" not in text
    assert "Прогон даёт решение по прежнему правилу" not in text


def test_report_stale_rule_caveats_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Унаследованный статус не выглядит зелёным светом.

    Иначе первая строка отчёта говорит «готово», а раздел «Вывод» ниже
    уточняет, что статус посчитан прежним правилом. Порядок чтения важнее:
    оговорка стоит там, где читатель решает, можно ли опираться на вывод.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    _session(
        tmp_path,
        monkeypatch,
        [_strategy_record(confirmed=2, rejected=0, evidence=[ROW0, ROW1])],
    )

    text = build_session_report("session-conclusion").as_markdown()

    assert "Прогон даёт решение по прежнему правилу" in text
    assert "make replay" in text


# ── Допущения вывода ──────────────────────────────────────────────────────────
#
# Вывод работает, пока верны его допущения. В `cloud-deepseek-2` обещание
# «удержание вырастет» держалось на допущении «отток не ускорится», которого в
# документе не было ни в каком виде: читатель не мог его ни принять, ни
# отвергнуть, а проверять собирался только результат.


def test_report_shows_assumptions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Допущения стоят рядом с выводом, а не в приложении.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    _session(
        tmp_path,
        monkeypatch,
        [
            _strategy_record(
                confirmed=2,
                rejected=0,
                evidence=[ROW0, ROW1],
                rule=CONCLUSION_RULE,
                assumptions=["Отток не ускорится", "Цена останется в прежней вилке"],
            )
        ],
    )

    text = build_session_report("session-conclusion").as_markdown()

    assert "Что должно быть верно, чтобы вывод работал" in text
    assert "Отток не ускорится" in text
    assert "Допущения вывода не названы" not in text


def test_report_names_missing_assumptions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Прогон текущим правилом без допущений называет пробел.

    Унаследованный журнал не судится: поля тогда не было, и требовать его
    задним числом — наказывать за отсутствие разметки. Для нового прогона
    молчание недопустимо: отсутствие допущений спряталось бы в «всё хорошо».

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    _session(
        tmp_path,
        monkeypatch,
        [
            _strategy_record(
                confirmed=2,
                rejected=0,
                evidence=[ROW0, ROW1],
                rule=CONCLUSION_RULE,
            )
        ],
    )

    text = build_session_report("session-conclusion").as_markdown()

    assert "Допущения вывода не названы" in text


# ── Согласованность артефакта ─────────────────────────────────────────────────
#
# `cloud-deepseek-baseline-1` печатал «Шагов выполнено: 12» при десяти реальных
# шагах: в журнал писали два запуска, и номера 9 и 10 встречались дважды. На
# HEAD такой журнал уже не создать (`journal.DuplicateOrdinalError`), но
# унаследованные журналы читаются — и отчёт обязан сказать, что это два
# запуска, а не двенадцать шагов одного.


def _write_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, records: list[StepRecord]) -> None:
    """Записать журнал в обход защиты от дублей — так выглядит старый журнал.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
        records: Записи в том порядке, в каком они лежат в файле.
    """
    monkeypatch.setattr("pf.config.settings.journal_dir", tmp_path)
    with journal_path("session-conclusion").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def test_report_counts_steps_by_ordinal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Шаг считается по номеру: дубль номера — два запуска, а не два шага.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    records = [
        StepRecord(ordinal=1, role="intake", phase="intake", summary="постановка"),
        StepRecord(ordinal=1, role="intake", phase="intake", summary="постановка"),
        StepRecord(ordinal=2, role="planner", phase="planning", summary="план"),
    ]
    _write_raw(tmp_path, monkeypatch, records)

    report = build_session_report("session-conclusion")

    assert report.steps == 2
    text = report.as_markdown()
    assert "записан 2 раза" in text
    assert "| Шагов выполнено | 2 |" in text


def test_report_shows_considered_variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Отчёт показывает набор вариантов и его полюса.

    В `cloud-deepseek-2` вариантов было четыре, включая «Оставить как есть:
    работать только с коммуникацией», а в отчёте их не было вовсе: читатель
    видел «осознанные отказы» и не мог понять, из чего выбирали и был ли
    вообще набор решением, а не списком пожеланий.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    options = StepRecord(
        ordinal=3,
        role="options",
        phase="options",
        summary="варианты: 3",
        detail={
            "variants": [
                {
                    "title": "Оставить как есть: только коммуникация",
                    "summary": "Ничего не менять в тарифе и цене.",
                    "approach": "Переписать коммуникацию о том, что уже действует.",
                    "objections": ["Если падение вызвано условиями, текст не поможет"],
                    "distinct_axis": "что меняем: продукт или текст",
                },
                {
                    "title": "Гибкий тариф с персонализацией",
                    "summary": "Собрать тариф из блоков.",
                    "approach": "Дать выбор состава.",
                    "objections": ["Дорого и долго"],
                    "distinct_axis": "цена и состав",
                },
                {
                    "title": "Закрыть направление",
                    "summary": "Прекратить продукт.",
                    "approach": "Вывести тариф.",
                    "objections": ["Теряем базу"],
                    "distinct_axis": "оставаться ли в продукте",
                },
            ],
            "keep_current_marker": "Оставить как есть",
            "close_direction_marker": "Закрыть направление",
        },
    )
    _session(tmp_path, monkeypatch, [options, _strategy_record(), _keeper_record()])

    text = build_session_report("session-conclusion").as_markdown()

    assert "## Рассмотренные варианты" in text
    assert "Гибкий тариф с персонализацией" in text
    assert "Если падение вызвано условиями, текст не поможет" in text
    assert "оставить как есть" in text.lower()
    assert "закрыть направление" in text.lower()
