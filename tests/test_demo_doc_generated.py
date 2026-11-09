"""Проверки генератора документа демонстрации.

`demo/RESULT.md` — поставляемый артефакт. Его нельзя писать руками: один
раз это уже кончилось тем, что в таблице расчётов стояли числа без
формулы, а ссылка на строку, которую проверка отвергла, — в таблице
подтверждённых оснований.

Поэтому документ собирается из журнала прогона, и здесь проверяются три
свойства этой сборки:

1. документ совпадает с тем, что получается из его собственного журнала;
2. ссылка, которую проверка отвергла, не выдаётся за подтверждённую;
3. расчёт без формулы не выглядит пересчитываемым.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from pf.journal import Journal, StepRecord, journal_path

ROOT = Path(__file__).resolve().parent.parent
DEMO_DOC = ROOT / "demo" / "RESULT.md"


def _generator() -> ModuleType:
    """Загрузить генератор документа как модуль.

    Returns:
        Модуль `scripts/build_demo_doc.py`.
    """
    path = ROOT / "scripts" / "build_demo_doc.py"
    spec = importlib.util.spec_from_file_location("build_demo_doc", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _section(text: str, header: str) -> str:
    """Вырезать раздел документа.

    Args:
        text: Текст документа.
        header: Заголовок раздела.

    Returns:
        Текст раздела до следующего горизонтального разделителя.
    """
    start = text.index(header)
    end = text.find("\n---", start)
    return text[start : end if end != -1 else len(text)]


def _write_session(session_id: str, records: list[StepRecord]) -> None:
    """Записать синтетический журнал сессии.

    Args:
        session_id: Идентификатор сессии.
        records: Записи журнала.
    """
    journal = Journal(journal_path(session_id))
    for record in records:
        journal.append(record)


def test_document_matches_its_journal() -> None:
    """Документ — это ровно то, что собирается из его журнала.

    Если документ правили руками, он разойдётся с прогоном: в нём
    появятся основания, которых проверка не подтверждала, или опустеет
    колонка «Как считалось».
    """
    text = DEMO_DOC.read_text(encoding="utf-8")
    match = re.search(r"^Сессия:\s*(\S+)$", text, re.MULTILINE)
    assert match, "в документе не указано, из какой сессии он собран"

    generated = _generator().build_document(match.group(1))
    assert generated == text, "документ расходится с журналом своей сессии — пересоберите его"


def test_rejected_citation_is_not_presented_as_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Отвергнутая ссылка не попадает в подтверждённые основания.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    monkeypatch.setattr("pf.config.settings.journal_dir", tmp_path)
    citation = "project/data/numbers.csv#row0"

    _write_session(
        "demo-confirmed",
        [
            StepRecord(
                ordinal=0,
                role="collector",
                phase="collecting",
                summary="шаг",
                citations=[citation],
            )
        ],
    )
    _write_session(
        "demo-rejected",
        [
            StepRecord(
                ordinal=0,
                role="collector",
                phase="collecting",
                summary="шаг",
                citations=[citation],
                rejected_details=[f"{citation}: источник не относится к задаче"],
            )
        ],
    )

    generator = _generator()
    confirmed_doc = generator.build_document("demo-confirmed")
    rejected_doc = generator.build_document("demo-rejected")

    assert citation in _section(confirmed_doc, "## 2.")
    assert citation not in _section(rejected_doc, "## 2.")


def test_form_rejection_does_not_hide_confirmed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Отказ по форме цитаты не вычёркивает источник из подтверждённых.

    В прогоне `demo-v9` одна и та же строка таблицы подтверждала вывод
    (стратегия ссылалась на неё) и отклонялась в факте — «значение
    посчитано, а не процитировано». Если считать любой отказ отказом по
    источнику, раздел «Подтверждённые основания» пустеет, а раздел
    «Вывод» в том же документе говорит «подкреплено»: документ спорит сам
    с собой.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    monkeypatch.setattr("pf.config.settings.journal_dir", tmp_path)
    citation = "project/data/numbers.csv#row0"

    _write_session(
        "demo-form-reject",
        [
            StepRecord(
                ordinal=0,
                role="collector",
                phase="collecting",
                summary="шаг",
                citations=[citation],
                rejected_details=[
                    f"{citation}: значение посчитано, а не процитировано — "
                    "оформи как метрику в metrics"
                ],
            )
        ],
    )

    document = _generator().build_document("demo-form-reject")

    assert citation in _section(document, "## 2.")


def _conclusion_records() -> list[StepRecord]:
    """Журнал сессии, дошедшей до вывода.

    Returns:
        Записи шагов стратегии и фиксации решения.
    """
    return [
        StepRecord(
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
                },
                "conclusion_check": {
                    "checked": 2,
                    "confirmed": 1,
                    "rejected": 1,
                    "status": "подкреплено частично",
                },
            },
        ),
        StepRecord(
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
        ),
    ]


def test_document_states_the_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Документ отвечает на заданный вопрос: что решили и на чём держится.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    monkeypatch.setattr("pf.config.settings.journal_dir", tmp_path)
    _write_session("demo-conclusion", _conclusion_records())

    document = _generator().build_document("demo-conclusion")

    assert "## 6. Вывод" in document
    assert "Внедрить гибкие тарифные планы" in document
    assert "если через 3 месяца прироста нет" in document
    assert "подкреплено частично" in document
    assert "cbr_rates.csv" in document


def test_document_says_when_there_is_no_conclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Прогон без стратегии говорит об этом словами, а не молчанием.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    monkeypatch.setattr("pf.config.settings.journal_dir", tmp_path)
    _write_session(
        "demo-no-conclusion",
        [
            StepRecord(
                ordinal=0,
                role="collector",
                phase="collecting",
                summary="шаг",
                citations=["project/data/numbers.csv#row0"],
            )
        ],
    )

    document = _generator().build_document("demo-no-conclusion")

    assert "Вывод не получен" in document


def test_metric_without_formula_is_marked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Расчёт без формулы помечен: пересчитать его нечем.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    monkeypatch.setattr("pf.config.settings.journal_dir", tmp_path)
    _write_session(
        "demo-metrics",
        [
            StepRecord(
                ordinal=0,
                role="analyst",
                phase="analyzing",
                summary="шаг",
                detail={
                    "metrics": [
                        {
                            "name": "Абонентская база МТС",
                            "value": 82.0,
                            "unit": "млн",
                            "source_id": "project/data/numbers.csv",
                            "row": 0,
                            "formula": "значение из колонки «значение», строка 1",
                            "inputs": {"значение": "82"},
                        },
                        {
                            "name": "Доля оттока",
                            "value": 4.5,
                            "unit": "%",
                            "source_id": "project/data/numbers.csv",
                        },
                    ]
                },
            )
        ],
    )

    document = _generator().build_document("demo-metrics")
    section = _section(document, "## 3.")
    rows = [line for line in section.splitlines() if line.startswith("|")]

    with_formula = next(row for row in rows if "Абонентская база МТС" in row)
    without_formula = next(row for row in rows if "Доля оттока" in row)
    assert "значение из колонки" in with_formula
    assert "формула не указана" in without_formula


def test_rejection_grouping_dedups_repeats_and_groups_by_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Отклонения в документе: группа по виду причины, повторы схлопнуты.

    Плоский список читался как много разных отказов, хотя отказ один: одна
    и та же строка приходит с разных шагов. Документ обязан сгруппировать
    отказы по виду причины (а не по тексту) и схлопнуть повторы с «(×N)».

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    monkeypatch.setattr("pf.config.settings.journal_dir", tmp_path)
    repeated = "project/data/numbers.csv#row0: цитата не найдена"
    other = "project/data/numbers.csv#row1: число 5 не совпадает с источником"

    _write_session(
        "demo-rejection-grouping",
        [
            StepRecord(
                ordinal=0,
                role="analyst",
                phase="analyzing",
                summary="разбор",
                rejected_details=[repeated, other],
            ),
            StepRecord(
                ordinal=1,
                role="strategy",
                phase="strategy",
                summary="стратегия",
                rejected_details=[repeated],
            ),
        ],
    )

    document = _generator().build_document("demo-rejection-grouping")
    section = _section(document, "## 5.")

    # Разные отказы одного вида стоят под общим заголовком вида.
    assert "**цитата не найдена** (2)" in section
    assert "**значение не сошлось с источником** (1)" in section
    assert repeated in section
    # Повтор двух шагов схлопнут в одну строку со счётчиком.
    assert section.count(repeated) == 1
    assert f"- {repeated} (×2)" in section


def test_rejection_grouping_prefers_recorded_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Документ группирует по записанному виду, а не по разбору текста.

    Журнал append-only: новый отказ `indicator-demotion` хранит вид в поле
    `rejections`. Если документ угадывает класс по формулировке, отказ
    попадает в «прочее» — как это было до `D-48`.

    Args:
        tmp_path: Временный каталог журналов.
        monkeypatch: Подмена каталога журналов.
    """
    monkeypatch.setattr("pf.config.settings.journal_dir", tmp_path)
    detail = "формулировка, которой нет в таблице текстов"

    _write_session(
        "demo-rejection-kind",
        [
            StepRecord(
                ordinal=0,
                role="strategy",
                phase="strategy",
                summary="стратегия",
                rejected_details=[detail],
                rejections=[
                    {
                        "kind": "indicator-demotion",
                        "status": "no_data",
                        "detail": detail,
                        "claim": "вывод",
                    }
                ],
            )
        ],
    )

    document = _generator().build_document("demo-rejection-kind")
    section = _section(document, "## 5.")

    assert "**основание про другой показатель**" in section
    assert "**прочее**" not in section
