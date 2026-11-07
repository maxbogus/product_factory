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
                rejected_details=[f"{citation}: цитата не найдена в строке таблицы"],
            )
        ],
    )

    generator = _generator()
    confirmed_doc = generator.build_document("demo-confirmed")
    rejected_doc = generator.build_document("demo-rejected")

    assert citation in _section(confirmed_doc, "## 2.")
    assert citation not in _section(rejected_doc, "## 2.")


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
