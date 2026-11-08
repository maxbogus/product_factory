"""Проверки данных демонстрационной задачи.

Задача T-11 выросла из конкретного наблюдения: в демо-прогоне `demo-v9`
нашлось 2 уникальных основания, и оба — одна и та же строка про
абонентскую базу, а оба расчёта удержания были отклонены. Модель
подставила размер базы в формулу удержания и получила «87 процентов» там,
где пересчёт даёт 0. Вывод от этого не стал неверным — он стал
необоснованным: стратегия про удержание опиралась на данные про размер
базы.

Поэтому эти проверки не спрашивают «файл существует». Они спрашивают:

1. у каждого файла данных есть происхождение (`url`, `retrieved`, `note`)
   и записанный хеш — иначе пересмотр выводов (T-12) нечем делать;
2. таблица соблюдает контракт расчёта: колонка значения, колонка периода,
   ряд из двух и более точек — иначе `analysis.summarize_groups` не
   посчитает изменение;
3. из этих данных **выводится** расчёт, который подтверждается
   пересчётом, — то есть данные доходят до блока РАСЧЁТЫ и проверку
   проходят, а не просто лежат в репозитории;
4. если прямого ряда оттока в открытых источниках нет, это **названо**,
   а не замазано похожим показателем.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import pytest

from pf.knowledge_store import sha256_file

#: Корень репозитория: файл лежит в ``tests/``.
ROOT = Path(__file__).resolve().parent.parent

#: Данные проекта — то, на что опирается демонстрация.
PROJECT_DATA = ROOT / "project" / "data"

#: Сырые выгрузки: документ и снимки внешних источников.
KNOWLEDGE_RAW = ROOT / "knowledge" / "raw"

#: Файлы, добавленные задачей T-11. Проверяются по контракту расчёта.
NEW_FILES = ("telecom_arpu.csv", "mnp_stats.csv")

#: Формат даты получения материала.
RETRIEVED = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Период-год: «2018». Другие форматы («8 месяцев 2019») в том же ряду
#: дали бы другой класс ключа сортировки, и изменение посчиталось бы по
#: случайному порядку строк.
YEAR = re.compile(r"^\d{4}$")


def _rows(path: Path) -> list[dict[str, str]]:
    """Прочитать таблицу в словари «колонка → значение»."""
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _sidecar(path: Path) -> dict[str, Any]:
    """Прочитать боковой файл метаданных материала."""
    payload = json.loads((path.parent / f"{path.name}.meta.json").read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _value_column(columns: list[str]) -> str | None:
    """Найти колонку значения так же, как её ищет рабочий процесс."""
    return next(
        (name for name in columns if "знач" in name.lower() or "value" in name.lower()),
        None,
    )


def test_demo_data_files_have_provenance() -> None:
    """У каждого файла данных есть происхождение и записанный хеш.

    Хеш — не украшение: `pf review` сравнивает его с текущим файлом и
    отвечает на вопрос «устарел ли вывод». Без записанного хеша
    пересмотр превращается в «на глаз».
    """
    files = sorted(
        path
        for path in PROJECT_DATA.iterdir()
        if path.suffix in {".csv", ".json"} and not path.name.endswith(".meta.json")
    )
    assert files, "в project/data нет ни одного файла с данными"

    for path in files:
        sidecar_path = path.parent / f"{path.name}.meta.json"
        assert sidecar_path.is_file(), f"у {path.name} нет метаданных"

        meta = _sidecar(path)
        assert str(meta.get("url", "")).startswith("http"), f"у {path.name} не назван источник"
        assert RETRIEVED.match(str(meta.get("retrieved", ""))), (
            f"у {path.name} не названа дата получения"
        )
        assert str(meta.get("note", "")).strip(), f"у {path.name} нет пояснения"

        assert meta.get("sha256") == sha256_file(path), (
            f"хеш в метаданных {path.name} не совпадает с файлом: "
            "пересмотр выводов (T-12) сравнивал бы разные версии"
        )
        assert meta.get("bytes") == path.stat().st_size, f"размер {path.name} в метаданных не тот"


def test_demo_data_table_follows_calculation_contract() -> None:
    """Таблица данных соблюдает контракт расчёта.

    Расчёт ищет колонку значения по имени и отделяет период от подписи
    ряда (`analysis.label_column_for`). Файл, у которого нет ни колонки
    значения, ни периода, до расчёта не доходит — и это не ошибка
    загрузки, а молчаливая потеря данных.
    """
    from pf.analysis import PERIOD_COLUMNS

    for name in NEW_FILES:
        path = PROJECT_DATA / name
        assert path.is_file(), f"нет файла данных {name}"
        rows = _rows(path)
        assert rows, f"{name} пуст"

        columns = list(rows[0].keys())
        assert _value_column(columns), f"в {name} нет колонки значения"
        assert any(column.strip().lower() in PERIOD_COLUMNS for column in columns), (
            f"в {name} нет колонки периода: ряд во времени не построить"
        )


def test_demo_data_series_are_not_mixed() -> None:
    """Ряд — одна величина во времени, а не набор похожих строк.

    «Заявок за год» и «перенесено номеров накопленно» — разные
    показатели: сложенные в один ряд, они дают изменение, которого не
    было. Разные форматы периода в одном ряду дают то же самое, но
    незаметно: порядок строк становится случайным.
    """
    for name in NEW_FILES:
        rows = _rows(PROJECT_DATA / name)
        series: dict[tuple[str, str], list[str]] = {}
        for row in rows:
            key = (row.get("оператор", ""), row.get("показатель", ""))
            series.setdefault(key, []).append(row.get("период", ""))

        assert any(len(periods) >= 2 for periods in series.values()), (
            f"в {name} нет ни одного ряда из двух точек: изменение не посчитать"
        )
        for key, periods in series.items():
            kinds = {bool(YEAR.match(period)) for period in periods}
            assert len(kinds) == 1, (
                f"в {name} ряд {key} смешивает форматы периода {periods}: "
                "порядок наблюдений станет случайным"
            )


def test_demo_data_change_is_confirmed_by_recalculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Из данных проекта выводится расчёт, который подтверждается пересчётом.

    Это и есть критерий задачи: не «файл лежит», а «число посчитано кодом
    и проверку прошло». Считается тем же путём, что в рабочем процессе, —
    `Workflow._computed_metric_records` наполняет блок РАСЧЁТЫ.
    """
    from pf.ground import SupportStatus
    from pf.materials import load_materials
    from pf.sources import verify_metric
    from pf.workflow import Workflow

    monkeypatch.setattr("pf.config.settings.project_dir", ROOT / "project")
    monkeypatch.setattr("pf.config.settings.include_knowledge_raw", False)

    metrics = Workflow._computed_metric_records()
    materials = load_materials(ROOT / "project", include_knowledge=False)

    changes = [
        metric for metric in metrics if "изменение" in metric.name.lower() and "ARPU" in metric.name
    ]
    assert changes, (
        "из ряда ARPU не посчитано изменение: данные есть, а расчёта нет. "
        f"Посчитано: {[metric.name for metric in metrics][:5]}"
    )

    support = verify_metric(changes[0], materials)
    assert support.status is SupportStatus.CONFIRMED, (
        f"расчёт {changes[0].name} не подтверждён пересчётом: {support.detail}"
    )


def test_demo_data_churn_gap_is_named() -> None:
    """Отсутствие прямого ряда оттока названо, а не заполнено похожим.

    Бриф требует называть границы, а не заполнять их. Поэтому либо ряд
    оттока есть (и тогда это отдельный файл), либо пробел назван словами —
    в данных или в описании демонстрации.
    """
    churn = PROJECT_DATA / "telecom_churn_open.csv"
    if churn.is_file():
        assert _rows(churn), "файл оттока есть, но пуст"
        return

    texts = [
        (PROJECT_DATA / "telecom_open_metrics.json").read_text(encoding="utf-8"),
        (ROOT / "demo" / "BRIEF.md").read_text(encoding="utf-8"),
    ]
    named = [text for text in texts if "отток" in text.lower() and "не" in text.lower()]
    assert named, (
        "прямого ряда оттока нет и об этом нигде не сказано: "
        "читатель примет отсутствие данных за их наличие"
    )


def test_demo_data_document_is_readable_or_named() -> None:
    """Фрагмент документа либо читается, либо назван в пропуске.

    Чтение PDF/DOCX требует отдельного набора библиотек. Пропуск без
    причины выглядит как отсутствие документа в проекте — поэтому он
    должен быть назван (механика T-09), а не проглочен.
    """
    from pf.materials import DOCUMENT_SUFFIXES, scan_materials

    scan = scan_materials(ROOT / "project", knowledge_root=KNOWLEDGE_RAW)
    documents = [
        material for material in scan.materials if Path(material.path).suffix in DOCUMENT_SUFFIXES
    ]
    skipped = {
        Path(item.path).name: item.reason
        for item in scan.skipped
        if Path(item.path).suffix in DOCUMENT_SUFFIXES
    }

    assert documents or skipped, (
        "фрагмента документа нет ни среди материалов, ни среди пропусков: "
        "чтение документов в демонстрации не показано"
    )
    if documents:
        assert any(material.passages for material in documents), (
            "документ прочитан, но цитировать из него нечего"
        )
    else:
        reasons = " ".join(skipped.values())
        assert "uv sync --extra documents" in reasons, f"пропуск документа не объяснён: {skipped}"
