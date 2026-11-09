"""Проверки документа демонстрации.

Документ «Результат демонстрационной задачи» — поставляемый артефакт.
Раньше он был написан руками, и в нём расчёты, отклонённые проверкой,
стояли в таблице как посчитанные кодом, а факт со ссылкой на строку,
которую проверка отвергла, — в таблице подтверждённых оснований.

Отсюда правило: документ не пишется, а **генерируется** из журнала
прогона. Тогда он не может разойтись с тем, что агент действительно
сделал и что код действительно проверил.

Здесь проверяется именно это: каждая строка таблицы оснований должна
найти подтверждение в журнале, а каждая отклонённая ссылка не должна
попасть в подтверждённые.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pf.config import settings

DEMO_DOC = Path("demo/RESULT.md")

#: Строка с идентификатором сессии в шапке документа. Документ сам
#: называет прогон, из которого собран: после повторного демо-прогона
#: тесты не нужно править руками.
SESSION_LINE = re.compile(r"^Сессия:\s*(\S+)$", re.MULTILINE)


def _session_id(text: str) -> str:
    """Идентификатор сессии, из которой собран документ.

    Args:
        text: Текст документа.

    Returns:
        Идентификатор сессии.
    """
    match = SESSION_LINE.search(text)
    assert match, "в документе не указана сессия, из которой он собран"
    return match.group(1)


def _journal_records(session_id: str) -> list[dict[str, object]]:
    """Записи журнала сессии.

    Args:
        session_id: Идентификатор сессии.

    Returns:
        Список записей.
    """
    path = settings.journal_dir / f"session-{session_id}.jsonl"
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue
    return records


def _document_citations(text: str) -> list[str]:
    """Ссылки из таблицы подтверждённых оснований.

    Args:
        text: Текст документа.

    Returns:
        Список ссылок вида ``путь#якорь``.
    """
    section = re.search(
        r"## 2\. Подтверждённые основания(.*?)(?=\n---)",
        text,
        re.DOTALL,
    )
    if not section:
        return []
    return re.findall(r"`([^`]+#[^`]+)`", section.group(1))


def test_demo_document_lists_only_confirmed_grounds() -> None:
    """В документе нет источника, который проверка отвергла по существу.

    Это главный смысл документа: он показывает, что именно удалось
    подтвердить. Ссылка на источник, который не относится к задаче или
    которого нет в материалах, в таблице подтверждённых оснований —
    ложь.

    Отказ по форме цитаты («значение посчитано, а не процитировано»)
    источник не дискредитирует: в прогоне `demo-v9` одна и та же строка
    подтверждала вывод и отклонялась в факте, и вычёркивать её из
    подтверждённых значило бы спорить с разделом «Вывод» того же
    документа.
    """
    text = DEMO_DOC.read_text(encoding="utf-8")
    records = _journal_records(_session_id(text))
    assert records, "журнал демо-прогона не найден"

    source_rejections = (
        "не относится к задаче",
        "не найден среди материалов",
        "вне среза задачи",
        "цитировать нечего",
        "ввод человека",
    )
    rejected: set[str] = set()
    for record in records:
        for line in record.get("rejected_details", []):
            if not any(marker in str(line) for marker in source_rejections):
                continue
            match = re.match(r"([^:]+):", str(line))
            if match:
                rejected.add(match.group(1).strip())

    for citation in _document_citations(text):
        assert citation not in rejected, (
            f"в документе основание {citation} помечено как подтверждённое, а проверка его отвергла"
        )


def test_demo_document_citations_exist_in_journal() -> None:
    """Каждое основание в документе подтверждено в журнале.

    Args:
        tmp_path: Не используется.
    """
    text = DEMO_DOC.read_text(encoding="utf-8")
    records = _journal_records(_session_id(text))
    assert records, "журнал демо-прогона не найден"

    confirmed: set[str] = set()
    for record in records:
        for citation in record.get("citations", []):
            confirmed.add(str(citation))

    for citation in _document_citations(text):
        assert citation in confirmed, (
            f"основание {citation} есть в документе, но в журнале не подтверждено"
        )


def test_demo_document_has_no_empty_formula_cell() -> None:
    """У расчёта в документе заполнена колонка «Как считалось».

    Пустая колонка означает, что расчёт нечем пересчитать: либо формулы
    нет, либо она не показана. И то и другое делает число unverifiable.
    """
    text = DEMO_DOC.read_text(encoding="utf-8")
    section = re.search(r"## 3\. Расчёты по данным(.*?)(?=\n---)", text, re.DOTALL)
    assert section, "в документе нет раздела с расчётами"

    rows = [line for line in section.group(1).splitlines() if line.strip().startswith("|")]
    assert len(rows) > 2, "в разделе расчётов нет строк"

    for row in rows[2:]:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        assert len(cells) >= 4, f"в строке расчёта не хватает колонок: {row}"
        assert cells[3], f"у расчёта не указано, как он считался: {row}"


def test_demo_document_numbers_match_report() -> None:
    """Числа в документе совпадают с отчётом по прогону.

    Документ и отчёт читают один и тот же журнал, но собираются
    разными путями. Если числа разошлись, документ врёт.
    """
    from pf.report import build_session_report

    text = DEMO_DOC.read_text(encoding="utf-8")
    report = build_session_report(_session_id(text))
    assert report.steps > 0, "отчёт по демо-прогону пуст"

    assert f"| Шагов выполнено | {report.steps} |" in text, (
        "число шагов в документе не совпадает с отчётом"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
