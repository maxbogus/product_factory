#!/usr/bin/env python3
"""Собрать demo/RESULT.md из журнала прогона.

Документ демонстрации — поставляемый артефакт, и он обязан быть
следствием прогона, а не текстом, написанным руками. Иначе в нём
появляются числа без формулы и основания, которые проверка отвергла.

Правила сборки:

- в подтверждённые основания попадает ссылка, которую код подтвердил и
  ни разу не отверг;
- у каждого расчёта видна формула; если её нет, это сказано прямо;
- числа шагов и оснований берутся из журнала, а не из пересказа.

Запуск:
    uv run python scripts/build_demo_doc.py --session demo-v4
    uv run python scripts/build_demo_doc.py --session demo-v4 --out /tmp/RESULT.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pf.config import settings
from pf.journal import Journal, StepRecord, journal_path
from pf.report import SessionReport, build_session_report


def _citation_key(citation: str) -> str:
    """Ключ основания: путь и первый якорь.

    Якорь может продублироваться (``путь#row0#p0``) — для счёта и
    сравнения это одно и то же основание.

    Args:
        citation: Ссылка на источник.

    Returns:
        Ссылка без повторного якоря.
    """
    path, _, fragment = citation.partition("#")
    parts = [part for part in fragment.split("#") if part]
    return f"{path}#{parts[0]}" if parts else path


def _rejected_keys(records: list[StepRecord]) -> set[str]:
    """Ключи ссылок, которые проверка отвергла.

    Args:
        records: Записи журнала.

    Returns:
        Множество ключей отклонённых ссылок.
    """
    rejected: set[str] = set()
    for record in records:
        for line in record.rejected_details:
            rejected.add(_citation_key(line.split(":", 1)[0].strip()))
    return rejected


def _confirmed_grounds(records: list[StepRecord]) -> list[tuple[str, str]]:
    """Подтверждённые основания в порядке появления.

    Основание подтверждено, если код нашёл его в источнике и ни один шаг
    не отверг ту же ссылку. Отклонение сильнее: ссылку, которую хотя бы
    раз не подтвердили, выдавать за подтверждённую нельзя.

    Args:
        records: Записи журнала.

    Returns:
        Пары «где подтверждено — ссылка».
    """
    rejected = _rejected_keys(records)
    seen: set[str] = set()
    grounds: list[tuple[str, str]] = []
    for record in records:
        for citation in record.citations:
            key = _citation_key(citation)
            if key in seen or key in rejected:
                continue
            seen.add(key)
            grounds.append((f"{record.role}, {record.phase}", citation))
    return grounds


def _render(report: SessionReport, records: list[StepRecord]) -> str:
    """Собрать текст документа.

    Args:
        report: Отчёт по сессии.
        records: Записи журнала.

    Returns:
        Текст документа в markdown.
    """
    lines: list[str] = []
    lines.append("# Результат демонстрационной задачи")
    lines.append("")
    if report.question:
        lines.append(f"Задача: **«{report.question}»**")
    else:
        lines.append("Задача: не записана в состоянии сессии")
    lines.append("")
    lines.append(f"Сессия: {report.session_id}")
    lines.append("")
    lines.append("Документ собран из журнала шагов: основания и расчёты взяты")
    lines.append("оттуда, а статусы проставил код. Поэтому он не может разойтись")
    lines.append("с тем, что агент действительно сделал.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 1. Что измерено")
    lines.append("")
    lines.append("| Показатель | Значение |")
    lines.append("|---|---:|")
    lines.append(f"| Шагов выполнено | {report.steps} |")
    lines.append(f"| Подтверждённых записей | {report.confirmed_citations} |")
    lines.append(f"| Подтверждённых оснований (уникальных) | {report.unique_citations} |")
    lines.append(f"| Использовано источников | {report.unique_sources} |")
    lines.append(f"| Оснований отклонено | {report.rejected_citations} |")
    lines.append(f"| Доля подтверждённых | {report.grounding_rate:.0%} |")
    lines.append(f"| Токенов | {sum(report.tokens.values())} |")
    lines.append(f"| Расход | {report.total_cost_units:.6f} условных единиц |")
    lines.append("")
    report_file = settings.reports_dir / f"session-{report.session_id}.md"
    if report_file.is_file():
        lines.append(f"Полный отчёт с разбивкой расхода по ролям — `{report_file}`.")
        lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 2. Подтверждённые основания")
    lines.append("")
    lines.append("Основание подтверждает код: цитата должна найтись в источнике.")
    lines.append("Ссылка, которую проверка хотя бы однажды отвергла, сюда не попадает.")
    lines.append("")
    grounds = _confirmed_grounds(records)
    if grounds:
        lines.append("| Подтверждено на шаге | Источник |")
        lines.append("|---|---|")
        for where, citation in grounds:
            lines.append(f"| {where} | `{citation}` |")
    else:
        lines.append("Подтверждённых оснований нет: проверка не нашла ни одной")
        lines.append("цитаты, дословно совпавшей с источником.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 3. Расчёты по данным")
    lines.append("")
    lines.append("Числа посчитаны кодом по строкам таблицы, а не взяты из текста")
    lines.append("модели. Колонка «Статус» показывает, что из этого прошло проверку")
    lines.append("пересчётом.")
    lines.append("")
    if report.metrics:
        lines.append("| Метрика | Значение | Строка источника | Как считалось | Статус |")
        lines.append("|---|---:|---|---|---|")
        for metric in report.metrics:
            citation = f"`{metric.citation}`" if metric.citation else "—"
            lines.append(
                f"| {metric.name} | {metric.formatted} | {citation} | "
                f"{metric.formula_note} | {metric.status} |"
            )
    else:
        lines.append("В этом прогоне расчётов нет: агент не вернул ни одной метрики")
        lines.append("с источником и строкой.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 4. Контроль качества")
    lines.append("")
    lines.append("| Что проверено | Значение |")
    lines.append("|---|---:|")
    lines.append(
        f"| Расчётов с формулой | {report.metrics_with_formula} из {len(report.metrics)} |"
    )
    for kind, count in sorted(report.rejection_kinds.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| Отклонено: {kind} | {count} |")
    if report.fact_check:
        lines.append(f"| Повторная проверка (fact-check) | {report.fact_check.get('summary', '—')} |")
    lines.append("")
    if report.routing_notes:
        lines.append("**Решения роутера.**")
        lines.append("")
        for note in report.routing_notes:
            lines.append(f"- {note}")
        lines.append("")
    lines.append("---")
    lines.append("")

    if report.rejected_details:
        lines.append("## 5. Отклонённые основания")
        lines.append("")
        lines.append("Что проверка не приняла и почему. Отклонение — это результат,")
        lines.append("а не сбой: лучше видимый отказ, чем правдоподобное число.")
        lines.append("")
        for detail in report.rejected_details:
            lines.append(f"- {detail}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## 6. Чего этот замер не показывает")
    lines.append("")
    for item in report.limitations:
        lines.append(f"1. {item}")
    lines.append("")
    return "\n".join(lines)


def build_document(session_id: str) -> str:
    """Собрать документ демонстрации по журналу сессии.

    Args:
        session_id: Идентификатор сессии.

    Returns:
        Текст документа.
    """
    report = build_session_report(session_id)
    records = Journal(journal_path(session_id)).read()
    return _render(report, records)


def main(argv: list[str] | None = None) -> int:
    """Собрать документ и записать его в файл.

    Args:
        argv: Аргументы командной строки.

    Returns:
        Код возврата: 0 — документ записан.
    """
    parser = argparse.ArgumentParser(description="Собрать demo/RESULT.md из журнала сессии")
    parser.add_argument("--session", required=True, help="идентификатор сессии")
    parser.add_argument(
        "--out",
        default="demo/RESULT.md",
        help="куда записать документ (по умолчанию demo/RESULT.md)",
    )
    args = parser.parse_args(argv)

    document = build_document(str(args.session))
    target = Path(str(args.out))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    print(f"[OK] {target} собран из сессии {args.session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
