"""Проверка расчётов из блока РАСЧЁТЫ: проходит ли круг проверки.

Блок РАСЧЁТЫ — то, что агент показывает модели. Модель копирует оттуда
значение, единицу, ссылку с якорем, формулу и входы агрегата. Значит,
круг проверки надо прогнать именно по тому, что видит модель: иначе
проверяется одно, а агент получает другое.

Скрипт отвечает на вопрос: сколько расчётов из блока проходит
проверку и что именно с теми, что не проходят.

Запуск:
    uv run python scripts/check_grounds.py
    uv run python scripts/check_grounds.py --verbose
"""

from __future__ import annotations

import argparse
import re
from typing import Any

from pf.materials import load_materials
from pf.schemas import MetricRef
from pf.sources import verify_metric
from pf.state import SessionState
from pf.workflow import Workflow

#: Строка блока расчётов: имя, значение, единица, ссылка, формула и
#: входы агрегата (если есть).
LINE_PATTERN = re.compile(
    r"- (?P<name>.+?): (?P<value>-?[\d.]+) ?(?P<unit>\S*)? "
    r"\[(?P<source>.+?)#row(?P<row>\d+)\] \((?P<rest>.+)\)$"
)

#: Входы агрегата в конце строки: «база: 87; новое: 87» или
#: «наблюдения: 6, 7, 10». Ключ — слово перед двоеточием.
INPUT_PATTERN = re.compile(r"(?:^|; )(?P<key>[а-яё][\w ]*): (?P<value>[^;]+)")


def parse_inputs(raw: str) -> tuple[str, dict[str, Any]]:
    """Разобрать скобки строки расчётов на формулу и входы.

    Входы агрегата идут в конце скобок и перечислены через ``;``.
    Разделять надо по ключам входов, а не по первой ``;``: ключ всегда
    виден — это слово перед двоеточием.

    Args:
        raw: Содержимое скобок строки.

    Returns:
        Пара ``(формула, входы)``.
    """
    inputs: dict[str, Any] = {}
    for match in INPUT_PATTERN.finditer(raw):
        key = match.group("key").strip()
        value = match.group("value").strip()
        if "," in value:
            inputs[key] = [item.strip() for item in value.split(",") if item.strip()]
        else:
            inputs[key] = value
    if not inputs:
        return raw, {}
    first = INPUT_PATTERN.search(raw)
    formula = raw[: first.start()].rstrip(" ;") if first else raw
    return formula, inputs


def main() -> int:
    """Прогнать круг проверки по блоку расчётов.

    Returns:
        Код возврата: 0, если все расчёты подтверждены.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="показать все расчёты")
    args = parser.parse_args()

    state = SessionState(session_id="check", phase="collecting", question="удержание")
    block = Workflow._computed_metrics(state)
    materials = load_materials()

    confirmed = 0
    rejected: list[tuple[str, str]] = []
    total = 0

    for line in block.split("\n"):
        match = LINE_PATTERN.match(line.strip())
        if not match:
            continue
        total += 1
        name = match.group("name")
        value = float(match.group("value"))
        unit = match.group("unit") or ""
        source_id = f"{match.group('source')}#row{match.group('row')}"
        row = int(match.group("row"))
        formula, inputs = parse_inputs(match.group("rest"))

        metric = MetricRef(
            name=name,
            value=value,
            unit=unit,
            source_id=source_id,
            row=row,
            formula=formula,
            inputs=inputs,
        )
        support = verify_metric(metric, materials)
        if support.is_usable:
            confirmed += 1
            if args.verbose:
                print(f"  ok   {name[:60]:62} {support.detail}")
        else:
            rejected.append((name, support.detail))
            if args.verbose:
                print(f"  FAIL {name[:60]:62} {support.detail}")

    print()
    print(f"Расчётов в блоке: {total}")
    print(f"Подтверждено:     {confirmed}")
    print(f"Отклонено:        {len(rejected)}")
    if rejected:
        print()
        print("Отклонённые:")
        for name, detail in rejected:
            print(f"  - {name[:70]}")
            print(f"    {detail}")
    return 0 if not rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
