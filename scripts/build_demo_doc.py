#!/usr/bin/env python3
"""Собрать demo/RESULT.md из журнала прогона.

Документ демонстрации — поставляемый артефакт, и он обязан быть следствием
прогона, а не текстом, написанным руками. Иначе в нём появляются числа без
формулы и основания, которые проверка отвергла.

Сборка живёт в `pf.artifacts`: этим же кодом документ собирает `pf report` и
проверка «документ совпадает с журналом». Скрипт — тонкая обёртка, второй
копии правил здесь нет (`T-63`).

Запуск:
    uv run python scripts/build_demo_doc.py --session demo-v4
    uv run python scripts/build_demo_doc.py --session demo-v4 --out /tmp/RESULT.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pf.artifacts import build_document, provenance, recheck, recheck_markdown

__all__ = ["build_document", "main", "provenance", "recheck", "recheck_markdown"]


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
