#!/usr/bin/env python3
"""Гейт поставляемых документов: числа, цели и пути сходятся с кодом.

Зачем. Документ расходится с кодом молча: на 2026-11-08 README обещал
«259 тестов», описание — «433» и «259», при факте 448, а окно контекста
называлось «8192» при умолчании 12288. Такое расхождение не ломает
сборку, поэтому живёт месяцами — и именно его читает жюри.

Источник истины — ``docs/canon.py``. Документ обязан называть число
оттуда; гейт падает, если:

- документ называет устаревшее значение (реестр ``STALE_DOC_NUMBERS``);
- число в документе не совпадает с каноном или пропало совсем;
- ``AGENTS.md`` ссылается на ``make``-цель, которой нет в Makefile;
- ``AGENTS.md`` ссылается на путь в репозитории, которого не существует.

Принцип тот же, что у остальных гейтов проекта: проверка, которая никогда
не падает, хуже отсутствия проверки. Красные прогоны — в
``tests/test_docs_check.py``.

Запуск:

    uv run python scripts/docs_check.py
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

#: Корень репозитория.
ROOT = Path(__file__).resolve().parent.parent

#: Где лежит канон чисел.
CANON_PATH = ROOT / "docs" / "canon.py"

#: Файл карты проекта.
AGENTS_FILE = "AGENTS.md"

#: Цель Makefile: `target: ## описание`.
MAKEFILE_TARGET_RE = re.compile(r"^(?P<target>[a-zA-Z][a-zA-Z0-9_-]*):", re.MULTILINE)

#: Команда `make …` в тексте.
MAKE_CALL_RE = re.compile(r"\bmake\s+(?P<target>[a-zA-Z][a-zA-Z0-9_-]*)")

#: Путь в обратных кавычках: `src/pf/ground.py`, `prompts/roles/`.
PATH_IN_BACKTICKS_RE = re.compile(r"`(?P<path>[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.*-]*)*/?)`")

#: Каталоги, пути из которых обязаны существовать. Остальные упоминания
#: (например, `state/` в тексте про git) гейт не трогает: он проверяет
#: карту проекта, а не каждую строку описания.
TRACKED_PREFIXES = (
    "src/",
    "tests/",
    "scripts/",
    "docs/",
    "prompts/",
    "methods/",
    "skills/",
    "hooks/",
    "eval/",
    "mcp/",
    "architecture/",
    "project/",
    "knowledge/",
    "data/",
    "reports/",
)

#: Пути, которые появляются только во время работы. В свежем клоне их нет
#: и быть не должно: очередь распознания слов — след прогона, она в
#: `.gitignore`. Карта проекта вправе о ней упоминать.
RUNTIME_ARTIFACTS = frozenset({"knowledge/lexicon/oov.jsonl"})


def load_canon(path: Path | None = None) -> ModuleType:
    """Загрузить канон чисел как модуль.

    Args:
        path: Путь к файлу канона. По умолчанию — ``docs/canon.py``.

    Returns:
        Модуль канона.
    """
    target = path or CANON_PATH
    spec = importlib.util.spec_from_file_location("pf_docs_canon", target)
    if spec is None or spec.loader is None:
        raise ImportError(f"канон не загружается: {target}")
    module = importlib.util.module_from_spec(spec)
    # Модуль обязан быть в sys.modules до исполнения: иначе dataclasses не
    # находят его при разборе `from __future__ import annotations`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read(path: str, root: Path | None = None) -> str:
    """Прочитать документ.

    Args:
        path: Путь относительно корня репозитория.
        root: Корень. По умолчанию — корень репозитория.

    Returns:
        Текст документа; пустая строка, если файла нет.
    """
    target = (root or ROOT) / path
    if not target.is_file():
        return ""
    return target.read_text(encoding="utf-8")


def _number(raw: str) -> int:
    """Разобрать число, записанное с пробелами: «8 066» → 8066.

    Args:
        raw: Строка с числом.

    Returns:
        Значение числа.
    """
    cleaned = re.sub(r"[^\d]", "", raw)
    return int(cleaned) if cleaned else -1


def check_stale_numbers(canon: ModuleType | None = None, root: Path | None = None) -> list[str]:
    """В документах не осталось устаревших значений.

    Args:
        canon: Модуль канона. По умолчанию загружается сам.
        root: Корень репозитория.

    Returns:
        Список нарушений.
    """
    book = canon or load_canon()
    problems: list[str] = []
    cache: dict[str, str] = {}
    for entry in book.STALE_DOC_NUMBERS:
        if entry.path not in cache:
            cache[entry.path] = _read(entry.path, root)
        text = cache[entry.path]
        if not text:
            problems.append(f"{entry.path}: документ не найден")
            continue
        if entry.text in text:
            problems.append(
                f"{entry.path}: устаревшее «{entry.text}» — должно быть «{entry.replaced_by}»"
            )
    return problems


def check_doc_numbers(canon: ModuleType | None = None, root: Path | None = None) -> list[str]:
    """Числа в документах совпадают с каноном.

    Проверяются все вхождения образца, а не первое: одно и то же число
    называется в разных разделах, и правка одного из них — самая частая
    причина расхождения.

    Args:
        canon: Модуль канона. По умолчанию загружается сам.
        root: Корень репозитория.

    Returns:
        Список нарушений.
    """
    book = canon or load_canon()
    problems: list[str] = []
    cache: dict[str, str] = {}
    for entry in book.DOC_NUMBERS:
        if entry.path not in cache:
            cache[entry.path] = _read(entry.path, root)
        text = cache[entry.path]
        if not text:
            problems.append(f"{entry.path}: документ не найден")
            continue
        matches = list(re.finditer(entry.pattern, text))
        if not matches:
            expected = ", ".join(str(value) for value in entry.expected)
            problems.append(
                f"{entry.path}: не найдено «{entry.hint}» (ожидалось {expected}) — "
                "число исчезло из документа"
            )
            continue
        for match in matches:
            groups = match.groups()
            if len(groups) != len(entry.expected):
                problems.append(
                    f"{entry.path}: {entry.hint} — образец дал {len(groups)} чисел, "
                    f"ожидалось {len(entry.expected)}"
                )
                continue
            for raw, expected in zip(groups, entry.expected, strict=True):
                if raw is None:
                    continue
                actual = _number(raw)
                if actual != expected:
                    problems.append(
                        f"{entry.path}: {entry.hint} — написано {actual}, в каноне {expected}"
                    )
    return problems


def check_make_targets(root: Path | None = None) -> list[str]:
    """Каждая `make`-цель из карты проекта существует в Makefile.

    Args:
        root: Корень репозитория.

    Returns:
        Список нарушений.
    """
    base = root or ROOT
    makefile = base / "Makefile"
    if not makefile.is_file():
        return ["Makefile не найден"]
    targets = set(MAKEFILE_TARGET_RE.findall(makefile.read_text(encoding="utf-8")))
    text = _read(AGENTS_FILE, base)
    if not text:
        return [f"{AGENTS_FILE}: документ не найден"]
    problems: list[str] = []
    for target in sorted(set(MAKE_CALL_RE.findall(text))):
        if target not in targets:
            problems.append(f"{AGENTS_FILE}: цель «make {target}» не объявлена в Makefile")
    return problems


def check_doc_paths(root: Path | None = None) -> list[str]:
    """Пути, названные в карте проекта, существуют.

    Проверяются только пути из отслеживаемых каталогов и без шаблонов:
    карта проекта обещает, что по названному пути что-то лежит.

    Args:
        root: Корень репозитория.

    Returns:
        Список нарушений.
    """
    base = root or ROOT
    text = _read(AGENTS_FILE, base)
    if not text:
        return [f"{AGENTS_FILE}: документ не найден"]
    problems: list[str] = []
    seen: set[str] = set()
    for match in PATH_IN_BACKTICKS_RE.finditer(text):
        path = match.group("path")
        if path in seen or "*" in path or "…" in path:
            continue
        if path in RUNTIME_ARTIFACTS:
            continue
        if not path.startswith(TRACKED_PREFIXES):
            continue
        seen.add(path)
        if not (base / path).exists():
            problems.append(f"{AGENTS_FILE}: путь `{path}` не существует")
    return problems


CHECKS = {
    "stale_numbers": check_stale_numbers,
    "doc_numbers": check_doc_numbers,
    "make_targets": check_make_targets,
    "doc_paths": check_doc_paths,
}


def main(argv: list[str] | None = None) -> int:
    """Точка входа гейта документов.

    Args:
        argv: Аргументы командной строки (не используются).

    Returns:
        0 — расхождений нет; 1 — найдены.
    """
    _ = argv
    try:
        canon = load_canon()
    except ImportError as exc:
        print(f"[FAIL] docs-check: {exc}")
        return 1

    problems: list[str] = []
    problems.extend(check_stale_numbers(canon=canon))
    problems.extend(check_doc_numbers(canon=canon))
    problems.extend(check_make_targets())
    problems.extend(check_doc_paths())

    if not problems:
        print(f"[OK] docs-check: {len(CHECKS)} проверок, расхождений нет")
        return 0

    print(f"[FAIL] docs-check: нарушений — {len(problems)}")
    for problem in problems:
        print(f"  - {problem}")
    print()
    print("  Как чинить: числа берутся из docs/canon.py — он источник истины.")
    print("  Если число изменилось в коде, сначала поправьте канон (make canon),")
    print("  затем документ.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

