"""Проверки гейта поставляемых документов.

Гейт ловит расхождение документа с кодом: устаревшее число, неверное
число, `make`-цель или путь, которых нет. Каждая проверка обязана уметь
падать — поэтому у каждой здесь красный прогон.

Интеграционная проверка `test_real_documents_pass` держит сами документы:
после правки чисел в README и `docs/DESCRIPTION.md` гейт обязан быть
зелёным.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import docs_check  # noqa: E402  (путь к scripts добавляется выше)

CANON = docs_check.load_canon()


def _write(root: Path, path: str, text: str) -> None:
    """Записать файл во временный репозиторий.

    Args:
        root: Корень временного репозитория.
        path: Путь относительно корня.
        text: Содержимое.
    """
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_canon_exposes_numbers() -> None:
    """Канон читается и содержит числа."""
    assert CANON.TESTS_DETERMINISTIC > 0
    assert CANON.OLLAMA_NUM_CTX > CANON.TESTS_LIVE
    assert CANON.DOC_NUMBERS


def test_number_with_spaces_is_parsed() -> None:
    """«8 066» — это 8066: разряды разделены пробелом."""
    assert docs_check._number("8 066") == 8066
    assert docs_check._number("8066") == 8066


def test_stale_number_is_caught(tmp_path: Path) -> None:
    """Устаревшее число в документе — нарушение."""
    _write(tmp_path, "README.md", "make test        # 259 тестов, без обращения к модели\n")
    problems = docs_check.check_stale_numbers(canon=CANON, root=tmp_path)
    assert problems and "259 тестов" in problems[0]


def test_stale_number_absent_when_fixed(tmp_path: Path) -> None:
    """Исправленное число устаревшим не считается."""
    _write(tmp_path, "README.md", "make test        # 448 тестов, без обращения к модели\n")
    _write(
        tmp_path,
        "docs/DESCRIPTION.md",
        "448 детерминированных + 4 живых теста с моделью\n",
    )
    problems = docs_check.check_stale_numbers(canon=CANON, root=tmp_path)
    assert problems == []


def test_wrong_number_is_caught(tmp_path: Path) -> None:
    """Число, которого нет в каноне, — нарушение."""
    _write(
        tmp_path,
        "docs/DESCRIPTION.md",
        "make test          # 999 детерминированных тестов\n",
    )
    problems = docs_check.check_doc_numbers(canon=CANON, root=tmp_path)
    assert any("999" in problem for problem in problems)


def test_missing_number_is_caught(tmp_path: Path) -> None:
    """Число пропало из документа — это тоже расхождение."""
    _write(tmp_path, "docs/DESCRIPTION.md", "Отчёт о работе агента.\n")
    problems = docs_check.check_doc_numbers(canon=CANON, root=tmp_path)
    assert any("исчезло" in problem for problem in problems)


def test_missing_document_is_reported(tmp_path: Path) -> None:
    """Документа нет — гейт говорит об этом, а не молчит."""
    problems = docs_check.check_stale_numbers(canon=CANON, root=tmp_path)
    assert problems and "не найден" in problems[0]


def test_make_target_must_exist(tmp_path: Path) -> None:
    """Цель `make`, названная в карте проекта, должна быть в Makefile."""
    _write(tmp_path, "Makefile", "test: ## тесты\n\t@echo ok\n")
    _write(tmp_path, "AGENTS.md", "Запуск: `make test` и `make nosuch`.\n")
    problems = docs_check.check_make_targets(tmp_path)
    assert problems and "nosuch" in problems[0]


def test_make_target_that_exists_passes(tmp_path: Path) -> None:
    """Существующая цель жалоб не вызывает."""
    _write(tmp_path, "Makefile", "test: ## тесты\n\t@echo ok\n")
    _write(tmp_path, "AGENTS.md", "Запуск: `make test`.\n")
    assert docs_check.check_make_targets(tmp_path) == []


def test_doc_path_must_exist(tmp_path: Path) -> None:
    """Путь из карты проекта, которого нет, — расхождение карты с кодом."""
    (tmp_path / "src" / "pf").mkdir(parents=True)
    _write(tmp_path, "AGENTS.md", "Модуль: `src/pf/nope.py`.\n")
    problems = docs_check.check_doc_paths(tmp_path)
    assert problems and "src/pf/nope.py" in problems[0]


def test_doc_path_that_exists_passes(tmp_path: Path) -> None:
    """Существующий путь принимается."""
    (tmp_path / "src" / "pf").mkdir(parents=True)
    (tmp_path / "src" / "pf" / "module.py").write_text("", encoding="utf-8")
    _write(tmp_path, "AGENTS.md", "Модуль: `src/pf/module.py`.\n")
    assert docs_check.check_doc_paths(tmp_path) == []


def test_real_documents_pass() -> None:
    """Документы репозитория проходят собственный гейт."""
    problems: list[str] = []
    problems.extend(docs_check.check_stale_numbers())
    problems.extend(docs_check.check_doc_numbers())
    problems.extend(docs_check.check_make_targets())
    problems.extend(docs_check.check_doc_paths())
    assert problems == [], "; ".join(problems)
