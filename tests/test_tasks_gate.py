"""Проверки гейта индекса задач.

Гейт существует, чтобы «задача закрыта» нельзя было написать без
доказательства. Значит и сам гейт обязан падать: у каждой проверки здесь
есть красный прогон. Проверка, которая никогда не падает, хуже отсутствия
проверки — на этом проекте так уже вышло с eval-кейсами `D-05` и `D-06`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import next_task  # noqa: E402  (путь к scripts добавляется выше)
import tasks_gate  # noqa: E402


def _index(rows: str) -> str:
    """Собрать индекс с правильным заголовком.

    Args:
        rows: Строки таблицы.

    Returns:
        Текст индекса с маркерами.
    """
    header = "| ID | Фаза | Задача | SP | RICE | Статус | Зависит | Критерий приёмки |"
    separator = "|---|---|---|---|---|---|---|---|"
    return f"{tasks_gate.BEGIN_MARK}\n{header}\n{separator}\n{rows}\n{tasks_gate.END_MARK}\n"


def _row(
    task_id: str = "T-01",
    sp: int = 1,
    rice: str = "10.0",
    status: str = "⬜",
    depends: str = "—",
    acceptance: str = "`make test`",
) -> str:
    """Собрать строку задачи.

    Args:
        task_id: Идентификатор.
        sp: Оценка трудоёмкости.
        rice: Оценка RICE.
        status: Статус.
        depends: Зависимости.
        acceptance: Критерий приёмки.

    Returns:
        Строка Markdown-таблицы.
    """
    return f"| {task_id} | 1 | задача | {sp} | {rice} | {status} | {depends} | {acceptance} |"


def test_index_parses_into_tasks() -> None:
    """Индекс разбирается в задачи с полями."""
    tasks, problems = tasks_gate.parse_index(_index(_row()))
    assert not problems
    assert len(tasks) == 1
    task = tasks[0]
    assert task.task_id == "T-01"
    assert task.sp == 1
    assert task.rice == 10.0
    assert task.status == "⬜"
    assert task.depends == ()


def test_missing_markers_are_reported() -> None:
    """Индекс без маркеров — ошибка, а не «пусто и ладно»."""
    tasks, problems = tasks_gate.parse_index("| ID | Фаза |\n")
    assert not tasks
    assert problems and "маркеров" in problems[0]


def test_wrong_header_is_reported() -> None:
    """Заголовок таблицы обязан совпадать: иначе колонки поедут."""
    text = (
        f"{tasks_gate.BEGIN_MARK}\n| ID | Задача |\n|---|---|\n| T-01 | задача |\n"
        f"{tasks_gate.END_MARK}\n"
    )
    tasks, problems = tasks_gate.parse_index(text)
    assert not tasks
    assert problems and "заголовок" in problems[0]


def test_acceptance_without_command_is_rejected() -> None:
    """«Посмотреть глазами» — не критерий приёмки."""
    tasks, _ = tasks_gate.parse_index(_index(_row(acceptance="посмотреть глазами")))
    problems = tasks_gate.check_acceptance(tasks)
    assert problems and "без команды" in problems[0]


def test_acceptance_must_be_runnable_command() -> None:
    """Критерий приёмки — команда, а не название модуля."""
    tasks, _ = tasks_gate.parse_index(_index(_row(acceptance="`tests/test_x.py`")))
    problems = tasks_gate.check_acceptance(tasks)
    assert problems and "не похож на команду" in problems[0]


def test_unknown_status_is_rejected() -> None:
    """Статус вне набора ломает выбор следующей задачи."""
    tasks, _ = tasks_gate.parse_index(_index(_row(status="готово")))
    problems = tasks_gate.check_rows(tasks)
    assert problems and "статус" in problems[0]


def test_duplicate_ids_are_rejected() -> None:
    """Два раза один ID — индекс перестаёт быть источником истины."""
    tasks, _ = tasks_gate.parse_index(_index(_row() + "\n" + _row()))
    problems = tasks_gate.check_rows(tasks)
    assert problems and "повторяется" in problems[0]


def test_missing_dependency_is_rejected() -> None:
    """Зависимость на несуществующую задачу — опечатка, а не порядок."""
    tasks, _ = tasks_gate.parse_index(_index(_row(depends="T-99")))
    problems = tasks_gate.check_dependencies(tasks)
    assert problems and "T-99" in problems[0]


def _make_root(tmp_path: Path, rows: str, cards: dict[str, str] | None = None) -> Path:
    """Собрать временный репозиторий с индексом и карточками.

    Args:
        tmp_path: Временный каталог.
        rows: Строки таблицы задач.
        cards: Карточки: имя файла → текст.

    Returns:
        Корень временного репозитория.
    """
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "TASKS.md").write_text(_index(rows), encoding="utf-8")
    cards_dir = docs / "tasks"
    cards_dir.mkdir()
    for name, text in (cards or {}).items():
        (cards_dir / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_big_task_without_card_is_rejected(tmp_path: Path) -> None:
    """Задача от 3 SP без карточки — «сделаю потом» без места, где это видно."""
    root = _make_root(tmp_path, _row(sp=3))
    problems = tasks_gate.check_cards(tasks_gate.load_tasks(root / "docs" / "TASKS.md")[0], root)
    assert problems and "без карточки" in problems[0]


def test_big_task_with_card_passes(tmp_path: Path) -> None:
    """Карточка с критерием и той же оценкой RICE — достаточна."""
    card = "**RICE:** 10.0\n\n**Критерий приёмки** — `make test`\n"
    root = _make_root(tmp_path, _row(sp=3), {"T-01.md": card})
    problems = tasks_gate.check_cards(tasks_gate.load_tasks(root / "docs" / "TASKS.md")[0], root)
    assert problems == []


def test_card_rice_must_match_index(tmp_path: Path) -> None:
    """Расхождение RICE между индексом и карточкой — выбор задачи врёт."""
    card = "**RICE:** 5.0\n\n**Критерий приёмки** — `make test`\n"
    root = _make_root(tmp_path, _row(sp=3), {"T-01.md": card})
    problems = tasks_gate.check_cards(tasks_gate.load_tasks(root / "docs" / "TASKS.md")[0], root)
    assert problems and "RICE" in problems[0]


def test_closed_task_needs_proof(tmp_path: Path) -> None:
    """Закрытая задача без доказательства — обещание, а не результат."""
    card = "**RICE:** 10.0\n\n**Критерий приёмки** — `make test`\n"
    root = _make_root(tmp_path, _row(sp=3, status="✅"), {"T-01.md": card})
    problems = tasks_gate.check_cards(tasks_gate.load_tasks(root / "docs" / "TASKS.md")[0], root)
    assert problems and "Доказательство" in problems[0]


def test_proof_path_must_exist(tmp_path: Path) -> None:
    """Доказательство ссылается на файл, которого нет."""
    card = (
        "**RICE:** 10.0\n\n**Критерий приёмки** — `make test`\n\n"
        "**Доказательство:** `src/pf/nope.py`\n"
    )
    root = _make_root(tmp_path, _row(sp=3, status="✅"), {"T-01.md": card})
    problems = tasks_gate.check_cards(tasks_gate.load_tasks(root / "docs" / "TASKS.md")[0], root)
    assert problems and "его нет" in problems[0]


def test_proof_path_present_passes(tmp_path: Path) -> None:
    """Доказательство существует — задача закрыта по-настоящему."""
    (tmp_path / "src" / "pf").mkdir(parents=True)
    (tmp_path / "src" / "pf" / "module.py").write_text("", encoding="utf-8")
    card = (
        "**RICE:** 10.0\n\n**Критерий приёмки** — `make test`\n\n"
        "**Доказательство:** `src/pf/module.py`\n"
    )
    root = _make_root(tmp_path, _row(sp=3, status="✅"), {"T-01.md": card})
    problems = tasks_gate.check_cards(tasks_gate.load_tasks(root / "docs" / "TASKS.md")[0], root)
    assert problems == []


def test_real_index_passes() -> None:
    """Индекс репозитория проходит собственный гейт."""
    problems = tasks_gate.check_all()
    assert problems == [], "; ".join(problems)


def test_next_task_prefers_higher_rice() -> None:
    """Сначала берётся задача с большим RICE."""
    tasks, _ = tasks_gate.parse_index(
        _index(_row(task_id="T-01", rice="1.0") + "\n" + _row(task_id="T-02", rice="9.0"))
    )
    picked = next_task.pick(tasks, 1)
    assert [task.task_id for task in picked] == ["T-02"]


def test_next_task_keeps_started_task_first() -> None:
    """Начатая задача идёт раньше неначатой: возвращаться дешевле."""
    tasks, _ = tasks_gate.parse_index(
        _index(
            _row(task_id="T-01", rice="1.0", status="🔄")
            + "\n"
            + _row(task_id="T-02", rice="1.5", status="⬜")
        )
    )
    picked = next_task.pick(tasks, 1)
    assert [task.task_id for task in picked] == ["T-01"]


def test_next_task_skips_blocked_task() -> None:
    """Задача с незакрытой зависимостью не предлагается."""
    tasks, _ = tasks_gate.parse_index(
        _index(
            _row(task_id="T-01", rice="1.0", status="⬜")
            + "\n"
            + _row(task_id="T-02", rice="9.0", status="⬜", depends="T-01")
        )
    )
    picked = next_task.pick(tasks, 2)
    assert [task.task_id for task in picked] == ["T-01"]
