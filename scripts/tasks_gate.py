#!/usr/bin/env python3
"""Гейт индекса задач: задачи, оценки и доказательства сходятся.

Зачем. Индекс задач — единственное место, где видно, что осталось сделать и
чем задача закрыта. Без гейта он расходится с кодом так же, как расходились
числа в описании: задача помечена закрытой, а модуля нет. Поэтому проверки
жёсткие и машинные:

- индекс разбирается: у каждой строки обязательные колонки;
- ID уникален и в допустимом виде; статус из известного набора;
- зависимости ссылаются на существующие задачи;
- критерий приёмки — команда, которая падает, если задача не сделана
  (иначе «сделано» нельзя проверить);
- задача на 3 SP и больше имеет карточку в ``docs/tasks/``;
- карточка закрытой задачи называет доказательство, и пути из него
  существуют в репозитории.

Устройство повторяет ``architecture/rules/implementation-status.yaml``:
реестр — источник истины, код только исполняет его.

Запуск:

    uv run python scripts/tasks_gate.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

#: Корень репозитория.
ROOT = Path(__file__).resolve().parent.parent

#: Индекс задач.
INDEX_PATH = ROOT / "docs" / "TASKS.md"

#: Каталог карточек задач.
CARDS_DIR = ROOT / "docs" / "tasks"

#: Маркеры машинно-читаемой части индекса.
BEGIN_MARK = "<!-- tasks:begin -->"
END_MARK = "<!-- tasks:end -->"

#: Обязательные колонки в порядке следования.
COLUMNS = ("ID", "Фаза", "Задача", "SP", "RICE", "Статус", "Зависит", "Критерий приёмки")

#: Допустимые статусы.
STATUSES = frozenset({"⬜", "🔄", "✅", "⏸"})

#: Идентификатор задачи: `T-01`, `T-12a`.
TASK_ID_RE = re.compile(r"^T-\d{2}[a-z]?$")

#: Фрагмент в обратных кавычках.
BACKTICK_RE = re.compile(r"`([^`]+)`")

#: С чего начинается исполняемая команда критерия приёмки.
COMMAND_PREFIXES = ("make ", "uv ", "pf ", "python ", "python3 ", "bash ", "PF_")

#: С чего начинается путь, который обязан существовать.
PATH_PREFIXES = ("src/", "tests/", "scripts/", "docs/", "prompts/", "methods/", "skills/")


@dataclass(frozen=True, slots=True)
class Task:
    """Строка индекса задач.

    Attributes:
        task_id: Идентификатор.
        phase: Номер фазы.
        title: Формулировка.
        sp: Оценка трудоёмкости в story points.
        rice: Оценка RICE либо ``None``, если задача мета-уровня.
        status: Статус из :data:`STATUSES`.
        depends: Идентификаторы задач-предшественников.
        acceptance: Критерий приёмки — команда в обратных кавычках.
        line: Номер строки в индексе — для сообщения об ошибке.
    """

    task_id: str
    phase: str
    title: str
    sp: int
    rice: float | None
    status: str
    depends: tuple[str, ...]
    acceptance: str
    line: int

    @property
    def is_closed(self) -> bool:
        """Задача закрыта."""
        return self.status == "✅"


def _cells(line: str) -> list[str]:
    """Разобрать строку Markdown-таблицы на ячейки.

    Args:
        line: Строка таблицы.

    Returns:
        Список ячеек без внешних разделителей.
    """
    stripped = line.strip()
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _is_separator(line: str) -> bool:
    """Строка — разделитель заголовка таблицы.

    Args:
        line: Строка таблицы.

    Returns:
        ``True``, если все ячейки состоят из дефисов и двоеточий.
    """
    cells = _cells(line)
    return bool(cells) and all(cell and set(cell) <= set("-: ") for cell in cells)


def _parse_rice(raw: str) -> float | None:
    """Разобрать оценку RICE.

    Args:
        raw: Ячейка таблицы.

    Returns:
        Значение либо ``None`` для прочерка.
    """
    cleaned = raw.strip().replace(",", ".")
    if not cleaned or cleaned in {"—", "-", "–"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_index(text: str) -> tuple[list[Task], list[str]]:
    """Разобрать машинно-читаемую часть индекса.

    Args:
        text: Содержимое ``docs/TASKS.md``.

    Returns:
        Пары «задачи» и «ошибки разбора».
    """
    problems: list[str] = []
    if BEGIN_MARK not in text or END_MARK not in text:
        return [], [f"в индексе нет маркеров {BEGIN_MARK} … {END_MARK}"]

    block = text.split(BEGIN_MARK, 1)[1].split(END_MARK, 1)[0]
    lines = [line for line in block.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return [], ["таблица задач пуста: нужен заголовок, разделитель и строки"]

    header = tuple(_cells(lines[0]))
    if header != COLUMNS:
        problems.append(
            "заголовок таблицы не совпадает с ожидаемым: "
            + " | ".join(COLUMNS)
            + " (получено: "
            + " | ".join(header)
            + ")"
        )
        return [], problems

    tasks: list[Task] = []
    for offset, line in enumerate(lines[1:], start=1):
        if _is_separator(line):
            continue
        cells = _cells(line)
        if len(cells) != len(COLUMNS):
            problems.append(f"строка {offset + 1}: ячеек {len(cells)}, ожидалось {len(COLUMNS)}")
            continue
        raw_id, phase, title, raw_sp, raw_rice, status, depends, acceptance = cells
        try:
            sp = int(raw_sp)
        except ValueError:
            problems.append(f"{raw_id or '?'}: SP «{raw_sp}» — не число")
            continue
        deps = tuple(part.strip() for part in depends.split(",") if part.strip() not in {"", "—"})
        tasks.append(
            Task(
                task_id=raw_id,
                phase=phase,
                title=title,
                sp=sp,
                rice=_parse_rice(raw_rice),
                status=status,
                depends=deps,
                acceptance=acceptance,
                line=offset + 1,
            )
        )
    return tasks, problems


def _card_path(task_id: str, root: Path | None = None) -> Path:
    """Путь к карточке задачи.

    Args:
        task_id: Идентификатор задачи.
        root: Корень репозитория.

    Returns:
        Путь к файлу карточки.
    """
    base = root or ROOT
    return base / "docs" / "tasks" / f"{task_id}.md"


def check_rows(tasks: list[Task]) -> list[str]:
    """Строки индекса сами по себе корректны.

    Args:
        tasks: Разобранные задачи.

    Returns:
        Список нарушений.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        if not TASK_ID_RE.match(task.task_id):
            problems.append(f"строка {task.line}: идентификатор «{task.task_id}» не в виде T-01")
        if task.task_id in seen:
            problems.append(f"строка {task.line}: идентификатор «{task.task_id}» повторяется")
        seen.add(task.task_id)
        if task.status not in STATUSES:
            problems.append(
                f"{task.task_id}: статус «{task.status}» неизвестен "
                f"(допустимо: {' '.join(sorted(STATUSES))})"
            )
        if not 1 <= task.sp <= 13:
            problems.append(f"{task.task_id}: SP {task.sp} вне шкалы 1..13")
    return problems


def check_acceptance(tasks: list[Task]) -> list[str]:
    """У каждой задачи есть критерий приёмки — исполняемая команда.

    Критерий вида «посмотреть глазами» не принимается: его нельзя
    проверить повторно, а значит нельзя и закрыть задачу доказуемо.

    Args:
        tasks: Разобранные задачи.

    Returns:
        Список нарушений.
    """
    problems: list[str] = []
    for task in tasks:
        commands = BACKTICK_RE.findall(task.acceptance)
        if not commands:
            problems.append(f"{task.task_id}: критерий приёмки без команды в обратных кавычках")
            continue
        if not any(command.strip().startswith(COMMAND_PREFIXES) for command in commands):
            problems.append(
                f"{task.task_id}: критерий приёмки «{commands[0]}» не похож на команду "
                f"(ожидается начало из {', '.join(prefix.strip() for prefix in COMMAND_PREFIXES)})"
            )
    return problems


def check_dependencies(tasks: list[Task]) -> list[str]:
    """Зависимости ссылаются на существующие задачи.

    Args:
        tasks: Разобранные задачи.

    Returns:
        Список нарушений.
    """
    known = {task.task_id for task in tasks}
    problems: list[str] = []
    for task in tasks:
        for dep in task.depends:
            if dep not in known:
                problems.append(f"{task.task_id}: зависимость «{dep}» не найдена в индексе")
    return problems


def check_cards(tasks: list[Task], root: Path | None = None) -> list[str]:
    """У крупных задач есть карточка, у закрытых — доказательство.

    Args:
        tasks: Разобранные задачи.
        root: Корень репозитория.

    Returns:
        Список нарушений.
    """
    problems: list[str] = []
    for task in tasks:
        if task.sp < 3:
            continue
        card = _card_path(task.task_id, root)
        if not card.is_file():
            problems.append(f"{task.task_id}: задача на {task.sp} SP без карточки docs/tasks/")
            continue
        text = card.read_text(encoding="utf-8")
        if "**Критерий приёмки**" not in text:
            problems.append(f"{task.task_id}: в карточке нет раздела «Критерий приёмки»")
        if task.rice is not None:
            match = re.search(r"\*\*RICE:\*\*\s*([0-9]+(?:[.,][0-9]+)?)", text)
            if not match:
                problems.append(f"{task.task_id}: в карточке нет оценки RICE")
            elif abs(float(match.group(1).replace(",", ".")) - task.rice) > 1e-9:
                problems.append(
                    f"{task.task_id}: RICE в карточке {match.group(1)}, в индексе {task.rice}"
                )
        if task.is_closed:
            problems.extend(_check_proof(task, text, root))
    return problems


def _check_proof(task: Task, card_text: str, root: Path | None = None) -> list[str]:
    """Доказательство закрытой задачи существует в репозитории.

    Args:
        task: Задача.
        card_text: Текст карточки.
        root: Корень репозитория.

    Returns:
        Список нарушений.
    """
    base = root or ROOT
    marker = "**Доказательство:**"
    if marker not in card_text:
        return [f"{task.task_id}: закрыта, но в карточке нет раздела «Доказательство»"]
    line = card_text.split(marker, 1)[1].splitlines()[0]
    problems: list[str] = []
    paths = [item for item in BACKTICK_RE.findall(line) if item.startswith(PATH_PREFIXES)]
    if not paths:
        problems.append(f"{task.task_id}: доказательство без пути в обратных кавычках")
    for path in paths:
        if not (base / path).exists():
            problems.append(f"{task.task_id}: доказательство ссылается на `{path}` — его нет")
    return problems


def load_tasks(path: Path | None = None) -> tuple[list[Task], list[str]]:
    """Прочитать индекс задач с диска.

    Args:
        path: Путь к индексу. По умолчанию — ``docs/TASKS.md``.

    Returns:
        Пары «задачи» и «ошибки разбора».
    """
    target = path or INDEX_PATH
    if not target.is_file():
        return [], [f"индекс задач не найден: {target}"]
    return parse_index(target.read_text(encoding="utf-8"))


def check_all(path: Path | None = None, root: Path | None = None) -> list[str]:
    """Выполнить все проверки индекса.

    Args:
        path: Путь к индексу.
        root: Корень репозитория.

    Returns:
        Список нарушений.
    """
    tasks, problems = load_tasks(path)
    if not tasks:
        return problems or ["в индексе нет ни одной задачи"]
    problems.extend(check_rows(tasks))
    problems.extend(check_acceptance(tasks))
    problems.extend(check_dependencies(tasks))
    problems.extend(check_cards(tasks, root))
    return problems


def main(argv: list[str] | None = None) -> int:
    """Точка входа гейта индекса задач.

    Args:
        argv: Аргументы командной строки (не используются).

    Returns:
        0 — нарушений нет; 1 — найдены.
    """
    _ = argv
    tasks, _ = load_tasks()
    problems = check_all()

    if not problems:
        opened = sum(1 for task in tasks if task.status != "✅")
        print(f"[OK] tasks-check: задач {len(tasks)}, открыто {opened}, нарушений нет")
        return 0

    print(f"[FAIL] tasks-check: нарушений — {len(problems)}")
    for problem in problems:
        print(f"  - {problem}")
    print()
    print("  Как чинить: индекс — docs/TASKS.md, карточки — docs/tasks/.")
    print("  Критерий приёмки — команда, которая падает, пока задача не сделана.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
