#!/usr/bin/env python3
"""Выбор 1–2 задач на сессию по оценке RICE.

Зачем. Индекс задач отвечает «что осталось», но не отвечает «что делать
сейчас». Порядок работы проекта: за сессию берётся 1–2 задачи, чтобы
контекст не расползался, и каждая следующая сессия начинается с выбора, а не
с чтения всего индекса.

Правило выбора:

1. незакрытая задача с наибольшим RICE;
2. начатая задача (🔄) идёт раньше неначатой с тем же RICE — возвращаться
   дешевле, чем начинать заново;
3. при равном RICE берётся меньшая по SP — она быстрее закроется;
4. задача, у которой не закрыты зависимости, не выбирается.

Запуск:

    uv run python scripts/next_task.py            # 1–2 задачи
    uv run python scripts/next_task.py --limit 1
    uv run python scripts/next_task.py --all      # все готовые к работе
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasks_gate import Task, load_tasks  # noqa: E402  (путь добавляется выше)

#: Сколько задач по умолчанию предлагается на сессию.
DEFAULT_LIMIT = 2


def ready_tasks(tasks: list[Task]) -> list[Task]:
    """Задачи, которые можно брать в работу прямо сейчас.

    Args:
        tasks: Разобранный индекс.

    Returns:
        Отсортированный список: начатые раньше, затем по RICE и по SP.
    """
    closed = {task.task_id for task in tasks if task.is_closed}
    ready = [
        task
        for task in tasks
        if task.status in {"⬜", "🔄"} and all(dep in closed for dep in task.depends)
    ]
    return sorted(
        ready,
        key=lambda task: (
            task.status != "🔄",
            -(task.rice if task.rice is not None else 0.0),
            task.sp,
        ),
    )


def pick(tasks: list[Task], limit: int = DEFAULT_LIMIT) -> list[Task]:
    """Выбрать задачи на сессию.

    Args:
        tasks: Разобранный индекс.
        limit: Сколько задач взять.

    Returns:
        Выбранные задачи.
    """
    return ready_tasks(tasks)[: max(limit, 1)]


def main(argv: list[str] | None = None) -> int:
    """Точка входа выбора задачи.

    Args:
        argv: Аргументы командной строки.

    Returns:
        0 — задачи найдены; 1 — брать нечего.
    """
    parser = argparse.ArgumentParser(description="Выбрать задачи на сессию")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="сколько задач взять")
    parser.add_argument("--all", action="store_true", help="показать все готовые к работе")
    args = parser.parse_args(argv)

    tasks, problems = load_tasks()
    if not tasks:
        for problem in problems:
            print(f"[FAIL] next-task: {problem}")
        return 1

    ready = ready_tasks(tasks)
    if not ready:
        print("[OK] next-task: готовых к работе задач нет")
        return 1

    chosen = ready if args.all else pick(tasks, args.limit)
    total_sp = sum(task.sp for task in chosen)
    print(f"Сессия: {len(chosen)} задач, {total_sp} SP")
    print()
    for task in chosen:
        rice = f"{task.rice:.1f}" if task.rice is not None else "—"
        print(f"  {task.status} {task.task_id}  {task.title}")
        print(f"     RICE {rice} · SP {task.sp} · фаза {task.phase}")
        print(f"     приёмка: {task.acceptance}")
        if task.sp >= 3:
            print(f"     карточка: docs/tasks/{task.task_id}.md")
        print()

    blocked = [task for task in tasks if task.status in {"⬜", "🔄"} and task not in ready]
    if blocked:
        print(f"Ждут зависимостей: {', '.join(task.task_id for task in blocked)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
