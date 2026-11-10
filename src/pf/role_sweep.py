"""Свип модели, окна и бюджета контекста: решение по числу, а не на глаз.

Зачем отдельный модуль. Выбор «какая модель на ярус» уже делался однажды —
по одному прогону (`D-57`), и оказался обещанием: 12b держит схему, но
уходит на CPU и оставляет пустые ``evidence``; 8b быстрее в 2,5×. Один
прогон локальной модели ничего не доказывает: разброс велик. Поэтому выбор
фиксируется замером — матрицей «модель × окно × бюджет контекста», по
нескольку повторов на клетку.

Здесь живёт **правило решения** (чистые функции), а не сам замер:
``scripts/grid_sweep.py`` прогоняет матрицу и пишет снимок, а модуль решает
по нему. Тесты ``tests/test_role_sweep.py`` (критерий — ``-k sweep_decision``)
проверяют правило без живой модели: замер идёт часами и требует GPU, а
решение должно быть воспроизводимым за миллисекунды.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Ярусы ролей — те же имена, что в ``pf.roles.ROLE_TIERS``.
TIERS: tuple[str, ...] = ("extraction", "planning", "synthesis")

#: Минимальная доля повторов, при которой проход схемы считается состоявшимся.
#: Один удачный прогон из трёх — ещё не «держит схему»: у локальной модели
#: разброс велик, и выбор по одному попаданию был бы выбором по шуму.
SCHEMA_PASS_RATE = 2 / 3

#: Текущее умолчание окна (``Settings.ollama_num_ctx``). Замер **подтверждает**
#: его, а не меняет без нужды: если умолчание чистое по обрезке, оно и
#: остаётся. Меньшее окно берётся, только когда умолчание обрезает промпт.
#: Согласованность с настройками держит ``tests/test_role_sweep.py``.
DEFAULT_WINDOW = 12288

#: Признак того, что часть модели ушла на CPU: ``ollama ps`` пишет долю
#: видеопамяти. Строка, а не число процентов: замер записывает то, что
#: сказал Ollama, и не выдаёт оценку за факт.
CPU_MARK = "CPU"


@dataclass(frozen=True, slots=True)
class TierStats:
    """Итог яруса в одном прогоне.

    Attributes:
        roles: Сколько ролей яруса дошло до журнала.
        schema_ok: Все роли яруса прошли схему (с ретраями `T-74`).
        truncation_steps: Сколько шагов яруса упёрлось в окно контекста.
        duration_sec: Суммарное время ролей яруса.
    """

    roles: int = 0
    schema_ok: bool = True
    truncation_steps: int = 0
    duration_sec: float = 0.0


@dataclass(frozen=True, slots=True)
class SweepCell:
    """Одна клетка матрицы: модель × окно × бюджет контекста × повтор.

    Attributes:
        model: Имя модели.
        num_ctx: Окно контекста (``PF_OLLAMA_NUM_CTX``).
        context_passages: Бюджет абзацев на роль (``PF_CONTEXT_PASSAGES``).
        repeat: Номер повтора.
        status: ``ok``, ``failed`` или ``interrupted``.
        grounding_rate: Доля подтверждённых записей — свойство модели.
        phantom: Сколько отказов пришлось на выдуманные источники и
            ненайденные цитаты (виды ``source_not_named`` / ``source_unknown``
            / ``fabrication``).
        duration_sec: Время прогона целиком.
        processor: Как модель лежала в памяти (``100% GPU`` / ``…% CPU``).
        conclusion_status: Статус оснований вывода.
        tiers: Итоги по ярусам: имя яруса → статистика.
    """

    model: str
    num_ctx: int
    context_passages: int
    repeat: int
    status: str = "ok"
    grounding_rate: float = 0.0
    phantom: int = 0
    duration_sec: float = 0.0
    processor: str = ""
    conclusion_status: str = ""
    tiers: dict[str, TierStats] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SweepDecision:
    """Замороженное решение свипа: что берём в настройки.

    Attributes:
        window: Окно контекста (``PF_OLLAMA_NUM_CTX``).
        context_passages: Бюджет абзацев на роль (``PF_CONTEXT_PASSAGES``).
        tiers: Модель на каждый ярус.
    """

    window: int
    context_passages: int
    tiers: dict[str, str]


def cell_key(model: str, num_ctx: int, context_passages: int, repeat: int) -> str:
    """Детерминированный ключ клетки — по нему продолжается замер.

    Args:
        model: Имя модели.
        num_ctx: Окно контекста.
        context_passages: Бюджет абзацев на роль.
        repeat: Номер повтора.

    Returns:
        Ключ, одинаковый в записи и при продолжении.
    """
    safe = model.replace(":", "-").replace(".", "-").replace("/", "-")
    return f"{safe}|ctx={num_ctx}|passages={context_passages}|rep={repeat}"


def _ok(cells: list[SweepCell]) -> list[SweepCell]:
    """Только завершённые клетки.

    Args:
        cells: Все клетки.

    Returns:
        Клетки со статусом ``ok``.
    """
    return [cell for cell in cells if cell.status == "ok"]


def _tier_summary(cells: list[SweepCell], tier: str) -> tuple[float, int, float, int]:
    """Сводка по ярусу для одной модели.

    Args:
        cells: Клетки одной модели в одной конфигурации.
        tier: Имя яруса.

    Returns:
        Доля прохода схемы, шагов с обрезкой, медиана времени, число
        прогонов с частью модели на CPU.
    """
    stats = [cell.tiers[tier] for cell in cells if tier in cell.tiers]
    if not stats:
        return 0.0, 0, 0.0, 0
    schema = sum(1 for stat in stats if stat.schema_ok) / len(stats)
    truncation = sum(stat.truncation_steps for stat in stats)
    duration = float(statistics.median(stat.duration_sec for stat in stats))
    cpu = sum(1 for cell in cells if CPU_MARK in cell.processor)
    return schema, truncation, duration, cpu


def decide_config(cells: list[SweepCell]) -> tuple[int, int]:
    """Выбрать окно и бюджет контекста.

    Правило. Сначала конфигурации без обрезки промпта. Среди чистых —
    **текущее умолчание окна** (``DEFAULT_WINDOW``): замер перепроверяет его,
    а не меняет без нужды. Если умолчание обрезает промпт, берётся чистое
    окно с наименьшим бюджетом абзацев (меньше материала — дешевле и
    безопаснее), затем меньшее окно. Если чистых нет — минимальная обрезка:
    поднимать окно «на всякий случай» без числа нельзя (``docs/decisions.md``,
    D-20: второй рычаг — сокращать контекст, а не окно).

    Args:
        cells: Клетки свипа.

    Returns:
        Пара ``(окно, бюджет абзацев)``.

    Raises:
        ValueError: Нет ни одной завершённой клетки.
    """
    ok = _ok(cells)
    if not ok:
        raise ValueError("нет завершённых клеток: свип нечего решать")
    groups: dict[tuple[int, int], list[SweepCell]] = {}
    for cell in ok:
        groups.setdefault((cell.num_ctx, cell.context_passages), []).append(cell)

    def sort_key(item: tuple[tuple[int, int], list[SweepCell]]) -> tuple[float, ...]:
        """Ключ предпочтения конфигурации.

        Args:
            item: Пара «(окно, бюджет) → клетки».

        Returns:
            Кортеж: нет обрезки, чистое умолчание, обрезка, бюджет, окно.
        """
        (window, passages), group = item
        truncation = sum(stat.truncation_steps for cell in group for stat in cell.tiers.values())
        schema_fails = sum(
            1 for cell in group for stat in cell.tiers.values() if not stat.schema_ok
        )
        clean = truncation == 0
        return (
            0.0 if clean else 1.0,
            0.0 if clean and window == DEFAULT_WINDOW else 1.0,
            float(truncation),
            float(schema_fails),
            float(passages),
            float(window),
        )

    (window, passages), _group = min(groups.items(), key=sort_key)
    return window, passages


def decide_tiers(cells: list[SweepCell], window: int, context_passages: int) -> dict[str, str]:
    """Выбрать модель на каждый ярус.

    Клетки фильтруются по выбранной конфигурации. Модель допускается к
    ярусу, если держит схему в ``SCHEMA_PASS_RATE`` повторов. Среди
    допущенных: меньше обрезки, меньше времени, меньше прогонов на CPU,
    меньше фантомов; имя — только для повторяемости при полном равенстве.

    Args:
        cells: Клетки свипа.
        window: Выбранное окно.
        context_passages: Выбранный бюджет абзацев.

    Returns:
        Модель на каждый ярус. Пусто, если завершённых клеток нет.
    """
    selected = [
        cell
        for cell in _ok(cells)
        if cell.num_ctx == window and cell.context_passages == context_passages
    ]
    if not selected:
        # Фолбэк: конфигурация без повторов не должна ронять решение.
        selected = _ok(cells)
    models = sorted({cell.model for cell in selected})
    if not models:
        return {}
    by_model: dict[str, list[SweepCell]] = {
        model: [cell for cell in selected if cell.model == model] for model in models
    }
    decision: dict[str, str] = {}
    for tier in TIERS:
        scored: list[dict[str, float | int | str]] = []
        for model in models:
            schema, truncation, duration, cpu = _tier_summary(by_model[model], tier)
            phantom = sum(cell.phantom for cell in by_model[model])
            scored.append(
                {
                    "model": model,
                    "schema": schema,
                    "eligible": int(schema >= SCHEMA_PASS_RATE),
                    "truncation": truncation,
                    "duration": duration,
                    "cpu": cpu,
                    "phantom": phantom,
                }
            )
        eligible = [row for row in scored if int(row["eligible"]) == 1]
        pool = eligible or scored

        def rank(row: dict[str, float | int | str]) -> tuple[float, int, float, int, int, str]:
            """Ключ выбора модели на ярус.

            Args:
                row: Оценка модели.

            Returns:
                Кортеж предпочтения (меньше — лучше).
            """
            return (
                -float(row["schema"]),
                int(row["truncation"]),
                float(row["duration"]),
                int(row["cpu"]),
                int(row["phantom"]),
                str(row["model"]),
            )

        best = min(pool, key=rank)
        decision[tier] = str(best["model"])
    return decision


def decide_sweep(cells: list[SweepCell]) -> SweepDecision:
    """Полное решение: окно, бюджет контекста и модель на каждый ярус.

    Args:
        cells: Клетки свипа.

    Returns:
        Замороженное решение.

    Raises:
        ValueError: Нет ни одной завершённой клетки.
    """
    window, passages = decide_config(cells)
    return SweepDecision(
        window=window,
        context_passages=passages,
        tiers=decide_tiers(cells, window, passages),
    )


def cell_as_dict(cell: SweepCell) -> dict[str, Any]:
    """Представить клетку для снимка.

    Args:
        cell: Клетка свипа.

    Returns:
        Словарь, который читает :func:`cell_from_dict`.
    """
    return {
        "model": cell.model,
        "num_ctx": cell.num_ctx,
        "context_passages": cell.context_passages,
        "repeat": cell.repeat,
        "status": cell.status,
        "grounding_rate": round(cell.grounding_rate, 6),
        "phantom": cell.phantom,
        "duration_sec": round(cell.duration_sec, 3),
        "processor": cell.processor,
        "conclusion_status": cell.conclusion_status,
        "tiers": {
            name: {
                "roles": stat.roles,
                "schema_ok": stat.schema_ok,
                "truncation_steps": stat.truncation_steps,
                "duration_sec": round(stat.duration_sec, 3),
            }
            for name, stat in cell.tiers.items()
        },
    }


def cell_from_dict(item: dict[str, Any]) -> SweepCell:
    """Восстановить клетку из снимка.

    Args:
        item: Словарь из :func:`cell_as_dict`.

    Returns:
        Клетка свипа.
    """
    tiers = {
        str(name): TierStats(
            roles=int(stats.get("roles", 0)),
            schema_ok=bool(stats.get("schema_ok", True)),
            truncation_steps=int(stats.get("truncation_steps", 0)),
            duration_sec=float(stats.get("duration_sec", 0.0)),
        )
        for name, stats in (item.get("tiers") or {}).items()
    }
    return SweepCell(
        model=str(item["model"]),
        num_ctx=int(item["num_ctx"]),
        context_passages=int(item["context_passages"]),
        repeat=int(item["repeat"]),
        status=str(item.get("status", "ok")),
        grounding_rate=float(item.get("grounding_rate", 0.0)),
        phantom=int(item.get("phantom", 0)),
        duration_sec=float(item.get("duration_sec", 0.0)),
        processor=str(item.get("processor", "")),
        conclusion_status=str(item.get("conclusion_status", "")),
        tiers=tiers,
    )


def load_snapshot(path: str | Path) -> list[SweepCell]:
    """Прочитать клетки снимка свипа.

    Args:
        path: Файл снимка.

    Returns:
        Клетки в порядке записи.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("cells") if isinstance(payload, dict) else payload
    return [cell_from_dict(item) for item in raw or []]


def load_frozen(path: str | Path) -> SweepDecision:
    """Прочитать замороженное решение из снимка.

    Args:
        path: Файл снимка.

    Returns:
        Замороженное решение.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frozen = payload["frozen"] if isinstance(payload, dict) else {}
    return SweepDecision(
        window=int(frozen["window"]),
        context_passages=int(frozen["context_passages"]),
        tiers={str(name): str(model) for name, model in (frozen.get("tiers") or {}).items()},
    )


def snapshot_payload(
    cells: list[SweepCell],
    decision: SweepDecision,
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Собрать снимок: клетки и замороженное решение.

    Решение пишется тем же правилом, которым проверяется тестом: снимок не
    правится руками, иначе правило и число разойдутся.

    Args:
        cells: Клетки свипа.
        decision: Решение, посчитанное :func:`decide_sweep`.
        meta: Метаданные замера (дата, модели, окна, бюджет, повторы).

    Returns:
        Словарь для записи в JSON.
    """
    return {
        "meta": meta or {},
        "cells": [cell_as_dict(cell) for cell in cells],
        "frozen": {
            "window": decision.window,
            "context_passages": decision.context_passages,
            "tiers": decision.tiers,
        },
    }
