"""Правило свипа: модель на ярус, окно и бюджет контекста выбираются по числу.

Зачем тест без живой модели. Замер (``scripts/grid_sweep.py``) идёт часами и
требует GPU; тест проверяет **правило решения** на подготовленных клетках,
поэтому гоняется в общем наборе за миллисекунды. Живой замер нужен, чтобы
получить числа; правило проверяется здесь.

Критерий приёмки T-75 — ``-k sweep_decision``: все проверки этого файла
названы так, чтобы индекс задач выбирал их целиком, а не одну подстрокой.
"""

from __future__ import annotations

from pathlib import Path

from pf.config import Settings
from pf.role_sweep import (
    DEFAULT_WINDOW,
    SweepCell,
    TierStats,
    decide_sweep,
    load_frozen,
    load_snapshot,
)

#: Снимок замера. Пишет ``scripts/grid_sweep.py``; руками не правится.
SNAPSHOT = Path(__file__).resolve().parents[1] / "reports" / "role_sweep" / "decision.json"

TIERS = ("extraction", "planning", "synthesis")


def _tiers(
    *,
    schema_ok: bool = True,
    truncation: int = 0,
    duration: float = 10.0,
) -> dict[str, TierStats]:
    """Итоги ярусов для клетки.

    Args:
        schema_ok: Все роли яруса прошли схему.
        truncation: Шагов с обрезкой в каждом ярусе.
        duration: Время каждого яруса.

    Returns:
        Статистика по всем трём ярусам.
    """
    return {
        name: TierStats(
            roles=2,
            schema_ok=schema_ok,
            truncation_steps=truncation,
            duration_sec=duration,
        )
        for name in TIERS
    }


def _cell(
    model: str,
    *,
    num_ctx: int = 12288,
    passages: int = 5,
    repeat: int = 0,
    tiers: dict[str, TierStats] | None = None,
    processor: str = "100% GPU",
    phantom: int = 0,
) -> SweepCell:
    """Собрать клетку свипа.

    Args:
        model: Имя модели.
        num_ctx: Окно контекста.
        passages: Бюджет абзацев.
        repeat: Номер повтора.
        tiers: Итоги ярусов; по умолчанию чистые.
        processor: Как модель лежала в памяти.
        phantom: Отклонённых «фантомов».

    Returns:
        Клетка свипа.
    """
    return SweepCell(
        model=model,
        num_ctx=num_ctx,
        context_passages=passages,
        repeat=repeat,
        grounding_rate=0.5,
        phantom=phantom,
        duration_sec=60.0,
        processor=processor,
        conclusion_status="подкреплено частично",
        tiers=tiers if tiers is not None else _tiers(),
    )


def test_sweep_decision_extraction_takes_cheapest_clean_model() -> None:
    """На извлечении берём самую быструю модель, если обе чисты по схеме."""
    fast = [_cell("qwen3:8b", repeat=i, tiers=_tiers(duration=8.0)) for i in range(3)]
    slow = [_cell("mistral-nemo:12b", repeat=i, tiers=_tiers(duration=40.0)) for i in range(3)]

    decision = decide_sweep(fast + slow)

    assert decision.tiers["extraction"] == "qwen3:8b"
    assert decision.tiers["synthesis"] == "qwen3:8b"


def test_sweep_decision_synthesis_rejects_model_that_fails_schema() -> None:
    """Синтез не отдаём модели, которая не держит схему, даже если она сильнее."""
    good = [_cell("qwen3:8b", repeat=i) for i in range(3)]
    broken = {
        **_tiers(),
        "synthesis": TierStats(roles=2, schema_ok=False, duration_sec=10.0),
    }
    bad = [_cell("mistral-nemo:12b", repeat=i, tiers=broken) for i in range(3)]

    decision = decide_sweep(good + bad)

    assert decision.tiers["synthesis"] == "qwen3:8b"


def test_sweep_decision_window_prefers_clean_default() -> None:
    """Замер подтверждает умолчание: при обрезке 8192 остаётся 12288."""
    cells: list[SweepCell] = []
    for window in (8192, 12288, 16384):
        truncation = 4 if window == 8192 else 0
        for model in ("qwen3:8b", "mistral-nemo:12b"):
            for repeat in range(3):
                cells.append(
                    _cell(model, num_ctx=window, repeat=repeat, tiers=_tiers(truncation=truncation))
                )

    decision = decide_sweep(cells)

    assert decision.window == 12288


def test_sweep_decision_keeps_default_window_when_clean() -> None:
    """Если все окна чистые, умолчание не меняется без нужды."""
    cells: list[SweepCell] = []
    for window in (8192, 12288, 16384):
        for repeat in range(3):
            cells.append(_cell("qwen3:8b", num_ctx=window, repeat=repeat))

    assert decide_sweep(cells).window == DEFAULT_WINDOW


def test_sweep_decision_widens_when_default_truncates() -> None:
    """Если умолчание обрезает промпт, берётся чистое окно с запасом — 16384."""
    cells: list[SweepCell] = []
    for window, truncation in {12288: 2, 16384: 0}.items():
        for repeat in range(3):
            cells.append(
                _cell(
                    "qwen3:8b", num_ctx=window, repeat=repeat, tiers=_tiers(truncation=truncation)
                )
            )

    assert decide_sweep(cells).window == 16384


def test_sweep_decision_default_window_matches_config() -> None:
    """Константа умолчания совпадает с настройками — иначе правило судит не то."""
    assert Settings.model_fields["ollama_num_ctx"].default == DEFAULT_WINDOW


def test_sweep_decision_window_falls_back_to_least_truncation() -> None:
    """Если обрезают все окна — берём с наименьшей обрезкой, а не большее «на всякий»."""
    cells: list[SweepCell] = []
    for window, truncation in {8192: 5, 12288: 1, 16384: 1}.items():
        for repeat in range(3):
            cells.append(
                _cell(
                    "qwen3:8b", num_ctx=window, repeat=repeat, tiers=_tiers(truncation=truncation)
                )
            )

    decision = decide_sweep(cells)

    assert decision.window == 12288


def test_sweep_decision_prefers_clean_context_budget() -> None:
    """Бюджет контекста — наименьший чистый: 5 абзацев, а не 8."""
    cells: list[SweepCell] = []
    for passages, truncation in {3: 3, 5: 0, 8: 0}.items():
        for repeat in range(3):
            cells.append(
                _cell(
                    "qwen3:8b",
                    passages=passages,
                    repeat=repeat,
                    tiers=_tiers(truncation=truncation),
                )
            )

    decision = decide_sweep(cells)

    assert decision.context_passages == 5


def test_sweep_decision_prefers_gpu_when_otherwise_equal() -> None:
    """При равенстве схемы и времени берём модель, которая целиком на GPU."""
    gpu = [_cell("qwen3:8b", repeat=i, processor="100% GPU") for i in range(3)]
    cpu = [_cell("mistral-nemo:12b", repeat=i, processor="20% CPU / 80% GPU") for i in range(3)]

    decision = decide_sweep(gpu + cpu)

    assert decision.tiers["synthesis"] == "qwen3:8b"


def test_sweep_decision_matches_frozen_snapshot() -> None:
    """Замороженное решение в снимке совпадает с правилом и чисто по окну.

    Снимок пишет сам замер. Если его правят руками, это видно: правило,
    пересчитанное по клеткам, обязано дать тот же выбор.
    """
    assert SNAPSHOT.is_file(), f"нет снимка свипа: {SNAPSHOT}"
    cells = load_snapshot(SNAPSHOT)
    frozen = load_frozen(SNAPSHOT)

    assert decide_sweep(cells) == frozen
    clean = {
        cell.num_ctx
        for cell in cells
        if cell.status == "ok" and all(stat.truncation_steps == 0 for stat in cell.tiers.values())
    }
    if clean:
        assert frozen.window in clean
