#!/usr/bin/env python3
"""Замеры: агент против обычного чата, несколько моделей.

Скрипт прогоняет одну и ту же задачу в двух режимах:

- **agent** — полный рабочий процесс с проверкой оснований;
- **baseline** — тот же вопрос тому же модели одним промптом.

и повторяет прогон несколько раз: разброс между прогонами у локальной
модели велик, а один замер ничего не доказывает.

Запуск:

    python scripts/bench.py --repeats 3
    python scripts/bench.py --models qwen2.5:7b qwen3:8b --repeats 2
    python scripts/bench.py --baseline-only

Результаты пишутся в ``reports/bench/`` и в MLflow (если он установлен).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pf.bench import (  # noqa: E402
    DEFAULT_TASK,
    baseline_metrics,
    run_agent,
    run_baseline,
    settings_for_model,
)
from pf.tracking import Accumulator, RunMetrics  # noqa: E402


def _session_id(prefix: str, model: str, index: int) -> str:
    """Идентификатор сессии для прогона.

    Args:
        prefix: Префикс.
        model: Имя модели.
        index: Номер прогона.

    Returns:
        Идентификатор сессии.
    """
    stamp = time.strftime("%H%M%S")
    safe_model = model.replace(":", "-").replace(".", "-")
    return f"bench-{prefix}-{safe_model}-{stamp}-{index}"


def _averages(rows: list[RunMetrics], mode: str) -> dict[str, float]:
    """Средние по прогонам одного режима.

    Args:
        rows: Все измерения.
        mode: Режим прогона.

    Returns:
        Средние значения метрик.
    """
    selected = [row for row in rows if row.mode == mode]
    if not selected:
        return {}
    keys = ("citations", "total_tokens", "duration_sec", "cost_units", "facts", "variants")
    result: dict[str, float] = {"runs": len(selected)}
    for key in keys:
        values = [float(row.as_metrics().get(key, 0.0)) for row in selected]
        result[key] = round(statistics.fmean(values), 4)
    return result


def main() -> int:
    """Выполнить замеры и сохранить отчёт.

    Returns:
        Код возврата.
    """
    parser = argparse.ArgumentParser(description="Замеры агента против обычного чата")
    parser.add_argument("--task", default=DEFAULT_TASK, help="вопрос руководителя")
    parser.add_argument("--repeats", type=int, default=3, help="сколько раз повторить")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="модели через пробел (по умолчанию — текущая)",
    )
    parser.add_argument("--baseline-only", action="store_true", help="только обычный чат")
    parser.add_argument("--agent-only", action="store_true", help="только агент")
    parser.add_argument(
        "--out",
        default=str(REPO / "reports" / "bench"),
        help="каталог для отчётов",
    )
    args = parser.parse_args()

    from pf.config import settings

    models = [m.strip() for m in args.models if m.strip()]
    if not models:
        models = [settings.effective_model]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    accumulator = Accumulator(path=str(out_dir / "runs.jsonl"))
    all_rows: list[RunMetrics] = []

    for model in models:
        config = settings_for_model(model)
        print(f"=== модель: {model} ===", flush=True)
        for index in range(1, args.repeats + 1):
            if not args.baseline_only:
                session = _session_id("agent", model, index)
                agent_row = run_agent(args.task, session, config=config)
                accumulator.add(agent_row)
                all_rows.append(agent_row)
                print(
                    f"  agent    #{index}: оснований={agent_row.citations} "
                    f"токенов={agent_row.total_tokens} "
                    f"время={agent_row.duration_sec:.1f}с "
                    f"фактов={agent_row.facts} вариантов={agent_row.variants}"
                    + (f" ОШИБКА: {agent_row.errors}" if agent_row.errors else ""),
                    flush=True,
                )
            if not args.agent_only:
                from pf.llm import build_client

                base_result = run_baseline(args.task, client=build_client(config))
                base_row = baseline_metrics(args.task, base_result, config=config)
                accumulator.add(base_row)
                all_rows.append(base_row)
                print(
                    f"  baseline #{index}: токенов={base_row.total_tokens} "
                    f"время={base_row.duration_sec:.1f}с",
                    flush=True,
                )

    agent_avg = _averages(all_rows, "agent")
    base_avg = _averages(all_rows, "baseline")
    report: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "task": args.task,
        "models": models,
        "repeats": args.repeats,
        "agent_avg": agent_avg,
        "baseline_avg": base_avg,
    }
    if agent_avg and base_avg:
        report["comparison"] = {
            "token_ratio": round(
                agent_avg.get("total_tokens", 0) / max(base_avg.get("total_tokens", 1), 1), 2
            ),
            "time_ratio": round(
                agent_avg.get("duration_sec", 0) / max(base_avg.get("duration_sec", 1), 1), 2
            ),
            "citations_delta": agent_avg.get("citations", 0) - base_avg.get("citations", 0),
        }

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"summary-{stamp}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = out_dir / f"comparison-{stamp}.md"
    md_path.write_text(_markdown(report), encoding="utf-8")

    print()
    print(f"Отчёт: {md_path}")
    print(f"Данные: {json_path}")
    print(f"Накопитель: {accumulator.path}")
    return 0


def _markdown(report: dict[str, object]) -> str:
    """Оформить отчёт в Markdown.

    Args:
        report: Данные отчёта.

    Returns:
        Текст отчёта.
    """
    agent = report.get("agent_avg") or {}
    baseline = report.get("baseline_avg") or {}
    comparison = report.get("comparison") or {}
    lines = [
        "# Агент против обычного чата с ИИ",
        "",
        f"_Задача: {report.get('task', '')}_",
        "",
        f"_Модели: {', '.join(report.get('models') or [])} · повторов: {report.get('repeats', 0)}_",
        "",
        "| Метрика | Агент | Обычный чат |",
        "|---|---:|---:|",
    ]
    for key, title in (
        ("citations", "Подтверждённых оснований"),
        ("facts", "Фактов"),
        ("variants", "Вариантов"),
        ("total_tokens", "Токенов (среднее)"),
        ("duration_sec", "Время, с (среднее)"),
        ("cost_units", "Расход, усл. ед. (среднее)"),
    ):
        agent_value = agent.get(key, "—")
        baseline_value = baseline.get(key, "—")
        lines.append(f"| {title} | {agent_value} | {baseline_value} |")
    lines.extend(
        [
            "",
            "## Отношения",
            "",
            f"- Токенов агента к обычному чату: **{comparison.get('token_ratio', '—')}×**",
            f"- Времени агента к обычному чату: **{comparison.get('time_ratio', '—')}×**",
            f"- Разница в подтверждённых основаниях: **{comparison.get('citations_delta', '—')}**",
            "",
            "## Как это читать",
            "",
            "- Подтверждённых оснований у обычного чата быть не может: их "
            "проверяет код агента дословным поиском, а не модель.",
            "- Токены агента больше — это цена проверки, журнала шагов и фиксации обязательств.",
            "- Разброс между прогонами у локальных моделей велик, поэтому "
            "каждая величина — среднее по нескольким прогонам.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
