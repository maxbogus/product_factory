#!/usr/bin/env python3
"""Свип модели, окна и бюджета контекста: матрица, прогресс, продолжение.

Зачем отдельный скрипт. Замер идёт часами, требует GPU и может упасть по
памяти: 12b с окном 16384 на 8 ГБ VRAM выливается на CPU или падает. Поэтому
каждая клетка матрицы выполняется **отдельным процессом**: OOM убивает
ребёнка, а не харнесс, и продолжение начинается с того места, где оборвалось.

Устройство:

- **клетка** — ``(модель, окно, бюджет абзацев, повтор)``;
- **состояние** — append-only ``runs.jsonl``: одна строка на попытку. По нему
  видно, что уже сделано, и по нему же продолжается замер;
- **накопитель** — ``progress.log`` обновляется не реже, чем раз в 30 секунд;
- **решение** — ``decision.json``: клетки и замороженный выбор, который
  проверяет ``tests/test_role_sweep.py -k sweep_decision``.

Запуск:

    uv run python scripts/grid_sweep.py --dry-run
    uv run python scripts/grid_sweep.py --models qwen3:8b mistral-nemo:12b \\
        --num-ctx 8192 12288 16384 --context 3 5 8 --repeats 3
    make grid-sweep                     # 2 модели × 3 окна × 3 повтора
    make grid-sweep CONTEXT="3 5 8"     # полная ось бюджета контекста

Продолжение — тем же вызовом: готовые клетки (``status=ok``) пропускаются,
упавшие и прерванные повторяются в новом ``session_id``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from pf.journal import StepRecord  # noqa: E402
from pf.role_sweep import (  # noqa: E402
    TierStats,
    cell_from_dict,
    cell_key,
    decide_sweep,
    snapshot_payload,
)
from pf.roles import ROLE_TIERS, tier_of  # noqa: E402

#: Вопрос демонстрации — тот же, что в ``make demo``.
QUESTION = "Почему падает удержание у мобильного тарифа и что с этим делать?"

#: Ответ человеку, одинаковый для всех прогонов: иначе сравнение моделей
#: сместится вводом, а не моделью (так же, как в сессии 31).
ANSWER = (
    "Внутренних данных нет — работаем только на открытых источниках: "
    "корпус Википедии и таблицы project/. Бюджет и сроки не заданы. "
    "Менять можно продукт, условия тарифа и коммуникацию; цену снижать "
    "нельзя — это ограничение по марже. Вариант «оставить как есть» допустим."
)

#: Виды отказов, которые считаются фантомами: источник придуман или цитата
#: не найдена. Остальные отказы (число не сошлось, пересказ) — расхождения,
#: а не выдумка.
PHANTOM_KINDS = frozenset({"source_not_named", "source_unknown", "fabrication"})

#: Предохранитель от бесконечного цикла шагов.
MAX_STEPS = 30

#: Как часто обновлять прогресс.
PROGRESS_EVERY = 15.0

#: Как в ошибке называется роль, на которой упал прогон.
FAILED_ROLE_RE = re.compile(r"роль «([^»]+)»")


@dataclass(frozen=True, slots=True)
class CellSpec:
    """Клетка матрицы: что запускаем.

    Attributes:
        model: Имя модели Ollama.
        num_ctx: Окно контекста.
        context_passages: Бюджет абзацев на роль.
        repeat: Номер повтора.
        attempt: Номер попытки (0 — первая). Прерванная клетка повторяется
            с новым ``session_id``, иначе журнал отвергнет дубль ``ordinal``.
    """

    model: str
    num_ctx: int
    context_passages: int
    repeat: int
    attempt: int = 0


def build_grid(
    models: list[str],
    windows: list[int],
    budgets: list[int],
    repeats: int,
) -> list[CellSpec]:
    """Построить матрицу клеток.

    Порядок — от меньшего окна к большему и от меньшего бюджета к большему:
    дешёвые и безопасные клетки идут первыми, а опасные (большое окно на
    12b) — последними.

    Args:
        models: Модели.
        windows: Окна контекста.
        budgets: Бюджеты абзацев.
        repeats: Число повторов на клетку.

    Returns:
        Список клеток.
    """
    grid: list[CellSpec] = []
    # Повтор — внешний цикл: первый проход даёт по клетке на каждое окно и
    # модель, и картина обрезки видна раньше, чем добор повторов. Иначе все
    # девять клеток одного окна шли бы часами, прежде чем появится второе.
    for repeat in range(repeats):
        for model in models:
            for window in windows:
                for budget in sorted(budgets):
                    grid.append(CellSpec(model, window, budget, repeat))
    return grid


def read_records(path: Path) -> list[dict[str, Any]]:
    """Прочитать записи накопителя.

    Args:
        path: Файл ``runs.jsonl``.

    Returns:
        Записи в порядке появления; битая последняя строка пропускается.
    """
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Дописать запись в накопитель.

    Args:
        path: Файл ``runs.jsonl``.
        record: Запись о клетке.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _key(spec: CellSpec) -> str:
    """Ключ клетки.

    Args:
        spec: Клетка матрицы.

    Returns:
        Детерминированный ключ без номера попытки.
    """
    return cell_key(spec.model, spec.num_ctx, spec.context_passages, spec.repeat)


def _truncated(record: StepRecord) -> bool:
    """Упёрся ли шаг в окно контекста.

    Args:
        record: Запись журнала.

    Returns:
        ``True``, если признак обрезки записан.
    """
    detail = record.detail if isinstance(record.detail, dict) else {}
    context = detail.get("context")
    return bool(isinstance(context, dict) and context.get("truncated"))


def tier_stats(records: list[StepRecord], failed_role: str = "") -> dict[str, TierStats]:
    """Свести журнал по ярусам.

    Ярус считается пройденным, если его роли дошли до журнала и прогон не
    упал на одной из них. Недостигнутый ярус — не «мало данных», а провал:
    модель не довела работу до него.

    Args:
        records: Записи журнала.
        failed_role: Роль, на которой упал прогон (пусто — не падал).

    Returns:
        Статистика по трём ярусам.
    """
    present = {record.role for record in records}
    by_tier: dict[str, list[StepRecord]] = {}
    for record in records:
        by_tier.setdefault(tier_of(record.role), []).append(record)
    failed_tier = tier_of(failed_role) if failed_role else ""
    stats: dict[str, TierStats] = {}
    for tier in ("extraction", "planning", "synthesis"):
        expected = [role for role, name in ROLE_TIERS.items() if name == tier]
        tier_records = by_tier.get(tier, [])
        schema_ok = (
            bool(tier_records) and tier != failed_tier and all(role in present for role in expected)
        )
        stats[tier] = TierStats(
            roles=len(tier_records),
            schema_ok=schema_ok,
            truncation_steps=sum(1 for record in tier_records if _truncated(record)),
            duration_sec=round(sum(record.duration_ms for record in tier_records) / 1000.0, 3),
        )
    return stats


def phantom_count(records: list[StepRecord]) -> int:
    """Сколько отказов пришлось на выдуманные источники и цитаты.

    Args:
        records: Записи журнала.

    Returns:
        Число отказов видов :data:`PHANTOM_KINDS`.
    """
    total = 0
    for record in records:
        for rejection in record.rejections or []:
            if isinstance(rejection, dict) and rejection.get("kind") in PHANTOM_KINDS:
                total += 1
    return total


def run_worker(spec: CellSpec, session_id: str, result_path: Path) -> int:
    """Прогнать одну клетку до ``DONE`` и записать результат.

    Автоответ человеку повторяет сценарий сессии 31: один и тот же текст на
    всех прогонах, чтобы сравнение не смещалось вводом.

    Args:
        spec: Клетка матрицы.
        session_id: Идентификатор сессии.
        result_path: Куда записать результат.

    Returns:
        ``0`` — клетка дошла до ``DONE``; ``1`` — ошибка или остановка.
    """
    from pf.config import Settings
    from pf.journal import Journal, journal_path
    from pf.report import build_session_report
    from pf.state import Phase, SessionState, record_human_input, save
    from pf.workflow import Workflow

    settings = Settings(
        llm_provider="ollama",
        ollama_model=spec.model,
        ollama_num_ctx=spec.num_ctx,
        context_passages=spec.context_passages,
        max_tokens=4096,
        # Матрица меряет одну модель на всех ролях. Имена ярусов из `.env`
        # (например, синтез на 12b) подменили бы модель клетки и смешали
        # замер; здесь они гасятся явно.
        role_model_extraction="",
        role_model_planning="",
        role_model_synthesis="",
        # След работы не нужен замеру: очередь распознания — не часть проверки.
        lexicon_enabled=False,
        enable_web=False,
    )
    workflow = Workflow(config=settings)
    state = SessionState(session_id=session_id, phase=Phase.INTAKE, question=QUESTION)
    save(state)
    bundle = None
    error = ""
    failed_role = ""
    started = time.perf_counter()
    try:
        for _step in range(1, MAX_STEPS + 1):
            if state.phase is Phase.DONE:
                break
            if state.pending_question:
                question = state.pending_question
                state.pending_question = ""
                record_human_input(state, ANSWER, question=question)
                save(state)
            if state.phase is Phase.COLLECTING:
                bundle = workflow.collect_sources(state)
            result = workflow.run_step(state, bundle)
            prompt = workflow.human_question_to_stop(state, result)
            if prompt:
                state.pending_question = prompt
                state.results.setdefault("human_questions", []).append(prompt)
                save(state)
                continue
            if state.phase is Phase.KEEPER:
                state.phase = Phase.DONE
            elif state.phase is Phase.ROUTING:
                state.phase = workflow.route_next(state, result)
            else:
                state.phase = state.phase.next
            save(state)
    except Exception as exc:  # noqa: BLE001 — клетка должна оставить след
        error = f"{type(exc).__name__}: {exc}"
        match = FAILED_ROLE_RE.search(error)
        failed_role = match.group(1) if match else ""
        save(state)

    duration = time.perf_counter() - started
    records = Journal(journal_path(session_id)).read()
    report = build_session_report(session_id)
    status = "ok" if state.phase is Phase.DONE and not error else "failed"
    payload: dict[str, Any] = {
        "key": _key(spec),
        "model": spec.model,
        "num_ctx": spec.num_ctx,
        "context_passages": spec.context_passages,
        "repeat": spec.repeat,
        "attempt": spec.attempt,
        "session_id": session_id,
        "status": status,
        "error": error,
        "failed_role": failed_role,
        "phase": state.phase.value,
        "grounding_rate": round(report.grounding_rate, 6),
        "phantom": phantom_count(records),
        "conclusion_status": report.conclusion.status if report.conclusion else "",
        "duration_sec": round(duration, 3),
        "tiers": {
            name: {
                "roles": stat.roles,
                "schema_ok": stat.schema_ok,
                "truncation_steps": stat.truncation_steps,
                "duration_sec": stat.duration_sec,
            }
            for name, stat in tier_stats(records, failed_role).items()
        },
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if status == "ok" else 1


def ollama_processor(base_url: str, model: str) -> str:
    """Как модель лежит в памяти: доля видео- и системной памяти.

    ``/api/ps`` отдаёт размер модели и размер в видеопамяти. Это измерение,
    а не оценка: на 8 ГБ VRAM 12b с большим окном действительно уезжает на
    CPU, и это видно по доле.

    Args:
        base_url: Адрес Ollama.
        model: Имя модели.

    Returns:
        ``100% GPU``, ``NN% CPU / MM% GPU`` или пустая строка, если модель
        не загружена.
    """
    import httpx

    try:
        response = httpx.get(base_url.rstrip("/") + "/api/ps", timeout=2.0)
        payload = response.json()
    except Exception:  # noqa: BLE001 — опрос памяти не должен ронять замер
        return ""
    for item in payload.get("models") or []:
        name = str(item.get("name", ""))
        if name != model and not name.startswith(model.split(":", 1)[0]):
            continue
        size = int(item.get("size", 0) or 0)
        vram = int(item.get("size_vram", 0) or 0)
        if not size:
            return ""
        percent = round(vram / size * 100)
        if percent >= 100:
            return "100% GPU"
        return f"{100 - percent}% CPU / {percent}% GPU"
    return ""


def preflight() -> str:
    """Короткая сводка железа для ``progress.log``.

    Returns:
        Строка с RAM, диском и видеопамятью.
    """
    parts: list[str] = []
    for command in (
        ["free", "-g"],
        ["df", "-h", "/"],
        ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader"],
    ):
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=10)
            text = (done.stdout or done.stderr).strip().splitlines()
            parts.append(f"{command[0]}: {' | '.join(text[:2])}")
        except Exception:  # noqa: BLE001
            parts.append(f"{command[0]}: недоступен")
    return "; ".join(parts)


def run_cell(
    spec: CellSpec,
    session_id: str,
    result_path: Path,
    timeout: int,
    progress_path: Path | None = None,
) -> dict[str, Any]:
    """Выполнить клетку в дочернем процессе.

    OOM убивает ребёнка, а не харнесс: если файла результата нет, клетка
    считается прерванной и повторится при следующем запуске.

    Args:
        spec: Клетка матрицы.
        session_id: Идентификатор сессии.
        result_path: Куда ребёнок пишет результат.
        timeout: Предел времени клетки в секундах.
        progress_path: Файл прогресса; раз в 30 секунд пишется heartbeat —
            иначе долгая клетка выглядит зависшей (правило 08).

    Returns:
        Запись о клетке.
    """
    processor = {"value": ""}
    stop = threading.Event()
    started = time.monotonic()

    def poll() -> None:
        last_beat = time.monotonic()
        while not stop.is_set():
            value = ollama_processor("http://localhost:11434", spec.model)
            if value:
                processor["value"] = value
            now = time.monotonic()
            if progress_path is not None and now - last_beat >= 30.0:
                last_beat = now
                log_progress(
                    progress_path,
                    f"… {_key(spec)} идёт {now - started:.0f}с "
                    f"processor={processor['value'] or '—'}",
                )
            stop.wait(PROGRESS_EVERY)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--spec",
        json.dumps(asdict(spec), ensure_ascii=False),
        "--session",
        session_id,
        "--result",
        str(result_path),
    ]
    stderr = ""
    try:
        done = subprocess.run(
            command, cwd=str(REPO), capture_output=True, text=True, timeout=timeout
        )
        stderr = (done.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        stderr = f"timeout {timeout}s"
    finally:
        stop.set()
        thread.join(timeout=2.0)

    if result_path.is_file():
        payload: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        payload = {
            "key": _key(spec),
            "model": spec.model,
            "num_ctx": spec.num_ctx,
            "context_passages": spec.context_passages,
            "repeat": spec.repeat,
            "attempt": spec.attempt,
            "session_id": session_id,
            "status": "interrupted",
            "error": stderr,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    payload["processor"] = processor["value"]
    return payload


def log_progress(path: Path, text: str) -> None:
    """Дописать строку прогресса и продублировать в stdout.

    Args:
        path: Файл ``progress.log``.
        text: Что записать.
    """
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    line = f"[{stamp}] {text}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def log_memory(record: dict[str, Any]) -> None:
    """Записать шаг в память ML-сессий (правило 08).

    Args:
        record: Запись о клетке.
    """
    target = Path.home() / ".cline" / "memory" / "ml-sessions.jsonl"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "task": "role_sweep",
            "status": record.get("status", ""),
            "note": (
                f"{record.get('key', '')} {record.get('model', '')} "
                f"ctx={record.get('num_ctx', '')} p={record.get('context_passages', '')}"
            ),
        }
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _meta(args: argparse.Namespace) -> dict[str, Any]:
    """Метаданные замера для снимка.

    Args:
        args: Разобранные аргументы.

    Returns:
        Словарь метаданных.
    """
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "question": QUESTION,
        "models": args.models,
        "num_ctx": args.num_ctx,
        "context": args.context,
        "repeats": args.repeats,
    }


def write_snapshot(out: Path, *, log_path: Path | None, meta: dict[str, Any]) -> int:
    """Записать ``decision.json`` по завершённым клеткам.

    Читаются только ``status=ok``: решение не должно держаться на упавших
    прогонах. Функция отделена от запуска, чтобы решение можно было
    пересобрать по уже готовым клеткам (``--snapshot-only``), не гоняя
    матрицу заново.

    Args:
        out: Каталог замера.
        log_path: Файл прогресса (``None`` — не писать).
        meta: Метаданные замера.

    Returns:
        ``0`` — снимок записан; ``1`` — завершённых клеток нет.
    """
    by_key: dict[str, dict[str, Any]] = {}
    for record in read_records(out / "runs.jsonl"):
        # Последняя попытка на клетку. Снимок — запись замера, а не только
        # удачных прогонов: таймаут 12b (25 % CPU) тоже результат, и без него
        # решение «8b на синтез» выглядело бы взятым с потолка. Решение
        # считается только по ``ok`` — это делает :func:`decide_sweep`.
        by_key[str(record.get("key", ""))] = record
    cells = [cell_from_dict(record) for record in by_key.values()]
    if not any(cell.status == "ok" for cell in cells):
        if log_path is not None:
            log_progress(log_path, "нет завершённых клеток — решение не пишется")
        return 1
    decision = decide_sweep(cells)
    payload = snapshot_payload(cells, decision, meta=meta)
    path = out / "decision.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if log_path is not None:
        log_progress(
            log_path,
            f"решение: окно {decision.window}, бюджет {decision.context_passages}, "
            f"ярусы {decision.tiers} → {path}",
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Точка входа харнесса.

    Args:
        argv: Аргументы командной строки.

    Returns:
        ``0`` — решение записано; ``1`` — завершённых клеток нет.
    """
    parser = argparse.ArgumentParser(description="Свип модели, окна и бюджета контекста")
    parser.add_argument("--worker", action="store_true", help="выполнить одну клетку")
    parser.add_argument("--spec", default="", help="клетка для режима --worker")
    parser.add_argument("--session", default="", help="сессия для режима --worker")
    parser.add_argument("--result", default="", help="файл результата для режима --worker")
    parser.add_argument("--models", nargs="+", default=["qwen3:8b", "mistral-nemo:12b"])
    parser.add_argument("--num-ctx", nargs="+", type=int, default=[12288, 8192, 16384])
    parser.add_argument("--context", nargs="+", type=int, default=[5])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--out", default=str(REPO / "reports" / "role_sweep"))
    parser.add_argument("--dry-run", action="store_true", help="только напечатать матрицу")
    parser.add_argument("--smoke", action="store_true", help="одна клетка — проверка харнесса")
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="не запускать клетки, только пересобрать decision.json из runs.jsonl",
    )
    args = parser.parse_args(argv)

    if args.worker:
        spec = CellSpec(**json.loads(args.spec))
        return run_worker(spec, args.session, Path(args.result))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"
    progress_path = out / "progress.log"
    grid = build_grid(args.models, args.num_ctx, args.context, args.repeats)
    if args.smoke:
        grid = grid[:1]
    if args.dry_run:
        for spec in grid:
            print(f"{_key(spec)}")
        print(f"клеток: {len(grid)}")
        return 0

    if args.snapshot_only:
        return write_snapshot(out, log_path=progress_path, meta=_meta(args))

    log_progress(progress_path, f"старт: клеток {len(grid)}; {preflight()}")
    records = read_records(runs_path)
    latest = {str(record.get("key", "")): record for record in records}
    attempts: dict[str, int] = {}
    for record in records:
        key = str(record.get("key", ""))
        attempts[key] = attempts.get(key, 0) + 1

    for spec in grid:
        key = _key(spec)
        if latest.get(key, {}).get("status") == "ok":
            log_progress(progress_path, f"пропуск {key}: уже сделано")
            continue
        attempt = attempts.get(key, 0)
        current = CellSpec(spec.model, spec.num_ctx, spec.context_passages, spec.repeat, attempt)
        # Нонс в идентификаторе: убитый прогон (OOM, timeout) оставляет
        # частичный журнал, и повтор с тем же именем отверг бы дубль
        # ``ordinal``. Продолжение идёт по ``runs.jsonl``, а не по имени.
        session_id = (
            f"sweep-{key.replace('|', '-').replace('=', '-')}-a{attempt}-{int(time.time())}"
        )
        result_path = out / "results" / f"{session_id}.json"
        log_progress(progress_path, f"старт {key} attempt={attempt}")
        payload = run_cell(current, session_id, result_path, args.timeout, progress_path)
        append_record(runs_path, payload)
        latest[key] = payload
        attempts[key] = attempt + 1
        log_memory(payload)
        error = f" error={str(payload.get('error'))[:120]}" if payload.get("error") else ""
        log_progress(
            progress_path,
            f"{payload.get('status')} {key}: {payload.get('duration_sec', '?')}с "
            f"grounding={payload.get('grounding_rate', '?')} "
            f"processor={payload.get('processor', '') or '—'}{error}",
        )

    by_key: dict[str, dict[str, Any]] = {}
    for record in read_records(runs_path):
        if record.get("status") == "ok":
            by_key[str(record.get("key", ""))] = record
    cells = [cell_from_dict(record) for record in by_key.values()]
    if not cells:
        log_progress(progress_path, "нет завершённых клеток — решение не пишется")
        return 1

    decision = decide_sweep(cells)
    payload = snapshot_payload(cells, decision, meta=_meta(args))
    decision_path = out / "decision.json"
    decision_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_progress(
        progress_path,
        f"решение: окно {decision.window}, бюджет {decision.context_passages}, "
        f"ярусы {decision.tiers} → {decision_path}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
