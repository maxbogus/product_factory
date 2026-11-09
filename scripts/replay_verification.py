#!/usr/bin/env python3
"""Повторная проверка оснований по сохранённым журналам: A/B вместо «до/после».

Зачем это нужно. Сравнивать «прогон против прогона» бессмысленно: на одной и
той же задаче число подтверждённых оснований различается в разы из-за модели
(в проекте зафиксирован разброс 18 против 0). Поэтому проверка
воспроизводится на **сохранённых** результатах ролей: данные одинаковые,
меняется только код проверки — и дельта принадлежит правкам, а не модели.

Откуда берутся данные: ``state/session-<id>.json`` — результаты завершённых
шагов. Они восстанавливаются по схемам ролей (``pf.roles.validate``), а
источники собираются заново тем же детерминированным способом, что и в живом
прогоне (``Workflow.collect_sources``: TF-IDF по корпусу и материалы проекта,
без обращения к модели).

Что печатается: подтверждено / нерелевантный источник / нет данных / требует
проверки, разбивка по ролям и классам отказов. С ``--json`` — снимок для
сравнения двух версий кода:

    git worktree add /tmp/old HEAD
    (cd /tmp/old && uv run python scripts/replay_verification.py --json /tmp/base.json)
    uv run python scripts/replay_verification.py --json /tmp/new.json --compare /tmp/base.json

Контрольные пробы (``--probes``, включены по умолчанию) — восемь известных
расхождений, на которых проверка когда-то ошибалась: округление числа,
словоформа, смешанная раскладка, подмена бренда, иностранная страница,
добавленное отрицание, пересказ строки таблицы и короткая цитата. Они не зависят от журналов и ловят регресс сразу, без
живого прогона.

Важно: ``state/`` и ``journal/`` в ``.gitignore``. На чистом клоне журналов
нет — скрипт скажет об этом прямо, а не покажет ноль: ноль оснований и
отсутствие данных — разные вещи.

Запуск:
    uv run python scripts/replay_verification.py
    uv run python scripts/replay_verification.py --sessions demo-v7 demo-v9
    uv run python scripts/replay_verification.py --json /tmp/replay.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pf.config import settings
from pf.ground import SupportStatus, verify_claim, verify_quote
from pf.replay import replay_session
from pf.state import list_sessions

#: Прогоны, на которых замер уже делался. Используются по умолчанию, если они
#: есть: в ``state/`` лежат и сессии бенчмарка, и служебные прогоны тестов.
MEASURED_SESSIONS = ("demo-v7", "demo-v9", "demo-final", "cloud-deepseek-1", "smoke-lexicon")

#: Пояснение отказа, по которому основание попадает в отдельную корзину:
#: источник настоящий, но не про эту задачу.
IRRELEVANT = "источник не относится к задаче"


@dataclass(frozen=True, slots=True)
class Probe:
    """Контрольная проба: известное расхождение и ожидаемый вердикт.

    Attributes:
        id: Идентификатор пробы.
        title: Что проверяется — для человека.
        source: Текст источника.
        claim: Утверждение или цитата из ответа модели.
        expected: Ожидаемый статус.
        via_claim: Проверять покрытием абзаца (как факт), а не цитатой.
        quote: Цитата из ответа модели, если она есть. Недословная цитата не
            отменяет проверку полями — это и держат пробы ниже.
    """

    id: str
    title: str
    source: str
    claim: str
    expected: SupportStatus
    via_claim: bool = False
    quote: str = ""


#: Шесть расхождений, которые проверка когда-то путала. Две последние строки —
#: не строгость, а пропущенная ложь: подмена бренда и добавленное отрицание
#: проходили как подтверждённые основания.
PROBES: tuple[Probe, ...] = (
    Probe(
        id="rounded-number",
        title="округление числа: 77,3 против 77,26",
        source="По состоянию на июнь 2024 года компания обслужила 77,26 миллиона абонентов.",
        claim="компания обслужила 77,3 миллиона абонентов",
        expected=SupportStatus.CONFIRMED,
    ),
    Probe(
        id="synonym",
        title="словоформа: «клиентов» против «абонентов»",
        source="Компания обслуживает абонентов связи.",
        claim="Компания обслуживает клиентов связи",
        expected=SupportStatus.CONFIRMED,
        via_claim=True,
    ),
    Probe(
        id="mixed-script",
        title="смешанная раскладка: абонентoв с латинской o",
        source="Компания обслужила 77,26 миллиона абонентов.",
        claim="обслужила 77,26 миллиона абонентoв",
        expected=SupportStatus.CONFIRMED,
    ),
    Probe(
        id="brand-substitution",
        title="подмена бренда: МТС вместо МегаФона",
        source="«МегаФон» — российская телекоммуникационная компания.",
        claim="МТС — российская телекоммуникационная компания",
        expected=SupportStatus.NO_DATA,
        via_claim=True,
    ),
    Probe(
        id="row-paraphrase",
        title="пересказ строки таблицы: цитата без меток колонок",
        source=(
            "оператор: МТС, показатель: абонентская база в России, значение: 82, "
            "период: конец 3-го квартала 2024"
        ),
        claim="Абонентская база МТС в России составляет 82 млн человек",
        expected=SupportStatus.CONFIRMED,
        via_claim=True,
        quote="МТС, абонентская база в России: 82",
    ),
    Probe(
        id="short-quote",
        title="короткая цитата: «38,9» при верном числе",
        source=(
            "оператор: Tele2, показатель: абонентская база, значение: 38.9, период: конец июля"
        ),
        claim="Абонентская база Tele2 составляет 38,9 млн человек",
        expected=SupportStatus.CONFIRMED,
        via_claim=True,
        quote="38,9",
    ),
    Probe(
        id="foreign-page",
        title="русская цитата на английской странице",
        source="Megafon served 77.26 million subscribers in June 2024.",
        claim="МегаФон обслужил 77,26 миллиона абонентов",
        expected=SupportStatus.NEEDS_REVIEW,
    ),
    Probe(
        id="added-negation",
        title="добавленное отрицание: «не обслуживает»",
        source="Компания обслуживает абонентов связи.",
        claim="Компания не обслуживает абонентов связи",
        expected=SupportStatus.NO_DATA,
        via_claim=True,
    ),
)


class _QuietLexicon:
    """Выключить очередь распознания на время замера.

    Очередь — след работы, а не часть проверки: реплей не должен дописывать
    в ``knowledge/lexicon/oov.jsonl`` слова из старых прогонов.
    """

    def __enter__(self) -> _QuietLexicon:
        """Запомнить прежнее значение и выключить накопление.

        Returns:
            Себя — для использования в ``with``.
        """
        self._previous = settings.lexicon_enabled
        settings.lexicon_enabled = False
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Вернуть настройку как была.

        Args:
            *exc_info: Стандартные аргументы выхода из ``with``.
        """
        settings.lexicon_enabled = self._previous


def run_probe(probe: Probe) -> SupportStatus:
    """Выполнить контрольную пробу.

    Args:
        probe: Проба.

    Returns:
        Фактический статус проверки.
    """
    if probe.via_claim:
        support = verify_claim(
            probe.source,
            probe.claim,
            "probe/source",
            paragraph_index=0,
            quote=probe.quote,
        )
    else:
        support = verify_quote(probe.source, probe.claim, "probe/source")
    return support.status


def check_probes() -> list[dict[str, Any]]:
    """Проверить все контрольные пробы.

    Returns:
        Список результатов ``{id, title, expected, actual, passed}``.
    """
    results: list[dict[str, Any]] = []
    for probe in PROBES:
        actual = run_probe(probe)
        results.append(
            {
                "id": probe.id,
                "title": probe.title,
                "expected": probe.expected.value,
                "actual": actual.value,
                "passed": actual is probe.expected,
            }
        )
    return results


def _totals(sessions: list[dict[str, Any]]) -> dict[str, int]:
    """Свод по всем сессиям.

    Args:
        sessions: Снимки по сессиям.

    Returns:
        Счётчики по корзинам и общее число оснований.
    """
    totals: dict[str, int] = {
        "подтверждено": 0,
        "опора на ввод пользователя": 0,
        "нерелевантный источник": 0,
        "нет данных": 0,
        "требует проверки": 0,
    }
    for payload in sessions:
        for kind, count in payload["kinds"].items():
            if kind.startswith("причина: "):
                continue
            totals[kind] = totals.get(kind, 0) + count
    totals["всего оснований"] = sum(len(payload["verdicts"]) for payload in sessions)
    # Величины-факты плана без своего расчёта: мера того, сколько раз статус
    # вывода понижен числом, а не основанием (`T-48b`).
    totals["величин-фактов без расчёта"] = sum(
        int((payload.get("plan_numbers") or {}).get("unsupported") or 0) for payload in sessions
    )
    return totals


def _role_totals(sessions: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Свод по ролям.

    Args:
        sessions: Снимки по сессиям.

    Returns:
        Счётчики по ролям, отсортированные по имени роли.
    """
    roles: dict[str, dict[str, int]] = {}
    for payload in sessions:
        for role, counts in payload["roles"].items():
            bucket = roles.setdefault(
                role,
                {
                    "подтверждено": 0,
                    "отклонено": 0,
                    "требует проверки": 0,
                    "опора на ввод пользователя": 0,
                    "оснований": 0,
                },
            )
            bucket["подтверждено"] += counts["confirmed"]
            bucket["отклонено"] += counts["rejected"]
            bucket["требует проверки"] += counts["needs_review"]
            bucket["опора на ввод пользователя"] += int(counts.get("user_input") or 0)
            bucket["оснований"] += counts["claims"]
    return dict(sorted(roles.items()))


def _verdict_key(verdict: dict[str, Any]) -> tuple[str, str, str, str]:
    """Ключ сопоставления вердиктов между версиями кода.

    Args:
        verdict: Вердикт из снимка.

    Returns:
        Кортеж «сессия — роль — утверждение — источник».
    """
    return (
        str(verdict["session"]),
        str(verdict["role"]),
        str(verdict["claim"]),
        str(verdict["source_id"]),
    )


def _flips(baseline: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Вердикты, которые перевернулись между версиями.

    Args:
        baseline: Прежний снимок.
        current: Текущий снимок.

    Returns:
        Строки «что было — что стало» для каждого перевёрнутого вердикта.
    """
    before = {_verdict_key(item): item for item in baseline.get("verdicts", [])}
    after = {_verdict_key(item): item for item in current.get("verdicts", [])}
    changed: list[str] = []
    for key, was in before.items():
        now = after.get(key)
        if now is None or now["status"] == was["status"]:
            continue
        changed.append(
            f"{key[0]}/{key[1]}: «{key[2][:60]}» → {key[3]}: {was['status']} стало {now['status']}"
        )
    return changed


def _commit() -> str:
    """Короткий хеш текущей версии кода — чтобы снимок был привязан к коду.

    Returns:
        Хеш или пустая строка, если git недоступен.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _selected(ids: list[str] | None) -> list[str]:
    """Выбрать сессии для повтора.

    Args:
        ids: Явно заданные идентификаторы.

    Returns:
        Список сессий: явно заданные, иначе замеренные, иначе все доступные.
    """
    if ids:
        return ids
    available = set(list_sessions())
    measured = [session for session in MEASURED_SESSIONS if session in available]
    return measured or sorted(available)


def _current_snapshot(
    snapshots: list[dict[str, Any]],
    probes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Собрать снимок замера для записи в файл.

    Вердикты нужны, чтобы следующее сравнение отвечало не только «сколько»,
    но и «какие именно основания переехали».

    Args:
        snapshots: Снимки по сессиям.
        probes: Результаты контрольных проб.

    Returns:
        Снимок с настройками, счётчиками и вердиктами.
    """
    verdicts = [verdict for payload in snapshots for verdict in payload["verdicts"]]
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _commit(),
        "settings": {
            "claim_min_overlap": settings.claim_min_overlap,
            "metric_abs_tolerance": settings.metric_abs_tolerance,
            "metric_rel_tolerance": settings.metric_rel_tolerance,
        },
        "sessions": {payload["session_id"]: payload for payload in snapshots},
        "totals": _totals(snapshots),
        "roles": _role_totals(snapshots),
        "probes": probes or [],
        "verdicts": verdicts,
    }


def _print_totals(totals: dict[str, int], baseline: dict[str, int] | None = None) -> None:
    """Напечатать свод, при наличии — с дельтой к прежнему снимку.

    Args:
        totals: Текущие счётчики.
        baseline: Счётчики прежнего снимка.
    """
    print("=== Проверка оснований по сохранённым журналам ===")
    print()
    if baseline is None:
        for kind, count in totals.items():
            print(f"  {kind:<24} {count}")
        print()
        return

    print(f"  {'исход':<24} {'было':>6} {'стало':>6} {'дельта':>7}")
    for kind, count in totals.items():
        was = baseline.get(kind, 0)
        print(f"  {kind:<24} {was:>6} {count:>6} {count - was:>+7}")
    print()


def main(argv: list[str] | None = None) -> int:
    """Выполнить замер по сохранённым журналам.

    Args:
        argv: Аргументы командной строки (по умолчанию — из окружения).

    Returns:
        Код возврата: 0 — замер выполнен и пробы сошлись; 1 — проба не
        сошлась, перевёрнутый вердикт при ``--strict`` либо нет данных для
        повтора.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        metavar="ID",
        help="сессии для повтора (по умолчанию — замеренные, иначе все сохранённые)",
    )
    parser.add_argument("--json", dest="json_path", metavar="PATH", help="записать снимок")
    parser.add_argument("--compare", metavar="PATH", help="сравнить с прежним снимком")
    parser.add_argument(
        "--no-probes", dest="probes", action="store_false", help="не выполнять пробы"
    )
    parser.add_argument(
        "--strict", action="store_true", help="код возврата 1 при перевёрнутых вердиктах"
    )
    parser.add_argument("--verbose", action="store_true", help="показать вердикты и причины")
    args = parser.parse_args(argv)

    session_ids = _selected(args.sessions)
    with _QuietLexicon():
        snapshots = [payload for sid in session_ids if (payload := replay_session(sid))]
        probes = check_probes() if args.probes else []

    found = {payload["session_id"] for payload in snapshots}
    missing = [sid for sid in session_ids if sid not in found]
    if missing:
        print("Нет состояния на диске (state/ в .gitignore — прогон был на другой машине):")
        for sid in missing:
            print(f"  - {sid}")
        print()

    if not snapshots:
        print("Повторять нечего: сохранённых прогонов нет.")
        print('Сначала выполните прогон: make demo  (или uv run pf run "…")')
        return 1

    baseline: dict[str, Any] | None = None
    if args.compare:
        path = Path(args.compare)
        if not path.is_file():
            print(f"Снимок для сравнения не найден: {path}")
            return 1
        baseline = json.loads(path.read_text(encoding="utf-8"))

    _print_totals(_totals(snapshots), baseline.get("totals") if baseline else None)

    print("По ролям:")
    for role, counts in _role_totals(snapshots).items():
        print(
            f"  {role:<10} оснований {counts['оснований']:>4}  "
            f"подтверждено {counts['подтверждено']:>4}  "
            f"отклонено {counts['отклонено']:>4}  "
            f"ждёт проверки {counts['требует проверки']:>3}"
        )
    print()

    # Замер `T-48b`: величина-факт без своего расчёта понижает статус вывода.
    # Печатается отдельно от исходов проверки оснований: это другой вопрос —
    # не «нашлась ли цитата», а «из чего следует число в плане».
    print("Числа плана (величина-факт без расчёта понижает статус):")
    for payload in snapshots:
        block = payload.get("plan_numbers") or {}
        for role, counts in (block.get("roles") or {}).items():
            status = counts.get("status") or "—"
            print(
                f"  {payload['session_id']:<22} {role:<9} замечаний {counts['notes']:>2}  "
                f"величин-фактов без расчёта {counts['unsupported']:>2}  статус {status}"
            )
    total_unsupported = sum(
        int((payload.get("plan_numbers") or {}).get("unsupported") or 0) for payload in snapshots
    )
    print(f"  всего величин-фактов без расчёта: {total_unsupported}")
    print()

    kinds: dict[str, int] = {}
    for payload in snapshots:
        for kind, count in payload["kinds"].items():
            kinds[kind] = kinds.get(kind, 0) + count
    print("Исходы и причины:")
    for kind, count in sorted(kinds.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {kind:<40} {count}")
    print()

    for payload in snapshots:
        if payload["skipped"]:
            print(f"Пропущено в {payload['session_id']} (схема ответа изменилась):")
            for item in payload["skipped"]:
                print(f"  - {item}")
            print()

    snapshot = _current_snapshot(snapshots, probes)
    flips: list[str] = []
    if baseline is not None:
        flips = _flips(baseline, snapshot)
        print(f"Перевёрнутых вердиктов: {len(flips)}")
        for line in flips if args.verbose else []:
            print(f"  - {line}")
        print()

    if args.probes:
        passed = sum(1 for probe in probes if probe["passed"])
        print(f"Контрольные пробы: {passed} из {len(probes)}")
        for probe in probes:
            mark = "ok  " if probe["passed"] else "FAIL"
            line = f"  {mark} {probe['id']:<20} ожидалось {probe['expected']}"
            if not probe["passed"]:
                line += f", получено {probe['actual']}"
            print(line)
        print()

    if args.json_path:
        target = Path(args.json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Снимок записан: {target}")

    if args.probes and any(not probe["passed"] for probe in probes):
        return 1
    if args.strict and flips:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
