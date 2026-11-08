"""Единственный источник чисел, которые называются в поставляемых документах.

Зачем модуль. Числа дублировались в трёх местах: ``README.md``,
``docs/DESCRIPTION.md`` и разделе «Как запустить». На 2026-11-08 в README
стояло «259 тестов», в описании — «433» и «259», при факте 448: читатель не
мог понять, какое из чисел настоящее, а жюри проверяет именно описание.
Теперь числа берутся отсюда, а гейт ``make docs-check``
(``scripts/docs_check.py``) падает, если в документе появилось значение,
расходящееся с каноном.

Проверка самого канона — ``make canon``. Числа, которые пересчитываются
дешёвой командой, пересчитываются: иначе канон сам станет ещё одним местом,
где написано неправда. Что дорого пересчитать (число статей корпуса —
175 МБ JSON), проверяется по количеству разделов, а статьи помечаются как
непроверенные явно.

Запуск:

    uv run python docs/canon.py            # сверить канон с деревом
    uv run python docs/canon.py --quiet    # только итог
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: Корень репозитория: файл лежит в ``docs/``.
ROOT = Path(__file__).resolve().parent.parent

# ── Числа ────────────────────────────────────────────────────────────────────
#: Детерминированные тесты: ``pytest -m "not slow"``.
TESTS_DETERMINISTIC = 666

#: Тесты с живой моделью: ``pytest -m slow``.
TESTS_LIVE = 4

#: Кейсов в eval-наборе (``eval/cases/scenarios.jsonl``).
EVAL_CASES = 22

#: BDD-сценариев в ``tests/features/*.feature``.
SCENARIOS_BDD = 12

#: Рабочих фаз (``pf.state.Phase`` без терминальной ``DONE``).
WORKFLOW_PHASES = 9

#: Окно контекста локальной модели по умолчанию (``PF_OLLAMA_NUM_CTX``).
OLLAMA_NUM_CTX = 12288

#: Статей в локальном снимке Википедии.
CORPUS_ARTICLES = 8066

#: Разделов (файлов) корпуса.
CORPUS_SECTIONS = 11


@dataclass(frozen=True, slots=True)
class DocNumber:
    """Число, которое документ обязан называть именно так.

    Attributes:
        path: Документ относительно корня репозитория.
        pattern: Регулярное выражение с одной или несколькими группами-числами.
        expected: Ожидаемые значения групп по порядку.
        hint: Что это за число — для сообщения об ошибке.
    """

    path: str
    pattern: str
    expected: tuple[int, ...]
    hint: str


@dataclass(frozen=True, slots=True)
class StaleNumber:
    """Значение, которое из документа обязано исчезнуть.

    Нужно, чтобы исправленное число не вернулось при следующей правке
    текста: гейт ловит не только «неверное сейчас», но и «вернулось».

    Attributes:
        path: Документ относительно корня репозитория.
        text: Подстрока, которой быть не должно.
        replaced_by: Чем это значение заменено — для сообщения.
    """

    path: str
    text: str
    replaced_by: str


#: Числа, которые называют поставляемые документы.
DOC_NUMBERS: tuple[DocNumber, ...] = (
    DocNumber(
        path="README.md",
        pattern=r"make test\s+#\s+(\d+)\s+тест",
        expected=(TESTS_DETERMINISTIC,),
        hint="число детерминированных тестов",
    ),
    DocNumber(
        path="README.md",
        pattern=r"(\d[\d\s\u00a0]*\d)\s+статей из\s+(\d+)\s+разделов",
        expected=(CORPUS_ARTICLES, CORPUS_SECTIONS),
        hint="размер корпуса",
    ),
    DocNumber(
        path="docs/DESCRIPTION.md",
        pattern=r"(\d[\d\s\u00a0]*\d)\s+статей,\s+(\d+)\s+разделов",
        expected=(CORPUS_ARTICLES, CORPUS_SECTIONS),
        hint="размер корпуса",
    ),
    DocNumber(
        path="docs/DESCRIPTION.md",
        pattern=r"(\d+)\s+детерминированных\s*\+\s*(\d+)\s+жив",
        expected=(TESTS_DETERMINISTIC, TESTS_LIVE),
        hint="число тестов: детерминированные и с живой моделью",
    ),
    DocNumber(
        path="docs/DESCRIPTION.md",
        pattern=r"make test\s+#\s+(\d+)\s+детерминированных",
        expected=(TESTS_DETERMINISTIC,),
        hint="число детерминированных тестов в разделе «Как запустить»",
    ),
    DocNumber(
        path="docs/DESCRIPTION.md",
        pattern=r"`PF_OLLAMA_NUM_CTX`,\s*по умолчанию\s+(\d+)",
        expected=(OLLAMA_NUM_CTX,),
        hint="окно контекста по умолчанию",
    ),
)

#: Значения, которые обязаны исчезнуть из документов.
STALE_DOC_NUMBERS: tuple[StaleNumber, ...] = (
    StaleNumber(path="README.md", text="259 тестов", replaced_by=f"{TESTS_DETERMINISTIC} тестов"),
    StaleNumber(
        path="docs/DESCRIPTION.md",
        text="433 детерминированных",
        replaced_by=f"{TESTS_DETERMINISTIC} детерминированных",
    ),
    StaleNumber(
        path="docs/DESCRIPTION.md",
        text="259 детерминированных",
        replaced_by=f"{TESTS_DETERMINISTIC} детерминированных",
    ),
    StaleNumber(
        path="docs/DESCRIPTION.md",
        text="по умолчанию 8192",
        replaced_by=f"по умолчанию {OLLAMA_NUM_CTX}",
    ),
)

#: Документы поставки: их читает жюри, за ними и следит гейт.
DELIVERY_DOCS: tuple[str, ...] = (
    "README.md",
    "docs/DESCRIPTION.md",
    "docs/decisions.md",
    "demo/RESULT.md",
    "demo/BRIEF.md",
    "docs/VIDEO_SCRIPT.md",
    "AGENTS.md",
)


def _count_collected(marker: str) -> int | None:
    """Сколько тестов собирает pytest по маркеру.

    Args:
        marker: Выражение маркера, например ``not slow``.

    Returns:
        Число собранных тестов либо ``None``, если pytest недоступен.
    """
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-m",
        marker,
        "-q",
        "--collect-only",
        "-p",
        "no:cacheprovider",
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    match = re.search(r"(\d+)/\d+ tests collected", result.stdout)
    return int(match.group(1)) if match else None


def _count_eval_cases() -> int:
    """Сколько кейсов лежит в eval-наборе.

    Returns:
        Число непустых строк в наборе.
    """
    path = ROOT / "eval" / "cases" / "scenarios.jsonl"
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _count_scenarios() -> int:
    """Сколько BDD-сценариев описано в feature-файлах.

    Returns:
        Суммарное число строк «Сценарий:».
    """
    total = 0
    for path in sorted((ROOT / "tests" / "features").glob("*.feature")):
        text = path.read_text(encoding="utf-8")
        total += sum(1 for line in text.splitlines() if re.match(r"\s*Сценарий:", line))
    return total


def _count_phases() -> int | None:
    """Сколько рабочих фаз объявлено в ``pf.state.Phase``.

    Returns:
        Число фаз без терминальной ``DONE`` либо ``None``, если модуль
        не импортируется.
    """
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from pf.state import Phase
    except ImportError:
        return None
    return len([phase for phase in Phase if phase.value != "done"])


def _default_num_ctx() -> int | None:
    """Значение окна контекста по умолчанию из настроек.

    Берётся именно умолчание, а не действующее значение: ``.env`` у
    проверяющего может содержать своё, и гейт не должен из-за этого
    краснеть.

    Returns:
        Умолчание ``ollama_num_ctx`` либо ``None``, если настройки
        не импортируются.
    """
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from pf.config import Settings
    except ImportError:
        return None
    default = Settings.model_fields["ollama_num_ctx"].default
    return int(default) if isinstance(default, int) else None


def _count_corpus_sections() -> int:
    """Сколько файлов-разделов лежит в снимке корпуса.

    Returns:
        Число JSON-файлов в ``data/wiki``.
    """
    wiki = ROOT / "data" / "wiki"
    if not wiki.is_dir():
        return 0
    return len(list(wiki.glob("*.json")))


def verify() -> list[str]:
    """Сверить числа канона с деревом репозитория.

    Returns:
        Список расхождений. Пустой список — канон верен.
    """
    problems: list[str] = []

    tests = _count_collected("not slow")
    if tests is None:
        problems.append("не удалось собрать тесты: pytest недоступен")
    elif tests != TESTS_DETERMINISTIC:
        problems.append(
            f"TESTS_DETERMINISTIC = {TESTS_DETERMINISTIC}, а pytest собирает {tests}"
        )

    live = _count_collected("slow")
    if live is not None and live != TESTS_LIVE:
        problems.append(f"TESTS_LIVE = {TESTS_LIVE}, а pytest собирает {live}")

    cases = _count_eval_cases()
    if cases != EVAL_CASES:
        problems.append(f"EVAL_CASES = {EVAL_CASES}, а в наборе {cases}")

    scenarios = _count_scenarios()
    if scenarios != SCENARIOS_BDD:
        problems.append(f"SCENARIOS_BDD = {SCENARIOS_BDD}, а описано {scenarios}")

    phases = _count_phases()
    if phases is not None and phases != WORKFLOW_PHASES:
        problems.append(f"WORKFLOW_PHASES = {WORKFLOW_PHASES}, а объявлено {phases}")

    num_ctx = _default_num_ctx()
    if num_ctx is not None and num_ctx != OLLAMA_NUM_CTX:
        problems.append(f"OLLAMA_NUM_CTX = {OLLAMA_NUM_CTX}, а умолчание {num_ctx}")

    sections = _count_corpus_sections()
    if sections and sections != CORPUS_SECTIONS:
        problems.append(f"CORPUS_SECTIONS = {CORPUS_SECTIONS}, а файлов {sections}")

    return problems


def main(argv: list[str] | None = None) -> int:
    """Точка входа самопроверки канона.

    Args:
        argv: Аргументы командной строки.

    Returns:
        0 — канон сходится с деревом; 1 — расхождения.
    """
    args = argv if argv is not None else sys.argv[1:]
    quiet = "--quiet" in args

    problems = verify()
    if not quiet:
        print(
            f"Канон: тесты {TESTS_DETERMINISTIC} + {TESTS_LIVE} живых, "
            f"кейсов eval {EVAL_CASES}, сценариев {SCENARIOS_BDD}, "
            f"фаз {WORKFLOW_PHASES}, окно {OLLAMA_NUM_CTX}"
        )
        print(
            f"Корпус: {CORPUS_ARTICLES} статей в {CORPUS_SECTIONS} разделах — "
            "число статей не пересчитывается (175 МБ JSON)"
        )

    if problems:
        print(f"[FAIL] canon: расхождений — {len(problems)}")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("[OK] canon: числа сходятся с деревом")
    return 0


if __name__ == "__main__":
    sys.exit(main())
