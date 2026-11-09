"""Проверки харнесса A/B: `scripts/replay_verification.py`.

Харнесс — инструмент замера, а не украшение: если он сломается молча, следующий
A/B покажет «ничего не изменилось» независимо от правок. Поэтому проверяется
три вещи:

1. контрольные пробы совпадают с ожиданиями проверки (регресс виден сразу);
2. повтор по сохранённому состоянию считает исходы правильно — выдумка
   отклоняется, честная цитата подтверждается;
3. замер не оставляет следов: очередь распознания не пополняется, а сессия без
   состояния называется прямо, а не показывается нулём.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from pf.config import settings
from pf.corpus import Article
from pf.sources import SourceBundle
from pf.state import Phase, SessionState, save
from pf.workflow import Workflow

ROOT = Path(__file__).resolve().parent.parent

#: Текст источника: честная цитата и число, которого в нём нет.
SOURCE_TEXT = "По состоянию на июнь 2024 года компания обслужила 77,26 миллиона абонентов."
REAL_QUOTE = "обслужила 77,26 миллиона абонентов"


def _harness() -> ModuleType:
    """Загрузить харнесс как модуль.

    Returns:
        Модуль `scripts/replay_verification.py`.
    """
    path = ROOT / "scripts" / "replay_verification.py"
    spec = importlib.util.spec_from_file_location("replay_verification", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_probes_match_expectations() -> None:
    """Каждая контрольная проба даёт ожидаемый вердикт."""
    results = _harness().check_probes()

    assert len(results) == 8
    failed = [item for item in results if not item["passed"]]
    assert failed == [], f"пробы не сошлись: {failed}"


def _analysis(facts: list[dict[str, object]]) -> dict[str, object]:
    """Результат роли «аналитик» в том виде, в каком он лежит в состоянии.

    Args:
        facts: Факты с основаниями.

    Returns:
        Словарь результата, пригодный для `pf.roles.validate`.
    """
    return {
        "facts": facts,
        "hypotheses": [],
        "forecasts": [],
        "gaps": ["нет данных об удержании"],
        "metrics": [],
    }


@pytest.fixture
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Изолировать состояние, журнал и словарь; источники — один материал.

    Args:
        tmp_path: Временный каталог.
        monkeypatch: Подмена настроек и сборки источников.

    Returns:
        Корень временного каталога.
    """
    monkeypatch.setattr(settings, "state_dir", tmp_path / "state")
    monkeypatch.setattr(settings, "journal_dir", tmp_path / "journal")
    monkeypatch.setattr(settings, "lexicon_dir", tmp_path / "lexicon")
    # Материалы проекта — свои: сверка чисел плана считает расчёты по ним, и
    # замер не должен зависеть от содержимого репозитория.
    project = tmp_path / "project"
    project.mkdir()
    (project / "base.csv").write_text(
        "показатель,значение\nабонентская база,87\n", encoding="utf-8"
    )
    monkeypatch.setattr(settings, "project_dir", project)
    article = Article(
        source_id="project/data/base.csv",
        title="base.csv",
        section="project",
        content=SOURCE_TEXT,
    )
    bundle = SourceBundle(passages=[], articles=[article], query="удержание абонентов")
    monkeypatch.setattr(Workflow, "collect_sources", lambda self, state: bundle)
    return tmp_path


def _save_session(session_id: str) -> None:
    """Сохранить сессию с честной цитатой и выдуманным числом.

    Args:
        session_id: Идентификатор сессии.
    """
    state = SessionState(
        session_id=session_id,
        phase=Phase.DONE,
        question="Почему падает удержание у мобильного тарифа?",
    )
    state.results[Phase.ANALYZING.value] = _analysis(
        [
            {
                "kind": "факт",
                "text": "Компания обслужила 77,26 миллиона абонентов",
                "sources": [{"source_id": "project/data/base.csv", "quote": REAL_QUOTE}],
            },
            {
                "kind": "факт",
                "text": "Компания обслужила 99,9 миллиона абонентов",
                "sources": [
                    {"source_id": "project/data/base.csv", "quote": "обслужила 99,9 миллиона"}
                ],
            },
        ]
    )
    save(state)


def test_replay_counts_confirmed_and_fabricated(_isolated: Path) -> None:
    """Повтор различает честную цитату и выдуманное число.

    Args:
        _isolated: Изоляция состояния и источников.
    """
    _save_session("replay-test")

    payload = _harness().replay_session("replay-test")

    assert payload is not None
    assert payload["kinds"]["подтверждено"] == 1
    assert payload["kinds"]["нет данных"] == 1
    assert {verdict["status"] for verdict in payload["verdicts"]} == {"confirmed", "no_data"}


def test_replay_does_not_touch_lexicon_queue(_isolated: Path) -> None:
    """Замер не пополняет очередь распознания.

    Args:
        _isolated: Изоляция состояния и словаря.
    """
    _save_session("replay-quiet")

    _harness().replay_session("replay-quiet")

    assert not (_isolated / "lexicon" / "oov.jsonl").exists()


def test_missing_session_is_reported(_isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Сессия без состояния называется прямо, а не показывается нулём.

    Args:
        _isolated: Изоляция состояния.
        capsys: Перехват вывода.
    """
    code = _harness().main(["--sessions", "no-such-session"])

    assert code == 1
    captured = capsys.readouterr()
    assert "no-such-session" in captured.out
    assert "повторять нечего" in captured.out.lower()


def test_compare_detects_flipped_verdict(_isolated: Path) -> None:
    """Сравнение снимков находит перевёрнутый вердикт.

    Это и есть смысл харнесса: «58 против 58» не отвечает на вопрос, какие
    основания переехали.

    Args:
        _isolated: Изоляция состояния (нужна для единообразия с другими тестами).
    """
    module = _harness()
    verdict = {
        "session": "demo-v7",
        "role": "collector",
        "claim": "Компания обслужила 77,26 миллиона абонентов",
        "source_id": "project/data/base.csv",
        "status": "confirmed",
    }
    flipped = dict(verdict, status="no_data")

    changed = module._flips({"verdicts": [verdict]}, {"verdicts": [flipped]})

    assert len(changed) == 1
    assert "confirmed" in changed[0]
    assert "no_data" in changed[0]


def _save_strategy_session(session_id: str, compare_against: str) -> None:
    """Сохранить сессию, дошедшую до стратегии.

    Args:
        session_id: Идентификатор сессии.
        compare_against: База сравнения плана проверки — текст с числом.
    """
    from pf.schemas import CommunicationStrategy, ProductStrategy, Strategy, VerificationPlan

    state = SessionState(
        session_id=session_id,
        phase=Phase.DONE,
        question="Почему падает удержание у мобильного тарифа?",
    )
    strategy = Strategy(
        product=ProductStrategy(
            audience="клиенты тарифа",
            chosen_direction="снизить цену",
            evidence=[{"source_id": "project/data/base.csv", "quote": REAL_QUOTE}],
        ),
        communication=CommunicationStrategy(
            audience="клиенты тарифа",
            barrier="цена выше альтернатив",
            key_message="мы держим цену под контролем",
        ),
        verification=VerificationPlan(
            expected_change="Удержание клиентов вырастет на 2 п.п.",
            compare_against=compare_against,
            observation_period="3 месяца",
            adverse_effects=["маржа снизится"],
            success_criteria=["Удержание клиентов вырастет на 2 п.п."],
        ),
    )
    state.results[Phase.STRATEGY.value] = Workflow._serialize(strategy)
    save(state)


def test_replay_reports_plan_number_cap(_isolated: Path) -> None:
    """Замер показывает число-факт без расчёта и статус, который оно понизило.

    Без этого блока `T-48b` нечем проверить: «статус стал частичным» ничего не
    говорит о том, сколько величин его понизили и была ли среди них цель плана.

    Args:
        _isolated: Изоляция состояния и источников.
    """
    _save_strategy_session("replay-cap", "Удержание на уровне 87%")

    payload = _harness().replay_session("replay-cap")

    assert payload is not None
    counts = payload["plan_numbers"]["roles"]["strategy"]
    # Три замечания: база сравнения (факт) и две цели — ожидаемое изменение и
    # критерий. Понижает статус только факт.
    assert counts["notes"] == 3
    assert counts["unsupported"] == 1
    assert counts["kinds"] == {"foreign_indicator": 1, "no_calculation": 2}
    assert payload["plan_numbers"]["unsupported"] == 1


def test_replay_plan_goal_is_not_counted(_isolated: Path) -> None:
    """Цель плана попадает в замечания, но не в понижающие величины.

    Args:
        _isolated: Изоляция состояния и источников.
    """
    _save_strategy_session("replay-goal", "уровень прошлого квартала")

    payload = _harness().replay_session("replay-goal")

    assert payload is not None
    counts = payload["plan_numbers"]["roles"]["strategy"]
    assert counts["notes"] >= 1
    assert counts["unsupported"] == 0
