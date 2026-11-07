"""Проверки оснований вывода.

Стратегия и решение опираются на источники так же, как факты и расчёты:
у `Strategy.product.evidence`, `Strategy.communication.promise_grounding` и
`DecisionRecord.commitments[].basis` лежат те же ссылки на материалы.

Раньше проверка оснований смотрела только `facts` и `metrics`. Из-за
этого вывод «внедрить гибкие тарифные планы» мог опираться на
`project/data/cbr_rates.csv#row1` — файл, которого нет в материалах, —
и в отчёте это никак не проявлялось: у шага стратегии было ноль
отклонённых оснований и ноль подтверждённых.

Здесь проверяется, что основание вывода проходит **ту же** проверку:
дословность цитаты и релевантность источника задаче.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pf.ground import SupportStatus
from pf.schemas import (
    Commitment,
    CommunicationStrategy,
    DecisionRecord,
    ProductStrategy,
    SourceRef,
    Strategy,
    VerificationPlan,
)
from pf.sources import SourceBundle, collect
from pf.workflow import Workflow

#: Цитата, которая дословно есть в строке таблицы материалов теста.
REAL_QUOTE = "МТС, абонентская база: 87"


def _bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SourceBundle:
    """Собрать источники с одним материалом проекта.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.

    Returns:
        Набор источников, где материал проекта релевантен задаче.
    """
    monkeypatch.setattr("pf.config.settings.project_dir", tmp_path)
    (tmp_path / "numbers.csv").write_text("оператор,абонентская база\nМТС,87\n", encoding="utf-8")
    return collect("абоненты оператора связи тарифы удержание")


def _strategy(evidence: list[SourceRef]) -> Strategy:
    """Стратегия с заданными основаниями вывода.

    Args:
        evidence: Основания продуктовой стратегии.

    Returns:
        Заполненная стратегия.
    """
    return Strategy(
        product=ProductStrategy(
            audience="клиенты мобильного тарифа",
            chosen_direction="снизить цену на тариф",
            evidence=evidence,
        ),
        communication=CommunicationStrategy(
            audience="клиенты мобильного тарифа",
            barrier="цена выше, чем у альтернатив",
            key_message="мы держим цену под контролем",
        ),
        verification=VerificationPlan(
            expected_change="удержание вырастет на 2 п.п.",
            compare_against="уровень прошлого квартала",
            observation_period="3 месяца",
            adverse_effects=["маржа снизится"],
        ),
    )


def _decision(basis: list[SourceRef]) -> DecisionRecord:
    """Решение с заданными основаниями обязательства.

    Args:
        basis: Основания обязательства.

    Returns:
        Заполненное решение.
    """
    return DecisionRecord(
        decision="снизить цену на тариф",
        commitments=[
            Commitment(
                statement="удержание вырастет на 2 п.п.",
                owner="команда тарифа",
                metric="удержание клиентов",
                review_condition="если через 3 месяца прироста нет",
                review_by="2024-06-30",
                basis=basis,
            )
        ],
    )


def test_strategy_evidence_is_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Основание стратегии проверяется, а не принимается на веру.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.
    """
    bundle = _bundle(tmp_path, monkeypatch)
    strategy = _strategy(
        [SourceRef(source_id="project/data/nonexistent.csv#row1", quote="Ключевая ставка: 8.0")]
    )

    grounding = Workflow._verify_claims(strategy, bundle)

    assert grounding.rejected == 1
    assert any("nonexistent.csv" in detail for detail in grounding.details)


def test_keeper_commitment_basis_is_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Основание обязательства в решении проверяется так же, как факт.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.
    """
    bundle = _bundle(tmp_path, monkeypatch)
    decision = _decision(
        [SourceRef(source_id="project/data/nonexistent.csv#row1", quote="Ключевая ставка: 8.0")]
    )

    grounding = Workflow._verify_claims(decision, bundle)

    assert grounding.rejected == 1
    assert any("nonexistent.csv" in detail for detail in grounding.details)


def test_conclusion_on_phantom_source_is_not_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Вывод без подтверждённого основания не выдаётся за подкреплённый.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.
    """
    bundle = _bundle(tmp_path, monkeypatch)
    strategy = _strategy(
        [SourceRef(source_id="project/data/nonexistent.csv#row1", quote="Ключевая ставка: 8.0")]
    )

    grounding = Workflow._verify_claims(strategy, bundle)

    assert grounding.conclusion is not None
    assert grounding.conclusion["status"] == "неподкреплено"
    assert grounding.conclusion["confirmed"] == 0
    assert grounding.conclusion["checked"] == 1


def test_confirmed_conclusion_evidence_counts_as_grounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Настоящее основание вывода подтверждается — иначе правило бессмысленно.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.
    """
    bundle = _bundle(tmp_path, monkeypatch)
    strategy = _strategy([SourceRef(source_id="project/numbers.csv#row0", quote=REAL_QUOTE)])

    grounding = Workflow._verify_claims(strategy, bundle)

    assert grounding.conclusion is not None
    assert grounding.conclusion["status"] == "подкреплено"
    assert grounding.conclusion["confirmed"] == 1
    assert grounding.citations


def test_verdicts_are_kept_for_every_ground(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """По каждому основанию сохраняется вердикт, а не только счётчик.

    Замеру нужно сопоставлять вердикты по утверждению и источнику: «58
    против 58» не отвечает на вопрос, какие основания переехали из
    подтверждённых в отклонённые.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.
    """
    bundle = _bundle(tmp_path, monkeypatch)
    strategy = _strategy(
        [
            SourceRef(source_id="project/numbers.csv#row0", quote=REAL_QUOTE),
            SourceRef(source_id="project/numbers.csv#row0", quote="база выросла до 99,9 млн"),
        ]
    )

    grounding = Workflow._verify_claims(strategy, bundle)

    statuses = [verdict.status for verdict in grounding.verdicts]
    assert statuses == [SupportStatus.CONFIRMED, SupportStatus.NO_DATA]
    assert grounding.verdicts[0].is_conclusion is True
    assert grounding.verdicts[0].source_id == "project/numbers.csv#row0"
    # Утверждение — это вывод стратегии, а не отдельный факт: именно его
    # подменяет вердикт, и именно по нему сопоставляются версии.
    assert grounding.verdicts[0].claim == "снизить цену на тариф"


def test_facts_only_result_has_no_conclusion_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """У результата без вывода нет и блока «на чём держится».

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.
    """
    from pf.schemas import Analysis

    bundle = _bundle(tmp_path, monkeypatch)
    analysis = Analysis(
        facts=[], hypotheses=[], forecasts=[], gaps=["нет данных об удержании"], metrics=[]
    )

    grounding = Workflow._verify_claims(analysis, bundle)

    assert grounding.conclusion is None


def _external_state(quote_source: str) -> object:
    """Состояние сессии с одной внешней страницей.

    Args:
        quote_source: Текст страницы.

    Returns:
        Состояние сессии с найденной внешней страницей.
    """
    from types import SimpleNamespace

    from pf.state import Phase, SessionState

    state = SessionState(session_id="external", phase=Phase.STRATEGY)
    state.results["web_pages"] = [
        SimpleNamespace(
            title="Новости",
            url="https://example.com/news",
            text=quote_source,
        )
    ]
    return state


def test_external_source_is_not_rejected_but_waits_for_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ссылка на веб-источник — не выдумка, а основание на проверку.

    Раньше такая ссылка отклонялась как «источник не найден среди
    собранных»: страница честно скачана, но в набор локальных
    источников не входит. Отказ и «ждёт человека» — разные вещи.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена каталога материалов.
    """
    bundle = _bundle(tmp_path, monkeypatch)
    state = _external_state("МегаФон обслужил 77,26 миллиона абонентов в июне.")
    strategy = _strategy(
        [
            SourceRef(
                source_id="web/Новости",
                quote="МегаФон обслужил 77,26 миллиона абонентов",
            )
        ]
    )

    grounding = Workflow._verify_claims(strategy, bundle, state=state)

    assert grounding.needs_review
    assert grounding.rejected == 0
    assert grounding.conclusion is not None
    assert grounding.conclusion["status"] == "ждёт проверки"


def test_external_source_with_wrong_quote_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Цитаты, которой нет на странице, нет и в основании.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена каталога материалов.
    """
    bundle = _bundle(tmp_path, monkeypatch)
    state = _external_state("МегаФон обслужил 77,26 миллиона абонентов в июне.")
    strategy = _strategy(
        [
            SourceRef(
                source_id="web/Новости",
                quote="МегаФон обслужил 88,88 миллиона абонентов",
            )
        ]
    )

    grounding = Workflow._verify_claims(strategy, bundle, state=state)

    assert grounding.needs_review == []
    assert grounding.rejected == 1
    assert grounding.conclusion is not None
    assert grounding.conclusion["status"] == "неподкреплено"


def test_oov_words_from_claims_reach_the_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Нераспознанные слова шага попадают в очередь распознания.

    Args:
        tmp_path: Временный каталог лексикона.
        monkeypatch: Подмена настроек.
    """
    from pf import lexicon

    monkeypatch.setattr("pf.config.settings.lexicon_dir", tmp_path)

    Workflow._queue_oov(
        {"криптовалюта"},
        context="Криптовалюта растёт у абонентов",
        role="analyst",
        session="demo-v1",
    )

    records = lexicon.read_oov()
    assert records[0].token == "криптовалюта"
    assert records[0].sessions == ["demo-v1"]
    assert "Криптовалюта растёт" in records[0].contexts[0]
