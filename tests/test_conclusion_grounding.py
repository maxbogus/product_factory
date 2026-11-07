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
