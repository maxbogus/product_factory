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

from pf.ground import (
    CONCLUSION_RULE,
    CONCLUSION_RULE_PARTS,
    GroundTally,
    SupportStatus,
    conclusion_status,
)
from pf.schemas import (
    Analysis,
    Commitment,
    CommunicationStrategy,
    DecisionRecord,
    MetricRef,
    ProductStrategy,
    SourceRef,
    Strategy,
    VerificationPlan,
)
from pf.sources import SourceBundle, collect
from pf.state import Phase, SessionState
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


def _strategy(evidence: list[SourceRef], direction: str = "снизить цену на тариф") -> Strategy:
    """Стратегия с заданными основаниями вывода.

    Args:
        evidence: Основания продуктовой стратегии.
        direction: Выбранное направление — по нему проверяется основание.

    Returns:
        Заполненная стратегия.
    """
    return Strategy(
        product=ProductStrategy(
            audience="клиенты мобильного тарифа",
            chosen_direction=direction,
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


def test_rejected_metric_on_same_anchor_weakens_conclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Якорь, на котором расчёт отклонён, не подкрепляет вывод.

    В прогоне `demo-v9` обе метрики были отклонены («пересчёт по входам
    даёт 0, а заявлено 87 тыс.»), а вывод при этом назывался
    «подкреплено»: он опирался на те же две строки таблицы. Один якорь не
    может быть одновременно непригодным для расчёта и достаточным для
    решения.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.
    """
    bundle = _bundle(tmp_path, monkeypatch)
    state = SessionState(session_id="metric-conflict", phase=Phase.STRATEGY)
    state.results["rejected_metric_keys"] = ["project/numbers.csv#row0"]
    strategy = _strategy([SourceRef(source_id="project/numbers.csv#row0", quote=REAL_QUOTE)])

    grounding = Workflow._verify_claims(strategy, bundle, state=state)

    assert grounding.conclusion is not None
    assert grounding.conclusion["status"] == "неподкреплено"
    assert grounding.conclusion["confirmed"] == 0


def test_rejected_metric_on_other_anchor_keeps_conclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Отказ на другом якоре вывод не гасит.

    Проверка сравнивает якоря, а не «есть ли в прогоне хоть один отказ»:
    иначе одна неудачная строка обнуляла бы весь вывод.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.
    """
    bundle = _bundle(tmp_path, monkeypatch)
    state = SessionState(session_id="metric-other", phase=Phase.STRATEGY)
    state.results["rejected_metric_keys"] = ["project/numbers.csv#row9"]
    strategy = _strategy([SourceRef(source_id="project/numbers.csv#row0", quote=REAL_QUOTE)])

    grounding = Workflow._verify_claims(strategy, bundle, state=state)

    assert grounding.conclusion is not None
    assert grounding.conclusion["status"] == "подкреплено"


def test_rejected_metric_anchor_is_remembered_in_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Отклонённый расчёт запоминается по якорю — для следующей фазы.

    Вывод проверяется позже разбора, поэтому решение «этому якорю верить
    нельзя» должно переехать между шагами, а не остаться в шаге разбора.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.
    """
    bundle = _bundle(tmp_path, monkeypatch)
    state = SessionState(session_id="metric-memory", phase=Phase.ANALYZING)
    analysis = Analysis(
        metrics=[
            MetricRef(
                name="Абонентская база",
                value=50.0,
                unit="млн",
                source_id="project/numbers.csv#row0",
                formula="(87 + 111) / 2",
                inputs={"first": "87", "second": "111"},
            )
        ]
    )

    grounding = Workflow._verify_claims(analysis, bundle, state=state)

    assert grounding.rejected == 1
    assert state.results["rejected_metric_keys"] == ["project/numbers.csv#row0"]


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


def test_conclusion_rule_is_named_in_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Статус вывода записывается вместе с именем правила.

    Без имени унаследованный статус не отличить от текущего: `demo-v9`
    печатал «подкреплено» по правилу, которое считало посчитанное значение в
    строке расчёта цитатой, и делал это уже после того, как правило сняли.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.
    """
    bundle = _bundle(tmp_path, monkeypatch)
    strategy = _strategy([SourceRef(source_id="project/numbers.csv#row0", quote=REAL_QUOTE)])

    grounding = Workflow._verify_claims(strategy, bundle)

    assert grounding.conclusion is not None
    assert grounding.conclusion["rule"] == CONCLUSION_RULE
    # Части правила перечислены буквально, а не выведены из самой константы:
    # сравнение `CONCLUSION_RULE.split("+")` с `CONCLUSION_RULE_PARTS` зелено
    # при любом составе и не заметило бы, что часть из правила выпала.
    assert list(CONCLUSION_RULE_PARTS) == [
        "unique-grounds",
        "calc-line-demotion",
        "indicator-demotion",
        "unsupported-number-cap",
    ]


def test_off_topic_demotes_status() -> None:
    """Чужое основание понижается счётом, а не отдельным признаком статуса.

    Раньше `conclusion_status` держал вход `off_topic`, и статус считался
    дважды: в рабочем процессе и в отчёте. Второй вход разошёлся бы с первым
    на первом же прогоне, где понижено не всё основание, а одно из двух
    (`D-46`). Теперь чужое основание понижается там, где собираются вердикты,
    и статус остаётся функцией трёх чисел.
    """
    assert conclusion_status(1, 0, 1) == "неподкреплено"
    assert conclusion_status(2, 1, 1) == "подкреплено частично"
    with pytest.raises(TypeError):
        conclusion_status(1, 1, 0, off_topic=True)  # type: ignore[call-arg]


def test_unsupported_number_caps_status() -> None:
    """Число плана без своего расчёта — потолок статуса, а не заметка.

    `_check_plan_numbers` уже называет такую величину в `rejected_details`,
    но на статус вывода это не влияло: в `demo-v9` «удержание на уровне
    87 %» (87 — абонентская база МТС) стояло рядом со статусом
    «подкреплено». Потолок получает `conclusion_status`, чтобы правило
    счёта оставалось одним и было видно в имени правила.

    Потолок умеет только понижать: «неподкреплено» и «ждёт проверки» он не
    поднимает — иначе это была бы подмена статуса, а не ограничение.
    """
    assert conclusion_status(2, 2, 0) == "подкреплено"
    assert conclusion_status(2, 2, 0, unsupported_numbers=1) == "подкреплено частично"
    assert conclusion_status(1, 0, 0, unsupported_numbers=1) == "неподкреплено"
    assert conclusion_status(1, 0, 0, needs_review=1, unsupported_numbers=1) == "ждёт проверки"


def test_unsupported_number_is_written_to_journal() -> None:
    """Причина потолка едет в журнал вместе со статусом.

    Без ключа читатель видит «подкреплено частично» и не знает, почему:
    причина осталась бы только в `rejected_details` шага, а статус вывода
    живёт отдельной записью. Ноль пишется тоже — пропуск ключа читался бы
    как «не проверяли».
    """
    tally = GroundTally(checked=2, confirmed=2, rejected=0, needs_review=0)

    plain = Workflow._conclusion_verdict(tally, {})

    assert plain is not None
    assert plain["unsupported_numbers"] == 0
    assert plain["status"] == "подкреплено"

    capped = Workflow._conclusion_verdict(tally, {}, unsupported_numbers=1)

    assert capped is not None
    assert capped["unsupported_numbers"] == 1
    assert capped["status"] == "подкреплено частично"
    assert "unsupported-number-cap" in capped["rule"].split("+")


def test_conclusion_counts_unique_grounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Повторная ссылка на тот же источник — одно основание, а не два.

    В прогоне `demo-v9` стратегия и фиксация решения объявляли по четыре
    основания вывода, и все вели на две строки таблицы: `#row0` и `#row1`
    по два раза. В журнале стояло «проверено 4, подтверждено 4», отчёт
    складывал шаги и печатал «проверено 8, подтверждено 8», а раздел «Что
    измерено» того же документа — «уникальных оснований 2». Документ
    спорил сам с собой: доля выглядела лучше, чем была.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.
    """
    bundle = _bundle(tmp_path, monkeypatch)
    strategy = _strategy(
        [
            SourceRef(source_id="project/numbers.csv#row0", quote=REAL_QUOTE),
            SourceRef(source_id="project/numbers.csv#row0", quote=REAL_QUOTE),
            SourceRef(source_id="project/numbers.csv#row0", quote=REAL_QUOTE),
            SourceRef(source_id="project/numbers.csv#row0", quote=REAL_QUOTE),
        ]
    )

    grounding = Workflow._verify_claims(strategy, bundle)

    assert grounding.conclusion is not None
    assert grounding.conclusion["checked"] == 1
    assert grounding.conclusion["confirmed"] == 1
    assert grounding.conclusion["status"] == "подкреплено"


def test_repeated_ground_with_single_rejection_is_not_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Повтор подтверждённой цитаты не перекрывает отказ по той же строке.

    Основание — это строка источника. Если по ней прошли и подтверждение,
    и выдумка, основание считается отклонённым: иначе достаточно повторить
    верную цитату, чтобы рядом стоящая выдумка перестала быть видна.

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

    assert grounding.conclusion is not None
    assert grounding.conclusion["checked"] == 1
    assert grounding.conclusion["confirmed"] == 0
    assert grounding.conclusion["rejected"] == 1
    assert grounding.conclusion["status"] == "неподкреплено"


def test_conclusion_grounds_are_named_in_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Шаг называет основания вывода ключами, а не только счётчиками.

    Отчёт собирает вывод из двух шагов — стратегии и фиксации решения, —
    и ссылки у них одни и те же. Без ключей в журнале отчёт не может
    отличить повтор от нового основания и складывает записи.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.
    """
    bundle = _bundle(tmp_path, monkeypatch)
    strategy = _strategy(
        [
            SourceRef(source_id="project/numbers.csv#row0", quote=REAL_QUOTE),
            SourceRef(source_id="project/numbers.csv#row0", quote=REAL_QUOTE),
        ]
    )

    grounding = Workflow._verify_claims(strategy, bundle)

    assert grounding.conclusion is not None
    grounds = grounding.conclusion["grounds"]
    assert grounds["confirmed"] == ["project/numbers.csv#row0"]
    assert grounds["rejected"] == []
    assert grounds["needs_review"] == []


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


# ── Основание не про тот показатель ─────────────────────────────────────────
#
# Дословность цитаты не говорит, о **чём** она. В прогоне `cloud-deepseek-2`
# вывод был про удержание, а держался на статье об операторе и на уровне ARPU;
# в `cloud-deepseek-baseline-1` — на абонентской базе. Проверка этого не
# видела: она подтверждала цитату, а не отвечала на вопрос, тот ли показатель
# подтверждён.
#
# Показатель есть только у основания-строки таблицы (колонка «показатель»).
# У абзаца статьи его нет, и правило про такие основания молчит: иначе оно
# срабатывало бы на каждом выводе и превратилось бы в шум.

#: Таблица с колонкой «показатель»: основание про абонентскую базу.
BASE_ROW_QUOTE = "показатель: абонентская база, значение: 87"

#: Таблица с колонкой «показатель»: основание про удержание.
RETENTION_ROW_QUOTE = "показатель: удержание клиентов, значение: 82"


def _indicator_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SourceBundle:
    """Собрать источники с двумя таблицами, у которых назван показатель.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.

    Returns:
        Набор источников с основаниями «абонентская база» и «удержание».
    """
    monkeypatch.setattr("pf.config.settings.project_dir", tmp_path)
    (tmp_path / "base.csv").write_text(
        "показатель,значение\nабонентская база,87\n", encoding="utf-8"
    )
    (tmp_path / "retention.csv").write_text(
        "показатель,значение\nудержание клиентов,82\n", encoding="utf-8"
    )
    return collect("абоненты оператора связи тарифы удержание")


def test_off_topic_demotes_confirmed_ground(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Основание про другой показатель не считается подтверждённым.

    Вывод обещает изменение удержания, а основание подтверждает абонентскую
    базу. Цитата дословна — и всё равно она не основание вывода: подтверждён
    не тот показатель. Потолка «подкреплено частично» мало: при единственном
    основании статус обязан быть «неподкреплено», иначе счётчик подтверждённых
    выдаёт чужое основание за опору (`T-57`).

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.
    """
    bundle = _indicator_bundle(tmp_path, monkeypatch)
    strategy = _strategy(
        [SourceRef(source_id="project/base.csv#row0", quote=BASE_ROW_QUOTE)],
        direction="абонентская база вырастет",
    )

    grounding = Workflow._verify_claims(strategy, bundle)

    assert grounding.conclusion is not None
    assert grounding.conclusion["confirmed"] == 0
    assert grounding.conclusion["rejected"] == 1
    assert grounding.conclusion["status"] == "неподкреплено"
    assert grounding.conclusion["grounds"]["confirmed"] == []
    assert grounding.conclusion["grounds"]["rejected"] == ["project/base.csv#row0"]
    off_topic = grounding.conclusion["off_topic"]
    assert off_topic["grounds"] == ["абонентская база"]
    assert any("удержание" in item for item in off_topic["conclusion"])


def test_off_topic_demotes_only_foreign_ground(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Понижается только чужое основание, а своё остаётся подтверждённым.

    Правило «всё или ничего» ошибалось в обе стороны: одно основание про
    удержание и другое про абонентскую базу давали молчание, хотя половина
    опоры вывода не про то. Понижение считается по каждому основанию.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.
    """
    bundle = _indicator_bundle(tmp_path, monkeypatch)
    strategy = _strategy(
        [
            SourceRef(source_id="project/base.csv#row0", quote=BASE_ROW_QUOTE),
            SourceRef(source_id="project/retention.csv#row0", quote=RETENTION_ROW_QUOTE),
        ],
        # Утверждение называет оба показателя: иначе проверка цитаты отклонит
        # основание до правила о показателе, и чужое основание не дойдёт до
        # понижения. Показатель вывода берётся из ожидаемого изменения.
        direction="абонентская база и удержание клиентов вырастут",
    )

    grounding = Workflow._verify_claims(strategy, bundle)

    assert grounding.conclusion is not None
    assert grounding.conclusion["checked"] == 2
    assert grounding.conclusion["confirmed"] == 1
    assert grounding.conclusion["rejected"] == 1
    assert grounding.conclusion["status"] == "подкреплено частично"
    assert grounding.conclusion["grounds"]["confirmed"] == ["project/retention.csv#row0"]
    assert grounding.conclusion["grounds"]["rejected"] == ["project/base.csv#row0"]
    assert grounding.conclusion["off_topic"]["grounds"] == ["абонентская база"]


def test_ground_about_same_indicator_keeps_conclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Основание про тот же показатель оставляет вывод подкреплённым.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.
    """
    bundle = _indicator_bundle(tmp_path, monkeypatch)
    strategy = _strategy(
        [SourceRef(source_id="project/retention.csv#row0", quote=RETENTION_ROW_QUOTE)],
        direction="удержание клиентов вырастет",
    )

    grounding = Workflow._verify_claims(strategy, bundle)

    assert grounding.conclusion is not None
    assert grounding.conclusion["status"] == "подкреплено"
    assert "off_topic" not in grounding.conclusion


def test_ground_anchor_from_paragraph_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Номер строки берётся из поля ``paragraph``, если его нет в ссылке.

    Модель часто присылает ``source_id`` путём, а номер строки — отдельным
    полем. В журнале от такого основания остаётся только путь, и без склейки
    двух полей показатель строки не найти.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.
    """
    bundle = _indicator_bundle(tmp_path, monkeypatch)
    strategy = _strategy(
        [SourceRef(source_id="project/base.csv", paragraph=0, quote=BASE_ROW_QUOTE)],
        direction="абонентская база вырастет",
    )

    grounding = Workflow._verify_claims(strategy, bundle)

    assert grounding.conclusion is not None
    assert grounding.conclusion["confirmed"] == 0
    assert grounding.conclusion["status"] == "неподкреплено"
    assert grounding.conclusion["off_topic"]["grounds"] == ["абонентская база"]


def test_ground_without_named_indicator_is_not_judged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """У основания без названного показателя правило молчит.

    В таблице теста нет колонки «показатель», и у абзацев статей её тоже
    нет. Объявлять такой вывод неподкреплённым значило бы наказывать за
    отсутствие разметки, а не за подмену показателя.

    Args:
        tmp_path: Временный каталог материалов.
        monkeypatch: Подмена настроек.
    """
    bundle = _bundle(tmp_path, monkeypatch)
    strategy = _strategy([SourceRef(source_id="project/numbers.csv#row0", quote=REAL_QUOTE)])

    grounding = Workflow._verify_claims(strategy, bundle)

    assert grounding.conclusion is not None
    assert grounding.conclusion["status"] == "подкреплено"
    assert "off_topic" not in grounding.conclusion


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
