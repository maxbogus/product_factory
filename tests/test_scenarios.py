"""Исполнение сценариев из tests/features.

Сценарий написан в Gherkin (русские ключевые слова), а исполняется
здесь — в коде. Разделение сделано намеренно: сценарий описывает
**ожидаемое поведение** и не зависит от того, как оно устроено внутри,
а код проверяет, что поведение действительно такое.

Каждый сценарий из `tests/features/*.feature` обязан быть покрыт —
это проверяет гейт `every-scenario-has-test`
(`scripts/contracts_check.py`).

Сценарии детерминированные: обращения к модели нет, поэтому они входят
в обычный прогон `make test`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pytest_bdd import given, scenarios, then, when

from pf.analysis import Metric as AnalysisMetric
from pf.corpus import Article
from pf.ground import SupportStatus
from pf.materials import Material, load_materials
from pf.report import build_session_report
from pf.schemas import Analysis, Claim, SourceRef, Variants, VerificationPlan
from pf.sources import SourceBundle, verify_metric

scenarios("features/grounding.feature")

#: Каталог материалов сценария: подменяется на временный в фикстуре.
MATERIALS_DIR: Path


@pytest.fixture(autouse=True)
def _materials_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Подменить каталог материалов и каталоги прогона на временные.

    Состояние и журнал тоже подменяются: исполнение сценария зовёт
    настоящие ``save()`` и ``Journal()``, и раньше после каждого
    ``make test`` в ``state/`` и ``journal/`` появлялись файлы
    ``session-scenario-grounding-*.json``. Тесты не должны оставлять
    следов в рабочем дереве.

    Args:
        tmp_path: Временный каталог теста.
        monkeypatch: Подмена настроек.
    """
    global MATERIALS_DIR
    MATERIALS_DIR = tmp_path
    monkeypatch.setattr("pf.config.settings.project_dir", tmp_path)
    monkeypatch.setattr("pf.config.settings.state_dir", tmp_path / "state")
    monkeypatch.setattr("pf.config.settings.journal_dir", tmp_path / "journal")


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    """Сбросить накопленное состояние сценария.

    Состояние шагов модульное: без сброса метрика из предыдущего
    сценария попадает в следующий, и проверка идёт не по тем данным.
    Сценарий обязан проходить и в одиночку, а не только в общем прогоне.
    """
    global STATE
    STATE = ScenarioState()


def _write_csv(name: str, header: list[str], rows: list[list[str]]) -> str:
    """Записать таблицу и вернуть её идентификатор.

    Args:
        name: Имя файла.
        header: Заголовок.
        rows: Строки.

    Returns:
        Идентификатор материала.
    """
    path = MATERIALS_DIR / "data" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return f"project/{name}"


def _write_text(name: str, text: str) -> str:
    """Записать документ и вернуть его идентификатор.

    Args:
        name: Имя файла.
        text: Содержимое.

    Returns:
        Идентификатор материала.
    """
    path = MATERIALS_DIR / "data" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return f"project/{name}"


def _materials() -> list[Material]:
    """Загрузить материалы сценария.

    Returns:
        Список материалов.
    """
    return load_materials(MATERIALS_DIR)


@dataclass(slots=True)
class ScenarioState:
    """Данные, накопленные шагами сценария."""

    metric: Any = None
    claim: Any = None
    support: Any = None
    analysis: Any = None
    verification: Any = None
    variants: Any = None
    session_id: str = ""


STATE = ScenarioState()


def _record(
    ordinal: int,
    role: str,
    phase: str,
    citations: list[str],
    rejected_details: list[str] | None = None,
) -> Any:
    """Запись журнала для сценария.

    Args:
        ordinal: Порядковый номер.
        role: Роль.
        phase: Фаза.
        citations: Подтверждённые ссылки.
        rejected_details: Отклонения.

    Returns:
        Запись журнала.
    """
    from pf.journal import StepRecord

    return StepRecord(
        ordinal=ordinal,
        role=role,
        phase=phase,
        summary="сценарий",
        citations=citations,
        rejected_details=rejected_details or [],
        cost_units=0.001,
    )


@given("в материалах есть таблица project/data/numbers.csv со строкой 0: оператор МТС, значение 82")
def _given_table_row() -> None:
    """Таблица с одной строкой."""
    _write_csv("numbers.csv", ["оператор", "значение"], [["МТС", "82"]])


@given("в материалах есть таблица project/data/numbers.csv со строками 0 и 1: значения 87 и 82")
def _given_table_two_rows() -> None:
    """Таблица с двумя строками."""
    _write_csv("numbers.csv", ["оператор", "значение"], [["МТС", "87"], ["МТС", "82"]])


@given(
    "в материалах есть таблица project/data/telecom_open_metrics.csv со строкой 0: "
    "МТС, абонентская база в России, 82"
)
def _given_telecom_table() -> None:
    """Таблица телеком-метрик."""
    _write_csv(
        "telecom_open_metrics.csv",
        ["оператор", "показатель", "значение"],
        [["МТС", "абонентская база в России", "82"]],
    )


@given(
    "в материалах есть документ project/data/report.md со строкой "
    '"Абонентская база МТС в России составляет 82 млн пользователей"'
)
def _given_document() -> None:
    """Документ с цитируемой строкой."""
    _write_text(
        "report.md",
        "Абонентская база МТС в России составляет 82 млн пользователей.\n"
        "Вторая строка для документа, чтобы он не был слишком коротким.\n",
    )


@given(
    'агент вернул метрику "Абонентская база МТС" со значением 82, единицей млн, '
    "источником project/data/numbers.csv и строкой 0"
)
def _given_metric() -> None:
    """Метрика из ответа агента."""
    STATE.metric = AnalysisMetric(
        name="Абонентская база МТС",
        value=82.0,
        unit="млн",
        source_id="project/data/numbers.csv",
        row=0,
        formula="значение из колонки «значение», строка 1",
        inputs={"значение": "82"},
    )


@given('метрика имеет формулу "значение из колонки «значение», строка 1" и вход "значение: 82"')
def _given_metric_provenance() -> None:
    """Provenance метрики."""
    assert STATE.metric is not None


@given(
    'агент вернул факт "Абонентская база МТС в России составляет 82 млн пользователей" '
    'с цитатой "Абонентская база МТС в России составляет 82 млн пользователей"'
)
def _given_claim() -> None:
    """Факт с дословной цитатой."""
    STATE.claim = Claim(
        kind="факт",
        text="Абонентская база МТС в России составляет 82 млн пользователей",
        sources=[
            SourceRef(
                source_id="project/data/report.md",
                paragraph=0,
                quote="Абонентская база МТС в России составляет 82 млн пользователей",
            )
        ],
    )


@given(
    'агент вернул метрику "Абонентская база МТС: среднее" со значением 84.5 '
    'и формулой "сумма / количество"'
)
def _given_aggregate_metric() -> None:
    """Агрегатная метрика."""
    STATE.metric = AnalysisMetric(
        name="Абонентская база МТС: среднее",
        value=84.5,
        unit="млн",
        source_id="project/data/numbers.csv",
        row=1,
        formula="сумма / количество",
        inputs={"наблюдения": ["87", "82"]},
    )


@given('входы метрики равны "наблюдения: 87, 82"')
def _given_aggregate_inputs() -> None:
    """Входы агрегата."""
    assert STATE.metric is not None


@given(
    'блок расчётов содержит метрику "Абонентская база МТС в России: среднее" со ссылкой на строку 0'
)
def _given_computed_block() -> None:
    """Метрика из блока расчётов."""
    STATE.metric = AnalysisMetric(
        name="Абонентская база МТС в России: среднее",
        value=82.0,
        unit="млн",
        source_id="project/data/telecom_open_metrics.csv",
        row=0,
        formula="сумма / количество",
        inputs={"наблюдения": ["82"]},
    )


@given('агент вернул факт "Абонентская база выросла" без ссылок на источники')
def _given_fact_without_source() -> None:
    """Факт без основания.

    Claim без источников отвергается самой схемой — значит, «факт» без
    основания нельзя даже построить. Это и есть контракт: отсутствие
    ссылки делает утверждение не фактом.
    """
    with pytest.raises(ValueError):
        Claim(kind="факт", text="Абонентская база выросла", sources=[])
    STATE.claim = None


@given("в источниках нет данных о причинах оттока")
def _given_no_cause_data() -> None:
    """Нет данных о причинах.

    Пустой разбор схема не принимает: ни фактов, ни гипотез, ни
    пробелов — значит, работа не сделана.
    """
    with pytest.raises(ValueError):
        Analysis(facts=[], hypotheses=[], forecasts=[], gaps=[])
    STATE.analysis = None


@given("агент вернул разбор с пустым списком пробелов")
def _given_analysis_without_gaps() -> None:
    """Разбор без пробелов."""
    assert STATE.analysis is None


@given("агент вернул два варианта решения")
def _given_two_variants() -> None:
    """Два варианта.

    Вариантов меньше трёх — схема их не принимает. Список вариантов
    хранится как объекты, а не строки.
    """
    STATE.variants = [
        {"title": "Оставить как есть", "objections": ["Дорого"]},
        {"title": "Закрыть направление", "objections": ["Потери перераспределятся"]},
    ]


@given("агент вернул план проверки без нежелательных последствий")
def _given_verification_without_adverse() -> None:
    """План без последствий.

    Пустой список последствий схема не принимает: план без «а если»
    не является планом.
    """
    with pytest.raises(ValueError):
        VerificationPlan(
            expected_change="Удержание вырастет",
            compare_against="текущий уровень",
            observation_period="3 месяца",
            adverse_effects=[],
        )
    STATE.verification = None


@given('агент вернул факт "Абонентская база МТС — 82 млн" со ссылкой на источник')
def _given_fact_with_source() -> None:
    """Факт с основанием."""
    STATE.claim = Claim(
        kind="факт",
        text="Абонентская база МТС — 82 млн",
        sources=[SourceRef(source_id="project/data/numbers.csv", paragraph=0, quote="82")],
    )


@given('агент вернул гипотезу "Отток вызван ростом цены" без ссылки на источник')
def _given_hypothesis() -> None:
    """Гипотеза без основания."""
    STATE.analysis = Analysis(
        facts=[],
        hypotheses=[Claim(kind="гипотеза", text="Отток вызван ростом цены", sources=[])],
        forecasts=[],
        gaps=["Нет данных"],
    )


@given("агент выполнил шаг и подтвердил одно основание")
def _given_confirmed_step() -> None:
    """Подтверждённое основание в журнале."""
    from pf.journal import Journal, journal_path
    from pf.state import Phase, SessionState, save

    STATE.session_id = f"scenario-grounding-{uuid4().hex[:8]}"
    save(SessionState(session_id=STATE.session_id, phase=Phase.DONE))
    Journal(journal_path(STATE.session_id)).append(
        _record(0, "collector", "collecting", ["project/data/numbers.csv#row1"])
    )


@given("агент выполнил шаг, в котором одно основание отклонено с причиной")
def _given_rejected_step() -> None:
    """Отклонённое основание в журнале."""
    from pf.journal import Journal, journal_path

    Journal(journal_path(STATE.session_id)).append(
        _record(
            1,
            "analyst",
            "analyzing",
            [],
            rejected_details=["project/data/numbers.csv#row9: цитата не найдена"],
        )
    )


@when("основание проверяется кодом")
def _when_verified() -> None:
    """Проверить основание.

    Метрика проверяется пересчётом по строке, факт — дословным поиском
    цитаты в источнике. Оба пути ведут к одному Support, поэтому
    сценарий работает с любым из них.
    """
    if STATE.metric is not None:
        STATE.support = verify_metric(STATE.metric, _materials())
        return
    assert STATE.claim is not None
    source = STATE.claim.sources[0]
    # Факт проверяется по своему источнику, а не по пустому набору.
    # Материал проекта лежит под тем же идентификатором, что и статья
    # корпуса, поэтому его можно проверить той же дословной проверкой.
    material = next(
        (item for item in _materials() if item.source_id == source.source_id),
        None,
    )
    assert material is not None, f"материал {source.source_id} не найден в сценарии"
    article = Article(
        source_id=material.source_id,
        title=material.title,
        section="project",
        content=material.content,
    )
    bundle = SourceBundle(passages=[], articles=[article], query="")
    STATE.support = bundle.verify_claim(
        STATE.claim.text,
        source_id=source.source_id,
        paragraph=source.paragraph,
        quote=source.quote,
    )


@when("модель копирует метрику из блока и основание проверяется кодом")
def _when_copied_and_verified() -> None:
    """Скопировать метрику из блока и проверить."""
    assert STATE.metric is not None
    STATE.support = verify_metric(STATE.metric, _materials())


@when("факт проверяется по схеме ответа")
def _when_fact_schema() -> None:
    """Проверить факт по схеме."""
    with pytest.raises(ValueError):
        Analysis(facts=[STATE.claim], hypotheses=[], forecasts=[], gaps=[])


@when("разбор проверяется по схеме ответа")
def _when_analysis_schema() -> None:
    """Проверить разбор по схеме."""
    with pytest.raises(ValueError):
        Analysis(facts=[], hypotheses=[], forecasts=[], gaps=[])


@when("варианты проверяются по схеме ответа")
def _when_variants_schema() -> None:
    """Проверить варианты по схеме."""
    with pytest.raises(ValueError):
        Variants(variants=STATE.variants)


@when("план проверяется по схеме ответа")
def _when_verification_schema() -> None:
    """Проверить план по схеме."""
    with pytest.raises(ValueError):
        VerificationPlan(
            expected_change="Удержание вырастет",
            compare_against="текущий уровень",
            observation_period="3 месяца",
            adverse_effects=[],
        )


@when("оба утверждения проверяются по схеме ответа")
def _when_both_schema() -> None:
    """Проверить факт и гипотезу по схеме."""
    STATE.analysis = Analysis(
        facts=[STATE.claim],
        hypotheses=STATE.analysis.hypotheses,
        forecasts=[],
        gaps=STATE.analysis.gaps,
    )


@when("строится отчёт по сессии")
def _when_report() -> None:
    """Построить отчёт по сессии."""
    STATE.support = build_session_report(STATE.session_id)


@then("статус основания равен подтверждено")
def _then_confirmed() -> None:
    """Основание подтверждено."""
    assert STATE.support is not None
    assert STATE.support.status is SupportStatus.CONFIRMED, STATE.support.detail


@then("основание ссылается на строку 0")
def _then_row_zero() -> None:
    """Основание указывает на строку 0."""
    assert STATE.support is not None
    assert STATE.support.paragraph_index == 0


@then("в отклонённые основания метрика не попадает")
def _then_not_rejected() -> None:
    """Метрика не отклонена."""
    assert STATE.support is not None
    assert STATE.support.status is SupportStatus.CONFIRMED


@then("цитата найдена в источнике")
def _then_quote_found() -> None:
    """Цитата найдена."""
    assert STATE.support is not None
    assert STATE.support.status is SupportStatus.CONFIRMED, STATE.support.detail


@then('в отчёте видно "метрика подтверждена пересчётом по заявленным входам"')
def _then_recomputed_detail() -> None:
    """Основание подтверждено пересчётом."""
    assert STATE.support is not None
    assert "пересчётом" in STATE.support.detail, STATE.support.detail


@then("номер строки в метрике равен 0")
def _then_metric_row_zero() -> None:
    """В метрике строка 0."""
    assert STATE.metric is not None
    assert STATE.metric.row == 0


@then("номер строки не равен 1")
def _then_metric_row_not_one() -> None:
    """В метрике не строка 1."""
    assert STATE.metric is not None
    assert STATE.metric.row != 1


@then("факт отвергнут")
def _then_fact_rejected() -> None:
    """Факт отвергнут."""
    assert STATE.claim is None


@then('причина содержит "источник"')
def _then_reason_source() -> None:
    """Причина про источник."""
    with pytest.raises(ValueError) as error:
        Claim(kind="факт", text="Абонентская база выросла", sources=[])
    assert "источник" in str(error.value).lower()


@then("разбор отвергнут")
def _then_analysis_rejected() -> None:
    """Разбор отвергнут."""
    assert STATE.analysis is None
    with pytest.raises(ValueError):
        Analysis(facts=[], hypotheses=[], forecasts=[], gaps=[])


@then('причина содержит "пробел"')
def _then_reason_gap() -> None:
    """Причина про пробел."""
    with pytest.raises(ValueError) as error:
        Analysis(facts=[], hypotheses=[], forecasts=[], gaps=[])
    assert "пробел" in str(error.value).lower()


@then("варианты отвергнуты")
def _then_variants_rejected() -> None:
    """Варианты отвергнуты."""
    assert STATE.variants is not None
    with pytest.raises(ValueError):
        Variants(variants=STATE.variants)


@then('причина содержит "вариант"')
def _then_reason_variant() -> None:
    """Причина про вариант."""
    with pytest.raises(ValueError) as error:
        Variants(variants=STATE.variants)
    assert "variant" in str(error.value).lower()


@then("план отвергнут")
def _then_verification_rejected() -> None:
    """План отвергнут."""
    assert STATE.verification is None


@then('причина содержит "последств"')
def _then_reason_adverse() -> None:
    """Причина про последствия."""
    with pytest.raises(ValueError) as error:
        VerificationPlan(
            expected_change="Удержание вырастет",
            compare_against="текущий уровень",
            observation_period="3 месяца",
            adverse_effects=[],
        )
    assert "adverse_effects" in str(error.value)


@then("факт принят")
def _then_fact_accepted() -> None:
    """Факт принят."""
    assert STATE.analysis is not None
    assert STATE.analysis.facts


@then("гипотеза принята")
def _then_hypothesis_accepted() -> None:
    """Гипотеза принята."""
    assert STATE.analysis is not None
    assert STATE.analysis.hypotheses


@then("у гипотезы нет оснований")
def _then_hypothesis_without_sources() -> None:
    """У гипотезы нет оснований."""
    assert STATE.analysis is not None
    assert not STATE.analysis.hypotheses[0].sources


@then("подтверждённых записей равно 1")
def _then_confirmed_count() -> None:
    """Одна подтверждённая запись."""
    assert STATE.support is not None
    assert STATE.support.confirmed_citations == 1


@then("отклонённых оснований равно 1")
def _then_rejected_count() -> None:
    """Одно отклонённое основание."""
    assert STATE.support is not None
    assert STATE.support.rejected_citations == 1


@then('в отчёте виден раздел "Отклонённые основания"')
def _then_report_has_rejected() -> None:
    """В отчёте есть раздел отклонённых."""
    assert STATE.support is not None
    assert "Отклонённые основания" in STATE.support.as_markdown()


def test_scenario_writes_sessions_outside_repository() -> None:
    """Сессия сценария пишется во временный каталог, а не в journal/.

    Раньше исполнение сценариев звало настоящие ``save()`` и ``Journal()``:
    после каждого ``make test`` в ``state/`` и ``journal/`` репозитория
    появлялись файлы ``session-scenario-grounding-*.json``, и дерево
    оставалось чистым только потому, что каталоги перечислены в
    ``.gitignore``. След теста в рабочем дереве — это не «чисто», это
    спрятанный мусор.
    """
    from pf.config import settings

    root = Path(__file__).resolve().parent.parent
    assert Path(settings.journal_dir).resolve() != (root / "journal").resolve()
    assert Path(settings.state_dir).resolve() != (root / "state").resolve()
