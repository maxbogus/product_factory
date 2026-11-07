"""Проверки сбора источников.

Здесь проверяется то, что делает агента честным: подтверждённое основание
должно быть и дословным, и относящимся к задаче.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pf.corpus import Article
from pf.materials import load_materials
from pf.sources import SourceBundle, collect, score_passages


def _article(source_id: str, title: str, content: str) -> Article:
    return Article(source_id=source_id, title=title, section="test", content=content)


def test_passage_keeps_source_identity() -> None:
    """Абзац помнит, откуда он взят."""
    article = _article(
        "security/Тест",
        "Тест",
        "Первый абзац статьи достаточно длинный, чтобы его можно было процитировать.\n"
        "Второй абзац тоже достаточно длинный для цитирования в тестах агента.",
    )
    passages = score_passages(article, {"первый", "абзац"}, max_paragraphs=5)
    assert passages
    assert passages[0].source_id == "security/Тест"
    assert passages[0].paragraph_index == 0
    assert passages[0].citation == "security/Тест#p0"


def test_short_paragraphs_are_skipped() -> None:
    """Слишком короткий абзац процитировать нечем — он не берётся."""
    article = _article("security/Тест", "Тест", "Коротко.\n" + "Длинный абзац " * 20)
    passages = score_passages(article, {"длинный"}, max_paragraphs=5)
    assert all(len(p.text) >= 60 for p in passages)


def test_bundle_reports_relevance_of_sources() -> None:
    """Источник из отобранного среза релевантен, чужой — нет."""
    article = _article("security/Тест", "Тест", "Абзац " * 40)
    bundle = SourceBundle(
        passages=[],
        articles=[article],
        query="тест",
    )
    assert bundle.is_relevant("security/Тест")
    assert bundle.is_relevant("security/Тест#p0")
    assert not bundle.is_relevant("mathematics/Чужая")


def test_bundle_context_marks_passages_with_ids() -> None:
    """В контексте каждый абзац помечен идентификатором для цитирования."""
    article = _article("security/Тест", "Тест", "Абзац " * 40)
    bundle = SourceBundle(
        passages=[
            __import__("pf.sources", fromlist=["Passage"]).Passage(
                source_id="security/Тест",
                title="Тест",
                section="test",
                paragraph_index=3,
                text="Текст абзаца",
                relevance=1.0,
            )
        ],
        articles=[article],
        query="тест",
    )
    context = bundle.context()
    assert "[security/Тест#p3]" in context
    assert "Текст абзаца" in context


def test_bundle_limit_caps_passages() -> None:
    """Лимит контекста работает: в модель уходит не всё."""
    from pf.sources import Passage

    passages = [
        Passage(
            source_id=f"security/Т{i}",
            title=f"Т{i}",
            section="test",
            paragraph_index=i,
            text="Абзац " * 10,
            relevance=1.0,
        )
        for i in range(10)
    ]
    bundle = SourceBundle(passages=passages, articles=[], query="тест")
    context = bundle.context(limit=3)
    # Каждый абзац начинается с «[раздел/Заголовок#pN]».
    assert context.count("#p") == 3


def test_collect_returns_bundle_for_real_query() -> None:
    """Сбор по телеком-запросу находит телеком-статьи."""
    bundle = collect("удержание абонентов оператора связи тариф")
    assert bundle.articles
    assert bundle.passages
    # В отобранном срезе не должно быть статей из несвязанных разделов:
    # доменный фильтр отсекает химию, математику и лингвистику. Раздел
    # project — это материалы проекта: свои документы и таблицы, они
    # относятся к задаче напрямую.
    for passage in bundle.passages:
        assert passage.section in {"security", "economics", "law", "telecom", "project"}


def test_verify_claim_rejects_irrelevant_but_real_source() -> None:
    """Настоящая цитата из чужой статьи не подтверждает утверждение по задаче."""
    bundle = collect("удержание абонентов оператора связи тариф")
    assert bundle.articles
    foreign = "mathematics/Испытание (техника)"
    assert not bundle.is_relevant(foreign)


def test_short_source_name_is_resolved() -> None:
    """Короткое имя файла вместо полного пути должно находиться.

    Модель часто отвечает «telecom_open_metrics» вместо
    «project/data/telecom_open_metrics.csv». Строгий поиск источник не
    находил, и все ссылки на собственные материалы отклонялись с
    сообщением «источник не относится к задаче».
    """
    bundle = collect("отток абонентов операторов связи удержание")
    assert bundle.is_relevant("telecom_open_metrics")


def test_short_source_name_resolves_article() -> None:
    """Цитата из материала по короткому имени подтверждается."""
    bundle = collect("отток абонентов операторов связи удержание")
    project = [a for a in bundle.articles if a.source_id.startswith("project/")]
    assert project, "материал проекта должен быть в срезе"
    article = project[0]
    text = article.content.splitlines()[0]
    support = bundle.verify_claim(
        text[:60],
        source_id="telecom_open_metrics",
        paragraph=None,
        quote=text[:60],
    )
    assert support.is_usable, f"цитата должна подтверждаться: {support.detail}"


def test_quote_from_table_row_is_confirmed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Цитата из расчёта подтверждается по строке таблицы.

    Модель цитирует посчитанную величину («...: среднее: 82»), а в
    строке таблицы лежит исходное число. Искать цитату надо в строке,
    а не в склеенном тексте материала: иначе подтверждения не будет
    никогда, и все ссылки на расчёты отклоняются.
    """
    monkeypatch.setattr("pf.config.settings.project_dir", tmp_path)
    (tmp_path / "numbers.csv").write_text(
        "оператор,абонентская база\nМТС,87\nTele2,38.9\n", encoding="utf-8"
    )
    bundle = collect("абоненты оператора связи тарифы удержание")
    project = [a for a in bundle.articles if a.source_id.startswith("project/")]
    assert project, "материал должен быть в срезе"

    support = bundle.verify_claim(
        "Абонентская база МТС",
        source_id="project/numbers.csv#row0",
        paragraph=None,
        quote="МТС, абонентская база: 87",
    )
    assert support.is_usable, f"цитата из строки должна подтверждаться: {support.detail}"


def test_wrong_number_in_table_row_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Число, которого нет в строке, не подтверждается."""
    monkeypatch.setattr("pf.config.settings.project_dir", tmp_path)
    (tmp_path / "numbers.csv").write_text("оператор,значение\nМТС,87\n", encoding="utf-8")
    bundle = collect("абоненты оператора связи тарифы удержание")
    support = bundle.verify_claim(
        "Абонентская база МТС",
        source_id="project/numbers.csv#row0",
        paragraph=None,
        quote="МТС, абонентская база: 99",
    )
    assert not support.is_usable


def test_verify_metric_accepts_metric_from_answer_schema(tmp_path: Path) -> None:
    """Метрика из схемы ответа агента проверяется, а не отбрасывается.

    Агент отвечает по схеме ``pf.schemas.Analysis``, и её метрика —
    ``pf.schemas.MetricRef``. Проверка же ждала ``pf.analysis.Metric``.
    Это разные классы с одинаковыми полями, поэтому isinstance-фильтр
    отбрасывал **все** метрики из ответа агента: в журнале демо-прогона
    10 отклонений из 12 — это метрики с текстом «метрика не имеет
    структуры с provenance».

    Тест строит метрику через схему ответа — туда, откуда она приходит
    на деле. Проверка через внутренний класс тестировала другой путь
    и потому оставалась зелёной.
    """
    from pf.roles import schema_for
    from pf.sources import verify_metric

    (tmp_path / "numbers.csv").write_text("оператор,значение\nМТС,82\n", encoding="utf-8")
    analysis_schema = schema_for("analyst")
    analysis = analysis_schema.model_validate(
        {
            "facts": [],
            "hypotheses": [],
            "forecasts": [],
            "gaps": [],
            "metrics": [
                {
                    "name": "Абонентская база МТС",
                    "value": 82.0,
                    "unit": "млн",
                    "source_id": "project/numbers.csv",
                    "row": 0,
                    "formula": "значение из колонки «значение», строка 1",
                    "inputs": {"значение": "82"},
                }
            ],
        }
    )

    support = verify_metric(analysis.metrics[0], load_materials(tmp_path))
    assert support.is_usable, f"метрика из схемы должна подтверждаться: {support.detail}"


def test_verify_metric_accepts_internal_metric(tmp_path: Path) -> None:
    """Внутренний класс метрики проверяется по-прежнему."""
    from pf.analysis import Metric
    from pf.sources import verify_metric

    (tmp_path / "numbers.csv").write_text("оператор,значение\nМТС,82\n", encoding="utf-8")
    materials = load_materials(tmp_path)
    metric = Metric(
        name="Абонентская база МТС",
        value=82.0,
        unit="млн",
        source_id=materials[0].source_id,
        row=0,
        formula="значение из колонки «значение», строка 1",
        inputs={"значение": "82"},
    )
    support = verify_metric(metric, materials)
    assert support.is_usable, f"внутренняя метрика должна подтверждаться: {support.detail}"


def test_verify_metric_accepts_anchor_inside_source_id(tmp_path: Path) -> None:
    """Якорь в source_id метрики не мешает найти материал.

    Модель копирует ``source_id`` из блока РАСЧЁТЫ вместе с якорем
    ``#row3``. Материал лежит под чистым путём, поэтому поиск по строке
    с якорем не находит его, и метрика отбрасывалась с текстом
    «источник не найден среди материалов» — хотя всё было верно.
    """
    from pf.sources import verify_metric

    (tmp_path / "numbers.csv").write_text("оператор,значение\nМТС,82\n", encoding="utf-8")
    materials = load_materials(tmp_path)

    from pf.schemas import MetricRef

    schema_metric = MetricRef(
        name="Абонентская база МТС",
        value=82.0,
        unit="млн",
        source_id=f"{materials[0].source_id}#row0",
        row=0,
        formula="значение из колонки «значение», строка 1",
        inputs={"значение": "82"},
    )
    support = verify_metric(schema_metric, materials)
    assert support.is_usable, f"метрика с якорем должна подтверждаться: {support.detail}"


def test_verify_metric_rejects_conflicting_anchor(tmp_path: Path) -> None:
    """Якорь, противоречащий номеру строки, не подтверждается.

    Если в source_id указан якорь одной строки, а в поле ``row`` — другая,
    значения расходятся. Подтверждать такое нельзя: неизвестно, какая
    строка имелась в виду, а подтверждение «по счастливой строке»
    пропустило бы расчёт по неверным данным.
    """
    from pf.schemas import MetricRef
    from pf.sources import verify_metric

    (tmp_path / "numbers.csv").write_text(
        "оператор,значение\nМТС,82\nTele2,38.9\n", encoding="utf-8"
    )
    materials = load_materials(tmp_path)
    schema_metric = MetricRef(
        name="Абонентская база МТС",
        value=82.0,
        unit="млн",
        source_id=f"{materials[0].source_id}#row1",
        row=0,
        formula="значение из колонки «значение», строка 1",
        inputs={"значение": "82"},
    )
    support = verify_metric(schema_metric, materials)
    assert not support.is_usable, "противоречивый якорь не должен подтверждаться"


def test_rounded_display_value_still_verifies(tmp_path: Path) -> None:
    """Округлённое для показа значение сверяется со строкой.

    Блок РАСЧЁТЫ показывает число в формате ``%.4g``: из 78,7913
    получается 78,79. Модель копирует именно это округлённое число в
    свою метрику, а проверка идёт с допуском 1e-6 — и верный расчёт
    отбрасывается с текстом «значение не совпадает со строкой».

    Значит проверка должна принимать и округлённое значение: расхождение
    в пределах точности показа — не расхождение.
    """
    from pf.analysis import Metric
    from pf.sources import verify_metric

    (tmp_path / "numbers.csv").write_text(
        "дата,значение_процентов\n13.01.2026,78.7913\n", encoding="utf-8"
    )
    materials = load_materials(tmp_path)
    metric = Metric(
        name="Ключевая ставка",
        value=78.79,  # округлено для показа
        unit="%",
        source_id=materials[0].source_id,
        row=0,
        formula="значение из колонки «значение_процентов», строка 1",
        inputs={"значение_процентов": "78.7913"},
    )
    support = verify_metric(metric, materials)
    assert support.is_usable, f"округлённое значение должно сверяться: {support.detail}"


def test_aggregate_metric_verifies_by_recomputation(tmp_path: Path) -> None:
    """Среднее по ряду проверяется пересчётом из заявленных наблюдений.

    Среднее, медиана, размах и изменение — величины, которых **нет** ни
    в одной строке таблицы: они считаются по нескольким наблюдениям.
    Проверка по одной строке их справедливо отвергала, и в демо-прогоне
    так отбрасывались все сводные метрики.

    Такая метрика проверяется пересчётом: если значение выводится из
    заявленных входов по указанной формуле — основание подтверждено.
    """
    from pf.analysis import Metric
    from pf.sources import verify_metric

    (tmp_path / "numbers.csv").write_text("оператор,значение\nМТС,87\nМТС,82\n", encoding="utf-8")
    materials = load_materials(tmp_path)
    metric = Metric(
        name="Абонентская база МТС: среднее",
        value=84.5,
        unit="млн",
        source_id=materials[0].source_id,
        row=0,
        formula="сумма / количество",
        inputs={"значения": [87.0, 82.0]},
    )
    support = verify_metric(metric, materials)
    assert support.is_usable, f"среднее должно подтверждаться пересчётом: {support.detail}"


def test_aggregate_metric_with_wrong_inputs_is_rejected(tmp_path: Path) -> None:
    """Среднее, не выводимое из входов, не подтверждается."""
    from pf.analysis import Metric
    from pf.sources import verify_metric

    (tmp_path / "numbers.csv").write_text("оператор,значение\nМТС,87\nМТС,82\n", encoding="utf-8")
    materials = load_materials(tmp_path)
    metric = Metric(
        name="Абонентская база МТС: среднее",
        value=99.0,
        unit="млн",
        source_id=materials[0].source_id,
        row=0,
        formula="сумма / количество",
        inputs={"значения": [87.0, 82.0]},
    )
    support = verify_metric(metric, materials)
    assert not support.is_usable, "неверное среднее не должно подтверждаться"


def test_recomputation_follows_formula_kind() -> None:
    """Вид пересчёта определяется формулой, а не порядком проверок.

    У формул разные слова: «максимум - минимум», «сумма / количество»,
    «(последнее - первое) / |первое| * 100». Если проверять их в неверном
    порядке, размах посчитается как изменение в процентах: из 78,79 и
    81,29 размах равен 2,5, а изменение — 3,18 %. Метрика с формулой
    «максимум - минимум» проверялась бы по чужему правилу и
    справедливо отбрасывалась.
    """
    from pf.sources import _aggregate_from_inputs

    observations = {"наблюдения": ["78.7913", "81.2955"]}

    assert _aggregate_from_inputs(2.5042, observations, "максимум - минимум") == pytest.approx(
        2.5042, abs=0.01
    )
    assert _aggregate_from_inputs(80.0434, observations, "сумма / количество") == pytest.approx(
        80.0434, abs=0.01
    )
    assert _aggregate_from_inputs(
        3.17827,
        {"база": "78.7913", "новое": "81.2955"},
        "(последнее - первое) / |первое| * 100",
    ) == pytest.approx(3.17827, abs=0.01)
