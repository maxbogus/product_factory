"""Проверки сбора источников.

Здесь проверяется то, что делает агента честным: подтверждённое основание
должно быть и дословным, и относящимся к задаче.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pf.corpus import Article
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
