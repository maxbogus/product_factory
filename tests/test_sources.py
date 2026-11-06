"""Проверки сбора источников.

Здесь проверяется то, что делает агента честным: подтверждённое основание
должно быть и дословным, и относящимся к задаче.
"""

from __future__ import annotations

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
    # В отобранном срезе не должно быть статей из несвязанных разделов.
    for passage in bundle.passages:
        assert passage.section in {"security", "economics", "law"}


def test_verify_claim_rejects_irrelevant_but_real_source() -> None:
    """Настоящая цитата из чужой статьи не подтверждает утверждение по задаче."""
    bundle = collect("удержание абонентов оператора связи тариф")
    assert bundle.articles
    foreign = "mathematics/Испытание (техника)"
    assert not bundle.is_relevant(foreign)
