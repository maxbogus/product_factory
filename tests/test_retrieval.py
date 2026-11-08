"""Проверки отбора источников: доменный запрос должен приводить к телеком-статьям.

Это тест на главный дефект, найденный в реальных прогонах: при запросе
про удержание абонентов срез источников содержал статьи по физике, а статья
«Мегафон (компания)» в него не попадала. Из-за этого модель получала
нерелевантный материал и выдумывала цитаты.

Проверки тут детерминированные — модель не нужна.
"""

from __future__ import annotations

import pytest

from pf.corpus import (
    Article,
    Corpus,
    domain_lemmas,
    is_domain_query,
    lemmatize,
    strong_domain_lemmas,
)


def _corpus(*articles: Article) -> Corpus:
    return Corpus(list(articles))


def _article(source_id: str, title: str, content: str) -> Article:
    return Article(source_id=source_id, title=title, section="test", content=content)


# ── Лемматизация словоформ ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("мобильном", "мобильный"),
        ("тарифе", "тариф"),
        ("удержания", "удержание"),
        ("удержанию", "удержание"),
        ("абонентов", "абонент"),
        ("связью", "связь"),
        ("пакетов", "пакет"),
        ("трафика", "трафик"),
        ("роуминга", "роуминг"),
    ],
)
def test_lemmatize_reduces_inflected_forms(word: str, expected: str) -> None:
    """Словоформы приводятся к общей лемме.

    Лемматизация взята из проверенного решения: pymorphy3. Эвристический
    срез суффиксов ошибается на «связью» → «связ» и на «удержание» →
    «удержание», поэтому он не годится.
    """
    assert lemmatize(word) == expected


def test_lemmatize_keeps_lemma_unchanged() -> None:
    """Лемма остаётся леммой."""
    assert lemmatize("тариф") == "тариф"
    assert lemmatize("связь") == "связь"


def test_is_domain_query_detects_inflected_forms() -> None:
    """Доменный запрос распознаётся, даже если слова в другой словоформе."""
    assert is_domain_query("Почему падает удержание у мобильном тарифе")
    assert is_domain_query("удержание клиентов на мобильном тарифе")


def test_is_domain_query_rejects_foreign_topic() -> None:
    """Чужая тема доменной не считается."""
    assert not is_domain_query("как устроен алгоритм сортировки слиянием")


def test_domain_lemmas_are_lemmas_of_clean_terms() -> None:
    """Набор доменных лемм совпадает с леммами чистых доменных терминов.

    Иначе фильтр запроса и словарь статей смотрели бы в разные множества.
    """
    lemmas = domain_lemmas()
    assert lemmas
    assert all(len(lemma) >= 4 for lemma in lemmas)
    assert "абонент" in lemmas
    assert "роуминг" in lemmas


def test_domain_lemmas_exclude_ambiguous_terms() -> None:
    """Двусмысленные термины не входят в доменные леммы.

    Замер по корпусу: «связь» встречается в 1 402 статьях чужих разделов
    (химическая «двойная связь», IT-«связь»), «удержание» — в 47.
    Оба прошли бы фильтр наравне с телекомом, поэтому в чистый набор
    не входят.
    """
    lemmas = domain_lemmas()
    assert "связь" not in lemmas
    assert "удержание" not in lemmas
    assert "интернет" not in lemmas


def test_strong_domain_lemmas_exclude_weak_terms() -> None:
    """Слабые термины не входят в сильные.

    «Оператор» в статье «Операторы в C++» — это лингвистика, а не телеком.
    Поэтому «оператор» не должен быть сильным признаком предметной области.
    """
    strong = strong_domain_lemmas()
    assert "оператор" not in strong
    assert "удержание" not in strong
    assert "абонент" in strong
    assert "тариф" in strong


# ── Доменный отбор ──────────────────────────────────────────────────────────


def test_telecom_article_is_selected_for_demo_query() -> None:
    """Статья об операторе попадает в срез по запросу про удержание."""
    corpus = _corpus(
        _article(
            "telecom/Мегафон (компания)",
            "Мегафон (компания)",
            "Компания обслужила 77,26 миллиона абонентов. "
            "Ввела специальный тариф «Транзитный» для абонентов в Сибири. " * 3,
        ),
        _article(
            "physics/Нерешённые проблемы",
            "Нерешённые проблемы современной физики",
            "Почему некоторые астрономические объекты испускают релятивистские струи? " * 4,
        ),
    )
    results = corpus.search("удержание клиентов на мобильном тарифе", top_k=3)
    source_ids = [article.source_id for article, _score in results]
    assert "telecom/Мегафон (компания)" in source_ids


def test_operator_outranks_other_domain_articles() -> None:
    """Статья про оператора приоритетнее прочей телеком-статьи."""
    corpus = _corpus(
        _article(
            "telecom/Мегафон (компания)",
            "Мегафон (компания)",
            "Компания обслужила 77,26 миллиона абонентов и ввела тарифы. " * 4,
        ),
        _article(
            "telecom/Мобильный ID",
            "Мобильный ID",
            "Технология идентификации абонента в мобильной сети. " * 4,
        ),
    )
    results = corpus.search("удержание клиентов на мобильном тарифе", top_k=2)
    assert results
    assert results[0][0].source_id == "telecom/Мегафон (компания)"


def test_foreign_topic_still_ranked_by_tfidf() -> None:
    """Для чужой темы доменный фильтр не мешает — работает обычный TF-IDF.

    Иначе агент перестанет отвечать на задачи вне телеком-тематики.
    """
    corpus = _corpus(
        _article(
            "cs/Алгоритм сортировки",
            "Алгоритм сортировки",
            "Сортировка слиянием использует массив и сортировку подмассивов. " * 5,
        ),
        _article(
            "cs/Интросорт",
            "Интросорт",
            "Интросорт сочетает быструю сортировку и сортировку вставками. " * 5,
        ),
        _article(
            "telecom/Мегафон (компания)",
            "Мегафон (компания)",
            "Компания обслужила 77,26 миллиона абонентов на мобильном тарифе. " * 5,
        ),
    )
    results = corpus.search("алгоритм сортировки слиянием", top_k=2)
    source_ids = [article.source_id for article, _score in results]
    assert "cs/Алгоритм сортировки" in source_ids
    assert "telecom/Мегафон (компания)" not in source_ids


def test_context_is_not_eaten_by_one_table() -> None:
    """Одна таблица не занимает всё окно контекста.

    В прогоне `demo-v9` первые пять абзацев контекста были строками одной
    таблицы (Tele2, Беларусь-2026, 2007 год, доля спутникового ТВ), и ни
    одного абзаца корпуса про отток модель не видела. Вопрос был «почему
    падает удержание», а материала по теме в промпте не оказалось: модель
    отвечала тем, что видела, — абонентской базой.
    """
    from pf.sources import collect

    bundle = collect("Почему падает удержание у мобильного тарифа и что с этим делать?")
    text = bundle.context(limit=5)

    sources = {block.split("]")[0].lstrip("[") for block in text.split("\n\n")}
    ids = {source.split("#")[0] for source in sources}

    assert len(ids) >= 3, f"окно занято {len(ids)} источником(ами): {sorted(ids)}"
    assert any(not source.startswith("project/") for source in ids), (
        f"в окне нет ни одного абзаца корпуса: {sorted(ids)}"
    )


def test_context_fills_window_when_source_is_single() -> None:
    """Если материал один, окно заполняется его абзацами.

    Ограничение на повторы не должно обрезать контекст: когда источник
    один, брать абзацы больше неоткуда, и окно остаётся заполненным.
    """
    from pf.sources import Passage, SourceBundle

    passages = [
        Passage(
            source_id="project/numbers.csv",
            title="numbers",
            section="project",
            paragraph_index=index,
            text=f"строка {index}",
            relevance=1.0 - index / 100,
        )
        for index in range(5)
    ]
    bundle = SourceBundle(passages=passages, articles=[], query="запрос")

    text = bundle.context(limit=5)

    assert text.count("\n\n") == 4, f"окно обрезано: {text!r}"


def test_empty_query_returns_nothing() -> None:
    """Пустой запрос не выбирает источники."""
    corpus = _corpus(_article("telecom/A", "A", "Абзац " * 20))
    assert corpus.search("   ", top_k=3) == []
