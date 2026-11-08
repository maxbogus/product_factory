"""Проверки отбора источников: доменный запрос должен приводить к телеком-статьям.

Это тест на главный дефект, найденный в реальных прогонах: при запросе
про удержание абонентов срез источников содержал статьи по физике, а статья
«Мегафон (компания)» в него не попадала. Из-за этого модель получала
нерелевантный материал и выдумывала цитаты.

Проверки тут детерминированные — модель не нужна.
"""

from __future__ import annotations

from pathlib import Path

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


def test_window_reserves_slot_for_corpus() -> None:
    """Окно не отдаётся целиком своим материалам.

    Проверка детерминированная: абзацы заданы вручную, настоящие данные и
    корпус не нужны. Смысл правила — модель должна видеть и строки своих
    таблиц, и статьи по теме вопроса; иначе на вопрос про отток она
    отвечает тем, что нашлось в данных: размером базы.
    """
    from pf.sources import Passage, SourceBundle

    local = [
        Passage(
            source_id=f"project/data/table{index}.csv",
            title=f"Таблица {index}",
            section="project",
            paragraph_index=0,
            text=f"оператор: МТС, значение: {80 + index}",
            relevance=1.0 - index * 0.01,
        )
        for index in range(4)
    ]
    corpus = Passage(
        source_id="telecom/Отток клиентов",
        title="Отток клиентов",
        section="telecom",
        paragraph_index=0,
        text="Отток клиентов — потеря клиентов или покупателей.",
        relevance=0.9,
    )
    bundle = SourceBundle(passages=[*local, corpus], articles=[], query="отток")

    text = bundle.context(limit=5)
    assert "telecom/Отток клиентов#p0" in text, f"корпус вытеснен из окна: {text}"


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


# ── Отбор строк таблицы ─────────────────────────────────────────────────────

#: Вопрос демонстрационного прогона.
DEMO_QUESTION = "Почему падает удержание у мобильного тарифа и что с этим делать?"


@pytest.fixture
def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Каталог с материалами проекта вместо настоящего.

    Args:
        tmp_path: Временный каталог теста.
        monkeypatch: Подмена настроек.

    Returns:
        Пустой каталог материалов.
    """
    directory = tmp_path / "project"
    directory.mkdir()
    monkeypatch.setattr("pf.config.settings.project_dir", directory)
    monkeypatch.setattr("pf.config.settings.knowledge_dir", tmp_path / "knowledge")
    return directory


def test_row_filter_drops_row_about_other_indicator() -> None:
    """Строка о другом показателе не попадает в срез демо-вопроса.

    В `demo-v9` окно занимали «МТС Беларусь 2026» и «доля на рынке
    спутникового ТВ». С вопросом у них не было ни одного общего слова,
    но материал был про ту же отрасль, и вес получала каждая строка — а
    порядок решала длина строки, то есть длина URL в колонке «источник».
    Отбор обязан идти по показателю строки.
    """
    from pf.sources import collect

    bundle = collect(DEMO_QUESTION)
    rows = [passage for passage in bundle.passages if passage.source_id.endswith(".csv")]

    assert rows, "строки про абонентскую базу должны остаться в срезе"
    assert not any("спутников" in passage.text for passage in rows), (
        "строка о другом показателе попала в срез: "
        f"{[passage.text for passage in rows if 'спутников' in passage.text]}"
    )

    window = bundle.context(limit=5)
    assert "МТС Беларусь" not in window, f"в окне чужая строка: {window!r}"
    # Состав окна из пяти слотов не фиксируется: он зависит от того, какие
    # файлы данных есть в проекте. Проверяется свойство, а не состав:
    # строки окна — про показатели задачи, а не про соседнюю отрасль.
    project_rows = [
        block
        for block in window.split("\n\n")
        if block.startswith("[project/") and ".csv" in block.split("]")[0]
    ]
    assert project_rows, f"в окне нет ни одной строки данных: {window!r}"
    assert any("абонентск" in block.lower() or "arpu" in block.lower() for block in project_rows), (
        f"в окне нет ни одной строки про показатель: {window!r}"
    )


def test_row_filter_ignores_provenance_length(_project: Path) -> None:
    """Длина URL в колонке «источник» не решает, какая строка важнее.

    Провенанс — это откуда взято, а не что строка утверждает. Пока он
    входил в оценку, «МТС Беларусь» выигрывала у всех строк таблицы
    только потому, что адрес Википедии у неё короче.
    """
    from pf.sources import collect

    long_url = "https://ru.wikipedia.org/wiki/" + "длинный_адрес_оператора_" * 4
    (_project / "metrics.csv").write_text(
        "оператор,показатель,значение,период,источник\n"
        "МТС,абонентская база,87,2024,https://a.ru\n"
        f"Т2,абонентская база,89,2024,{long_url}\n",
        encoding="utf-8",
    )
    bundle = collect("абонентская база операторов связи")

    rows = [passage for passage in bundle.passages if passage.source_id.endswith("metrics.csv")]
    assert len(rows) == 2, [passage.text for passage in rows]
    assert rows[0].relevance == pytest.approx(rows[1].relevance), (
        f"балл строки зависит от провенанса: {rows[0].relevance} и {rows[1].relevance}"
    )
    assert "МТС" in rows[0].text, "при равном балле порядок задаёт источник, а не адрес"


def test_row_filter_prefers_indicator_hit_over_domain_fallback(_project: Path) -> None:
    """Строка, чей показатель совпал с вопросом, стоит выше доменного добора.

    Иначе строка «про ту же отрасль» вытесняет строку, которая отвечает
    на заданный вопрос.
    """
    from pf.sources import collect

    (_project / "metrics.csv").write_text(
        "оператор,показатель,значение,период,источник\n"
        "МТС,абонентская база,87,2024,https://a.ru\n"
        "МТС,удержание клиентов,78,2024,https://a.ru/очень/длинный/адрес/строки\n",
        encoding="utf-8",
    )
    bundle = collect("почему падает удержание клиентов на мобильном тарифе")

    rows = [passage for passage in bundle.passages if passage.source_id.endswith("metrics.csv")]
    assert rows, "строки про удержание должны попасть в срез"
    assert "удержание" in rows[0].text, (
        f"первой стоит строка без совпадения по показателю: {rows[0].text!r}"
    )
