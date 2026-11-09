"""Тождество сущностей: прочтения аббревиатуры вместо одного имени.

«МТС» — не одна сущность с вариантами написания, а пучок прочтений: российский
оператор, белорусский, туркменский, банк, самолёт, система единиц «метр-тонна-
секунда», машинно-тракторная станция. Проверка сравнивала имя подстрокой, и
«МТС Беларусь» подтверждало утверждение про «МТС» (`support = 1.000`, измерено).

Здесь проверяется другое правило: прочтения берутся **из данных** — из снимка
страницы значений в хранилище знаний (`knowledge/raw/<дата>/`, запись в
манифесте с url, датой и sha256), а не из словаря в коде. Пересечение прочтений
утверждения и источника — **число**; ноль пересечений — «другая сущность»,
частичное — «требует проверки» со ссылкой на источник прочтений. Решение
принимает человек: `D-51`.

Критерий приёмки задачи — `uv run pytest tests/test_entities.py -q`: он обязан
выбирать **все** тесты файла, поэтому имена здесь не сужаются до `-k`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pf import entities, meanings
from pf.config import settings
from pf.ground import SupportStatus, verify_claim
from pf.knowledge_store import raw_dir, read_manifest, sha256_file, write_manifest

#: Снимок страницы значений для тестов: короткий, но с разными странами и с
#: прочтением вне домена задачи. Настоящий снимок проверяется отдельно — тем,
#: что лежит в репозитории, а не тем, что тест положил в `tmp_path`.
MEANINGS_TEXT = """\
МТС — страница значений Википедии.

Аббревиатура МТС может означать:

МТС (самолёт) — тактический военно-транспортный самолёт.
МТС (система единиц измерения) — система единиц «метр-тонна-секунда».
«МТС (компания)» — российский оператор связи.
«МТС Беларусь» — белорусский оператор связи.
«МТС Туркменистан» — туркменский оператор связи.
«МТС Банк» — российский универсальный банк.
"""

#: Строка белорусского оператора: субъект назван целиком.
ROW_BELARUS = "оператор: МТС Беларусь, показатель: абонентская база, значение: 5.7, период: 2026"

#: Строка российского оператора: страна названа в показателе.
ROW_MTS = (
    "оператор: МТС, показатель: абонентская база в России, значение: 82, "
    "период: конец 3-го квартала 2024"
)


@pytest.fixture
def shipped_knowledge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Смотреть на поставляемое хранилище знаний, а не на `tmp_path`.

    Набор тестов уводит ``knowledge_dir`` во временный каталог (`D-29`), а мерить
    надо поведение в поставке: страница значений «МТС» лежит в
    ``knowledge/raw/2026-11-09/``.

    Args:
        monkeypatch: Подмена настроек.
    """
    monkeypatch.setattr(
        settings, "knowledge_dir", Path(__file__).resolve().parent.parent / "knowledge"
    )


@pytest.fixture
def page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Положить снимок страницы значений в изолированное хранилище знаний.

    Args:
        tmp_path: Временный каталог теста.
        monkeypatch: Подмена настроек.

    Returns:
        Путь к файлу снимка.
    """
    monkeypatch.setattr(settings, "knowledge_dir", tmp_path / "knowledge")
    directory = raw_dir("2026-11-09")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "wikipedia_mts_meanings.txt"
    path.write_text(MEANINGS_TEXT, encoding="utf-8")
    write_manifest(
        directory,
        [
            {
                "name": path.name,
                "url": "https://ru.wikipedia.org/wiki/МТС",
                "retrieved": "2026-11-09",
                "kind": "meanings",
                "title": "МТС",
            }
        ],
    )
    return path


def test_entities_page_is_read_from_manifest(page: Path) -> None:
    """Прочтения читаются из снимка и манифеста, а не из словаря в коде."""
    loaded = meanings.pages()

    assert len(loaded) == 1
    content = loaded[0]
    assert content.abbreviation == "МТС"
    assert content.url == "https://ru.wikipedia.org/wiki/МТС"
    assert content.retrieved == "2026-11-09"
    assert content.sha256 == sha256_file(page)
    assert len(content.meanings) == 6


def test_entities_countries_come_from_the_page(page: Path) -> None:
    """Страна прочтения читается из названия и описания страницы значений."""
    content = meanings.pages()[0]
    countries = {item.title: item.country for item in content.meanings}

    assert countries["МТС Беларусь"] == "беларусь"
    assert countries["МТС Туркменистан"] == "туркменистан"
    assert countries["МТС (компания)"] == "россия"
    assert countries["МТС Банк"] == "россия"
    assert countries["МТС (самолёт)"] == ""
    assert countries["МТС (система единиц измерения)"] == ""


def test_entities_page_without_anchor_gives_no_readings() -> None:
    """Страница без перечня прочтений не выдумывает прочтения."""
    assert meanings.parse("Просто текст без перечня значений.") == ()


def test_entities_belarus_reading_does_not_confirm_plain_abbreviation(page: Path) -> None:
    """«МТС Беларусь» не подтверждает утверждение про «МТС»: прочтение не названо.

    Это и есть проверяемая правка: раньше такие пары подтверждались
    (`support = 1.000`), потому что имя искалось подстрокой.
    """
    support = verify_claim(ROW_BELARUS, "У МТС 5,7 млн абонентов", "project/data/rows.csv", 0)

    assert support.status is SupportStatus.NEEDS_REVIEW
    assert support.kind == entities.KIND_AMBIGUOUS
    assert "прочтений" in support.detail
    assert "проверить" in support.detail.lower()
    assert "wikipedia_mts_meanings.txt" in support.detail


def test_entities_named_reading_is_confirmed(page: Path) -> None:
    """Утверждение, назвавшее прочтение, подтверждается строкой того же прочтения."""
    support = verify_claim(
        ROW_BELARUS,
        "Абонентская база МТС Беларусь — 5,7 млн абонентов",
        "project/data/rows.csv",
        0,
    )

    assert support.status is SupportStatus.CONFIRMED
    assert support.kind == ""


def test_entities_named_country_is_confirmed(page: Path) -> None:
    """Страна в утверждении сужает прочтения так же, как название целиком."""
    support = verify_claim(
        ROW_BELARUS,
        "Абонентская база МТС в Беларуси — 5,7 млн абонентов",
        "project/data/rows.csv",
        0,
    )

    assert support.status is SupportStatus.CONFIRMED


def test_entities_other_country_is_mismatch(page: Path) -> None:
    """Утверждение о России отклоняется строкой Беларуси: ноль прочтений пересеклось."""
    support = verify_claim(
        ROW_BELARUS,
        "Абонентская база МТС в России — 5,7 млн абонентов",
        "project/data/rows.csv",
        0,
    )

    assert support.status is SupportStatus.NO_DATA
    assert support.kind == entities.KIND_MISMATCH
    assert "другая сущность" in support.detail


def test_entities_plain_claim_with_narrow_source_is_not_confirmed(page: Path) -> None:
    """Голое «МТС» против источника, называющего страну, — не подтверждение.

    Цена честности, названная заранее: раньше такая пара подтверждалась.
    Теперь прочтение утверждения не названо, и пересеклось 2 прочтения из 6.
    """
    support = verify_claim(ROW_MTS, "У МТС 82 млн абонентов", "project/data/rows.csv", 0)

    assert support.status is SupportStatus.NEEDS_REVIEW
    assert support.kind == entities.KIND_AMBIGUOUS
    assert "2 из 6" in support.detail


def test_entities_equal_reading_sets_are_compared_by_segment(page: Path) -> None:
    """Страна из соседнего предложения не переписывает прочтение этого."""
    claim = "Абонентская база МТС в России — 82 млн"
    source = (
        "Абонентская база МТС в России — 82 млн человек. "
        "Отдельно МТС Беларусь обслуживает 5,7 млн человек."
    )

    identity = entities.identity(claim, source)

    assert identity.kind == ""
    assert identity.share == 1.0
    assert "компания" in identity.note


def test_entities_mismatch_names_both_sides(page: Path) -> None:
    """Причина отказа называет и утверждение, и источник."""
    identity = entities.identity("Абонентская база МТС в России — 5,7 млн", ROW_BELARUS)

    assert identity.kind == entities.KIND_MISMATCH
    assert identity.share == 0.0
    assert "МТС Беларусь" in identity.note
    assert "россия" in identity.note.lower()


def test_entities_without_meaning_page_keeps_previous_behaviour() -> None:
    """Нет страницы значений — прочтения не выдумываются, проверка как раньше."""
    support = verify_claim(ROW_BELARUS, "У МТС 5,7 млн абонентов", "project/data/rows.csv", 0)

    assert support.status is SupportStatus.CONFIRMED
    assert support.kind == ""


def test_entities_shipped_snapshot_is_valid() -> None:
    """Поставляемый снимок разбирается и совпадает с манифестом.

    Читается настоящий путь репозитория, а не `settings.knowledge_dir`: набор
    тестов уводит хранилище в `tmp_path`, и без этой проверки снимок мог бы
    разойтись с манифестом молча.
    """
    root = Path(__file__).resolve().parent.parent
    directory = root / "knowledge" / "raw" / "2026-11-09"
    entries = read_manifest(directory)
    record = next(item for item in entries if item.get("kind") == meanings.MEANINGS_KIND)
    path = directory / str(record["name"])

    assert sha256_file(path) == record["sha256"]
    parsed = meanings.parse(path.read_text(encoding="utf-8"))

    assert len(parsed) >= 15
    assert any(item.title == "МТС Беларусь" for item in parsed)
    assert any(item.country == "" for item in parsed)


@pytest.mark.xfail(
    strict=True,
    reason="T-58b: is_usable схлопывает три состояния в два, у основания нет вклада",
)
def test_entities_ambiguous_basis_contributes_more_than_zero(page: Path) -> None:
    """Неоднозначное основание обязано давать вклад, а не ноль (`T-58b`)."""
    support = verify_claim(ROW_BELARUS, "У МТС 5,7 млн абонентов", "project/data/rows.csv", 0)
    contribution = getattr(support, "contribution", None)

    assert contribution is not None, "у основания нет числового вклада"
    assert contribution > 0.0


@pytest.mark.parametrize(
    "pair_id",
    ["p-001", "p-027", "p-043"],
)
def test_entities_pair_set_keeps_named_readings_confirmed(
    pair_id: str, shipped_knowledge: None
) -> None:
    """Пары, назвавшие прочтение, подтверждаются и после правки."""
    from pf.canonical import score
    from pf.semantic_bench import load_pairs, source_passage

    pair = next(item for item in load_pairs() if item.pair_id == pair_id)
    passage = source_passage(pair.source_id, pair.paragraph)
    assert passage is not None
    result = score(pair.claim, passage)

    assert result.support >= settings.claim_support_threshold, result.reason
    assert result.entity_kind == ""


@pytest.mark.parametrize(
    "pair_id",
    ["p-007", "p-015", "p-041", "p-053"],
)
def test_entities_pair_set_demotes_unnamed_reading(pair_id: str, shipped_knowledge: None) -> None:
    """Цена честности названа поимённо: «МТС» без страны — не подтверждение.

    Источник у этих пар называет прочтение (в колонке «источник» или в
    показателе), а утверждение — нет. Раньше они подтверждались целиком; теперь
    это «требует проверки» с числом и ссылкой. Регрессом было бы обратное.
    """
    from pf.canonical import score
    from pf.ground import field_verdict
    from pf.semantic_bench import load_pairs, source_passage

    pair = next(item for item in load_pairs() if item.pair_id == pair_id)
    passage = source_passage(pair.source_id, pair.paragraph)
    assert passage is not None
    result = score(pair.claim, passage)
    status, kind, _detail = field_verdict(result, settings.claim_support_threshold)

    assert result.entity_kind == entities.KIND_AMBIGUOUS, result.reason
    assert kind == entities.KIND_AMBIGUOUS
    # У p-001 степень поддержки высокая (0,25): остальные поля сходятся, спор
    # идёт только о прочтении. Поэтому статус понижается, а не отказывает.
    assert status is SupportStatus.NEEDS_REVIEW


#: Дословная цитата строки белорусского оператора: она подтверждает значение, но
#: не отвечает на вопрос «о какой сущности речь».
QUOTE_BELARUS = "оператор: МТС Беларусь, показатель: абонентская база, значение: 5.7, период: 2026"


def test_entities_verbatim_quote_from_other_reading_does_not_confirm(page: Path) -> None:
    """Дословная цитата чужой строки не подтверждает утверждение про «МТС»."""
    support = verify_claim(
        ROW_BELARUS,
        "У МТС 5,7 млн абонентов",
        "project/data/rows.csv",
        0,
        quote=QUOTE_BELARUS,
    )

    assert support.status is SupportStatus.NEEDS_REVIEW
    assert support.kind == entities.KIND_AMBIGUOUS
    assert "прочтений" in support.detail


def test_entities_verbatim_quote_with_named_country_is_confirmed(page: Path) -> None:
    """Если утверждение назвало страну, дословная цитата подтверждает его."""
    support = verify_claim(
        ROW_BELARUS,
        "Абонентская база МТС в Беларуси — 5,7 млн",
        "project/data/rows.csv",
        0,
        quote=QUOTE_BELARUS,
    )

    assert support.status is SupportStatus.CONFIRMED


def test_entities_verbatim_quote_for_other_country_is_refused(page: Path) -> None:
    """Цитата чужого прочтения при названной стране — отказ, а не подтверждение."""
    support = verify_claim(
        ROW_BELARUS,
        "Абонентская база МТС в России — 5,7 млн",
        "project/data/rows.csv",
        0,
        quote=QUOTE_BELARUS,
    )

    assert support.status is SupportStatus.NO_DATA
    assert support.kind == entities.KIND_MISMATCH
