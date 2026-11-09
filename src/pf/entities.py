"""Тождество сущностей: у аббревиатуры несколько прочтений, а не одно имя.

Ответ — **не вердикт.** Утверждение и источник дают по множеству прочтений, и
сравнивается их пересечение: доля от 0 до 1. Ноль пересечений — «в источнике
другая сущность», частичное — «прочтение не установлено», и тогда причину
называет число («совпало 1 из 20») и ссылка на источник прочтений.

Три исхода, а не два (`D-51`):

- ``""`` — прочтения утверждения целиком поддержаны источником. Это ещё не
  «подтверждено»: статус ставит `pf.ground` по степени поддержки;
- :data:`KIND_MISMATCH` — прочтения не пересеклись вовсе. Утверждение о
  российском операторе не подтверждается строкой белорусского;
- :data:`KIND_AMBIGUOUS` — пересеклись не все. Голое имя против строки с
  уточнением — это не «нет данных», а «не установлено»: 1 прочтение из 20.

Чего модуль не делает. Не угадывает прочтение и не выбирает за человека: он
называет варианты и даёт ссылку. Решение записывается отдельно (`T-58c`), а
вклад неоднозначного основания в вывод — `T-58b`: пока `is_usable` схлопывает
три состояния в два, у основания нет веса, и это названо `xfail`-тестом.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pf import meanings
from pf.lexical import lemma, term_present
from pf.verdicts import ENTITY_AMBIGUOUS, ENTITY_MISMATCH

#: Прочтения не пересеклись: в источнике другая сущность.
#:
#: Имена видов — из таблицы `pf.verdicts.KINDS`, а не свои: одно имя на
#: проверку, журнал и отчёт (иначе класс отказа считается по-разному).
KIND_MISMATCH = ENTITY_MISMATCH

#: Пересеклись не все прочтения утверждения: прочтение не установлено.
KIND_AMBIGUOUS = ENTITY_AMBIGUOUS

#: Нижняя граница доли поля: провал остаётся различимым по величине, а не
#: превращается в ноль — так же, как у остальных полей `pf.canonical`.
SHARE_FLOOR = 0.02

#: Граница предложения: страна из соседнего предложения не переписывает
#: прочтение этого.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

#: Слово для поиска страны в тексте.
_WORD = re.compile(r"[А-Яа-яЁёA-Za-z]{3,}")


@dataclass(frozen=True, slots=True)
class Readings:
    """Прочтения одного отрезка текста.

    Attributes:
        abbreviation: Аббревиатура, о которой речь.
        page: Страница значений — источник прочтений и ссылка для человека.
        titles: Нормализованные названия прочтений, которые допускает отрезок.
        narrowed: Назвал ли отрезок прочтение (названием или страной) или
            оставил все прочтения аббревиатуры.
    """

    abbreviation: str
    page: meanings.MeaningPage
    titles: frozenset[str]
    narrowed: bool


@dataclass(frozen=True, slots=True)
class Identity:
    """Исход сравнения прочтений утверждения и источника.

    Attributes:
        share: Доля прочтений утверждения, которые поддерживает источник.
        kind: :data:`KIND_MISMATCH`, :data:`KIND_AMBIGUOUS` либо пустая строка.
        note: Причина словами: что названо, сколько прочтений, где проверить.
        link: Ссылка на источник прочтений (снимок и адрес страницы).
        claim_narrowed: Назвало ли утверждение прочтение (названием или
            страной). Неназванное прочтение и есть причина «требует проверки».
    """

    share: float
    kind: str
    note: str
    link: str
    claim_narrowed: bool = False


def normalize(text: str) -> str:
    """Привести название к сравнимому виду — одним правилом со страницей.

    Args:
        text: Название или текст.

    Returns:
        Строка в нижнем регистре с одинарными пробелами.
    """
    return meanings.normalize(text)


def _sentences(text: str) -> tuple[str, ...]:
    """Разбить прозу на предложения.

    Проза читается по предложениям, а не целиком: упоминание страны в соседнем
    предложении иначе переписало бы прочтение этого.

    Args:
        text: Текст.

    Returns:
        Непустые предложения.
    """
    parts = (part.strip() for part in _SENTENCE.split(text))
    return tuple(part for part in parts if part)


def _phrase_present(normalized_title: str, normalized_segment: str) -> bool:
    """Назван ли заголовок прочтения подряд, а не по словам.

    Подряд, потому что «Компания А (компания)» иначе находилось бы в любом
    тексте со словами «компания» и «А» в любом порядке — это другая сущность,
    а не это прочтение.

    Args:
        normalized_title: Название прочтения, приведённое `normalize`.
        normalized_segment: Отрезок текста, приведённый `normalize`.

    Returns:
        Есть ли название целиком.
    """
    return f" {normalized_title} " in f" {normalized_segment} "


def _countries_in(segment: str, known: frozenset[str]) -> frozenset[str]:
    """Страны отрезка, которые называет хотя бы одно прочтение.

    Args:
        segment: Отрезок текста.
        known: Страны прочтений страницы.

    Returns:
        Леммы стран из ``known``, названные в отрезке.
    """
    lowered = segment.lower().replace("ё", "е")
    # Лемма, а не слово: «в России» и «Россия» — одна страна. Без этого
    # сужение по стране не срабатывало вовсе, и голое «МТС» оставалось
    # неоднозначным даже там, где страна названа прямо.
    found = {
        meanings.COUNTRIES[lemmatized]
        for lemmatized in (lemma(word) for word in _WORD.findall(lowered))
        if lemmatized in meanings.COUNTRIES
    }
    return frozenset(found & known)


def _group_in(page: meanings.MeaningPage, segment: str) -> Readings | None:
    """Прочтения, которые допускает отрезок текста.

    Сначала название целиком (составное имя), затем страна («компания А в
    Беларуси», «абонентская база в России»), иначе — все прочтения
    аббревиатуры.

    Args:
        page: Страница значений.
        segment: Отрезок текста.

    Returns:
        Прочтения отрезка либо ``None``, если аббревиатура не названа.
    """
    if not term_present(page.abbreviation, segment):
        return None
    normalized = normalize(segment)
    titled = frozenset(
        normalize(item.title)
        for item in page.meanings
        if _phrase_present(normalize(item.title), normalized)
    )
    if titled:
        return Readings(page.abbreviation, page, titled, True)
    known = frozenset(item.country for item in page.meanings if item.country)
    for country in _countries_in(segment, known):
        narrowed = frozenset(
            normalize(item.title) for item in page.meanings if item.country == country
        )
        if narrowed:
            return Readings(page.abbreviation, page, narrowed, True)
    return Readings(page.abbreviation, page, page.readings, False)


def readings(text: str, *, row: bool = False) -> tuple[Readings, ...]:
    """Прочтения, которые допускает текст: по отрезку на каждую страницу.

    Строка таблицы читается целиком: она описывает один субъект, и страна в
    соседней колонке («показатель: абонентская база в России») относится к тому
    же субъекту. Проза читается по предложениям.

    Args:
        text: Текст утверждения, строки таблицы или абзаца.
        row: Является ли текст строкой таблицы.

    Returns:
        Прочтения по отрезкам; пусто, если аббревиатур со страницами нет.
    """
    segments = (text,) if row else _sentences(text)
    groups: list[Readings] = []
    for page in meanings.pages():
        for segment in segments:
            group = _group_in(page, segment)
            if group is not None:
                groups.append(group)
    return tuple(groups)


def link_of(page: meanings.MeaningPage) -> str:
    """Ссылка на источник прочтений для человека.

    Args:
        page: Страница значений.

    Returns:
        Путь к снимку и адрес страницы: по ним прочтения можно проверить.
    """
    path = page.path
    try:
        shown: Path | str = path.relative_to(Path.cwd())
    except ValueError:
        shown = path
    return f"{shown} ({page.url})" if page.url else str(shown)


def _names(page: meanings.MeaningPage, titles: frozenset[str], *, limit: int = 4) -> str:
    """Назвать прочтения словами — со страной, если она известна.

    Args:
        page: Страница значений.
        titles: Нормализованные названия.
        limit: Сколько названий перечислить.

    Returns:
        Строка вида ``«Компания А (компания)» (россия), «Компания А Банк»``.
    """
    ordered = sorted(titles)
    parts: list[str] = []
    for title in ordered[:limit]:
        country = page.country_for(title)
        named = f"«{page.title_for(title)}»"
        parts.append(f"{named} ({country})" if country else named)
    joined = ", ".join(parts)
    return f"{joined} и ещё {len(ordered) - limit}" if len(ordered) > limit else joined


def identity(
    claim_text: str,
    source_text: str,
    *,
    claim_row: bool = False,
    source_row: bool = False,
) -> Identity:
    """Сравнить прочтения утверждения и источника.

    Источник берётся по лучшему отрезку: утверждение о России подтверждается
    тем предложением, которое говорит о России, даже если рядом упомянута
    Беларусь. Пересечение считается с прочтениями утверждения: важно, сколько
    его прочтений поддержано, а не сколько чужих рядом.

    Названное прочтение принимается, даже если источник перечислил не весь
    набор: утверждение «компания А в Беларуси» сузило себя страной, и источник,
    называющий составное имя, с ним согласен. Не названо прочтение — исход
    «прочтение не установлено», даже если цитата дословна.

    Args:
        claim_text: Текст утверждения.
        source_text: Текст источника (строка таблицы, абзац).
        claim_row: Является ли утверждение строкой таблицы.
        source_row: Является ли источник строкой таблицы.

    Returns:
        Долю, вид исхода, причину словами и ссылку на источник прочтений.
    """
    claim_groups = readings(claim_text, row=claim_row)
    source_groups = readings(source_text, row=source_row)
    if not claim_groups or not source_groups:
        # Прочтения не перечислены: сущность не судится, решают другие поля.
        return Identity(1.0, "", "", "")

    claim_titles: frozenset[str] = frozenset().union(*(group.titles for group in claim_groups))
    page = claim_groups[0].page
    link = link_of(page)
    total = len(claim_titles)

    best: Readings | None = None
    best_share = -1.0
    for group in source_groups:
        share = len(claim_titles & group.titles) / total
        # При равной доле выигрывает более узкое прочтение: «источник называет»
        # должно называть как можно точнее.
        if share > best_share or (
            share == best_share and best is not None and len(group.titles) < len(best.titles)
        ):
            best, best_share = group, share
    if best is None:  # pragma: no cover - source_groups непусто, значит best есть
        return Identity(1.0, "", "", link)

    common = claim_titles & best.titles
    claim_narrowed = any(group.narrowed for group in claim_groups)
    if not common:
        note = (
            f"в источнике другая сущность: {_names(page, best.titles)}, "
            f"а утверждение называет {_names(page, claim_titles)}. Проверить: {link}"
        )
        return Identity(0.0, KIND_MISMATCH, note, link, claim_narrowed)

    if common == claim_titles or claim_narrowed:
        note = ""
        if total > 1 and best.narrowed:
            note = (
                f"«{page.abbreviation}»: прочтение не названо (прочтений {total}); "
                f"источник называет {_names(page, best.titles)}"
            )
        elif total > 1:
            note = (
                f"«{page.abbreviation}»: прочтение не названо ни в утверждении, "
                f"ни в источнике (прочтений {total})"
            )
        return Identity(1.0, "", note, link, claim_narrowed)

    note = (
        f"«{page.abbreviation}»: прочтений {total}; источник называет {_names(page, best.titles)}; "
        f"совпало {len(common)} из {total}. Проверить: {link}"
    )
    return Identity(len(common) / total, KIND_AMBIGUOUS, note, link, claim_narrowed)
