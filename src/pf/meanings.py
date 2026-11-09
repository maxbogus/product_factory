"""Страница значений: перечень прочтений аббревиатуры как данные, а не код.

Зачем модуль. Аббревиатура — не одна сущность с вариантами написания, а пучок
прочтений: оператор в одной стране, оператор в другой, банк, самолёт, система
единиц, машинно-тракторная станция. Пока проверка сравнивала имя подстрокой,
строка с уточнением подтверждала утверждение про голое имя
(`support = 1.000`, измерено).

Прочтения нельзя держать в словаре кода: список пришлось бы править под каждую
аббревиатуру, и он всё равно был бы чьим-то мнением. Здесь прочтения читаются
из **снимка страницы значений** в хранилище знаний — так же, как любая другая
выгрузка (`pf.knowledge_store`): у записи в манифесте есть адрес, дата получения
и хеш содержимого. Новая аббревиатура — это новый снимок, а не правка кода.

Что модуль не делает. Не решает, какое прочтение верное: он перечисляет
прочтения и их страны. Тождество считает `pf.entities`, статус ставит
`pf.ground`, решение принимает человек (`D-51`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pf.config import settings
from pf.knowledge_store import read_manifest

#: Вид записи манифеста: страница значений (перечень прочтений аббревиатуры).
MEANINGS_KIND = "meanings"

#: Фраза, с которой начинается перечень прочтений. По ней перечень и
#: находится: заголовок страницы и её служебные строки лежат выше.
ANCHOR = "может означать"

#: Минимальная длина названия прочтения. Более короткое — не название, а мусор.
MIN_TITLE_CHARS = 3

#: Страны, которые называет прочтение: лемма. Список названный и короткий:
#: страна — единственный машинно читаемый признак прочтения.
COUNTRIES: dict[str, str] = {
    "россия": "россия",
    "беларусь": "беларусь",
    "украина": "украина",
    "сербия": "сербия",
    "туркменистан": "туркменистан",
    "казахстан": "казахстан",
}

#: Прилагательные стран в описании прочтения («российский оператор связи»).
#: Нужны, потому что страну называет описание, а название — не всегда:
#: «МТС (компания)» — российский оператор, хотя слова «Россия» в названии нет.
COUNTRY_ADJECTIVES: tuple[tuple[str, str], ...] = (
    ("российск", "россия"),
    ("белорусск", "беларусь"),
    ("украинск", "украина"),
    ("сербск", "сербия"),
    ("туркменск", "туркменистан"),
    ("казахстанск", "казахстан"),
)

#: Тире между названием прочтения и описанием.
_DASH = re.compile(r"\s+[—–]\s+")

#: Слово: прилагательное страны ищется по началу слова.
_WORD = re.compile(r"[А-Яа-яЁёA-Za-z]+")

#: Всё, что не буква и не цифра, — разделитель названия.
_NON_WORD = re.compile(r"[^0-9a-zа-я]+")

_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Привести название прочтения к сравнимому виду.

    Скобки, подчёркивания и дефисы становятся пробелами: «Имя (Страна)»,
    «Имя_Страна» и «Имя Страна» — одно название. Регистр и «ё» снимаются
    тем же правилом, что в остальной проверке.

    Args:
        text: Название или текст.

    Returns:
        Строка в нижнем регистре, разделённая одинарными пробелами.
    """
    lowered = text.lower().replace("ё", "е")
    return _SPACES.sub(" ", _NON_WORD.sub(" ", lowered)).strip()


def country_of(text: str, *, adjectives: bool = True) -> str:
    """Страна, которую называет текст прочтения.

    Args:
        text: Название прочтения или его описание.
        adjectives: Читать ли прилагательные («российский оператор»).

    Returns:
        Лемма страны либо пустая строка: не названа — не угадывается.
    """
    words = _WORD.findall(text.lower().replace("ё", "е"))
    for word in words:
        found = COUNTRIES.get(word)
        if found:
            return found
    if not adjectives:
        return ""
    for word in words:
        for stem, country in COUNTRY_ADJECTIVES:
            if word.startswith(stem):
                return country
    return ""


def _mentions(title: str, abbreviation: str) -> bool:
    """Названа ли аббревиатура в названии прочтения.

    По ней решается, можно ли брать страну из описания: «российский оператор
    связи» описывает страну прочтения «Имя (компания)», а «российская учебная
    дисциплина» — предмет, а не страну («Музыкально-теоретические системы»).

    Args:
        title: Название прочтения.
        abbreviation: Аббревиатура страницы.

    Returns:
        Есть ли аббревиатура в названии целым словом.
    """
    return f" {normalize(abbreviation)} " in f" {normalize(title)} "


@dataclass(frozen=True, slots=True)
class Meaning:
    """Одно прочтение аббревиатуры.

    Attributes:
        title: Название прочтения так, как оно записано на странице значений.
        country: Лемма страны либо пустая строка.
        description: Описание прочтения — для человека, в сравнении не участвует.
    """

    title: str
    country: str = ""
    description: str = ""


@dataclass(frozen=True, slots=True)
class MeaningPage:
    """Снимок страницы значений со своим провенансом.

    Attributes:
        abbreviation: Аббревиатура, которую описывает страница.
        path: Путь к снимку — по нему человек открывает источник прочтений.
        url: Адрес страницы в исходном источнике.
        retrieved: Дата получения снимка.
        sha256: Хеш содержимого: снимок не «уплывает» молча.
        meanings: Перечень прочтений.
    """

    abbreviation: str
    path: Path
    url: str
    retrieved: str
    sha256: str
    meanings: tuple[Meaning, ...]

    @property
    def readings(self) -> frozenset[str]:
        """Нормализованные названия всех прочтений страницы.

        Returns:
            Множество названий, приведённых `normalize`.
        """
        return frozenset(normalize(item.title) for item in self.meanings)

    def country_for(self, reading: str) -> str:
        """Страна прочтения по нормализованному названию.

        Args:
            reading: Нормализованное название.

        Returns:
            Лемма страны либо пустая строка.
        """
        for item in self.meanings:
            if normalize(item.title) == reading:
                return item.country
        return ""

    def title_for(self, reading: str) -> str:
        """Название прочтения в исходном написании.

        Args:
            reading: Нормализованное название.

        Returns:
            Название со страницы значений; исходная строка, если не нашлось.
        """
        for item in self.meanings:
            if normalize(item.title) == reading:
                return item.title
        return reading


def parse(text: str) -> tuple[Meaning, ...]:
    """Разобрать снимок страницы значений на прочтения.

    Перечень начинается после фразы :data:`ANCHOR`: выше лежат заголовок и
    служебные строки. Строка перечня — «Название — описание» либо просто
    «Название.» (так записаны прочтения без описания). Ничего не угадывается:
    нет фразы — нет прочтений.

    Args:
        text: Текст снимка.

    Returns:
        Прочтения в порядке страницы.
    """
    lines = text.splitlines()
    start = next((index for index, line in enumerate(lines) if ANCHOR in line.lower()), None)
    if start is None:
        return ()

    meanings: list[Meaning] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if len(stripped) < MIN_TITLE_CHARS:
            continue
        match = _DASH.search(stripped)
        if match:
            title = stripped[: match.start()].strip()
            description = stripped[match.end() :].strip()
        else:
            title = stripped.rstrip(".").strip()
            description = ""
        title = title.strip("«»\"' ").strip()
        if len(title) < MIN_TITLE_CHARS:
            continue
        meanings.append(
            Meaning(
                title=title,
                country=country_of(title, adjectives=False),
                description=description,
            )
        )
    return tuple(meanings)


def _with_description_countries(
    items: tuple[Meaning, ...],
    abbreviation: str,
) -> tuple[Meaning, ...]:
    """Добрать страну прочтения из описания, где она известна.

    «Имя (компания)» — российский оператор связи: слова «Россия» в названии
    нет, а прочтение всё равно сужено страной. Берётся только то прочтение,
    в названии которого есть аббревиатура: «Музыкально-теоретические системы»
    — российская учебная дисциплина, и «российская» там про предмет, а не про
    страну сущности.

    Args:
        items: Прочтения, разобранные из снимка.
        abbreviation: Аббревиатура страницы.

    Returns:
        Прочтения с добраной страной там, где она читается из описания.
    """
    enriched: list[Meaning] = []
    for item in items:
        country = item.country
        if not country and _mentions(item.title, abbreviation):
            country = country_of(item.description)
        enriched.append(Meaning(title=item.title, country=country, description=item.description))
    return tuple(enriched)


def page_of(directory: Path, entry: dict[str, Any]) -> MeaningPage | None:
    """Собрать страницу значений из записи манифеста и файла рядом с ним.

    Args:
        directory: Каталог снимков за дату.
        entry: Запись манифеста.

    Returns:
        Страница значений либо ``None``: нет названия, файла или прочтений.
    """
    name = str(entry.get("name", "")).strip()
    abbreviation = str(entry.get("title", "")).strip()
    if not name or not abbreviation:
        return None
    path = directory / name
    if not path.is_file():
        return None
    items = parse(path.read_text(encoding="utf-8"))
    if not items:
        return None
    return MeaningPage(
        abbreviation=abbreviation,
        path=path,
        url=str(entry.get("url", "")),
        retrieved=str(entry.get("retrieved", "")),
        sha256=str(entry.get("sha256", "")),
        meanings=_with_description_countries(items, abbreviation),
    )


def pages(knowledge_dir: Path | None = None) -> tuple[MeaningPage, ...]:
    """Все страницы значений из хранилища знаний.

    Каталоги читаются новыми первыми, и на одну аббревиатуру берётся только
    свежий снимок: два снимка одной страницы — это версии одного перечня, а не
    две разные страницы. Молча сложить их значило бы удвоить прочтения.

    Args:
        knowledge_dir: Корень знаний; по умолчанию — из настроек.

    Returns:
        Страницы значений в порядке появления.
    """
    root = (knowledge_dir or settings.knowledge_dir) / "raw"
    if not root.is_dir():
        return ()
    found: list[MeaningPage] = []
    seen: set[str] = set()
    for directory in sorted((item for item in root.iterdir() if item.is_dir()), reverse=True):
        for entry in read_manifest(directory):
            if str(entry.get("kind", "")) != MEANINGS_KIND:
                continue
            page = page_of(directory, entry)
            if page is None:
                continue
            key = normalize(page.abbreviation)
            if key in seen:
                continue
            seen.add(key)
            found.append(page)
    return tuple(found)
