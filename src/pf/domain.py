"""Домен проекта как данные: субъекты, термины и шаблоны заголовков.

Зачем модуль. Домен жил в коде: названия операторов в ``canonical.SUBJECTS``,
бренды конкурентов в ``schemas.COMPETITOR_BRANDS``, доменные слова в
``corpus`` и ``web``. Из этого следовало, что второй проект на другом домене
требует правки ``src/``, а «агент переносим» остаётся обещанием.

Что здесь, а что в данных:

- **в данных** (``knowledge/domain/entities.yaml``) — имена субъектов, их
  написания, доменные термины и слова отбора выдачи;
- **в коде** — структуры и правила: как искать субъекта в тексте
  (``pf.canonical``), как считать доменный счёт (``pf.corpus``), как строить
  шаблон заголовка по списку слов.

Три правила, без которых словарь приносит вред:

1. **Нет словаря — ошибка, а не пустой домен.** Пустой набор молчит: отбор
   статей перестаёт быть доменным, а тесты остаются зелёными. Такой отказ
   дороже падения, потому что выглядит как работа.
2. **Кэш привязан к пути и времени изменения файла**, как у словаря близости
   (``pf.lexicon``): пополнение домена действует сразу, без перезапуска
   процесса и без правки кода.
3. **Написания субъекта — часть домена.** Латиница и разговорная форма —
   тот же субъект: они попадают в словарь близости
   (``pf.lexicon.groups``) и в проверку сообщения о чужом бренде.

Запуск проверки: ``make doctor`` печатает, сколько субъектов и терминов
прочитано из файла.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from pf.config import settings

#: Имя файла домена внутри каталога ``domain_dir``.
DOMAIN_NAME = "entities.yaml"

#: Субъект — компания или организация: по нему строится шаблон заголовка.
KIND_OPERATOR = "operator"

#: Субъект — категория рынка («сотовая связь»): шаблоны на ней не строятся.
KIND_CATEGORY = "category"


class DomainError(RuntimeError):
    """Словарь домена отсутствует или непригоден для работы."""


@dataclass(frozen=True, slots=True)
class Subject:
    """Субъект домена: то, чьё имя утверждение может назвать.

    Attributes:
        canonical: Каноническое имя — как оно пишется в утверждении.
        kind: Вид: ``operator`` (компания) или ``category`` (категория рынка).
        aliases: Другие написания того же имени (латиница, разговорная форма).
        country: Страна, если имя без неё неоднозначно.
        ours: Наш ли это бренд. Коммуникация идёт от его имени.
        competitor: Считается ли имя чужим брендом в сообщении.
    """

    canonical: str
    kind: str = KIND_CATEGORY
    aliases: tuple[str, ...] = ()
    country: str = ""
    ours: bool = False
    competitor: bool = False

    @property
    def spellings(self) -> tuple[str, ...]:
        """Все написания имени: каноническое и остальные."""
        return (self.canonical, *self.aliases)


@dataclass(frozen=True, slots=True)
class Dictionary:
    """Прочитанный словарь домена.

    Attributes:
        domain: Имя домена из метаданных («telecom»).
        subjects: Субъекты в порядке файла: порядок задаёт порядок полей.
        strong: Сильные доменные термины — без них совпадение случайно.
        clean: Чистые термины — не встречаются в чужих областях.
        domain_terms: Доменные термины для взвешивания выдачи.
        term_records: Сколько записей терминов в файле, все признаки вместе.
        search_stems: Основы слов отбора новостей и трендов по теме.
        web_domains: Сайты субъектов домена, разрешённые к скачиванию.
        operator_pattern: Шаблон «заголовок называет оператора».
        company_pattern: Шаблон «заголовок называет именно компанию».
        groups: Группы близости «каноническое имя + написания».
        source: Путь файла — для отчёта и сообщений.
    """

    domain: str
    subjects: tuple[Subject, ...]
    strong: frozenset[str]
    clean: frozenset[str]
    domain_terms: frozenset[str]
    term_records: int
    search_stems: frozenset[str]
    web_domains: frozenset[str]
    operator_pattern: re.Pattern[str]
    company_pattern: re.Pattern[str]
    groups: tuple[tuple[str, ...], ...]
    source: Path


def domain_dir() -> Path:
    """Каталог домена проекта.

    Отдельная настройка, а не ``knowledge_dir``: набор тестов уводит
    ``knowledge_dir`` в ``tmp_path`` (`D-29`), и домен, лежащий там, исчез бы
    во всех проверках сразу.

    Returns:
        Каталог из настроек.
    """
    return settings.domain_dir


def entities_path() -> Path:
    """Путь к файлу домена."""
    return domain_dir() / DOMAIN_NAME


def _words(items: list[dict[str, Any]], flag: str) -> frozenset[str]:
    """Слова записей, у которых поднят признак.

    Args:
        items: Записи раздела ``terms``.
        flag: Имя признака (``strong``, ``clean``, ``domain``).

    Returns:
        Множество словоформ в нижнем регистре.
    """
    return frozenset(
        str(item.get("word", "")).strip().lower()
        for item in items
        if item.get(flag) and str(item.get("word", "")).strip()
    )


def _alternation(words: list[str]) -> str:
    """Шаблон «любое из слов», длинные первыми.

    Длинное слово идёт первым, чтобы составное имя находилось целиком, а не
    как короткое с хвостом: иначе проверка компании срабатывала бы на
    дочернюю структуру.

    Args:
        words: Слова в любом порядке.

    Returns:
        Часть регулярного выражения; заведомо несовпадающая, если слов нет.
    """
    ordered = sorted(
        {word.strip().lower() for word in words if word.strip()}, key=len, reverse=True
    )
    if not ordered:
        # Ничего не совпадает: пустая альтернатива совпала бы со всем.
        return r"(?!x)x"
    return "|".join(re.escape(word) for word in ordered)


def _parse(path: Path) -> Dictionary:
    """Прочитать словарь домена.

    Args:
        path: Файл домена.

    Returns:
        Разобранный словарь.

    Raises:
        DomainError: Файла нет, YAML испорчен, нет субъектов или терминов.
    """
    if not path.is_file():
        raise DomainError(
            f"нет словаря домена: {path}. Домен — данные проекта; "
            "без него поиск и проверка теряют предметную область"
        )
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise DomainError(f"словарь домена не читается ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise DomainError(f"словарь домена должен быть словарём YAML: {path}")

    subjects: list[Subject] = []
    for item in payload.get("subjects") or []:
        if not isinstance(item, dict):
            continue
        canonical = str(item.get("canonical", "")).strip()
        if not canonical:
            continue
        subjects.append(
            Subject(
                canonical=canonical,
                kind=str(item.get("kind", KIND_CATEGORY)).strip() or KIND_CATEGORY,
                aliases=tuple(
                    str(alias).strip() for alias in item.get("aliases") or [] if str(alias).strip()
                ),
                country=str(item.get("country", "")).strip(),
                ours=bool(item.get("ours", False)),
                competitor=bool(item.get("competitor", False)),
            )
        )
    if not subjects:
        raise DomainError(f"в словаре домена нет субъектов: {path}")

    terms = [item for item in payload.get("terms") or [] if isinstance(item, dict)]
    domain_terms = _words(terms, "domain")
    if not domain_terms:
        raise DomainError(f"в словаре домена нет терминов: {path}")

    patterns = payload.get("patterns") or {}
    search = payload.get("search") or {}
    return Dictionary(
        domain=str((payload.get("metadata") or {}).get("domain", "")).strip(),
        subjects=tuple(subjects),
        strong=_words(terms, "strong"),
        clean=_words(terms, "clean"),
        domain_terms=domain_terms,
        term_records=len(terms),
        search_stems=frozenset(
            str(word).strip().lower()
            for word in search.get("strong_stems") or []
            if str(word).strip()
        ),
        web_domains=frozenset(
            str(host).strip().lower()
            for host in (payload.get("web") or {}).get("allowed_domains") or []
            if str(host).strip()
        ),
        operator_pattern=re.compile(
            f"({_alternation(list(patterns.get('operator') or []))})", re.IGNORECASE
        ),
        company_pattern=re.compile(
            rf"^({_alternation(list(patterns.get('company') or []))})(?![^()]*\))"
            r"|\((компания|фирма)\)$",
            re.IGNORECASE,
        ),
        groups=tuple(subject.spellings for subject in subjects if subject.aliases),
        source=path,
    )


@lru_cache(maxsize=8)
def _load(path: str, stamp: int) -> Dictionary:
    """Прочитать и закэшировать словарь по версии файла.

    Args:
        path: Путь к файлу.
        stamp: Время изменения файла — ключ кэша.

    Returns:
        Разобранный словарь.
    """
    return _parse(Path(path))


def version() -> int:
    """Версия словаря домена: время изменения файла.

    Нужна там, где кэш строится по нескольким файлам: словарь близости
    (`pf.lexicon`) зависит и от своего файла, и от домена.

    Returns:
        Время изменения файла в наносекундах; ноль, если файла нет.
    """
    path = entities_path()
    return path.stat().st_mtime_ns if path.is_file() else 0


def dictionary() -> Dictionary:
    """Словарь домена с учётом правок файла.

    Returns:
        Разобранный словарь; кэш привязан к пути и времени изменения.

    Raises:
        DomainError: Словаря нет или он непригоден.
    """
    return _load(str(entities_path()), version())


def subjects() -> tuple[Subject, ...]:
    """Субъекты домена в порядке файла.

    Returns:
        Кортеж субъектов; порядок задаёт порядок поля «субъекты» при разборе.
    """
    return dictionary().subjects


def ours() -> Subject | None:
    """Наш бренд, от имени которого идёт коммуникация.

    Returns:
        Субъект с признаком ``ours`` либо ``None``, если домен его не называет.
    """
    return next((subject for subject in subjects() if subject.ours), None)


def competitors() -> tuple[Subject, ...]:
    """Чужие бренды, недопустимые в ключевом сообщении.

    Returns:
        Субъекты с признаком ``competitor``.
    """
    return tuple(subject for subject in subjects() if subject.competitor)


def subject_groups() -> tuple[tuple[str, ...], ...]:
    """Группы близости: каноническое имя и его написания.

    Returns:
        Кортежи слов, которые означают один субъект. Пополняют словарь
        близости (``pf.lexicon``), чтобы латиница и кириллица одного имени не
        считались разными понятиями при разборе утверждения.
    """
    return dictionary().groups


def strong_terms() -> frozenset[str]:
    """Сильные доменные термины: без них совпадение доменного слова случайно."""
    return dictionary().strong


def clean_terms() -> frozenset[str]:
    """Чистые доменные термины: отобраны замером как не встречающиеся в чужих областях."""
    return dictionary().clean


def domain_terms() -> frozenset[str]:
    """Доменные термины предметной области для взвешивания выдачи."""
    return dictionary().domain_terms


def search_stems() -> frozenset[str]:
    """Основы слов отбора новостей и трендов по теме."""
    return dictionary().search_stems


def web_domains() -> frozenset[str]:
    """Сайты субъектов домена, которые разрешено скачивать.

    Общие издания перечислены в настройках: это политика выборки, а не
    свойство домена. Сайты брендов — свойство домена, и второй проект
    приносит свои.

    Returns:
        Домены без ``www.``, в нижнем регистре.
    """
    return dictionary().web_domains


def operator_pattern() -> re.Pattern[str]:
    """Шаблон «текст называет оператора связи»."""
    return dictionary().operator_pattern


def company_pattern() -> re.Pattern[str]:
    """Шаблон «заголовок называет компанию, а не банк или смартфон с тем же именем»."""
    return dictionary().company_pattern


def summary() -> dict[str, Any]:
    """Сводка словаря для человека: ``pf doctor`` и отчёт.

    Returns:
        Имя домена, путь файла и количество записей.
    """
    table = dictionary()
    return {
        "domain": table.domain,
        "source": str(table.source),
        "subjects": len(table.subjects),
        "terms": table.term_records,
        "domain_terms": len(table.domain_terms),
    }
