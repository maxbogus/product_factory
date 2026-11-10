"""Тематический справочник: знание о применимости, отдельное от оснований.

Зачем модуль. Агенту не хватало экспертного слоя понятий домена: «что такое
MVNO», «когда eSIM уместна», «что показывает ARPU» — это рамка мышления, а
не факт о конкретном кейсе. Без неё модель опиралась на веса, а не на явный
источник знания о применимости.

Три правила, ради которых всё написано:

1. **Справочник — данные, а не код.** Понятия лежат в
   ``knowledge/domain/reference.yaml``; при смене темы правится файл, а не
   ``src/`` (то же правило, что у домена, `T-67`).
2. **Справочное знание не подтверждает факт.** Дословная цитата из
   справочника проверяется, но статус — «требует проверки»: правило
   ``domain-reference-is-not-case-fact``. Иначе знание о применимости
   выдавалось бы за факт о кейсе — подмена источника.
3. **Нет файла — пустой блок, а не ошибка.** Второй проект может не иметь
   справочника; тогда его просто нет в контексте. Это отличается от домена,
   где пустой словарь — ошибка: без домена поиск теряет предметную область.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from pf.config import settings

#: Имя файла справочника внутри каталога домена.
REFERENCE_NAME = "reference.yaml"

#: Префикс ссылки на понятие справочника: ``reference/<id>``.
REFERENCE_PREFIX = "reference/"


@dataclass(frozen=True, slots=True)
class ReferenceEntry:
    """Понятие справочника и условия его применимости.

    Attributes:
        id: Идентификатор понятия — попадает в ссылку ``reference/<id>``.
        term: Название понятия для человека.
        what: Что это такое.
        applies_when: Когда знание применимо.
        not_when: Когда знание неприменимо — чтобы им не подменяли факт.
    """

    id: str
    term: str
    what: str
    applies_when: str = ""
    not_when: str = ""


def reference_path() -> Path:
    """Путь к файлу справочника.

    Returns:
        ``<domain_dir>/reference.yaml``.
    """
    return settings.domain_dir / REFERENCE_NAME


def version() -> int:
    """Время изменения файла справочника.

    Returns:
        Наносекунды изменения; ноль, если файла нет.
    """
    path = reference_path()
    return path.stat().st_mtime_ns if path.is_file() else 0


@lru_cache(maxsize=8)
def _load(path_key: str, stamp: int) -> tuple[ReferenceEntry, ...]:
    """Прочитать справочник, привязав кэш к пути и времени изменения файла.

    Args:
        path_key: Путь к файлу строкой.
        stamp: Время изменения файла — часть ключа кэша.

    Returns:
        Понятия в порядке файла; пустой кортеж, если файла нет или он пуст.
    """
    del stamp  # ключ кэша: путь и время изменения
    path = Path(path_key)
    if not path.is_file():
        return ()
    payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = payload.get("reference") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return ()

    entries: list[ReferenceEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        entry_id = str(item.get("id", "")).strip()
        term = str(item.get("term", "")).strip()
        what = str(item.get("what", "")).strip()
        if not (entry_id and term and what):
            continue
        entries.append(
            ReferenceEntry(
                id=entry_id,
                term=term,
                what=what,
                applies_when=str(item.get("applies_when", "")).strip(),
                not_when=str(item.get("not_when", "")).strip(),
            )
        )
    return tuple(entries)


def entries() -> tuple[ReferenceEntry, ...]:
    """Понятия справочника с учётом правок файла.

    Returns:
        Кортеж понятий; пустой, если справочника нет.
    """
    return _load(str(reference_path()), version())


def source_id_of(entry_id: str) -> str:
    """Ссылка на понятие справочника.

    Args:
        entry_id: Идентификатор понятия.

    Returns:
        ``reference/<id>``.
    """
    return f"{REFERENCE_PREFIX}{entry_id}"


def is_reference(source_id: str) -> bool:
    """Ссылается ли идентификатор на понятие справочника.

    Args:
        source_id: Идентификатор из ответа модели, возможно с якорем.

    Returns:
        ``True``, если ссылка ведёт в справочник.
    """
    clean = str(source_id).split("#", 1)[0].strip()
    return clean.startswith(REFERENCE_PREFIX)


def text_of(source_id: str) -> str:
    """Текст понятия справочника по ссылке.

    Нужен проверке дословности: цитата сверяется с текстом понятия, а не с
    тем, что модель о нём помнит.

    Args:
        source_id: Ссылка ``reference/<id>`` (возможно, с якорем).

    Returns:
        Текст понятия либо пустая строка, если понятия нет.
    """
    clean = str(source_id).split("#", 1)[0].strip()
    if not clean.startswith(REFERENCE_PREFIX):
        return ""
    entry_id = clean[len(REFERENCE_PREFIX) :]
    for entry in entries():
        if entry.id == entry_id:
            return _entry_text(entry)
    return ""


def _entry_text(entry: ReferenceEntry) -> str:
    """Собрать текст понятия для контекста и проверки цитаты.

    Args:
        entry: Понятие справочника.

    Returns:
        Текст с условиями применимости.
    """
    parts = [f"{entry.term}. {entry.what}"]
    if entry.applies_when:
        parts.append(f"Применимо, когда: {entry.applies_when}")
    if entry.not_when:
        parts.append(f"Неприменимо, когда: {entry.not_when}")
    return " ".join(parts)


def as_lines() -> list[str]:
    """Строки блока «СПРАВОЧНИК» — по одной на понятие.

    Ссылка ``reference/<id>`` показана модели явно: если она захочет
    опереться на понятие, проверка увидит, что это справочник, а не факт о
    кейсе.

    Returns:
        Список строк; пустой, если справочника нет.
    """
    return [f"- [{source_id_of(entry.id)}] {_entry_text(entry)}" for entry in entries()]


def clear_cache() -> None:
    """Сбросить кэш — нужно после правки файла справочника."""
    _load.cache_clear()
