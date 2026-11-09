"""Слои знаний: постоянные инструкции, методики, скилы.

Бриф требует, чтобы знания хранились раздельно: постоянные инструкции
ролей, методические знания, материалы проекта, состояние задачи. Первые
три лежат в файлах — но методики и скилы не были подключены к работе
агента: они читались человеком, а модель их не видела.

Здесь они собираются и подставляются в инструкцию роли. Файлы остаются
правкой без перезапуска — просто теперь агент их читает.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pf.config import settings


@lru_cache(maxsize=1)
def _method_files() -> list[Path]:
    """Файлы методик.

    Returns:
        Список путей к методикам.
    """
    directory = settings.methods_dir
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.md"))


@lru_cache(maxsize=1)
def _skill_files() -> list[Path]:
    """Файлы скилов.

    Returns:
        Список путей к скилам.
    """
    directory = settings.skills_dir
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.md"))


def read_methods() -> str:
    """Собрать текст методик.

    Returns:
        Текст всех методик через разделитель.
    """
    blocks = []
    for path in _method_files():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            blocks.append(f"### {path.stem}\n\n{text}")
    return "\n\n---\n\n".join(blocks)


def read_skills() -> str:
    """Собрать текст скилов.

    Returns:
        Текст всех скилов через разделитель.
    """
    blocks = []
    for path in _skill_files():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            blocks.append(f"### {path.stem}\n\n{text}")
    return "\n\n---\n\n".join(blocks)


#: Методика — для каких ролей она своя. Подстановка всех методик в каждую
#: роль удваивала расход: промпт вырастал с 1,7 тыс. до 10,5 тыс. символов,
#: а токены — с 16 тыс. до 33 тыс. на прогон, при выигрыше в одну цитату.
#: Роли получают только свои методики.
ROLE_METHODS: dict[str, tuple[str, ...]] = {
    "intake": ("fact-hypothesis-forecast",),
    "planner": ("fact-hypothesis-forecast",),
    "collector": ("fact-hypothesis-forecast",),
    "analyst": ("fact-hypothesis-forecast",),
    "options": ("variants",),
    "redteam": ("variants",),
    "strategy": ("strategy-and-verification",),
    "keeper": ("decision-keeper",),
    "router": (),
}

#: Скилы — повторяемые процедуры. Они лежали в файлах, но модель их не
#: видела: скил был написан для человека, а не подключён к работе.
#: Теперь скил подставляется в инструкцию своей роли — как методика,
#: но описывает не «как думать», а «как выполнить процедуру».
ROLE_SKILLS: dict[str, tuple[str, ...]] = {
    "analyst": ("analyze-data", "mock-respondents", "detect-contradictions"),
    "collector": ("collect-sources",),
}


def skills_for(role_name: str) -> str:
    """Собрать скилы для роли.

    Args:
        role_name: Идентификатор роли.

    Returns:
        Текст скилов, пригодных для роли, либо пустая строка.
    """
    names = ROLE_SKILLS.get(role_name)
    if not names:
        return ""
    blocks: list[str] = []
    for name in names:
        path = settings.skills_dir / f"{name}.md"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            blocks.append(f"### {name}\n\n{text}")
    return "\n\n---\n\n".join(blocks)


def knowledge_for(role_name: str) -> str:
    """Собрать методические знания для роли.

    Args:
        role_name: Идентификатор роли.

    Returns:
        Текст методик, релевантных роли, либо пустая строка.
    """
    names = ROLE_METHODS.get(role_name)
    if names is None:
        return ""
    blocks = []
    for name in names:
        path = settings.methods_dir / f"{name}.md"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            blocks.append(f"### {name}\n\n{text}")
    return "\n\n---\n\n".join(blocks)


def clear_cache() -> None:
    """Сбросить кэш — нужно после правки файлов знаний.

    Сбрасывается и кэш собранных промптов: он тоже зависит от файлов
    знаний.
    """
    _method_files.cache_clear()
    _skill_files.cache_clear()
    # Кэш промптов тоже зависит от файлов знаний. Импорт внутри функции:
    # roles и knowledge ссылаются друг на друга.
    from pf import roles

    cache = getattr(roles, "_PROMPT_CACHE", None)
    if cache is not None:
        cache.clear()
