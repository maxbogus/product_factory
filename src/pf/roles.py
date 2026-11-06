"""Роли агента: инструкция + схема вывода.

Каждая роль — это **файл промпта** в ``prompts/roles/`` и **схема**,
в которую модель обязана уложиться. Промпты вынесены в файлы намеренно:
содержание работы агента должно правиться без перезапуска и без
перекомпиляции, иначе его невозможно отлаживать.

Роли не «говорят», а **производят артефакт**. Каждая отвечает на свой
вопрос и ничего сверх того:

- интик — что именно нужно понять;
- план — какие вопросы могут изменить решение;
- сбор — на какие источники опираться;
- анализ — что известно, что предположено, чего мы не знаем;
- варианты — какие есть содержательно разные пути;
- red-team — что не так с каждым вариантом;
- стратегия — что делаем и как это проверим;
- keeper — что зафиксировали и когда пересмотрим;
- роутер — что делать дальше.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from pf.config import settings
from pf.schemas import (
    Analysis,
    DecisionRecord,
    NextStep,
    RedTeamReport,
    ResearchPlan,
    Strategy,
    TaskFrame,
    Variants,
)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class Role:
    """Роль агента.

    Attributes:
        name: Идентификатор роли.
        prompt_file: Имя файла промпта в ``prompts/roles/``.
        schema: Схема вывода роли.
        title_ru: Человекочитаемое название роли.
    """

    name: str
    prompt_file: str
    schema: type[BaseModel]
    title_ru: str


ROLES: dict[str, Role] = {
    "intake": Role(
        name="intake",
        prompt_file="intake.md",
        schema=TaskFrame,
        title_ru="Постановка задачи",
    ),
    "planner": Role(
        name="planner",
        prompt_file="planner.md",
        schema=ResearchPlan,
        title_ru="План исследования",
    ),
    "collector": Role(
        name="collector",
        prompt_file="collector.md",
        schema=Analysis,
        title_ru="Сбор источников",
    ),
    "analyst": Role(
        name="analyst",
        prompt_file="analyst.md",
        schema=Analysis,
        title_ru="Разбор источников",
    ),
    "options": Role(
        name="options",
        prompt_file="options.md",
        schema=Variants,
        title_ru="Варианты решения",
    ),
    "redteam": Role(
        name="redteam",
        prompt_file="redteam.md",
        schema=RedTeamReport,
        title_ru="Критическая проверка",
    ),
    "strategy": Role(
        name="strategy",
        prompt_file="strategy.md",
        schema=Strategy,
        title_ru="Разработка стратегии",
    ),
    "keeper": Role(
        name="keeper",
        prompt_file="keeper.md",
        schema=DecisionRecord,
        title_ru="Фиксация решения",
    ),
    "router": Role(
        name="router",
        prompt_file="router.md",
        schema=NextStep,
        title_ru="Маршрутизация",
    ),
}


@lru_cache(maxsize=1)
def prompts_root() -> Path:
    """Каталог с промптами ролей."""
    return settings.prompts_dir / "roles"


def load_prompt(role_name: str) -> str:
    """Загрузить промпт роли вместе с методическими знаниями.

    Кэширование — по времени изменения файлов, а не по имени роли. Иначе
    правка промпта или методики не подхватывалась бы до перезапуска, а
    обещание «правится без перезапуска» оказалось бы неправдой.

    Args:
        role_name: Идентификатор роли.

    Returns:
        Текст инструкции.

    Raises:
        KeyError: Роли нет в реестре.
        FileNotFoundError: Файл промпта не найден.
    """
    role = ROLES[role_name]
    path = prompts_root() / role.prompt_file
    if not path.exists():
        raise FileNotFoundError(
            f"не найден файл промпта роли «{role.title_ru}»: {path}. "
            "Без него агент не знает, что делать."
        )
    prompt = path.read_text(encoding="utf-8")
    return _with_skills(_with_methods(prompt, role_name), role_name)


def _prompt_stamp(role_name: str) -> tuple[float, ...]:
    """Отметка времени изменения файлов знаний роли.

    Args:
        role_name: Идентификатор роли.

    Returns:
        Временные метки файлов: промпт, его методики и его скилы.
    """
    role = ROLES.get(role_name)
    if role is None:
        return ()
    path = prompts_root() / role.prompt_file
    stamps = [path.stat().st_mtime if path.exists() else 0.0]
    from pf.knowledge import ROLE_METHODS, ROLE_SKILLS

    for name in ROLE_METHODS.get(role_name, ()):
        method = settings.methods_dir / f"{name}.md"
        stamps.append(method.stat().st_mtime if method.exists() else 0.0)
    for name in ROLE_SKILLS.get(role_name, ()):
        skill = settings.skills_dir / f"{name}.md"
        stamps.append(skill.stat().st_mtime if skill.exists() else 0.0)
    return tuple(stamps)


def _with_methods(prompt: str, role_name: str) -> str:
    """Дополнить инструкцию роли методическими знаниями.

    Методики из ``methods/`` — второй слой знаний по брифу: постоянные
    инструкции говорят «что делать», методики — «по каким правилам».
    До подключения они лежали в файлах, но модель их не видела.

    Объём ограничен: локальная модель небольшого размера теряет формат
    ответа на длинной инструкции. Методики — самый дешёвый способ это
    испортить, поэтому они подставляются последними и коротко.

    Args:
        prompt: Текст инструкции роли.
        role_name: Идентификатор роли.

    Returns:
        Инструкция с методиками.
    """
    from pf.knowledge import knowledge_for

    methods = knowledge_for(role_name)
    if not methods:
        return prompt
    return (
        f"{prompt}\n\n---\n\n"
        "## Методические знания\n\n"
        "Это правила работы, общие для всех ролей. Они уточняют инструкцию "
        "выше, но не отменяют её.\n\n"
        f"{methods}"
    )


def _with_skills(prompt: str, role_name: str) -> str:
    """Дополнить инструкцию роли скилами.

    Скил — повторяемая процедура: не «как думать», а «как выполнить».
    Методики отвечают на первое, скилы — на второе. Без подключения
    скилы остаются файлами, которые читает человек, а модель их не видит.

    Args:
        prompt: Текст инструкции роли.
        role_name: Идентификатор роли.

    Returns:
        Инструкция со скилами.
    """
    from pf.knowledge import skills_for

    skills = skills_for(role_name)
    if not skills:
        return prompt
    return (
        f"{prompt}\n\n---\n\n"
        "## Процедуры (скилы)\n\n"
        "Это повторяемые процедуры работы. Выполняй подходящую под задачу; "
        "если ни одна не подходит — не упоминай их.\n\n"
        f"{skills}"
    )


def schema_for(role_name: str) -> type[BaseModel]:
    """Схема вывода роли.

    Args:
        role_name: Идентификатор роли.

    Returns:
        Класс схемы.

    Raises:
        KeyError: Роли нет в реестре.
    """
    return ROLES[role_name].schema


def validate(role_name: str, payload: Any) -> BaseModel:
    """Проверить ответ роли по её схеме.

    Args:
        role_name: Идентификатор роли.
        payload: Разобранный ответ модели.

    Returns:
        Валидированный объект схемы.

    Raises:
        KeyError: Роли нет в реестре.
        pydantic.ValidationError: Ответ не соответствует схеме.
    """
    return ROLES[role_name].schema.model_validate(payload)


def role_titles() -> list[str]:
    """Названия всех ролей — для документации и подсказок."""
    return [role.title_ru for role in ROLES.values()]
