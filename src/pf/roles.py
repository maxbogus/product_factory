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


@lru_cache(maxsize=32)
def load_prompt(role_name: str) -> str:
    """Загрузить промпт роли.

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
    return path.read_text(encoding="utf-8")


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
