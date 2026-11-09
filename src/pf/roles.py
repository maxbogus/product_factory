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

from pf.config import Settings, settings
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
        purpose: Что роль даёт конвейеру — одна строка для отчёта. Стоимость
            шага без этого описания не говорит, оправдан ли расход
            (`T-70`).
    """

    name: str
    prompt_file: str
    schema: type[BaseModel]
    title_ru: str
    purpose: str


ROLES: dict[str, Role] = {
    "intake": Role(
        name="intake",
        prompt_file="intake.md",
        schema=TaskFrame,
        title_ru="Постановка задачи",
        purpose="что именно нужно понять",
    ),
    "planner": Role(
        name="planner",
        prompt_file="planner.md",
        schema=ResearchPlan,
        title_ru="План исследования",
        purpose="какие вопросы могут изменить решение",
    ),
    "collector": Role(
        name="collector",
        prompt_file="collector.md",
        schema=Analysis,
        title_ru="Сбор источников",
        purpose="на какие источники опираться",
    ),
    "analyst": Role(
        name="analyst",
        prompt_file="analyst.md",
        schema=Analysis,
        title_ru="Разбор источников",
        purpose="что известно, что предположено, чего мы не знаем",
    ),
    "options": Role(
        name="options",
        prompt_file="options.md",
        schema=Variants,
        title_ru="Варианты решения",
        purpose="какие есть содержательно разные пути",
    ),
    "redteam": Role(
        name="redteam",
        prompt_file="redteam.md",
        schema=RedTeamReport,
        title_ru="Критическая проверка",
        purpose="что не так с каждым вариантом",
    ),
    "strategy": Role(
        name="strategy",
        prompt_file="strategy.md",
        schema=Strategy,
        title_ru="Разработка стратегии",
        purpose="что делаем и как это проверим",
    ),
    "keeper": Role(
        name="keeper",
        prompt_file="keeper.md",
        schema=DecisionRecord,
        title_ru="Фиксация решения",
        purpose="что зафиксировали и когда пересмотрим",
    ),
    "router": Role(
        name="router",
        prompt_file="router.md",
        schema=NextStep,
        title_ru="Маршрутизация",
        purpose="что делать дальше",
    ),
}


#: Ярус роли: от него зависит модель. Извлечение (intake, collector, analyst,
#: router, keeper) не требует сильной модели; план и противоречия (planner,
#: redteam) — средняя; синтез (options, strategy) — сильная. Это политика, а
#: не настройка: настройка — имя модели, потому что агент не знает, какие
#: модели скачаны в Ollama.
ROLE_TIERS: dict[str, str] = {
    "intake": "extraction",
    "collector": "extraction",
    "analyst": "extraction",
    "router": "extraction",
    "keeper": "extraction",
    "planner": "planning",
    "redteam": "planning",
    "options": "synthesis",
    "strategy": "synthesis",
}


def tier_of(role_name: str) -> str:
    """Ярус роли: от него зависит модель.

    Args:
        role_name: Идентификатор роли.

    Returns:
        ``extraction``, ``planning`` или ``synthesis``; неизвестная роль
        считается извлечением — дешёвый ярус безопасен по умолчанию.
    """
    return ROLE_TIERS.get(role_name, "extraction")


def model_for(role_name: str, config: Settings | None = None) -> str:
    """Модель для роли.

    Пустое значение яруса в настройках означает «модель по умолчанию»
    (``effective_model``). Так поведение без настройки не меняется, а
    экономия включается, когда человек задал имена моделей.

    Args:
        role_name: Идентификатор роли.
        config: Настройки. По умолчанию — из окружения.

    Returns:
        Имя модели провайдера для этой роли.
    """
    cfg = config or settings
    overrides = {
        "extraction": cfg.role_model_extraction,
        "planning": cfg.role_model_planning,
        "synthesis": cfg.role_model_synthesis,
    }
    return overrides.get(tier_of(role_name), "") or cfg.effective_model


#: Как роли называют себя в ответе. Модель отвечает по-русски и в
#: свободной форме («план», «сбор источников», «критик»), а реестр —
#: английские идентификаторы. Без этой таблицы решение роутера молча
#: отбрасывалось как «неизвестное», и работа шла по порядку.
ROLE_ALIASES: dict[str, str] = {
    "intake": "intake",
    "постановка": "intake",
    "постановка задачи": "intake",
    "уточнение": "intake",
    "planner": "planner",
    "план": "planner",
    "планировщик": "planner",
    "план исследования": "planner",
    "router": "router",
    "маршрут": "router",
    "маршрутизация": "router",
    "collector": "collector",
    "сбор": "collector",
    "сборщик": "collector",
    "сбор источников": "collector",
    "analyst": "analyst",
    "анализ": "analyst",
    "аналитик": "analyst",
    "разбор": "analyst",
    "разбор источников": "analyst",
    "options": "options",
    "варианты": "options",
    "варианты решения": "options",
    "генератор": "options",
    "redteam": "redteam",
    "критика": "redteam",
    "критик": "redteam",
    "критическая проверка": "redteam",
    "strategy": "strategy",
    "стратегия": "strategy",
    "стратег": "strategy",
    "keeper": "keeper",
    "решение": "keeper",
    "фиксация": "keeper",
    "фиксация решения": "keeper",
}


def resolve_role(name: str) -> str:
    """Привести название роли к идентификатору реестра.

    Сначала — таблица синонимов, затем — русские названия из реестра
    (``title_ru``), затем — само название: неизвестную роль вызывающий
    код обработает как ошибку маршрута.

    Args:
        name: Название роли из ответа модели.

    Returns:
        Идентификатор роли либо нормализованное название.
    """
    key = " ".join(name.strip().lower().split())
    if key in ROLE_ALIASES:
        return ROLE_ALIASES[key]
    for role in ROLES.values():
        if key == " ".join(role.title_ru.lower().split()):
            return role.name
    return key


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
