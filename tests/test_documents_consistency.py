"""Согласованность заявлений: документы против кода.

Документ обещает, как агент работает. Здесь проверяются расхождения,
которые уже случались:

- умолчание в `.env.example` разошлось с кодом (лимит ответа подняли,
  а файл, который копируют по инструкции, остался на старом значении);
- одна и та же установка описана двумя разными командами;
- документ называет неподключённым то, что реализовано, и обещает цикл
  работы, которого в CLI нет.

Две проверки — `test_env_example_has_no_unknown_variables` и
`test_commands_named_in_readme_exist` — сегодня зелёные: это страж против
регресса, а не починка. В журнале сессии они названы именно так, иначе
«проверка» ничего не проверяет.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from pf.cli import build_parser
from pf.config import Settings

ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = ROOT / ".env.example"
README = ROOT / "README.md"
BRIEF = ROOT / "demo" / "BRIEF.md"
MATERIALS_SOURCE = ROOT / "src" / "pf" / "materials.py"
MAKEFILE = ROOT / "Makefile"

#: Единственная инструкция для библиотек чтения документов. `README.md` и
#: текст ошибки в `materials.py` обязаны говорить одно и то же: две разные
#: команды означают, что человек выберет не ту.
DOCUMENTS_HINT = "uv sync --extra documents"

#: Настройки, которые сознательно не попадают в `.env.example`. Исключения
#: перечислены явно: молча пропущенных быть не должно.
NOT_IN_EXAMPLE = {
    "allowed_domains": "множество доменов: в формате KEY=VALUE читалось бы как одна строка",
    "user_agent": "технический заголовок запроса к сайтам, менять его незачем",
}

#: Формулировки, обещающие цикл пересмотра. Пока подкоманды `pf review` нет,
#: их в README быть не может: документ обещал бы работу, которой не существует.
REVIEW_CLAIMS = ("проверяется при пересмотре", "проверить при пересмотре")

ENV_LINE = re.compile(r"^\s*(PF_[A-Z0-9_]+)=(.*)$", re.MULTILINE)
MAKE_TARGET = re.compile(r"^([a-z][a-z0-9-]*):", re.MULTILINE)
MAKE_MENTION = re.compile(r"(?:^|\s|`)make ([a-z][a-z0-9-]*)", re.MULTILINE)
PF_MENTION = re.compile(r"(?:^|\s|`)pf ([a-z]+)", re.MULTILINE)


def _example_values() -> dict[str, str]:
    """Значения из `.env.example`.

    Returns:
        Словарь «переменная → значение». Строки-комментарии не читаются.
    """
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    return {key: value.strip() for key, value in ENV_LINE.findall(text)}


def _is_path(field: Any) -> bool:
    """Описание поля — путь, а не настройка поведения.

    Args:
        field: Поле модели настроек.

    Returns:
        ``True``, если поле объявлено как ``Path``.
    """
    annotation = field.annotation
    if annotation is Path:
        return True
    return Path in getattr(annotation, "__args__", ())


def _matches(raw: str, default: object) -> bool:
    """Значение из примера совпадает с умолчанием кода.

    Args:
        raw: Значение, записанное в `.env.example`.
        default: Умолчание поля настроек.

    Returns:
        ``True``, если это одно и то же значение.
    """
    if isinstance(default, bool):
        return raw.lower() == str(default).lower()
    if isinstance(default, int | float):
        try:
            return float(raw.replace(",", ".")) == float(default)
        except ValueError:
            return False
    return raw == str(default)


def _subcommands() -> set[str]:
    """Имена подкоманд ``pf`` из разборщика аргументов.

    Returns:
        Множество имён подкоманд.
    """
    parser = build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("у разборщика аргументов нет подкоманд")


def _make_targets() -> set[str]:
    """Цели, объявленные в Makefile.

    Returns:
        Множество имён целей.
    """
    return set(MAKE_TARGET.findall(MAKEFILE.read_text(encoding="utf-8")))


def test_env_example_matches_defaults() -> None:
    """Значение в примере равно умолчанию кода.

    Иначе человек, который делает `cp .env.example .env`, получает не то
    поведение, которое проверено тестами: так вышло с лимитом ответа
    (в примере 2400, в коде 3200 — обрезанный JSON стратегии).
    """
    values = _example_values()
    problems: list[str] = []
    for name, field in Settings.model_fields.items():
        key = f"PF_{name.upper()}"
        if key not in values:
            continue
        if not _matches(values[key], field.default):
            problems.append(f"{key}: в примере {values[key]!r}, в коде {field.default!r}")
    assert not problems, "пример .env разошёлся с умолчаниями: " + "; ".join(problems)


def test_env_example_documents_scalar_settings() -> None:
    """Каждая скалярная настройка названа в примере.

    Пути исключены правилом: это раскладка каталогов, и описание раскладки
    живёт в `AGENTS.md`, а не в файле переменных.
    """
    values = _example_values()
    missing = sorted(
        f"PF_{name.upper()}"
        for name, field in Settings.model_fields.items()
        if not _is_path(field) and name not in NOT_IN_EXAMPLE and f"PF_{name.upper()}" not in values
    )
    assert not missing, "в .env.example нет настроек: " + ", ".join(missing)


def test_env_example_has_no_unknown_variables() -> None:
    """В примере нет переменных, которых код не знает.

    `extra="ignore"` в настройках глотает опечатку в имени переменной молча:
    строка в `.env` выглядит рабочей и ничего не настраивает.
    """
    known = {f"PF_{name.upper()}" for name in Settings.model_fields}
    unknown = sorted(set(_example_values()) - known)
    assert not unknown, "в .env.example переменные, которых нет в коде: " + ", ".join(unknown)


def test_documents_extra_instruction_is_single() -> None:
    """Инструкция для библиотек документов одна на весь репозиторий."""
    readme = README.read_text(encoding="utf-8")
    materials = MATERIALS_SOURCE.read_text(encoding="utf-8")
    assert DOCUMENTS_HINT in readme, "README не называет установку набора documents"
    assert DOCUMENTS_HINT in materials, (
        "ошибка чтения документа не подсказывает набор documents: человек ищет "
        f"библиотеку по одной команде ({DOCUMENTS_HINT})"
    )
    assert "uv add " not in materials, (
        "вторая инструкция вместо набора documents: `uv add` ставит пакет мимо "
        "закреплённого набора зависимостей"
    )


def test_brief_does_not_call_web_unconnected() -> None:
    """Демо не называет веб-источники неподключёнными, пока они реализованы."""
    text = BRIEF.read_text(encoding="utf-8")
    assert "web" in _subcommands(), "веб-источников нет в CLI — тогда документ прав"
    assert Settings.model_fields["enable_web"].default is False
    lowered = text.lower()
    claim = (
        "demo/BRIEF.md называет веб-источники неподключёнными, хотя подкоманда "
        "`pf web` есть, а выключены они настройкой PF_ENABLE_WEB"
    )
    assert "не подключен" not in lowered, claim
    assert "не подключён" not in lowered, claim
    assert "PF_ENABLE_WEB" in text, "документ обязан сказать, как веб включается"


def test_readme_does_not_promise_review_cycle() -> None:
    """README не обещает цикл пересмотра, которого нет в CLI.

    Хеш версии данных — инфраструктура: по нему видно, что данные изменились.
    Сам пересмотр выводов — отдельная работа, и пока подкоманды `pf review`
    нет, документ не имеет права говорить, что что-то «проверяется при
    пересмотре».
    """
    readme = README.read_text(encoding="utf-8")
    lowered = readme.lower()
    if "review" in _subcommands():
        return
    promised = [phrase for phrase in REVIEW_CLAIMS if phrase in lowered]
    assert not promised, (
        "README говорит, что при пересмотре что-то проверяется, но подкоманды "
        "`pf review` нет: " + ", ".join(promised)
    )
    assert "`pf review`" not in readme, "README называет подкоманду `pf review`, которой нет"


def test_commands_named_in_readme_exist() -> None:
    """Команды, названные в README, существуют: подкоманда или цель Makefile.

    Страж против регресса: сегодня зелёный. Нужен, потому что README —
    инструкция, и её расхождение с деревом обнаруживается только руками.
    """
    readme = README.read_text(encoding="utf-8")
    unknown_pf = sorted(set(PF_MENTION.findall(readme)) - _subcommands())
    unknown_make = sorted(set(MAKE_MENTION.findall(readme)) - _make_targets())
    assert not unknown_pf, "README называет несуществующие подкоманды: " + ", ".join(unknown_pf)
    assert not unknown_make, "README называет несуществующие make-цели: " + ", ".join(unknown_make)
