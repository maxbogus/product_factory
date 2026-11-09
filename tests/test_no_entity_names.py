"""Анти-overfit гейт: имена субъектов — данные, а не код.

Зачем. Домен жил в логике: восемь названий операторов в ``canonical.SUBJECTS``,
пять в ``schemas.COMPETITOR_BRANDS``, восемь доменных слов в ``web``,
шаблоны заголовков в ``corpus``. Из этого следовало три вещи сразу:

1. второй проект на другом домене невозможен — проверка знает телеком-бренды
   из кода, а не из данных задачи;
2. пополнение домена — правка ``src/``, то есть каждая новая сущность
   проходит через код и тесты;
3. утверждение «агент переносим» становилось обещанием, а не фактом.

Теперь домен лежит в ``knowledge/domain/entities.yaml``, а гейт держит
обратное свойство: **ни одного имени в ``src/``**. Гейт смотрит строковые
литералы (то, что действительно исполняется), а не комментарии: комментарий —
пояснение для человека, имя в нём ничего не ломает. Легальный способ
оставить пример — комментарий, а не список исключений: список однажды
вырастет до размера проблемы.

Путь к словарю берётся **от корня репозитория**, а не из ``settings``: набор
тестов уводит ``knowledge_dir`` в ``tmp_path`` (`D-29`), и гейт, читающий
настройки, проверял бы пустое множество — то есть зеленел бы всегда.

Критерий приёмки задачи — ``-k entity_names``: ключ есть во всех проверках
файла, потому что критерий обязан выбирать все тесты задачи (`AGENTS.md`).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

from pf import corpus, domain
from pf.config import settings

#: Корень репозитория: файл лежит в ``tests/``.
ROOT = Path(__file__).resolve().parent.parent

#: Модули агента — то, что поставляется как код.
SRC = ROOT / "src" / "pf"

#: Словарь домена: субъекты и доменные термины.
DOMAIN_FILE = ROOT / "knowledge" / "domain" / "entities.yaml"

#: Инструкции ролей: имя там допустимо только как пример ответа.
ROLES = ROOT / "prompts" / "roles"


def _dictionary_names() -> list[str]:
    """Имена субъектов из файла домена — напрямую, без настроек и кэшей.

    Returns:
        Канонические имена и их написания.

    Raises:
        AssertionError: Файла нет или в нём нет субъектов. Тогда гейт
            проверял бы пустой список и всегда был бы зелёным.
    """
    assert DOMAIN_FILE.is_file(), f"нет словаря домена: {DOMAIN_FILE}"
    payload = yaml.safe_load(DOMAIN_FILE.read_text(encoding="utf-8")) or {}
    names: list[str] = []
    for subject in payload.get("subjects") or []:
        names.append(str(subject.get("canonical", "")).strip())
        names.extend(str(alias).strip() for alias in subject.get("aliases") or [])
    names = [name for name in names if name]
    assert names, "в словаре домена нет имён субъектов — гейт проверял бы пустое множество"
    return names


def _names_pattern() -> re.Pattern[str]:
    """Поиск любого имени домена в тексте.

    Длинные имена идут первыми, чтобы «МТС Беларусь» находилось целиком, а не
    как «МТС» с хвостом. Сравнение без учёта регистра: имя в коде пишут как
    придётся, а правило от этого не меняется.
    """
    names = sorted(set(_dictionary_names()), key=len, reverse=True)
    return re.compile("|".join(re.escape(name) for name in names), re.IGNORECASE)


def _docstring_ids(tree: ast.Module) -> set[int]:
    """Идентификаторы узлов-докстрингов: строки, а не код."""
    found: set[int] = set()
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, owners):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.add(id(first.value))
    return found


def _literals(path: Path, *, docstrings: bool) -> list[tuple[int, str]]:
    """Строковые литералы файла: код или докстринги.

    Комментарии сюда не попадают по устройству разбора: это не литералы.
    Именно поэтому они и служат допустимым местом для примера.

    Args:
        path: Файл модуля.
        docstrings: Взять докстринги (``True``) или исполняемые литералы.

    Returns:
        Пары «номер строки, значение».
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = _docstring_ids(tree)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if (id(node) in docs) is not docstrings:
            continue
        found.append((node.lineno, node.value))
    return found


def _offenders(*, docstrings: bool) -> list[str]:
    """Где имя домена попало в текст модулей."""
    pattern = _names_pattern()
    found: list[str] = []
    for path in sorted(SRC.glob("*.py")):
        for line, value in _literals(path, docstrings=docstrings):
            hit = pattern.search(value)
            if hit:
                found.append(f"{path.name}:{line} — «{hit.group(0)}»")
    return found


def test_entity_names_absent_from_code() -> None:
    """Ни одного имени субъекта в исполняемых литералах ``src/``.

    Имя в коде — это домен, зашитый в логику: второй проект на другом домене
    потребует правки ``src/``, и «переносимость» останется обещанием. Домен
    читается из ``knowledge/domain/entities.yaml``, поэтому имя в коде — не
    оформление, а дефект.
    """
    found = _offenders(docstrings=False)

    assert not found, (
        "имя субъекта в коде: домен должен читаться из knowledge/domain/entities.yaml.\n"
        + "\n".join(found)
        + "\nЕсли это пример для человека — перенесите его в комментарий (гейт читает литералы)."
    )


def test_entity_names_absent_from_docstrings() -> None:
    """Докстринги называют субъектов нейтрально: «Оператор А», «компания А».

    Докстринг — тоже текст, по которому читают механику. Пример с реальным
    брендом делает объяснение непереносимым: в проекте про другой домен он
    будет описывать чужую задачу.
    """
    found = _offenders(docstrings=True)

    assert not found, (
        "имя субъекта в докстринге: пример должен называть роль, а не бренд "
        "(«Оператор А», «компания А»).\n" + "\n".join(found)
    )


def test_entity_names_in_prompts_only_as_example() -> None:
    """В инструкциях ролей имя допустимо только внутри раздела «Пример».

    Инструкция — правило поведения; пример ответа — данные. Разделять их
    нужно потому, что пример задаёт форму JSON, а правило — что агент
    делает. Имя в правиле снова привязало бы агента к одному домену.
    """
    pattern = _names_pattern()
    problems: list[str] = []
    for path in sorted(ROLES.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            hit = pattern.search(line)
            if not hit:
                continue
            if not _inside_example(lines, number - 1):
                problems.append(f"{path.name}:{number} — «{hit.group(0)}»")

    assert not problems, (
        "имя субъекта в правиле промпта (допустимо только в разделе «Пример»):\n"
        + "\n".join(problems)
    )


def _inside_example(lines: list[str], index: int) -> bool:
    """Относится ли строка к разделу, названному «Пример».

    Args:
        lines: Строки файла.
        index: Номер строки с нуля.

    Returns:
        Ближайший заголовок выше называет пример.
    """
    for previous in reversed(lines[:index]):
        stripped = previous.strip()
        if stripped.startswith("#"):
            return "пример" in stripped.lower()
    return False


def test_entity_names_second_domain_needs_no_code_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Второй домен задаётся файлом: код и тесты править не нужно.

    Это обратная сторона гейта. Проверка, которая смотрит только на
    отсутствие имён, ничего не говорит о переносимости: словарь мог остаться
    в коде под другим именем. Поэтому здесь домен подменяется целиком, и
    проверка обязана увидеть биологию и перестать видеть телеком.
    """
    payload = {
        "version": 1,
        "metadata": {"description": "второй домен", "domain": "biology"},
        "subjects": [
            {"canonical": "диабет 2 типа", "kind": "category"},
            {"canonical": "инсулин", "kind": "category"},
        ],
        "terms": [
            {"word": "поджелудочная", "strong": True, "clean": True, "domain": True},
        ],
    }
    (tmp_path / "entities.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8"
    )
    monkeypatch.setattr(settings, "domain_dir", tmp_path)

    names = {subject.canonical for subject in domain.subjects()}

    assert names == {"диабет 2 типа", "инсулин"}, names
    assert "МТС" not in names
    assert corpus.is_domain_query("поджелудочная железа"), "доменный термин берётся из файла"
    assert not corpus.is_domain_query("абонентская база"), "телеком-терминов больше нет"


def test_entity_names_dictionary_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Нет словаря домена — ошибка, а не пустой домен.

    Пустой набор молчит: поиск деградирует, отбор статей перестаёт быть
    доменным, а `make ci` остаётся зелёным. Такой отказ дороже падения,
    потому что выглядит как работа.
    """
    monkeypatch.setattr(settings, "domain_dir", tmp_path / "пусто")

    with pytest.raises(domain.DomainError):
        domain.subjects()
