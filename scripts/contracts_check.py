#!/usr/bin/env python3
"""Гейт контрактов: реестры против кода.

Правила описаны в `architecture/rules/fitness-rules.yaml` — это источник
истины. Скрипт их исполняет и ничего не дублирует: правило, которого
нет в реестре, не проверяется.

Проверяется:
- зоны доверия объявлены, а потоки ссылаются на объявленные зоны;
- в графе потоков нет цикла с записью, пересекающего границу доверия;
- в журнал пишет только зона проверки оснований;
- у каждого заявленного числа есть источник и проверка;
- для реализованной возможности существуют модуль и тест;
- каждый сценарий из `tests/features` покрыт исполняемым тестом.

Запуск:
    uv run python scripts/contracts_check.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any

import yaml

try:  # pragma: no cover — зависит от окружения
    from parse import parse as _parse_library
except ImportError:  # pragma: no cover — pytest-bdd без parse не работает
    _parse_library = None

#: Разбор шаблонов шагов вида ``... {имя} ...``. Тип ``Any``: библиотека
#: необязательная, и ``None`` здесь — законное состояние.
parse_pattern: Any = _parse_library

ROOT = Path(__file__).resolve().parent.parent
RULES_DIR = ROOT / "architecture" / "rules"
FEATURES_DIR = ROOT / "tests" / "features"


def _load(name: str) -> dict[str, Any]:
    """Прочитать реестр из architecture/rules.

    Args:
        name: Имя файла реестра.

    Returns:
        Содержимое реестра.
    """
    path = RULES_DIR / name
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _zones() -> dict[str, dict[str, Any]]:
    """Объявленные зоны доверия.

    Returns:
        Словарь зон по идентификатору.
    """
    payload = _load("trust-zones.yaml")
    zones = payload.get("zones")
    if not isinstance(zones, list):
        return {}
    return {str(zone.get("id")): zone for zone in zones if isinstance(zone, dict)}


def _flows() -> list[dict[str, Any]]:
    """Объявленные потоки данных.

    Returns:
        Список потоков.
    """
    payload = _load("trust-flows.yaml")
    flows = payload.get("flows")
    if not isinstance(flows, list):
        return []
    return [flow for flow in flows if isinstance(flow, dict)]


def _rules() -> list[dict[str, Any]]:
    """Fitness-правила.

    Returns:
        Список правил.
    """
    payload = _load("fitness-rules.yaml")
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return []
    return [rule for rule in rules if isinstance(rule, dict)]


def check_zones_declared() -> list[str]:
    """Каждая зона в потоках объявлена в реестре зон.

    Returns:
        Список нарушений.
    """
    zones = _zones()
    problems: list[str] = []
    for flow in _flows():
        for side in ("from", "to"):
            zone = str(flow.get(side, ""))
            if zone and zone not in zones:
                problems.append(
                    f"поток {flow.get('from')} -> {flow.get('to')}: зона «{zone}» не объявлена"
                )
    return problems


def check_acyclic() -> list[str]:
    """Нет цикла с записью, пересекающего границу доверия.

    Returns:
        Список нарушений.
    """
    flows = _flows()
    edges = [
        (str(flow.get("from")), str(flow.get("to")))
        for flow in flows
        if flow.get("direction") == "write"
    ]
    problems: list[str] = []
    for start, target in edges:
        if start == target:
            continue
        # Ищем путь обратно: target -> ... -> start. Такой путь означает,
        # что данные проходят границу доверия без контроля.
        stack = [(target, [target])]
        while stack:
            node, path = stack.pop()
            for source, destination in edges:
                if source != node:
                    continue
                if destination == start:
                    problems.append("цикл с записью: " + " -> ".join([start, *path, destination]))
                elif destination not in path:
                    stack.append((destination, [*path, destination]))
    return problems


def check_write_sources_subset() -> list[str]:
    """Запись в журнал идёт только из зоны проверки оснований.

    Returns:
        Список нарушений.
    """
    rules = {str(rule.get("id")): rule for rule in _rules()}
    rule = rules.get("only-grounding-gate-writes-journal")
    if not rule:
        return []
    allowed = {str(zone) for zone in rule.get("allowed_write_zones") or []}
    problems: list[str] = []
    for flow in _flows():
        if flow.get("direction") != "write":
            continue
        source = str(flow.get("from", ""))
        if source not in allowed:
            problems.append(
                f"поток {source} -> {flow.get('to')}: запись разрешена только из {sorted(allowed)}"
            )
    return problems


def check_claims_have_verification() -> list[str]:
    """У каждого заявленного числа есть источник и проверка.

    Returns:
        Список нарушений.
    """
    payload = _load("claims.yaml")
    claims = payload.get("claims")
    if not isinstance(claims, list):
        return []
    problems: list[str] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        identifier = str(claim.get("id", "?"))
        kind = str(claim.get("kind", ""))
        source = str(claim.get("source", "")).strip()
        verified_by = str(claim.get("verified_by", "")).strip()
        if kind == "computed":
            if not source:
                problems.append(f"заявление {identifier}: у посчитанного числа нет источника")
            if not verified_by:
                problems.append(f"заявление {identifier}: у посчитанного числа нет проверки")
            elif "::" not in verified_by:
                problems.append(
                    f"заявление {identifier}: проверка должна быть именем теста «файл::тест»"
                )
            else:
                test_path = verified_by.split("::", 1)[0]
                if not (ROOT / test_path).is_file():
                    problems.append(f"заявление {identifier}: тест {test_path} не найден")
        elif kind == "target":
            if not verified_by:
                problems.append(f"заявление {identifier}: у цели нет проверки")
            if "цель" not in str(claim.get("note", "")).lower():
                problems.append(f"заявление {identifier}: цель обязана быть помечена словом «цель»")
    return problems


def check_implementation_status() -> list[str]:
    """Для реализованного существуют модуль и тест.

    Returns:
        Список нарушений.
    """
    payload = _load("implementation-status.yaml")
    methods = payload.get("methods")
    if not isinstance(methods, list):
        return []
    problems: list[str] = []
    for method in methods:
        if not isinstance(method, dict):
            continue
        identifier = str(method.get("id", "?"))
        status = str(method.get("status", ""))
        module = str(method.get("module", "")).strip()
        test = str(method.get("test", "")).strip()
        reason = str(method.get("reason", "")).strip()
        if status == "implemented":
            if not module:
                problems.append(f"возможность {identifier}: реализована, но не указан модуль")
            elif not (ROOT / module).is_file():
                problems.append(f"возможность {identifier}: модуль {module} не найден")
            if not test:
                problems.append(f"возможность {identifier}: реализована, но не указан тест")
            elif not (ROOT / test).is_file():
                problems.append(f"возможность {identifier}: тест {test} не найден")
        elif not reason:
            problems.append(f"возможность {identifier}: статус «{status}» без указания причины")
    return problems


#: Ключевые слова шагов Gherkin: русские и английские.
_STEP_KEYWORD = re.compile(r"^(Допустим|Если|Когда|Тогда|И|Но|Given|When|Then|And|But)\s+")


def _normalize(text: str) -> str:
    """Нормализовать текст шага или имя сценария для сравнения.

    Кавычки, запятые, дефисы и регистр — это оформление, а не смысл:
    шаг «И метрика имеет формулу "строка 1"» и определение
    ``@given('метрика имеет формулу "строка 1"')`` описывают одно и то
    же действие.

    Args:
        text: Текст шага, имя сценария или путь к файлу.

    Returns:
        Строка из строчных букв и цифр.
    """
    return re.sub(r"[^0-9a-zа-яё]", "", text.lower())


def _feature_steps(path: Path) -> dict[str, list[str]]:
    """Сценарии файла и тексты их шагов.

    Args:
        path: Файл сценариев.

    Returns:
        Словарь «имя сценария → список шагов».
    """
    result: dict[str, list[str]] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("Сценарий:", "Scenario:")):
            current = line.split(":", 1)[1].strip()
            result[current] = []
            continue
        if current is None:
            continue
        match = _STEP_KEYWORD.match(line)
        if match:
            result[current].append(line[match.end() :].strip())
    return result


def _step_definitions() -> set[str]:
    """Нормализованные тексты шагов, к которым есть определения.

    Определения — это декораторы ``@given``/``@when``/``@then`` в
    тестах. Разбор идёт по AST, а не по тексту: так соседние строковые
    литералы внутри одного декоратора склеиваются как в самом Python,
    и многострочный шаг не выглядит «ненайденным».

    Шаг может быть объявлен двумя способами: строкой (точный текст) и
    через ``parsers.parse("... {имя} ...")``. Второй способ нужен, когда
    в шаге есть значение — слово, число, ссылка. Гейт обязан понимать
    оба, иначе он объявляет корректный шаг «без определения»: ровно это
    и случилось со сценариями распознания слов.

    Returns:
        Множество текстов шагов и шаблонов (без нормализации: шаблоны
        ``parsers.parse`` нельзя нормализовать — вместе с кавычками
        пропали бы имена параметров).
    """
    definitions: set[str] = set()
    for path in (ROOT / "tests").glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call) or not decorator.args:
                    continue
                func = decorator.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                if name not in {"given", "when", "then"}:
                    continue
                argument = decorator.args[0]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    definitions.add(argument.value)
                    continue
                # @given(parsers.parse("...")) и @given(parsers.re("..."))
                if (
                    isinstance(argument, ast.Call)
                    and argument.args
                    and isinstance(argument.args[0], ast.Constant)
                    and isinstance(argument.args[0].value, str)
                ):
                    definitions.add(argument.args[0].value)
    return definitions


def _step_is_defined(step: str, definitions: set[str]) -> bool:
    """Есть ли у шага определение — точное или по шаблону.

    Два способа сравнения, потому что шаг объявляется двумя способами:

    1. точный текст — сравнивается нормализованно (кавычки и запятые не
       важны);
    2. шаблон ``parsers.parse`` с именованными параметрами — сравнивается
       как шаблон, по исходному тексту.

    Args:
        step: Текст шага из сценария.
        definitions: Тексты и шаблоны определений.

    Returns:
        Определение найдено.
    """
    normalized = _normalize(step)
    if normalized in {_normalize(item) for item in definitions}:
        return True
    for pattern in definitions:
        if "{" in pattern and parse_pattern is not None:
            if parse_pattern(pattern, step) is not None:
                return True
        elif pattern.startswith("^") and re.match(pattern, step):
            return True
    return False


def _registered_features() -> set[str]:
    """Файлы сценариев, подключённые через ``scenarios("...")``.

    Сценарий исполняется только если его файл подключён к тестам.
    Определённые шаги без подключения — это описание, а не проверка.

    Returns:
        Множество нормализованных путей.
    """
    registered: set[str] = set()
    for path in (ROOT / "tests").glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name != "scenarios":
                continue
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                registered.add(_normalize(argument.value))
    return registered


def check_scenarios_covered() -> list[str]:
    """Каждый сценарий исполняется: файл подключён, все шаги определены.

    Проверять «упоминание сценария в тестах» бессмысленно: pytest-bdd
    строит тесты из файла сценариев, и в исходниках их нет. Исполнение
    распадается на два условия, и оба проверяются здесь:

    1. файл подключён через ``scenarios(...)``;
    2. у каждого шага есть определение ``@given``/``@when``/``@then``.

    Второе условие — это ровно та ошибка, которой не видно глазами:
    шаг переименовали в feature, а определение осталось прежним.

    Returns:
        Список нарушений.
    """
    if not FEATURES_DIR.is_dir():
        return []
    feature_files = sorted(FEATURES_DIR.glob("*.feature"))
    if not feature_files:
        return []
    definitions = _step_definitions()
    registered = _registered_features()
    problems: list[str] = []
    for path in feature_files:
        if not any(_normalize(path.name) in item for item in registered):
            problems.append(f"файл {path.name}: не подключён через scenarios()")
        for name, steps in _feature_steps(path).items():
            if not steps:
                problems.append(f"сценарий «{name}» не содержит шагов")
            for step in steps:
                if not _step_is_defined(step, definitions):
                    problems.append(f"сценарий «{name}»: шаг «{step}» без определения")
    return problems


def check_untrusted_not_confirmed() -> list[str]:
    """Ненадёжные зоны не могут быть источником подтверждённого основания.

    Returns:
        Список нарушений.
    """
    rules = {str(rule.get("id")): rule for rule in _rules()}
    problems: list[str] = []
    for rule_id in ("external-sources-never-confirmed", "domain-reference-is-not-case-fact"):
        rule = rules.get(rule_id)
        if not rule:
            continue
        forbidden = {str(zone) for zone in rule.get("untrusted_zones") or []}
        for flow in _flows():
            source = str(flow.get("from", ""))
            target = str(flow.get("to", ""))
            if source in forbidden and target == "model-output":
                problems.append(
                    f"правило {rule_id}: данные зоны «{source}» идут в модель "
                    "без отметки о статусе «требует проверки»"
                )
    return problems


def _rule_by_id(identifier: str) -> dict[str, Any]:
    """Правило по идентификатору.

    Args:
        identifier: Идентификатор правила.

    Returns:
        Правило либо пустой словарь.
    """
    for rule in _rules():
        if str(rule.get("id")) == identifier:
            return rule
    return {}


def _workflow_source() -> str:
    """Текст исходника workflow — источник отметок о статусе.

    Returns:
        Текст файла.
    """
    path = ROOT / "src" / "pf" / "workflow.py"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def check_marking_present(identifier: str) -> list[str]:
    """В коде есть отметка о статусе, объявленная правилом.

    Args:
        identifier: Идентификатор правила.

    Returns:
        Список нарушений.
    """
    rule = _rule_by_id(identifier)
    if not rule:
        return []
    marking = str(rule.get("marking", ""))
    required = str(rule.get("required_note", ""))
    if not marking:
        return []
    source = _workflow_source()
    if not source:
        return [f"{identifier}: не найден исходник workflow.py"]
    problems: list[str] = []
    if marking not in source:
        problems.append(f"{identifier}: в коде нет отметки «{marking}»")
    elif required and required.lower() not in source.lower():
        problems.append(f"{identifier}: нет пояснения «{required}» рядом с отметкой")
    return problems


def check_external_sources_marked() -> list[str]:
    """Внешние источники помечены как требующие проверки.

    Returns:
        Список нарушений.
    """
    return check_marking_present("external-sources-never-confirmed")


def check_domain_reference_marked() -> list[str]:
    """Тематический справочник помечен как не являющийся фактом о кейсе.

    Returns:
        Список нарушений.
    """
    return check_marking_present("domain-reference-is-not-case-fact")


def check_conclusion_is_grounded() -> list[str]:
    """Основания вывода проверяются вместе с фактами.

    Правило статическое: проверка оснований обязана собирать ссылки не
    только из фактов, но и из стратегии (``evidence``,
    ``promise_grounding``) и решения (``basis``). Без этого вывод — «что
    делать» — остаётся единственным утверждением, которое никто не
    проверял: в демо-прогоне он опирался на файл, которого нет в
    материалах, и это не проявлялось.

    Returns:
        Список нарушений.
    """
    path = ROOT / "src" / "pf" / "workflow.py"
    if not path.is_file():
        return ["модуль src/pf/workflow.py не найден"]
    text = path.read_text(encoding="utf-8")
    required = ("evidence", "promise_grounding", "commitments", "basis", "conclusion")
    missing = [marker for marker in required if marker not in text]
    if missing:
        return ["проверка оснований не смотрит в: " + ", ".join(missing)]
    return []


CHECKS = {
    "zones_declared": check_zones_declared,
    "acyclic": check_acyclic,
    "write_sources_subset": check_write_sources_subset,
    "external_sources_marked": check_external_sources_marked,
    "domain_reference_marked": check_domain_reference_marked,
    "claims_have_verification": check_claims_have_verification,
    "implementation_status_valid": check_implementation_status,
    "scenarios_covered": check_scenarios_covered,
    "conclusion_is_grounded": check_conclusion_is_grounded,
}


def main() -> int:
    """Выполнить все fitness-проверки.

    Returns:
        0 — нарушений нет; 1 — нарушения найдены.
    """
    rules = _rules()
    if not rules:
        print("[FAIL] contracts-check: реестр fitness-правил не прочитан")
        return 1

    problems: list[str] = []
    planned: list[str] = []
    executed = 0
    for rule in rules:
        check_name = str(rule.get("check", ""))
        handler = CHECKS.get(check_name)
        if handler is None:
            problems.append(f"правило {rule.get('id')}: нет проверки «{check_name}»")
            continue
        if str(rule.get("status", "active")) == "planned":
            # Правило не проверяется, но и не молчит: гэп виден.
            reason = " ".join(str(rule.get("planned_reason", "")).split())
            planned.append(f"{rule.get('id')}: {reason}")
            continue
        executed += 1
        problems.extend(f"{rule.get('id')}: {problem}" for problem in handler())

    if planned:
        print(f"Известные гэпы (правила со статусом planned): {len(planned)}")
        for gap in planned:
            print(f"  ~ {gap}")
        print()

    if not problems:
        print(f"[OK] contracts-check: {executed} правил проверено, нарушений нет")
        return 0

    print(f"[FAIL] contracts-check: {len(problems)} нарушение(й)")
    for problem in problems:
        print(f"  - {problem}")
    print()
    print("  Как чинить: правило описано в architecture/rules/fitness-rules.yaml.")
    print("  Реестр — источник истины; код только исполняет его.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
