"""Покрытие сценариев по файлам.

Общий гейт ``every-scenario-has-test`` проверяет все feature-файлы сразу.
Точечный тест нужен, чтобы задача могла покраснеть сама: критерий приёмки
``-k workflow_features`` обязан выбирать все проверки задачи, а не одну.

Здесь же исполняется eval-кейс на автомат: feature-файл описывает
поведение, а кейс подтверждает, что оно действительно такое.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from pf.evalrun import AUTOMATON_CASE_ID, load_cases, run_all

ROOT = Path(__file__).resolve().parent.parent


def _contracts() -> ModuleType:
    """Загрузить гейт контрактов как модуль.

    Returns:
        Модуль ``scripts/contracts_check.py``.
    """
    path = ROOT / "scripts" / "contracts_check.py"
    spec = importlib.util.spec_from_file_location("contracts_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_workflow_features_covered() -> None:
    """Сценарии ``workflow.feature`` исполняются, и кейс на автомат проходит.

    Два условия задачи T-10a проверяются вместе: собственно покрытие
    feature-файла (файл подключён, у каждого шага есть определение) и
    eval-кейс на автомат (маршрут назад исправляется). Один критерий
    приёмки — ``-k workflow_features`` — выбирает обе проверки.
    """
    module = _contracts()
    problems = module.check_feature_covered("workflow.feature")
    assert problems == [], "сценарии workflow.feature не исполняются: " + "; ".join(problems)

    ids = {case.id for case in load_cases()}
    assert AUTOMATON_CASE_ID in ids, "кейс на автомат пропал из набора"
    result = next(item for item in run_all() if item.case_id == AUTOMATON_CASE_ID)
    assert result.passed, result.detail
