"""Проверки гейта контрактов.

Гейт `every-scenario-has-test` должен отвечать на вопрос «исполняется ли
каждый сценарий из `tests/features`». Проверка по дословному вхождению
идентификатора в текст тестов не работает: pytest-bdd строит имя теста
из названия сценария, выбрасывая дефисы и понижая регистр. Сценарий
«P-01-метрика-из-строки-подтверждается» живёт в тесте как
`test_p01метрикаизстрокиподтверждается` — дословного совпадения нет,
и гейт объявлял непокрытыми все сценарии сразу.

Здесь проверяется обратное: покрытые сценарии проходят, а сценарий без
исполняемого теста — по-прежнему нарушение.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent


def _contracts() -> ModuleType:
    """Загрузить гейт контрактов как модуль.

    Returns:
        Модуль `scripts/contracts_check.py`.
    """
    path = ROOT / "scripts" / "contracts_check.py"
    spec = importlib.util.spec_from_file_location("contracts_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_scenarios_are_covered() -> None:
    """Каждый сценарий из tests/features исполняется тестом."""
    problems = _contracts().check_scenarios_covered()
    assert problems == [], (
        "гейт считает сценарии непокрытыми, хотя pytest-bdd их исполняет: " + "; ".join(problems)
    )


def test_scenario_without_test_is_reported(tmp_path: Path, monkeypatch) -> None:
    """Сценарий без исполняемого теста остаётся нарушением.

    Args:
        tmp_path: Временный каталог.
        monkeypatch: Подмена путей гейта.
    """
    module = _contracts()
    features = tmp_path / "tests" / "features"
    features.mkdir(parents=True)
    (features / "orphan.feature").write_text(
        "Функция: Сирота\n\n  Сценарий: Ничей-сценарий-без-теста\n    Допустим что-то случилось\n",
        encoding="utf-8",
    )
    tests = tmp_path / "tests"
    (tests / "test_other.py").write_text(
        "def test_something() -> None:\n    assert True\n", encoding="utf-8"
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "FEATURES_DIR", features)

    problems = module.check_scenarios_covered()
    assert any("Ничей-сценарий-без-теста" in problem for problem in problems)
