"""Проверки согласованности CI и кода.

Живые тесты бьют по модели, указанной в настройках. CI поднимает
Ollama и скачивает конкретную модель. Если эти две вещи расходятся,
живые тесты в CI падают с «model not found» — и «запуск по инструкции»
перестаёт работать.

Та же проверка касается документации: инструкция по запуску должна
называть ту модель, которую агент просит по умолчанию.
"""

from __future__ import annotations

import re
from pathlib import Path

from pf.config import settings

ROOT = Path(__file__).resolve().parent.parent


def _ci_workflow() -> str:
    """Текст CI-файла."""
    return (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def test_ci_pulls_the_default_model() -> None:
    """CI должен поднимать модель по умолчанию."""
    workflow = _ci_workflow()
    pulled = re.findall(r"ollama pull ([\w:.\-]+)", workflow)
    assert pulled, "в CI должен быть хотя бы один ollama pull"
    assert settings.ollama_model in pulled, (
        f"CI тянет {pulled}, а агент по умолчанию просит {settings.ollama_model}. "
        "Живые тесты в CI упадут с «model not found»"
    )


def test_readme_names_the_default_model() -> None:
    """Инструкция по запуску должна называть модель по умолчанию."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"ollama pull {settings.ollama_model}" in readme, (
        f"в README нет команды «ollama pull {settings.ollama_model}» — "
        "инструкция по запуску ведёт к другой модели"
    )


def test_env_example_names_the_default_model() -> None:
    """Пример .env должен совпадать с моделью по умолчанию."""
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert f"PF_OLLAMA_MODEL={settings.ollama_model}" in example, (
        f"в .env.example указана не та модель, что просит агент ({settings.ollama_model})"
    )
