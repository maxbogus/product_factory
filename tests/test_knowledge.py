"""Проверки слоя знаний: методики и скилы.

Методики из ``methods/`` и скилы из ``skills/`` — это второй слой знаний
по брифу. До подключения они лежали в файлах, но модель их не видела:
слой был описан и не работал.

Проверяется, что знания действительно попадают в работу агента и что
подстановка не раздувает промпт.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pf.config import settings
from pf.knowledge import ROLE_METHODS, knowledge_for, read_methods, read_skills
from pf.roles import ROLES, load_prompt


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолировать каталоги знаний."""
    monkeypatch.setattr(settings, "methods_dir", tmp_path / "methods")
    monkeypatch.setattr(settings, "skills_dir", tmp_path / "skills")


def test_methods_are_read() -> None:
    """Методики читаются из файлов."""
    settings.methods_dir.mkdir(parents=True, exist_ok=True)
    (settings.methods_dir / "variants.md").write_text(
        "# Методика\n\nПравило про варианты.\n", encoding="utf-8"
    )
    assert "Правило про варианты" in read_methods()


def test_skills_are_read() -> None:
    """Скилы читаются из файлов."""
    settings.skills_dir.mkdir(parents=True, exist_ok=True)
    (settings.skills_dir / "collect.md").write_text(
        "# Skill\n\nПравило про сбор.\n", encoding="utf-8"
    )
    assert "Правило про сбор" in read_skills()


def test_method_is_attached_to_prompt() -> None:
    """Методика попадает в инструкцию роли."""
    settings.methods_dir.mkdir(parents=True, exist_ok=True)
    (settings.methods_dir / "fact-hypothesis-forecast.md").write_text(
        "# Методика\n\nФакт отличается от гипотезы.\n", encoding="utf-8"
    )
    from pf import knowledge

    knowledge.clear_cache()
    prompt = load_prompt("analyst")
    assert "Факт отличается от гипотезы" in prompt
    assert "Методические знания" in prompt


def test_only_own_methods_are_attached() -> None:
    """Роль получает только свои методики.

    Подстановка всех методик в каждую роль удваивала расход: токены
    на прогон выросли с 16 тыс. до 33 тыс. при выигрыше в одну цитату.
    """
    settings.methods_dir.mkdir(parents=True, exist_ok=True)
    (settings.methods_dir / "variants.md").write_text(
        "# Варианты\n\nПравило про варианты.\n", encoding="utf-8"
    )
    (settings.methods_dir / "fact-hypothesis-forecast.md").write_text(
        "# Факт\n\nПравило про факты.\n", encoding="utf-8"
    )
    from pf import knowledge

    knowledge.clear_cache()
    options_prompt = load_prompt("options")
    analyst_prompt = load_prompt("analyst")
    assert "Правило про варианты" in options_prompt
    assert "Правило про факты" not in options_prompt
    assert "Правило про факты" in analyst_prompt


def test_every_role_has_method_mapping() -> None:
    """Для каждой роли определено, какие методики её касаются."""
    for name in ROLES:
        assert name in ROLE_METHODS


def test_knowledge_is_empty_without_files() -> None:
    """Без файлов знаний подсказка пуста, а не ломает сборку."""
    assert knowledge_for("analyst") == ""
