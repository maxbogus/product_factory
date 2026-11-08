"""Общая изоляция каталогов на весь набор тестов.

Зачем. Материалы агента читаются из двух корней: ``project/`` (свои документы)
и ``knowledge/raw/<дата>/`` (сырые выгрузки, они коммитятся намеренно). Пока
читался только первый, изоляция каждым тестом отдельно работала. Со вторым
выходит иначе: тест, который не подменил ``knowledge_dir``, читает настоящий
снимок из репозитория — и зеленеет или краснеет от того, что в него положили.

Поэтому ``knowledge_dir`` уводится в ``tmp_path`` один раз на весь набор.
``project_dir`` здесь **не** подменяется: его тесты задают сами, а глобальная
подмена сломала бы проверки, которые читают настоящий ``project/`` (данные
демонстрации). Правило записано в ``docs/decisions.md`` (D-29).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pf.config import settings


@pytest.fixture(autouse=True, scope="session")
def _isolated_knowledge_dir(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Увести хранилище знаний во временный каталог на весь набор тестов."""
    directory = tmp_path_factory.mktemp("knowledge")
    patcher = pytest.MonkeyPatch()
    patcher.setattr(settings, "knowledge_dir", Path(directory))
    yield
    patcher.undo()
