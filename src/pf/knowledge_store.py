"""Хранилище знаний: сырые выгрузки и манифест.

Открытые данные полезны только вместе с provenance: по каждой выгрузке
должно быть видно, откуда она, когда получена и какая у неё версия.
Без этого пересмотреть выводы нельзя — нечем проверить, что цифры не
поменялись.

Поэтому каждая выгрузка кладётся в ``knowledge/raw/<дата>/`` и
описывается в манифесте: адрес источника, дата получения, хеш
содержимого. Хеш — ключевое: по нему видно, изменились ли данные с
момента, когда на них опирался вывод.

Манифест — append-only: записи не переписываются, а дописываются. Это
тот же принцип, что у журнала шагов: история работы не переделывается.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pf.config import settings

#: Имя файла манифеста.
MANIFEST_NAME = "manifest.jsonl"


def sha256_file(path: Path) -> str:
    """Посчитать хеш файла.

    Args:
        path: Путь к файлу.

    Returns:
        Хеш в шестнадцатеричном виде.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_dir(date: str | None = None) -> Path:
    """Каталог сырых выгрузок за дату.

    Args:
        date: Дата в формате ``ГГГГ-ММ-ДД``. По умолчанию — сегодня.

    Returns:
        Путь к каталогу.
    """
    day = date or datetime.now(UTC).date().isoformat()
    return settings.knowledge_dir / "raw" / day


def write_manifest(
    directory: Path,
    entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Дописать записи в манифест каталога.

    Манифест append-only: уже записанные строки не меняются. Иначе
    история перезаписывалась бы, а по ней как раз и видно, что изменилось.

    Args:
        directory: Каталог с выгрузками.
        entries: Записи для добавления.

    Returns:
        Записанные записи — с хешами и датой получения.
    """
    directory.mkdir(parents=True, exist_ok=True)
    prepared: list[dict[str, Any]] = []
    for entry in entries:
        name = str(entry.get("name", ""))
        path = directory / name
        record: dict[str, Any] = {
            "name": name,
            "url": str(entry.get("url", "")),
            "retrieved": str(entry.get("retrieved", "") or datetime.now(UTC).date().isoformat()),
            "kind": str(entry.get("kind", "")),
            "note": str(entry.get("note", "")),
            "sha256": sha256_file(path) if path.is_file() else "",
            "bytes": path.stat().st_size if path.is_file() else 0,
        }
        prepared.append(record)

    manifest_path = directory / MANIFEST_NAME
    with manifest_path.open("a", encoding="utf-8") as handle:
        for record in prepared:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return prepared


def read_manifest(directory: Path) -> list[dict[str, Any]]:
    """Прочитать манифест каталога.

    Args:
        directory: Каталог с выгрузками.

    Returns:
        Список записей. Пустой список, если манифеста нет или он пуст.
    """
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        return []

    entries: list[dict[str, Any]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            # Обрыв на границе записи не должен ломать чтение: как и
            # журнал шагов, манифест может остаться недописанным.
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def list_snapshots() -> list[str]:
    """Даты, за которые есть сырые выгрузки.

    Returns:
        Список дат, новые первыми.
    """
    raw = settings.knowledge_dir / "raw"
    if not raw.is_dir():
        return []
    return sorted((path.name for path in raw.iterdir() if path.is_dir()), reverse=True)
