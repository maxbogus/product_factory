"""Проверки материалов проекта: свои документы и таблицы.

Требование брифа: «работать как минимум с документами, таблицами и сайтами
из разрешённых источников» и «сохранять ссылку на источник каждого
существенного сведения».

Материалы проекта — третий слой знаний (рядом с постоянными инструкциями
и методиками). Они лежат в каталоге ``project/`` и должны попадать в
агента так же, как корпус: с тем же идентификатором, с тем же
механизмом проверки дословности цитаты.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pf.materials import Material, load_materials
from pf.sources import collect


@pytest.fixture
def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Каталог с материалами проекта."""
    directory = tmp_path / "project"
    directory.mkdir()
    monkeypatch.setattr("pf.config.settings.project_dir", directory)
    monkeypatch.setattr("pf.config.settings.knowledge_dir", tmp_path / "knowledge")
    return directory


def test_markdown_file_becomes_citable_passages(_project: Path) -> None:
    """Текстовый документ должен нарезаться на цитатуемые абзацы."""
    (_project / "brief.md").write_text(
        "Отток абонентов мобильного тарифа вырос до 8 % за первый квартал 2026 года.\n"
        "\n"
        "Одной из причин называют навязчивое подключение дополнительных услуг.\n",
        encoding="utf-8",
    )
    materials = load_materials()
    assert materials, "материалы должны быть загружены"
    material = materials[0]
    assert material.source_id.startswith("project/")
    assert "brief.md" in material.source_id
    assert any("8 %" in passage.text for passage in material.passages)


def test_csv_row_becomes_citable_passage(_project: Path) -> None:
    """Строка таблицы должна стать цитатуемым абзацем.

    Таблица — обязательный минимум брифа. Цитата из неё обязана быть
    дословной строкой, иначе агент не сможет на неё сослаться.
    """
    with (_project / "numbers.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["период", "отток", "абонентов"])
        writer.writerow(["2025 Q1", "7,2 %", "112,4 млн"])
        writer.writerow(["2026 Q1", "8,1 %", "110,7 млн"])
    materials = load_materials()
    assert materials, "таблица должна быть загружена"
    material = materials[0]
    rows = [p.text for p in material.passages]
    assert any("8,1" in row and "110,7" in row for row in rows), (
        f"строка таблицы должна сохраняться как абзац, получено: {rows}"
    )


def test_material_keeps_source_url_and_date(_project: Path) -> None:
    """У материала должен быть адрес источника и дата получения."""
    (_project / "report.json").write_text(
        json.dumps(
            {
                "title": "Отчёт по удержанию",
                "url": "https://example.org/report",
                "retrieved": "2026-11-01",
                "text": "Число активных абонентов снизилось на 1,7 млн за год.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    materials = load_materials()
    assert materials
    material = materials[0]
    assert material.url == "https://example.org/report"
    assert material.retrieved == "2026-11-01"


def test_material_has_content_hash(_project: Path) -> None:
    """У материала должен быть хеш содержимого.

    По нему видно, что вывод опирается на ту же версию данных: при
    пересмотре выводов важно знать, какие именно данные тогда были.
    """
    (_project / "note.txt").write_text(
        "Отток абонентов мобильного тарифа вырос до 8 % за первый квартал 2026 года.",
        encoding="utf-8",
    )
    materials = load_materials()
    assert materials
    material = materials[0]
    assert len(material.sha256) == 64


def test_material_citation_is_verified_verbatim(_project: Path) -> None:
    """Цитата из материала должна проходить проверку дословности.

    Механизм честности работает для всех источников: и для статей
    корпуса, и для собственных материалов проекта.
    """
    (_project / "brief.md").write_text(
        "Отток абонентов мобильного тарифа вырос до 8 % за первый квартал 2026 года.\n",
        encoding="utf-8",
    )
    bundle = collect("отток абонентов мобильного тарифа")
    assert bundle.passages, "материал должен попасть в срез источников"
    from_materials = [p for p in bundle.passages if p.source_id.startswith("project/")]
    assert from_materials, "материал проекта должен быть в срезе источников"

    passage = from_materials[0]
    support = bundle.verify_claim(
        "Отток вырос до 8 %",
        source_id=passage.source_id,
        paragraph=passage.paragraph_index,
        quote=passage.text,
    )
    assert support.is_usable, f"цитата должна подтверждаться, деталь: {support.detail}"


def test_material_without_quote_is_rejected(_project: Path) -> None:
    """Выдуманная цитата из материала должна отбрасываться."""
    (_project / "brief.md").write_text(
        "Отток абонентов мобильного тарифа вырос до 8 % за первый квартал 2026 года.\n",
        encoding="utf-8",
    )
    bundle = collect("отток абонентов мобильного тарифа")
    support = bundle.verify_claim(
        "Отток вырос до 47 %",
        source_id="project/brief.md",
        paragraph=0,
        quote="Отток вырос до 47 %",
    )
    assert not support.is_usable, "выдуманная цитата не должна подтверждаться"


def test_empty_project_directory_is_not_an_error(_project: Path) -> None:
    """Пустой каталог материалов — не ошибка, а отсутствие материалов."""
    assert load_materials() == []


def test_unsupported_format_is_skipped(_project: Path) -> None:
    """Неподдерживаемый формат пропускается, а не ломает загрузку."""
    (_project / "picture.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (_project / "brief.md").write_text(
        "Отток абонентов мобильного тарифа вырос до 8 % за первый квартал 2026 года.\n",
        encoding="utf-8",
    )
    materials = load_materials()
    assert len(materials) == 1
    assert materials[0].source_id.endswith("brief.md")


def test_material_exposes_kind(_project: Path) -> None:
    """У материала должен быть вид: документ, таблица или данные."""
    (_project / "numbers.csv").write_text("период,отток\n2026 Q1,8,1 %\n", encoding="utf-8")
    materials = load_materials()
    assert materials[0].kind == Material.Kind.TABLE


def test_readme_is_not_a_source_of_facts(_project: Path) -> None:
    """README проекта не должен попадать в источники.

    README описывает материалы, а не содержит данные. Если считать его
    источником, агент начнёт цитировать собственную документацию.
    """
    (_project / "README.md").write_text(
        "# Материалы проекта\n\nЗдесь лежат открытые метрики по операторам.\n",
        encoding="utf-8",
    )
    (_project / "metrics.csv").write_text(
        "оператор,значение\nМТС,87 млн человек\n", encoding="utf-8"
    )
    materials = load_materials()
    assert len(materials) == 1
    assert materials[0].source_id.endswith("metrics.csv")


def test_sidecar_metadata_is_applied(_project: Path) -> None:
    """Метаданные рядом с материалом применяются к нему."""
    (_project / "metrics.csv").write_text(
        "оператор,значение\nМТС,87 млн человек\n", encoding="utf-8"
    )
    (_project / "metrics.csv.meta.json").write_text(
        json.dumps(
            {
                "title": "Метрики по операторам",
                "url": "https://example.org/metrics",
                "retrieved": "2026-11-01",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    materials = load_materials()
    assert len(materials) == 1
    material = materials[0]
    assert material.title == "Метрики по операторам"
    assert material.url == "https://example.org/metrics"
    assert material.retrieved == "2026-11-01"


def test_metadata_file_is_not_loaded_as_data(_project: Path) -> None:
    """Файл метаданных не должен грузиться как самостоятельный материал."""
    (_project / "metrics.csv").write_text(
        "оператор,значение\nМТС,87 млн человек\n", encoding="utf-8"
    )
    (_project / "metrics.csv.meta.json").write_text(
        json.dumps({"title": "Метрики", "url": "https://example.org"}, ensure_ascii=False),
        encoding="utf-8",
    )
    materials = load_materials()
    assert [m.source_id for m in materials] == ["project/metrics.csv"]


def test_csv_value_with_comma_stays_whole(_project: Path) -> None:
    """Значение с запятой не должно разваливаться на части.

    Число «11,923 млн абонентов» — это одно значение. Без экранирования
    CSV разбирает его как два поля, и цитата получается неверной.
    """
    with (_project / "numbers.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["показатель", "значение"])
        writer.writerow(["численность абонентов", "11,923 млн абонентов"])
    materials = load_materials()
    rows = [p.text for p in materials[0].passages]
    assert any("11,923 млн абонентов" in row for row in rows), (
        f"значение с запятой должно остаться целым, получено: {rows}"
    )


def test_knowledge_manifest_records_source_and_date(tmp_path: Path) -> None:
    """Манифест должен хранить источник, дату и хеш.

    По манифесту видно, на какие данные опирался вывод. При пересмотре
    выводов первое, что нужно проверить, — та ли это была версия.
    """
    from pf.knowledge_store import read_manifest, write_manifest

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "cbr.xml").write_text("<xml>данные</xml>", encoding="utf-8")

    entries = write_manifest(
        raw,
        [
            {
                "name": "cbr.xml",
                "url": "https://www.cbr.ru/",
                "retrieved": "2026-11-06",
                "kind": "xml",
                "note": "Ключевая ставка",
            }
        ],
    )
    assert len(entries) == 1
    manifest = read_manifest(raw)
    assert manifest
    entry = manifest[0]
    assert entry["url"] == "https://www.cbr.ru/"
    assert entry["retrieved"] == "2026-11-06"
    assert len(entry["sha256"]) == 64


def test_knowledge_manifest_survives_missing_file(tmp_path: Path) -> None:
    """Файл, которого уже нет, не должен ломать чтение манифеста."""
    from pf.knowledge_store import read_manifest, write_manifest

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "a.txt").write_text("данные", encoding="utf-8")
    write_manifest(raw, [{"name": "a.txt", "url": "https://example.org/"}])
    (raw / "a.txt").unlink()

    manifest = read_manifest(raw)
    assert manifest
    assert manifest[0]["name"] == "a.txt"


def test_knowledge_manifest_is_not_empty_for_empty_dir(tmp_path: Path) -> None:
    """Пустой каталог — пустой манифест, без ошибки."""
    from pf.knowledge_store import read_manifest

    raw = tmp_path / "raw"
    raw.mkdir()
    assert read_manifest(raw) == []


def test_material_citation_resolves_to_url(tmp_path: Path) -> None:
    """Ссылка на материал должна вести на адрес источника.

    Идентификатор `project/файл#row3` — не ссылка: человек по нему
    ничего не откроет. Ссылка должна быть кликабельной и вести туда,
    откуда данные взяты.
    """
    from pf.materials import resolve_citation

    (tmp_path / "numbers.csv").write_text("период,значение\n2026 Q1,110,7\n", encoding="utf-8")
    (tmp_path / "numbers.csv.meta.json").write_text(
        json.dumps(
            {"title": "Метрики", "url": "https://example.org/metrics"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    resolved = resolve_citation("project/numbers.csv#row0", directory=tmp_path)
    assert resolved is not None
    assert resolved.url == "https://example.org/metrics"
    assert resolved.source_id == "project/numbers.csv"
    assert resolved.row == 0


def test_resolve_citation_for_corpus_article(tmp_path: Path) -> None:
    """Ссылка на статью корпуса должна разрешаться в статью Википедии."""
    from pf.materials import resolve_citation

    resolved = resolve_citation("telecom/Т2 (оператор сотовой связи)#p12")
    assert resolved is not None
    assert "wikipedia.org" in resolved.url
    assert resolved.source_id == "telecom/Т2 (оператор сотовой связи)"
    assert resolved.paragraph == 12


def test_resolve_citation_returns_none_for_unknown() -> None:
    """Неизвестный источник не разрешается — лучше пусто, чем выдумка."""
    from pf.materials import resolve_citation

    assert resolve_citation("project/нет-такого.csv#row0") is None
