"""Переносимость: второй проект на другом домене — факт, а не обещание.

Демонстрация одна и та же (телеком, удержание), а критерий «справляется с
новыми задачами» без второго проекта — обещание. Здесь второй проект на
незнакомом домене (кардиология и профилактика гипертензии) проходит ту же
механику: материалы читаются, домен берётся из файла, основания проверяются,
отчёт и документ строятся — **без единой правки `src/`**.

Критерий приёмки — `-k second_project`: ключ есть во всех проверках файла,
потому что критерий обязан выбирать все тесты задачи (`AGENTS.md`).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pf import corpus, domain
from pf.artifacts import build_document
from pf.config import settings
from pf.ground import SupportStatus, verify_claim
from pf.journal import Journal, StepRecord, journal_path
from pf.materials import Material, scan_materials
from pf.state import Phase, SessionState, save

#: Корень репозитория: файл лежит в ``tests/``.
ROOT = Path(__file__).resolve().parent.parent
#: Второй проект: материалы другого домена.
PROJECT2 = ROOT / "project2"
#: Домен второго проекта — файл, а не код.
DOMAIN2 = PROJECT2 / "domain"


def _material(name: str) -> Material:
    """Найти материал второго проекта по имени файла.

    Args:
        name: Часть идентификатора источника.

    Returns:
        Материал с абзацами.
    """
    scan = scan_materials(PROJECT2, include_knowledge=False)
    for material in scan.materials:
        if name in material.source_id:
            return material
    raise AssertionError(f"материал {name} не прочитан: {[m.source_id for m in scan.materials]}")


def test_second_project_materials_are_read() -> None:
    """Материалы второго проекта читаются тем же `scan_materials`."""
    scan = scan_materials(PROJECT2, include_knowledge=False)

    ids = [material.source_id for material in scan.materials]
    assert len(ids) >= 2, ids
    assert any("control.md" in source for source in ids), ids
    assert any("notes.md" in source for source in ids), ids
    # Словарь домена — не материал: он не цитируется, а читается парсером.
    skipped = {item.path for item in scan.skipped}
    assert any("entities.yaml" in path for path in skipped), skipped


def test_second_project_domain_is_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Домен второго проекта берётся из файла: телекома больше нет."""
    monkeypatch.setattr(settings, "domain_dir", DOMAIN2)

    names = {subject.canonical for subject in domain.subjects()}

    assert "артериальная гипертензия" in names, names
    assert "МТС" not in names
    assert corpus.is_domain_query("измерение артериального давления")
    assert not corpus.is_domain_query("абонентская база")


def test_second_project_grounding_works(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Дословная цитата второго домена подтверждается тем же правилом."""
    monkeypatch.setattr(settings, "lexicon_dir", tmp_path / "lexicon")
    monkeypatch.setattr(settings, "lexicon_enabled", False)
    material = _material("control.md")
    passage = material.passages[0]

    support = verify_claim(
        passage.text,
        passage.text,
        material.source_id,
        paragraph_index=0,
        quote=passage.text,
    )

    assert support.status is SupportStatus.CONFIRMED, support.detail


def test_second_project_without_numbers_reports_no_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """На материале без цифр число не подтверждается: «нет данных», не «да»."""
    monkeypatch.setattr(settings, "lexicon_dir", tmp_path / "lexicon")
    monkeypatch.setattr(settings, "lexicon_enabled", False)
    material = _material("notes.md")
    passage = material.passages[0]

    support = verify_claim(
        passage.text,
        "Домашнее измерение давления снижает риск осложнений на 30 %",
        material.source_id,
        paragraph_index=0,
    )

    assert support.status is SupportStatus.NO_DATA, support.detail
    assert "величин" in support.detail


def test_second_project_report_builds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Документ второго проекта собирается тем же `build_document` из журнала."""
    monkeypatch.setattr(settings, "state_dir", tmp_path / "state")
    monkeypatch.setattr(settings, "journal_dir", tmp_path / "journal")
    monkeypatch.setattr(settings, "reports_dir", tmp_path / "reports")
    direction = "Ввести домашний дневник давления с напоминаниями"
    state = SessionState(
        session_id="port2",
        phase=Phase.DONE,
        question="Как повысить контроль артериального давления у пациентов?",
    )
    save(state)
    journal = Journal(journal_path("port2"))
    journal.append(
        StepRecord(
            ordinal=1,
            role="collector",
            phase="collecting",
            summary="собрано",
            citations=["project/data/control.md#p0"],
        )
    )
    journal.append(
        StepRecord(
            ordinal=2,
            role="strategy",
            phase="strategy",
            summary="стратегия готова",
            detail={
                "product": {"chosen_direction": direction, "audience": "пациенты с гипертензией"},
                "verification": {
                    "expected_change": "доля контролируемого давления выросла",
                    "compare_against": "прошлый квартал",
                    "observation_period": "3 месяца",
                    "adverse_effects": ["нагрузка на врача"],
                },
            },
        )
    )

    document = build_document("port2")

    assert "## 6. Вывод" in document
    assert direction in document


def test_second_project_needs_no_src_changes() -> None:
    """Переносимость не потребовала правок `src/`: код не знает домена."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", "src/pf"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("дерево git недоступно — проверить отсутствие правок src/ нечем")
    changes = [line for line in result.stdout.splitlines() if line.strip()]

    assert not changes, "переносимость потребовала правки src/: " + ", ".join(changes)
