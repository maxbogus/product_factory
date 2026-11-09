"""Контракт передачи: каноническое утверждение и хеш источника.

Между агентами передаётся проза утверждения и цитата. Получатель толкует её
заново — и приходит к другому ответу, потому что понятия, числа и период
разбирает своим способом.

Здесь передаётся не проза, а поля: что утверждается (понятия, величины,
периоды, термины) и на каком источнике (sha256). Гарантия — проверяемость, а
не «модель поняла»: при том же хеше источника получатель пересчитывает поля
и получает тот же ответ, при другом — «требует пересмотра».
"""

from __future__ import annotations

import hashlib

import pytest

from pf.handoff import build_handoff, verify_handoff
from pf.report import ConclusionRecord, conclusion_markdown

CLAIM = "Абонентская база МТС — 82 млн"
SOURCE = "оператор: МТС\nпоказатель: абонентская база\nзначение: 82\nпериод: 2019"


def test_handoff_names_fields_and_source_hash() -> None:
    """Передаётся структура (поля) и хеш источника, а не проза."""
    handoff = build_handoff(CLAIM, SOURCE, source_id="project/data/base.csv#row0")

    assert handoff.source_hash == hashlib.sha256(SOURCE.encode("utf-8")).hexdigest()
    assert handoff.source_id == "project/data/base.csv#row0"
    assert "абонен" in handoff.fields["concepts"]
    assert handoff.fields["subjects"] == ["МТС"]
    assert any(
        item["value"] == 82.0 and item["scale"] == 1_000_000.0
        for item in handoff.fields["quantities"]
    )
    assert 0.0 <= handoff.score <= 1.0


def test_handoff_reproduces_same_verdict() -> None:
    """Тот же хеш источника — тот же ответ у получателя."""
    handoff = build_handoff(CLAIM, SOURCE)

    check = verify_handoff(handoff, SOURCE)

    assert check.status == "reproduced"
    assert check.support == pytest.approx(handoff.score)
    assert check.confirmed == (handoff.score >= 0.17)
    assert "совпал" in check.note


def test_handoff_marks_changed_source_for_review() -> None:
    """Другой хеш — «требует пересмотра», а не расхождение толкований."""
    handoff = build_handoff(CLAIM, SOURCE)
    changed = SOURCE.replace("значение: 82", "значение: 83")

    check = verify_handoff(handoff, changed)

    assert check.status == "stale"
    assert check.support == 0.0
    assert "пересмотр" in check.note.lower()


def test_handoff_basis_carries_fields_and_hash() -> None:
    """Основание несёт канонические поля и хеш того абзаца, где проверено."""
    from pf.ground import verify_claim
    from pf.handoff import source_digest

    support = verify_claim(SOURCE, CLAIM, "project/data/base.csv#row0", paragraph_index=0)

    assert support.source_hash == source_digest("оператор: МТС")
    assert support.canonical is not None
    assert support.canonical["subjects"] == ["МТС"]


def test_handoff_report_section_names_hash_and_fields() -> None:
    """Отчёт показывает, что именно передано и на каком хеше."""
    handoff = build_handoff(CLAIM, SOURCE, source_id="project/data/base.csv#row0")
    conclusion = ConclusionRecord(
        direction="удержать абонентскую базу",
        transmission=[handoff.as_record()],
    )

    text = "\n".join(conclusion_markdown(conclusion))

    assert "Передача" in text
    assert handoff.source_hash[:12] in text
    assert "абонен" in text
