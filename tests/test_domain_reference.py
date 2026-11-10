"""Проверки тематического справочника (`domain-reference`).

Справочник отвечает на вопрос «как это обычно бывает», а не «что было у
нас». Смешивать эти вещи — подмена источника: знание о применимости
выдаётся за факт о конкретном кейсе. Поэтому проверяется:

1. справочник лежит данными (`knowledge/domain/reference.yaml`), а не кодом;
2. блок «СПРАВОЧНИК» идёт в контекст отдельно от оснований и помечен
   «не факт о кейсе»;
3. цитата из справочника не подтверждает вывод: статус «требует проверки»,
   а не `CONFIRMED` (правило `domain-reference-is-not-case-fact`);
4. правило переведено в `active`/`implemented`, и это видит гейт контрактов.
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


def test_reference_is_data_and_loads() -> None:
    """Справочник лежит данными и читается без правки кода."""
    from pf import reference

    entries = reference.entries()
    assert entries, "справочник пуст: блоку «СПРАВОЧНИК» нечего показать"
    for entry in entries:
        assert entry.id and entry.term and entry.what
        assert entry.applies_when and entry.not_when
    assert reference.reference_path().name == "reference.yaml"


def test_reference_context_is_marked_not_case_fact() -> None:
    """Блок справочника отделён и помечен «не факт о кейсе»."""
    from pf.workflow import Workflow

    context = Workflow._reference_context()

    assert "СПРАВОЧНИК" in context
    assert "не факт о кейсе" in context
    assert "требует проверки" in context
    assert "reference/" in context, "в блоке нет ссылок, на которые можно сослаться"


def test_reference_quote_is_needs_review_not_confirmed() -> None:
    """Дословная цитата из справочника — «требует проверки», а не факт."""
    from pf import reference
    from pf.schemas import Analysis, Claim, SourceRef
    from pf.sources import SourceBundle
    from pf.workflow import Workflow

    entry = reference.entries()[0]
    data = Analysis(
        facts=[
            Claim(
                kind="факт",
                text=f"{entry.term}: {entry.what}",
                sources=[
                    SourceRef(
                        source_id=reference.source_id_of(entry.id),
                        paragraph=0,
                        quote=entry.what,
                    )
                ],
            )
        ]
    )
    bundle = SourceBundle(passages=[], articles=[], query="")

    grounding = Workflow._verify_claims(data, bundle)

    assert grounding.citations == [], "справочник не может подтвердить факт"
    assert grounding.rejected == 0, "дословная цитата — не отказ"
    assert grounding.needs_review, "цитата из справочника должна ждать проверки"
    assert any("справочник" in detail.lower() for detail in grounding.needs_review_details)


def test_reference_fabricated_quote_is_rejected() -> None:
    """Выдуманная цитата из справочника — отказ, а не «требует проверки»."""
    from pf import reference
    from pf.schemas import Analysis, Claim, SourceRef
    from pf.sources import SourceBundle
    from pf.workflow import Workflow

    entry = reference.entries()[0]
    data = Analysis(
        facts=[
            Claim(
                kind="факт",
                text=f"{entry.term}: выдуманное утверждение",
                sources=[
                    SourceRef(
                        source_id=reference.source_id_of(entry.id),
                        paragraph=0,
                        quote="такого текста в справочнике нет",
                    )
                ],
            )
        ]
    )
    bundle = SourceBundle(passages=[], articles=[], query="")

    grounding = Workflow._verify_claims(data, bundle)

    assert grounding.citations == []
    assert grounding.needs_review == []
    assert grounding.rejected == 1


def test_domain_reference_rule_is_active_and_implemented() -> None:
    """Правило справочника включено, и у реализованной возможности есть модуль и тест."""
    module = _contracts()

    assert module.check_domain_reference_marked() == []
    assert module.check_implementation_status() == []
