"""Контракт передачи: каноническое утверждение и хеш источника.

Зачем модуль. Между агентами передаётся **текст** утверждения и цитата.
Получатель (следующая фаза, субагент `fact-check`, человек через
`demo/RESULT.md`) толкует прозу заново — и приходит к другому ответу,
потому что понятия, числа и период разбирает своим способом.

Здесь передаётся не проза, а поля: что утверждается (понятия, величины,
периоды, термины) и на каком источнике (sha256). Получатель пересчитывает
поля по тому же источнику и получает тот же ответ. Гарантия — проверяемость,
а не «модель поняла»: смысл понятия задаёт словарь, подтверждённый человеком.

**Чего модуль не делает.** Он не поднимает и не понижает статусы (`pf.ground`
решает статус), не заменяет `pf.review` (там хеш про данные, здесь — про
переданное утверждение) и не обещает одинакового понимания: он даёт
одинаковый **пересчёт** при том же хеше.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pf.canonical import CanonicalText, canonicalize, score
from pf.config import settings


@dataclass(frozen=True, slots=True)
class Handoff:
    """Что передаётся получателю: поля утверждения и хеш источника.

    Attributes:
        claim: Исходный текст утверждения (для человека).
        source_id: Идентификатор источника.
        source_hash: sha256 текста источника, на котором проверялось.
        fields: Канонические поля утверждения: понятия, термины, величины,
            периоды, регионы, субъекты, отрицание.
        score: Степень поддержки на момент передачи.
    """

    claim: str
    source_id: str
    source_hash: str
    fields: Mapping[str, Any]
    score: float

    def as_record(self) -> dict[str, Any]:
        """Запись для журнала шага: только JSON-совместимые значения.

        Returns:
            Словарь с текстом, ссылкой, хешем, полями и степенью поддержки.
        """
        return {
            "claim": self.claim,
            "source_id": self.source_id,
            "source_hash": self.source_hash,
            "fields": dict(self.fields),
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class HandoffCheck:
    """Исход перепроверки передачи получателем.

    Attributes:
        status: ``reproduced`` — поля пересчитаны на том же хеше;
            ``stale`` — источник изменился, ответ не воспроизводится.
        support: Степень поддержки при перепроверке.
        confirmed: Подтверждено ли по текущему порогу.
        note: Пояснение для человека.
    """

    status: str
    support: float
    confirmed: bool
    note: str = ""


def source_digest(text: str) -> str:
    """sha256 текста источника.

    Args:
        text: Текст источника.

    Returns:
        Хеш в шестнадцатеричном виде.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fields_of(text: str) -> dict[str, Any]:
    """Канонические поля текста в JSON-совместимом виде.

    Args:
        text: Утверждение, строка таблицы или абзац.

    Returns:
        Понятия, термины, величины, периоды, регионы, субъекты, отрицание.
    """
    return _serialize(canonicalize(text))


def build_handoff(
    claim: str,
    source_text: str,
    *,
    source_id: str = "",
) -> Handoff:
    """Собрать запись передачи по утверждению и источнику.

    Args:
        claim: Текст утверждения.
        source_text: Полный текст источника, на котором оно проверено.
        source_id: Идентификатор источника.

    Returns:
        Передача с полями утверждения, хешем источника и степенью поддержки.
    """
    result = score(claim, source_text)
    return Handoff(
        claim=claim,
        source_id=source_id,
        source_hash=source_digest(source_text),
        fields=fields_of(claim),
        score=result.support,
    )


def verify_handoff(handoff: Handoff, source_text: str) -> HandoffCheck:
    """Перепроверить передачу по источнику.

    Если хеш источника не совпал — это «требует пересмотра» (`D-33`), а не
    расхождение толкований: пересчитывать нечего, текст уже другой.

    Args:
        handoff: Переданное утверждение с полями и хешем.
        source_text: Текущий текст источника у получателя.

    Returns:
        Исход перепроверки: воспроизведено или устарело.
    """
    if source_digest(source_text) != handoff.source_hash:
        return HandoffCheck(
            status="stale",
            support=0.0,
            confirmed=False,
            note="источник изменился: хеш не совпал — требует пересмотра",
        )
    result = score(handoff.claim, source_text)
    return HandoffCheck(
        status="reproduced",
        support=result.support,
        confirmed=result.supported(settings.claim_support_threshold),
        note="поля пересчитаны на том же хеше — ответ совпал",
    )


def handoff_markdown(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Раздел отчёта «Передача»: что передано и на каком хеше.

    Args:
        records: Записи `Handoff.as_record()` из журнала шага.

    Returns:
        Строки раздела Markdown; пустой список, если передач нет.
    """
    if not records:
        return []
    lines = [
        "### Передача: канонические поля и хеш источника",
        "",
        "Получатель сверяет поля, а не прозу: при том же хеше источника",
        "ответ воспроизводится, при другом — «требует пересмотра».",
        "",
        "| Утверждение | Источник | Хеш | Понятия | Величины | Периоды |",
        "|---|---|---|---|---|---|",
    ]
    for record in records:
        fields = record.get("fields") or {}
        lines.append(
            "| {claim} | {source} | `{digest}` | {concepts} | {quantities} | {periods} |".format(
                claim=_short(str(record.get("claim", ""))),
                source=str(record.get("source_id", "")) or "—",
                digest=str(record.get("source_hash", ""))[:12],
                concepts=_join(fields.get("concepts")),
                quantities=_quantities(fields.get("quantities")),
                periods=_periods(fields.get("periods")),
            )
        )
    lines.append("")
    return lines


def _serialize(parsed: CanonicalText) -> dict[str, Any]:
    """Поля текста в JSON-совместимый словарь.

    Args:
        parsed: Разобранный текст.

    Returns:
        Словарь полей; величины и периоды — списками словарей.
    """
    return {
        "concepts": sorted(parsed.concepts),
        "terms": list(parsed.terms),
        "quantities": [
            {
                "value": item.value,
                "scale": item.scale,
                "unit": item.unit,
                "modifier": item.modifier,
                "raw": item.raw,
            }
            for item in parsed.quantities
        ],
        "periods": [
            {"start": item.start, "end": item.end, "raw": item.raw} for item in parsed.periods
        ],
        "regions": sorted(parsed.regions),
        "subjects": list(parsed.subjects),
        "negated": parsed.negated,
    }


def _short(text: str, limit: int = 60) -> str:
    """Короткая версия текста для ячейки таблицы.

    Args:
        text: Любой текст.
        limit: Предел длины.

    Returns:
        Текст без переводов строк, обрезанный с многоточием.
    """
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _join(items: Any) -> str:
    """Список строк одной ячейкой.

    Args:
        items: Список значений либо None.

    Returns:
        Значения через запятую или прочерк.
    """
    if not items:
        return "—"
    return ", ".join(str(item) for item in items)


def _quantities(items: Any) -> str:
    """Величины одной ячейкой: значение, шкала, единица.

    Args:
        items: Список словарей величин либо None.

    Returns:
        Величины через запятую или прочерк.
    """
    if not items:
        return "—"
    parts = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        raw = str(item.get("raw", ""))
        scale = item.get("scale")
        unit = str(item.get("unit", ""))
        suffix = "" if scale in (1, 1.0, None) else f"×{scale:g}"
        parts.append(f"{raw}{suffix}{unit}")
    return ", ".join(parts) or "—"


def _periods(items: Any) -> str:
    """Периоды одной ячейкой.

    Args:
        items: Список словарей периодов либо None.

    Returns:
        Периоды через запятую или прочерк.
    """
    if not items:
        return "—"
    parts = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        raw = str(item.get("raw", ""))
        if raw:
            parts.append(raw)
            continue
        start = item.get("start")
        end = item.get("end")
        parts.append(f"{start:g}" if start == end else f"{start:g}–{end:g}")
    return ", ".join(parts) or "—"
