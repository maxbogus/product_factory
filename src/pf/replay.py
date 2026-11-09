"""Повторная проверка оснований на сохранённых ответах ролей.

Зачем это нужно. Сравнивать «прогон против прогона» бессмысленно: на одной и
той же задаче число подтверждённых оснований различается в разы из-за модели
(в проекте зафиксирован разброс 18 против 0). Поэтому проверка
воспроизводится на **сохранённых** результатах ролей: данные одинаковые,
меняется только код проверки — и дельта принадлежит правкам, а не модели.

Модуль один на два потребителя: замер (`scripts/replay_verification.py`) и
перепроверка артефакта (`pf report --recheck`). Двух копий правила быть не
должно: отчёт и замер разошлись бы в том, что считать опорой (`T-63`).
"""

from __future__ import annotations

from typing import Any

from pf.ground import SupportStatus
from pf.roles import validate
from pf.state import load
from pf.verdicts import classify_detail, label_of
from pf.workflow import _PHASE_BY_ROLE, Workflow

#: Фаза состояния → роль. Обратная карта к ``Workflow._PHASE_BY_ROLE``.
ROLE_BY_PHASE: dict[str, str] = {phase.value: role for role, phase in _PHASE_BY_ROLE.items()}

#: Старая формулировка причины отказа. Сейчас так не пишется (причина
#: называется лестницей `sources.absence_reason`), но корзина не должна
#: поехать на унаследованных снимках замера.
_LEGACY_IRRELEVANT = "источник не относится к задаче"


def bucket_of(status: SupportStatus, detail: str) -> str:
    """Корзина исхода: подтверждено, опора на ввод, проверка, нет данных.

    Нерелевантность отделена намеренно: это не расхождение в словах, а
    подмена источника, и лечится она контекстом, а не проверкой. Опоры на
    ввод пользователя отделены по той же причине: это не «нечем
    подтвердить», это слова человека, и доверие к ним решает он (`D-49`).

    Args:
        status: Статус основания.
        detail: Пояснение проверки (для метрик оно своё).

    Returns:
        Название корзины.
    """
    if status is SupportStatus.CONFIRMED:
        return "подтверждено"
    if status is SupportStatus.NEEDS_REVIEW:
        if detail.startswith("ввод пользователя"):
            return "опора на ввод пользователя"
        return "требует проверки"
    if _LEGACY_IRRELEVANT in detail:
        return "нерелевантный источник"
    return "нет данных"


def replay_session(session_id: str) -> dict[str, Any] | None:
    """Проверить основания одного сохранённого прогона.

    Читается **состояние** сессии (сохранённые ответы ролей), а источники
    собираются заново тем же детерминированным способом, что и в живом
    прогоне. Нет состояния — нет и перепроверки: возвращается ``None``, а не
    ноль подтверждённых (ноль и отсутствие данных — разные вещи).

    Args:
        session_id: Идентификатор сессии.

    Returns:
        Снимок по сессии либо ``None``, если состояния нет на диске.
    """
    state = load(session_id)
    if state is None:
        return None

    bundle = Workflow().collect_sources(state)
    payload: dict[str, Any] = {
        "session_id": session_id,
        "question": state.question,
        "sources": len(bundle.articles),
        "roles": {},
        "skipped": [],
        "kinds": {},
        "verdicts": [],
        # Замер `T-48b`: сколько величин-фактов плана осталось без своего
        # расчёта и что из-за них стало со статусом вывода. Без блока
        # «статус стал частичным» не отвечает, за что именно.
        "plan_numbers": {"roles": {}, "unsupported": 0, "kinds": {}},
    }

    for role, phase in _PHASE_BY_ROLE.items():
        raw = state.results.get(phase.value)
        if not isinstance(raw, dict):
            continue
        try:
            data = validate(role, raw)
        except Exception as exc:  # схема ответа могла измениться после прогона
            payload["skipped"].append(f"{role}: {exc}")
            continue

        plan_numbers = (
            Workflow._check_plan_numbers(data) if role in ("strategy", "keeper") else None
        )
        grounding = Workflow._verify_claims(
            data,
            bundle,
            state=state,
            role_name=role,
            session_id=session_id,
            unsupported_numbers=0 if plan_numbers is None else plan_numbers.unsupported,
        )
        payload["roles"][role] = {
            "confirmed": len(grounding.citations),
            "rejected": grounding.rejected,
            "needs_review": len(grounding.needs_review),
            "user_input": len(grounding.user_input),
            "claims": len(grounding.verdicts),
            "conclusion_status": (grounding.conclusion or {}).get("status", ""),
        }
        if plan_numbers is not None:
            kinds: dict[str, int] = {}
            for note in plan_numbers.notes:
                kinds[note.kind] = kinds.get(note.kind, 0) + 1
                payload["plan_numbers"]["kinds"][note.kind] = (
                    payload["plan_numbers"]["kinds"].get(note.kind, 0) + 1
                )
            payload["plan_numbers"]["roles"][role] = {
                "notes": len(plan_numbers.notes),
                "unsupported": plan_numbers.unsupported,
                "kinds": kinds,
                "status": (grounding.conclusion or {}).get("status", ""),
            }
            payload["plan_numbers"]["unsupported"] += plan_numbers.unsupported
        for verdict in grounding.verdicts:
            payload["verdicts"].append(
                {
                    "session": session_id,
                    "role": role,
                    "claim": verdict.claim,
                    "source_id": verdict.source_id,
                    "status": verdict.status.value,
                    "detail": verdict.detail,
                    "kind": verdict.kind,
                }
            )
            bucket = bucket_of(verdict.status, verdict.detail)
            payload["kinds"][bucket] = payload["kinds"].get(bucket, 0) + 1
            # Причина называется только у отказа: «требует проверки» — не
            # отказ, у него своя строка исхода. Иначе одна и та же запись
            # попадала в две строки, а «нужен взгляд человека» считалось
            # причиной отказа (`T-59`).
            if verdict.status is SupportStatus.NO_DATA:
                kind = verdict.kind or classify_detail(verdict.detail)
                reason = f"причина: {label_of(kind)}"
                payload["kinds"][reason] = payload["kinds"].get(reason, 0) + 1
    return payload
