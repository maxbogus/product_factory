"""Вид отказа называется по машинному имени, а не по русскому тексту.

Отчёт, снимок замера и документ демонстрации считали класс отказа по-разному:
отчёт разбирал русский текст причины (`report.rejection_kind`), замер — статус
и подстроку (`scripts/replay_verification.py:_bucket`). Машинный вид отказа при
этом существовал (`Support.kind`, `ABSENCE_KINDS`, `PlanNumberNote.kind`), но в
журнал не попадал — оставалась фраза, и каждый потребитель угадывал класс
заново. Новый отказ `indicator-demotion` поэтому попал в «прочее», а «прочее»
не отвечает на вопрос «что чинить».

Здесь проверяется единственная таблица видов: у каждого машинного вида есть
подпись и владелец исправления, известные тексты старых записей в «прочее» не
падают, а записанный вид сильнее разбора текста.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pf.config import settings
from pf.journal import Journal, StepRecord, journal_path
from pf.report import build_session_report
from pf.sources import (
    ABSENCE_KINDS,
    REASON_HUMAN_INPUT,
    REASON_NO_LINK,
    REASON_OUTSIDE_SLICE,
    REASON_UNKNOWN_SOURCE,
)
from pf.verdicts import (
    CONFIRMING_KINDS,
    FALLBACK_LABEL,
    INDICATOR_DEMOTION,
    KINDS,
    UNKNOWN_KIND,
    classify_detail,
    fix_of,
    label_of,
)
from pf.workflow import NUMBER_FOREIGN, NUMBER_MISMATCH, NUMBER_WITHOUT_CALC

#: Корень репозитория — нужен, чтобы гейт видел новые виды прямо в коде.
ROOT = Path(__file__).resolve().parent.parent

#: Тексты отказов, которые до таблицы попадали в «прочее» (из пяти журналов).
#: Формулировки записаны прогонами: журнал append-only, и переписать их нельзя.
LEGACY_DETAILS: tuple[str, ...] = (
    "project/data/telecom_arpu.csv: число 322.2 не совпадает с источником (допуск 0.1611)",
    "project/data/telecom_open_metrics.csv: термин «МТС» не найден в источнике",
    "project/data/telecom_open_metrics.csv: цитата короче 12 символов — подтвердить нечего",
    "ожидаемое изменение «Удержание вырастет на 5-7%»: число -7 % не совпадает "
    "с расчётом по показателю «заявки на перенос номера»",
    "критерий успеха «Удержание выросло на 5%»: число 5 % ничем не подтверждено — "
    "расчёта с таким показателем нет",
    "критерий успеха «Вовлеченность выросла на 10%»: число 10 % взято из расчёта "
    "«численность абонентов в России» — другой показатель",
    "в строке показатель «абонентская база (консолидированная)», а утверждение "
    "говорит о другом: число в строке есть, но подтверждает оно другой показатель",
)


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолировать журнал и состояние во временном каталоге."""
    monkeypatch.setattr(settings, "state_dir", tmp_path / "state")
    monkeypatch.setattr(settings, "journal_dir", tmp_path / "journal")


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_kind_has_label_and_fix(kind: str) -> None:
    """У вида есть подпись для человека и названный владелец исправления."""
    assert label_of(kind)
    assert label_of(kind) != FALLBACK_LABEL, "известный вид не может называться «прочее»"
    assert fix_of(kind)
    assert fix_of(kind) != "неизвестно", "класс нужен, чтобы знать, что чинить"


def test_machine_kinds_from_ground_are_in_the_table() -> None:
    """Новый вид отказа в `ground.py` без подписи валит этот тест."""
    source = (ROOT / "src" / "pf" / "ground.py").read_text(encoding="utf-8")
    used = set(re.findall(r'kind="([a-z_]+)"', source))
    assert used - CONFIRMING_KINDS <= set(KINDS), "вид проверки цитаты не описан в таблице"


def test_absence_kinds_are_in_the_table() -> None:
    """Виды отказа ссылки (`ABSENCE_KINDS`) описаны подписями."""
    assert set(ABSENCE_KINDS.values()) <= set(KINDS)


def test_plan_number_kinds_are_in_the_table() -> None:
    """Числа плана и обязательств: у каждого вида есть подпись."""
    assert {NUMBER_MISMATCH, NUMBER_FOREIGN, NUMBER_WITHOUT_CALC} <= set(KINDS)


@pytest.mark.parametrize("detail", LEGACY_DETAILS)
def test_known_details_never_fall_into_fallback(detail: str) -> None:
    """Запись прогона читается: причина названа, а не спрятана в «прочее»."""
    assert classify_detail(detail) in KINDS


def test_unknown_detail_is_named_unclassified() -> None:
    """Неизвестная причина называется прямо, а не выдаётся за класс."""
    assert classify_detail("причина, которой ещё нет в таблице") == UNKNOWN_KIND
    assert label_of(UNKNOWN_KIND) == FALLBACK_LABEL


def test_indicator_demotion_has_its_own_class() -> None:
    """Основание про другой показатель — не «прочее», а названный отказ."""
    detail = "в строке показатель «ARPU», а утверждение говорит о другом: число есть"
    assert classify_detail(detail) == INDICATOR_DEMOTION


def test_reasons_keep_their_labels() -> None:
    """Подписи прежних классов не поехали: по ним читают отчёты."""
    assert label_of(classify_detail(f"ввод человека: {REASON_HUMAN_INPUT}")) == (
        "ссылка на ввод человека"
    )
    assert label_of(classify_detail(f"источник не назван: {REASON_NO_LINK}")) == (
        "фантомный источник"
    )
    assert label_of(classify_detail(f"x: {REASON_OUTSIDE_SLICE}")) == "источник нерелевантен"
    assert label_of(classify_detail(f"x: {REASON_UNKNOWN_SOURCE}")) == "фантомный источник"


def test_journal_keeps_rejection_kind() -> None:
    """Журнал хранит машинный вид отказа: текст — для человека, вид — для счёта."""
    journal = Journal(settings.journal_dir / "s.jsonl")
    journal.append(
        StepRecord(
            ordinal=1,
            role="analyst",
            phase="analyzing",
            summary="факты",
            rejected_details=["не источник — это ввод человека, а не документ"],
            rejections=[
                {
                    "kind": "human_input",
                    "status": "no_data",
                    "detail": "не источник — это ввод человека, а не документ",
                    "claim": "Работаем только коммуникацией",
                }
            ],
        )
    )

    record = journal.read()[0]
    assert record.rejections[0]["kind"] == "human_input"


def test_report_prefers_recorded_kind_over_text() -> None:
    """Записанный вид сильнее разбора текста: иначе класс зависит от формулировки."""
    session_id = "kind-report"
    journal = Journal(journal_path(session_id))
    journal.append(
        StepRecord(
            ordinal=1,
            role="strategy",
            phase="strategy",
            summary="стратегия",
            rejected_details=["формулировка, которой нет в таблице текстов"],
            rejections=[
                {
                    "kind": "indicator-demotion",
                    "status": "no_data",
                    "detail": "формулировка, которой нет в таблице текстов",
                    "claim": "вывод",
                }
            ],
        )
    )

    report = build_session_report(session_id)

    assert report.rejection_kinds == {"основание про другой показатель": 1}
