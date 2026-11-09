"""Причина отказа называет то, что случилось, а не похожее.

В прогоне `cloud-deepseek-2` девять отказов из двенадцати были «модель
сослалась на „Ответ человека“», и все девять попадали в корзину «источник
нерелевантен». Отчёт отправлял чинить поиск и контекст, хотя чинить надо
ссылки в ответе модели. Здесь проверяется, что причина названа верно, а класс
отказа считается по ней, а не по первому знакомому слову.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pf.config import settings
from pf.report import rejection_kind
from pf.schemas import Analysis, Claim, SourceRef
from pf.sources import SourceBundle, collect
from pf.workflow import Workflow


@pytest.fixture(scope="session")
def materials_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Каталог материалов: один файл в срезе задачи, второй — вне его.

    Args:
        tmp_path_factory: Фабрика временных каталогов.

    Returns:
        Путь к каталогу материалов проекта.
    """
    path = tmp_path_factory.mktemp("rejection-materials")
    (path / "numbers.csv").write_text("оператор,абонентская база\nМТС,87\n", encoding="utf-8")
    # Материал проекта, который к задаче не относится: он существует, но в
    # срез не попадает. Ровно это и было с ключевой ставкой в `demo-v12`.
    (path / "weather.csv").write_text("город,температура\nМосква,-5\n", encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def _collected(materials_dir: Path) -> SourceBundle:
    """Собранные источники — один раз на модуль.

    Сбор идёт по всему корпусу и стоит секунды; пяти тестам незачем
    повторять его пять раз.
    """
    patch = pytest.MonkeyPatch()
    patch.setattr(settings, "project_dir", materials_dir)
    try:
        return collect("абоненты оператора связи тарифы удержание")
    finally:
        patch.undo()


@pytest.fixture()
def bundle(
    materials_dir: Path, _collected: SourceBundle, monkeypatch: pytest.MonkeyPatch
) -> SourceBundle:
    """Набор источников с подменённым каталогом материалов.

    Каталог подменяется на время теста: `absence_reason` читает материалы
    проекта по требованию, и без подмены он смотрел бы в настоящий `project/`.

    Args:
        materials_dir: Каталог материалов.
        _collected: Собранные источники.
        monkeypatch: Подмена настроек.

    Returns:
        Собранные источники по задаче про абонентов.
    """
    monkeypatch.setattr(settings, "project_dir", materials_dir)
    return _collected


def test_rejection_reason_missing_link_is_phantom(bundle: SourceBundle) -> None:
    """Пустая ссылка и имя поля — «ссылки нет», а не «источник не по теме»."""
    assert bundle.absence_reason("") == "ссылки нет — цитировать нечего"
    assert bundle.absence_reason("verification") == "ссылки нет — цитировать нечего"
    assert (
        rejection_kind(
            "источник не назван — вместо ссылки указано поле «verification»: "
            "ссылки нет — цитировать нечего"
        )
        == "фантомный источник"
    )


def test_rejection_reason_human_input_has_own_class(bundle: SourceBundle) -> None:
    """Ввод человека — не источник, и у отказа свой класс."""
    reason = bundle.absence_reason("Ответ человека")
    assert "ввод человека" in reason
    assert rejection_kind(f"ввод человека: {reason}") == "ссылка на ввод человека"


def test_rejection_reason_outside_slice_is_not_phantom(bundle: SourceBundle) -> None:
    """Материал проекта вне среза — не «фантом», но и не основание."""
    assert not bundle.is_relevant("project/data/weather.csv"), (
        "фикстура: файл не должен попасть в срез, иначе случай не проверен"
    )
    assert bundle.absence_reason("project/data/weather.csv") == "источник вне среза задачи"
    assert (
        rejection_kind("project/data/weather.csv: источник вне среза задачи")
        == "источник нерелевантен"
    )


def test_rejection_reason_unknown_source_is_phantom(bundle: SourceBundle) -> None:
    """Ссылки нет ни в проекте, ни в корпусе — это фантомный источник."""
    assert bundle.absence_reason("project/data/nope.csv") == "источник не найден среди материалов"
    assert (
        rejection_kind("project/data/nope.csv: источник не найден среди материалов")
        == "фантомный источник"
    )


def test_verdict_keeps_the_reason_of_the_rejection(bundle: SourceBundle) -> None:
    """Вердикт несёт машинный вид отказа, а не только текст для человека.

    Ссылка на ввод пользователя — свой канал: она не «фантомный источник» и
    не «источник нерелевантен». Но в этой фикстуре ввода в сессии нет вовсе
    (`state=None`), поэтому вид называется точно: ссылка на ввод, которого не
    было (`D-49`).
    """
    analysis = Analysis(
        facts=[
            Claim(
                kind="факт",
                text="Абонентская база МТС — 87 млн",
                sources=[SourceRef(source_id="Ответ человека", quote="")],
            )
        ]
    )

    grounding = Workflow._verify_claims(analysis, bundle)

    assert [verdict.kind for verdict in grounding.verdicts] == ["human_input_unrecorded"]
    assert grounding.rejected == 1
