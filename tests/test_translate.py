"""Перевод внешних материалов: кэш, сохранность чисел, деградация.

Перевод существует ради одной цели: английский источник перестаёт быть
бесполезным. Цитату можно искать дословно в переводе, а числа — сверять с
оригиналом. Поэтому у перевода два обязательных свойства, и оба проверяются
здесь:

- числа, единицы и идентификаторы не меняются — иначе перевод нельзя
  цитировать: «87,5 %» превратится в «88 %» и станет выдумкой;
- оригинал не трогается никогда — иначе проверка дословности потеряет опору,
  а кэш начнёт ключеваться переводом вместо источника.

Третий тест — про честную деградацию: если модель недоступна, перевода просто
нет. Это не ошибка прогона, а отсутствие перевода.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pf import translate
from pf.llm import LLMError

#: Английский абзац, пригодный для перевода и для цитирования.
ORIGINAL = (
    "Mobile ARPU in Russia declined by 4,7 percent in 2025 while subscriber "
    "retention stayed at 87,5 percent, see the operator table row #row3."
)

#: Хороший перевод: числа и идентификатор сохранены, разделитель сменился.
GOOD_RU = (
    "ARPU мобильной связи в России снизился на 4.7 процента в 2025 году, "
    "удержание абонентов осталось на уровне 87.5 процента, см. строку #row3."
)


class FakeClient:
    """Клиент модели с заготовленным ответом."""

    def __init__(self, answer: str) -> None:
        """Запомнить ответ.

        Args:
            answer: Что вернёт модель.
        """
        self.answer = answer
        self.calls = 0

    def complete(self, system: str, user: str) -> SimpleNamespace:
        """Вернуть заготовленный ответ.

        Args:
            system: Системная инструкция (не используется).
            user: Текст задачи (не используется).

        Returns:
            Объект с содержимым и расходом.
        """
        self.calls += 1
        return SimpleNamespace(content=self.answer, usage=SimpleNamespace(total_tokens=42))


class BrokenClient:
    """Клиент, который не отвечает."""

    def __init__(self) -> None:
        """Создать клиент-отказник."""
        self.calls = 0

    def complete(self, system: str, user: str) -> SimpleNamespace:
        """Отказать.

        Args:
            system: Системная инструкция (не используется).
            user: Текст задачи (не используется).

        Raises:
            LLMError: Всегда.
        """
        self.calls += 1
        raise LLMError("модель не ответила вовремя")


def test_numbers_preserved_allows_good_translation(tmp_path: Path) -> None:
    """Сохранённые числа и идентификатор — перевод пригоден."""
    store = translate.TranslationStore(tmp_path)

    result = translate.translate(ORIGINAL, client=FakeClient(GOOD_RU), store=store)

    assert result.ok
    assert result.is_usable
    assert "87.5" in result.text or "87,5" in result.text
    assert "#row3" in result.text


def test_numbers_preserved_rejects_lost_number(tmp_path: Path) -> None:
    """Потерянное число делает перевод непригодным."""
    store = translate.TranslationStore(tmp_path)
    lossy = "ARPU в России снизился на 4.7 процента, удержание — 87 процентов, см. #row3."

    result = translate.translate(ORIGINAL, client=FakeClient(lossy), store=store)

    assert not result.ok
    assert not result.is_usable
    assert any("87.5" in note for note in result.notes), result.notes


def test_numbers_preserved_rejects_invented_number(tmp_path: Path) -> None:
    """Придуманное переводчиком число — тоже отказ."""
    store = translate.TranslationStore(tmp_path)
    invented = GOOD_RU + " Дополнительно: 12.4 процента."

    result = translate.translate(ORIGINAL, client=FakeClient(invented), store=store)

    assert not result.ok
    assert any("12.4" in note for note in result.notes), result.notes


def test_original_untouched_by_translation(tmp_path: Path) -> None:
    """Оригинал не меняется, и кэш ключуется оригиналом."""
    store = translate.TranslationStore(tmp_path)
    before = str(ORIGINAL)

    translate.translate(ORIGINAL, client=FakeClient(GOOD_RU), store=store)

    assert str(ORIGINAL) == before, "перевод не имеет права менять источник"
    assert store.get(ORIGINAL) is not None, "кэш обязан помнить источник"
    assert store.get(GOOD_RU) is None, "перевод не становится ключом кэша"


def test_cache_hit_avoids_second_model_call(tmp_path: Path) -> None:
    """Второй перевод того же текста берётся из кэша, без вызова модели."""
    store = translate.TranslationStore(tmp_path)
    translate.translate(ORIGINAL, client=FakeClient(GOOD_RU), store=store)
    second = FakeClient(GOOD_RU)

    again = translate.translate(ORIGINAL, client=second, store=store)

    assert second.calls == 0, "кэш обязан снимать вызов модели"
    assert again.ok
    assert again.text == GOOD_RU


def test_model_down_degrades_without_exception(tmp_path: Path) -> None:
    """Недоступная модель — отсутствие перевода, а не падение шага."""
    store = translate.TranslationStore(tmp_path)

    result = translate.translate(ORIGINAL, client=BrokenClient(), store=store)

    assert not result.ok
    assert result.text == ""
    assert any("модель" in note for note in result.notes), result.notes


def test_russian_text_is_not_translated(tmp_path: Path) -> None:
    """Русский текст не переводим и модель не тратим."""
    client = BrokenClient()
    store = translate.TranslationStore(tmp_path)

    result = translate.translate(
        "Удержание абонентов снизилось до 87,5 процента за год.",
        client=client,
        store=store,
    )

    assert result.ok
    assert result.engine == "none"
    assert client.calls == 0


def test_preserved_accepts_decimal_separator_change() -> None:
    """Запятая и точка в числе — одно и то же число."""
    ok, notes = translate.preserved("87,5 процента", "87.5 percent")

    assert ok, notes


def test_preserved_counts_units_and_identifiers() -> None:
    """Единица измерения и идентификатор входят в сверку."""
    ok_units, _ = translate.preserved("рост 12 %", "growth 12 %")
    ok_lost, notes = translate.preserved("рост 12 %", "growth 12")

    assert ok_units
    assert not ok_lost
    assert any("%" in note for note in notes)
