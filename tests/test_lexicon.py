"""Проверки лексикона: словарь, очередь распознания и распознание.

Очередь — это «распознание»: слово, которого нет в словаре, не теряется и
не выдаётся за понятое. Оно копится, чтобы словарь можно было расширить
осознанно. Проверка оснований при этом не должна зависеть от очереди:
подтверждает словарь, а не модель.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pf import lexicon
from pf.lexical import overlap


@pytest.fixture
def lexicon_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Подменить каталог лексикона на временный.

    Args:
        tmp_path: Временный каталог теста.
        monkeypatch: Подмена настроек.

    Returns:
        Путь к временному каталогу лексикона.
    """
    monkeypatch.setattr("pf.config.settings.lexicon_dir", tmp_path)
    return tmp_path


class FakeClient:
    """Клиент модели с заранее заданным ответом."""

    def __init__(self, answer: str) -> None:
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
        return SimpleNamespace(content=self.answer, usage=SimpleNamespace(total_tokens=7))


def test_groups_include_approved_entries(lexicon_dir: Path) -> None:
    """Подтверждённая запись словаря участвует в канонизации."""
    (lexicon_dir / lexicon.SYNONYMS_NAME).write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "terms": [
                    {
                        "canonical": "оператор",
                        "aliases": ["виртуальный"],
                        "status": "approved",
                        "source": "manual",
                    }
                ],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    assert lexicon.groups()["виртуальный"] == "оператор"


def test_proposed_entry_is_not_used(lexicon_dir: Path) -> None:
    """Предложенная моделью запись в проверке не участвует."""
    lexicon.propose("виртуальный", "оператор", source="llm")

    assert "виртуальный" not in lexicon.groups()
    assert lexicon.approve("виртуальный", "оператор")
    assert lexicon.groups()["виртуальный"] == "оператор"


def test_oov_queue_is_append_only_and_counted(lexicon_dir: Path) -> None:
    """Повторная запись увеличивает счётчик, а не перезаписывает строку."""
    lexicon.record_oov("криптовалюта", reason="нет в словаре", session="s1")
    lexicon.record_oov("криптовалюта", reason="нет в словаре", session="s2")
    lexicon.record_oov("скроллинг", reason="нет в словаре", session="s1")

    records = lexicon.read_oov()
    assert [record.token for record in records] == ["криптовалюта", "скроллинг"]
    assert records[0].count == 2
    assert records[0].sessions == ["s1", "s2"]

    lines = (lexicon_dir / lexicon.OOV_NAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3, "очередь только дописывается"


def test_oov_contexts_are_capped(lexicon_dir: Path) -> None:
    """Контекст хранится ограниченно, чтобы очередь не распухала."""
    for index in range(6):
        lexicon.record_oov("криптовалюта", context=f"контекст {index}")

    record = lexicon.read_oov()[0]
    assert len(record.contexts) <= 3


def test_pending_respects_min_count(lexicon_dir: Path) -> None:
    """Редкие слова не тратят бюджет модели."""
    lexicon.record_oov("редкое")
    lexicon.record_oov("частое")
    lexicon.record_oov("частое")

    pending = lexicon.pending_oov(min_count=2)
    assert [record.token for record in pending] == ["частое"]


def test_junk_is_not_pending(lexicon_dir: Path) -> None:
    """Мусор выпадает из очереди."""
    lexicon.record_oov("ааа")
    lexicon.mark_junk("ааа")

    assert lexicon.pending_oov() == []


def test_enrich_dry_run_writes_nothing(lexicon_dir: Path) -> None:
    """Черновик показывает предложения и не меняет словарь."""
    lexicon.record_oov("виртуальный")
    client = FakeClient("оператор")

    result = lexicon.enrich(client=client, dry_run=True)

    assert result.proposals[0].action == lexicon.ACTION_SYNONYM
    assert result.proposals[0].canonical == "оператор"
    assert not (lexicon_dir / lexicon.SYNONYMS_NAME).exists()
    assert lexicon.read_oov()[0].status == lexicon.PENDING


def test_enrich_apply_records_proposal(lexicon_dir: Path) -> None:
    """Запись сохраняется как предложение, а не как подтверждение."""
    lexicon.record_oov("виртуальный")
    client = FakeClient("оператор")

    lexicon.enrich(client=client, dry_run=False)

    entries = lexicon.read_entries()
    assert entries[0].status == lexicon.PROPOSED
    assert entries[0].source == "llm"
    assert lexicon.read_oov()[0].status == lexicon.PROPOSED
    assert "виртуальный" not in lexicon.groups()


def test_parse_enrich_answer_understands_free_text() -> None:
    """Ответ модели разбирается терпимо."""
    known = {"абонент", "удержание"}

    assert lexicon.parse_enrich_answer("абонент", known) == (
        lexicon.ACTION_SYNONYM,
        "абонент",
    )
    assert lexicon.parse_enrich_answer("это термин", known)[0] == lexicon.ACTION_TERM
    assert lexicon.parse_enrich_answer("мусор", known)[0] == lexicon.ACTION_JUNK
    assert lexicon.parse_enrich_answer("не знаю", known)[0] == lexicon.ACTION_UNRESOLVED


def test_coverage_counts_oov(lexicon_dir: Path) -> None:
    """Покрытие показывает долю распознанных слов."""
    coverage = lexicon.coverage_report(["абоненты и криптовалюта"])

    assert coverage.total == 2
    assert coverage.oov == 1
    assert coverage.rate == pytest.approx(0.5)
    assert "криптовалюта" in coverage.oov_tokens


def test_pending_does_not_change_verification(lexicon_dir: Path) -> None:
    """Очередь не влияет на проверку: подтверждает словарь."""
    lexicon.record_oov("виртуальный")
    before = overlap("виртуальный оператор", "мобильный оператор")
    lexicon.propose("виртуальный", "оператор", source="llm")
    after_proposal = overlap("виртуальный оператор", "мобильный оператор")
    lexicon.approve("виртуальный", "оператор")
    after_approval = overlap("виртуальный оператор", "мобильный оператор")

    assert after_proposal.coverage == before.coverage
    assert after_approval.coverage > before.coverage


def test_stats_counts_entries_and_queue(lexicon_dir: Path) -> None:
    """Сводка считает и словарь, и очередь."""
    lexicon.record_oov("криптовалюта")
    lexicon.propose("криптовалюта", "криптовалюта", source="llm")

    data = lexicon.stats()

    assert data["entries"] >= 1
    assert data["oov_total"] == 1
    assert data["oov_pending"] == 0
