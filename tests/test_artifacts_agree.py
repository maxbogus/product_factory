"""Артефакты прогона из одного пути: документ, отчёт и перепроверка.

Документ демонстрации показывает **запись** прогона: статусы того кода, на
котором шла работа. На текущем коде те же ответы ролей дают другие числа, и
пока этого не видно, читатель принимает запись за проверку (`T-63`).

Здесь три свойства:

1. перепроверка (`pf report --recheck`) называет, что изменилось и какие
   основания переехали, и не переписывает журнал;
2. отсутствие сохранённых ответов называется словами, а не нулём;
3. документ собирается одной функцией (`pf.artifacts.build_document`) и несёт
   провенанс: журнал, его хеш, правило счёта.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from pf.artifacts import (
    build_document,
    provenance,
    recheck,
    recheck_markdown,
)
from pf.config import settings
from pf.corpus import Article
from pf.ground import CONCLUSION_RULE
from pf.journal import Journal, StepRecord, journal_path
from pf.sources import SourceBundle
from pf.state import Phase, SessionState, save
from pf.workflow import Workflow

ROOT = Path(__file__).resolve().parent.parent

#: Цитата, которая дословно есть в материале теста.
REAL_QUOTE = "обслужила 77,26 миллиона абонентов"

#: Утверждение, которое в записи было отклонено, а сейчас подтверждается.
CLAIM = "Компания обслужила 77,26 миллиона абонентов"


@pytest.fixture
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Изолировать состояние и журнал; источники — один материал.

    Args:
        tmp_path: Временный каталог.
        monkeypatch: Подмена настроек и сборки источников.

    Returns:
        Корень временного каталога.
    """
    monkeypatch.setattr(settings, "state_dir", tmp_path / "state")
    monkeypatch.setattr(settings, "journal_dir", tmp_path / "journal")
    monkeypatch.setattr(settings, "lexicon_dir", tmp_path / "lexicon")
    project = tmp_path / "project"
    project.mkdir()
    (project / "base.csv").write_text(
        "показатель,значение\nабонентская база,87\n", encoding="utf-8"
    )
    monkeypatch.setattr(settings, "project_dir", project)
    article = Article(
        source_id="project/data/base.csv",
        title="base.csv",
        section="project",
        content=f"МТС, абонентская база: 87. Компания {REAL_QUOTE}.",
    )
    bundle = SourceBundle(passages=[], articles=[article], query="удержание абонентов")
    monkeypatch.setattr(Workflow, "collect_sources", lambda self, state: bundle)
    return tmp_path


def _save_session(session_id: str) -> None:
    """Сохранить сессию с одним фактом, который проходит проверку.

    Args:
        session_id: Идентификатор сессии.
    """
    state = SessionState(
        session_id=session_id,
        phase=Phase.DONE,
        question="Почему падает удержание у мобильного тарифа?",
    )
    state.results[Phase.ANALYZING.value] = {
        "facts": [
            {
                "kind": "факт",
                "text": CLAIM,
                "sources": [{"source_id": "project/data/base.csv", "quote": REAL_QUOTE}],
            }
        ],
        "hypotheses": [],
        "forecasts": [],
        "gaps": [],
        "metrics": [],
    }
    save(state)


def _generator() -> ModuleType:
    """Загрузить скрипт сборки документа как модуль.

    Returns:
        Модуль `scripts/build_demo_doc.py`.
    """
    spec = importlib.util.spec_from_file_location(
        "build_demo_doc_under_test", ROOT / "scripts" / "build_demo_doc.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recheck_names_moved_grounds(_isolated: Path) -> None:
    """Перепроверка называет основание, которое переехало.

    В записи утверждение отклонено, на текущем коде — подтверждено. Молчать
    об этом нельзя: читатель документа считал бы запись проверкой.

    Args:
        _isolated: Изоляция состояния и источников.
    """
    _save_session("recheck-moved")
    Journal(journal_path("recheck-moved")).append(
        StepRecord(
            ordinal=0,
            role="analyst",
            phase="analyzing",
            summary="разбор",
            rejected_details=[f"project/data/base.csv: {CLAIM}"],
            rejections=[
                {
                    "kind": "number_mismatch",
                    "status": "no_data",
                    "detail": "значение не сошлось с источником",
                    "claim": CLAIM,
                }
            ],
        )
    )

    result = recheck("recheck-moved")
    text = "\n".join(recheck_markdown(result))

    assert result.available
    assert result.current_confirmed == 1
    assert result.recorded_confirmed == 0
    assert result.recorded_rejected == 1
    assert "переехали" in text
    assert CLAIM[:40] in text
    assert "сейчас «подтверждено»" in text


def test_recheck_without_state_is_named(_isolated: Path) -> None:
    """Без сохранённых ответов перепроверка говорит это словом, не нулём.

    Args:
        _isolated: Изоляция состояния.
    """
    Journal(journal_path("recheck-nostate")).append(
        StepRecord(ordinal=0, role="analyst", phase="analyzing", summary="разбор")
    )

    result = recheck("recheck-nostate")
    text = "\n".join(recheck_markdown(result))

    assert result.available is False
    assert "Перепроверить нечем" in text
    # Таблицы «сейчас» нет: сравнивать не с чем, и это сказано словами.
    assert "| Подтверждено |" not in text


def test_recheck_does_not_rewrite_journal(_isolated: Path) -> None:
    """Перепроверка — отдельный результат: журнал остаётся следом прогона.

    Args:
        _isolated: Изоляция состояния и источников.
    """
    _save_session("recheck-append-only")
    path = journal_path("recheck-append-only")
    Journal(path).append(StepRecord(ordinal=0, role="analyst", phase="analyzing", summary="разбор"))
    before = path.read_bytes()

    recheck("recheck-append-only")

    assert path.read_bytes() == before


def test_document_names_its_journal_hash(_isolated: Path) -> None:
    """Шапка документа называет журнал, его хеш и правило счёта.

    Без этого документ не отличить от проверки текущим кодом: числа в нём
    те, что были на момент прогона.

    Args:
        _isolated: Изоляция состояния и источников.
    """
    _save_session("doc-provenance")
    Journal(journal_path("doc-provenance")).append(
        StepRecord(ordinal=0, role="analyst", phase="analyzing", summary="разбор")
    )

    document = build_document("doc-provenance")
    origin = provenance("doc-provenance")

    assert origin.digest in document
    assert "sha256" in document
    assert origin.journal in document
    assert CONCLUSION_RULE in document
    assert "pf report --session doc-provenance --recheck" in document


def test_document_is_built_by_one_function() -> None:
    """Скрипт сборки не держит своей копии правил.

    Две сборки разошлись бы в том, что считать подтверждённым основанием, —
    так уже было у отчёта и документа.
    """
    module = _generator()

    assert module.build_document is build_document
    assert module.recheck is recheck
