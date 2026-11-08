"""Пересмотр выводов при новых данных.

Бриф, дополнительный уровень: «Агент находит, какие выводы затронуты
новыми сведениями, и обновляет их». Проверки держат три свойства,
без которых пересмотр превращается в имитацию:

1. **Молчание не считается проверкой.** Совпавший хеш называется словами
   («не изменилось»), а не пропускается.
2. **Три новости различаются.** «Изменилось», «файл исчез» и «хеш не
   записан» — разные состояния: у них разные последствия для вывода.
3. **Ничего не переписывается.** Журнал append-only: пересмотр его
   читает и возвращает отдельный результат.
"""

from __future__ import annotations

import json
from pathlib import Path

from pf.journal import Journal, StepRecord
from pf.knowledge_store import sha256_file, write_manifest
from pf.review import DataStatus, review_markdown, review_session

SESSION = "review-test"


def _project(tmp_path: Path) -> Path:
    """Каталог материалов проекта с одним файлом данных."""
    directory = tmp_path / "project" / "data"
    directory.mkdir(parents=True)
    (directory / "metrics.csv").write_text("оператор,значение\nМТС,87\n", encoding="utf-8")
    return tmp_path / "project"


def _seal(path: Path, *, sha: str | None = None, url: str = "https://example.org/data") -> None:
    """Записать метаданные материала с хешем (по умолчанию — верным)."""
    record = {
        "title": path.stem,
        "url": url,
        "retrieved": "2026-11-08",
        "note": "тестовая выгрузка",
        "sha256": sha if sha is not None else sha256_file(path),
        "bytes": path.stat().st_size,
    }
    path.with_name(path.name + ".meta.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8"
    )


def _write_journal(
    directory: Path,
    steps: list[StepRecord],
    session: str = SESSION,
) -> Path:
    """Записать журнал сессии в отдельный каталог."""
    journal = Journal(directory / f"session-{session}.jsonl")
    for step in steps:
        journal.append(step)
    return journal.path


def _step(
    ordinal: int,
    citations: list[str],
    *,
    role: str = "collector",
    phase: str = "collecting",
    commitments: list[dict[str, object]] | None = None,
) -> StepRecord:
    """Запись шага с основаниями и, если нужно, обязательствами."""
    detail: dict[str, object] = {}
    if commitments is not None:
        detail["commitments"] = commitments
    return StepRecord(
        ordinal=ordinal,
        role=role,
        phase=phase,
        summary="шаг",
        detail=detail,
        citations=citations,
    )


def test_review_reports_unchanged_data(tmp_path: Path) -> None:
    """Совпавший хеш называется словами, а не умалчивается.

    Иначе «ничего не напечатано» невозможно отличить от «ничего не
    проверено».
    """
    project = _project(tmp_path)
    path = project / "data" / "metrics.csv"
    _seal(path)
    journals = tmp_path / "journal"
    _write_journal(journals, [_step(1, ["project/data/metrics.csv#row0"])])

    result = review_session(
        SESSION, journal_dir=journals, project_dir=project, knowledge_dir=tmp_path / "knowledge"
    )
    text = review_markdown(result)

    assert result.checked, "источник должен быть проверен"
    assert result.checked[0].status is DataStatus.OK
    assert result.is_clean
    assert "не изменилось" in text


def test_review_names_changed_data_and_affected_steps(tmp_path: Path) -> None:
    """Изменённый файл назван вместе с шагами и обязательствами.

    Хеш меняется и при косметической правке, поэтому формулировка —
    «требует пересмотра», а не «вывод неверен». Проверяется и то, и
    другое: новость названа, а приговор не вынесен.
    """
    project = _project(tmp_path)
    path = project / "data" / "metrics.csv"
    _seal(path)
    path.write_text("оператор,значение\nМТС,88\n", encoding="utf-8")

    journals = tmp_path / "journal"
    _write_journal(
        journals,
        [
            _step(1, ["project/data/metrics.csv#row0"]),
            _step(
                2,
                ["project/data/metrics.csv#row0"],
                role="keeper",
                phase="keeper",
                commitments=[
                    {
                        "statement": "Поднять удержание на 5 %",
                        "owner": "маркетинг",
                        "basis": [{"source_id": "project/data/metrics.csv#row0"}],
                    }
                ],
            ),
        ],
    )

    result = review_session(
        SESSION, journal_dir=journals, project_dir=project, knowledge_dir=tmp_path / "knowledge"
    )
    text = review_markdown(result)

    assert result.changed, "изменение должно быть замечено"
    assert result.checked[0].status is DataStatus.CHANGED
    assert [step.ordinal for step in result.affected_steps] == [1, 2]
    assert result.affected_commitments, "обязательство должно быть названо"
    assert "Поднять удержание" in text
    assert "требует пересмотра" in text
    assert "вывод неверен" not in text.replace("«вывод неверен»", "")


def test_review_distinguishes_missing_file(tmp_path: Path) -> None:
    """Исчезнувший файл — это «файл отсутствует», а не «изменилось»."""
    project = _project(tmp_path)
    path = project / "data" / "metrics.csv"
    _seal(path)
    path.unlink()

    journals = tmp_path / "journal"
    _write_journal(journals, [_step(1, ["project/data/metrics.csv#row0"])])

    result = review_session(
        SESSION, journal_dir=journals, project_dir=project, knowledge_dir=tmp_path / "knowledge"
    )
    text = review_markdown(result)

    assert result.checked[0].status is DataStatus.MISSING
    assert "файл отсутствует" in text
    assert result.changed, "исчезнувший файл затрагивает вывод"


def test_review_reports_unrecorded_hash(tmp_path: Path) -> None:
    """Материал без записанного хеша называется, но выводов не касается.

    Хеш не записан — это пробел в происхождении данных, а не изменение.
    Смешать их значило бы объявить вывод устаревшим из-за отсутствия
    записи.
    """
    project = _project(tmp_path)
    path = project / "data" / "metrics.csv"
    _seal(path, sha="")

    journals = tmp_path / "journal"
    _write_journal(journals, [_step(1, ["project/data/metrics.csv#row0"])])

    result = review_session(
        SESSION, journal_dir=journals, project_dir=project, knowledge_dir=tmp_path / "knowledge"
    )
    text = review_markdown(result)

    assert result.checked[0].status is DataStatus.UNRECORDED
    assert result.is_clean, "отсутствие записи не делает вывод устаревшим"
    assert "хеш не записан" in text


def test_review_checks_download_manifest(tmp_path: Path) -> None:
    """Хеш выгрузки берётся из манифеста, а не из воздуха.

    Манифест append-only: у файла может быть несколько версий, и
    действующей считается последняя записанная.
    """
    knowledge = tmp_path / "knowledge"
    snapshot = knowledge / "raw" / "2026-11-08"
    snapshot.mkdir(parents=True)
    path = snapshot / "tariffs.csv"
    path.write_text("тариф,цена\nОптимальный,950\n", encoding="utf-8")
    write_manifest(
        snapshot,
        [{"name": "tariffs.csv", "url": "https://example.org/t", "kind": "csv", "note": "снимок"}],
    )
    path.write_text("тариф,цена\nОптимальный,1050\n", encoding="utf-8")

    journals = tmp_path / "journal"
    _write_journal(journals, [_step(1, ["knowledge/raw/2026-11-08/tariffs.csv#row0"])])

    result = review_session(
        SESSION,
        journal_dir=journals,
        project_dir=tmp_path / "project",
        knowledge_dir=knowledge,
    )
    text = review_markdown(result)

    assert result.checked[0].status is DataStatus.CHANGED
    assert result.checked[0].recorded, "записанный хеш должен быть виден"
    assert "требует пересмотра" in text


def test_review_does_not_rewrite_journal(tmp_path: Path) -> None:
    """Пересмотр читает журнал и не меняет его.

    Журнал append-only: история работы — след, а не черновик. Пересмотр,
    который «поправил» бы шаг, уничтожил бы доказательство того, что было
    сделано.
    """
    project = _project(tmp_path)
    _seal(project / "data" / "metrics.csv")
    journals = tmp_path / "journal"
    path = _write_journal(journals, [_step(1, ["project/data/metrics.csv#row0"])])
    before = path.read_bytes()

    review_session(
        SESSION, journal_dir=journals, project_dir=project, knowledge_dir=tmp_path / "knowledge"
    )

    assert path.read_bytes() == before, "журнал переписан пересмотром"


def test_review_marks_articles_outside_review(tmp_path: Path) -> None:
    """Статьи корпуса не пересматриваются, и это сказано словами."""
    project = _project(tmp_path)
    journals = tmp_path / "journal"
    _write_journal(
        journals,
        [_step(1, ["telecom/Отток клиентов#p2"])],
    )

    result = review_session(
        SESSION, journal_dir=journals, project_dir=project, knowledge_dir=tmp_path / "knowledge"
    )
    text = review_markdown(result)

    assert result.outside == ["telecom/Отток клиентов"]
    assert "не пересматриваются" in text
    assert not result.checked

    assert "требует пересмотра" not in text.split("### Что затронуто")[1].split("###")[0]
