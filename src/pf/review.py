"""Пересмотр выводов при новых данных.

Бриф, дополнительный уровень: «Пересмотр при новых данных. Агент находит,
какие выводы затронуты новыми сведениями, и обновляет их». Пока этого
модуля не было, в README был нарисован цикл «обязательства → новые
данные», а механизма не было: обещание без работы.

Что здесь делается и чего не делается:

- сравниваются **хеши**, а не скачивается заново. Вопрос пересмотра —
  «устарел ли вывод», а не «обновите данные»;
- хеш меняется и при косметической правке файла. Поэтому вывод получает
  пометку «требует пересмотра», а не «вывод неверен». Это тот случай,
  когда честная формулировка важнее красивой;
- журнал **не переписывается**: пересмотр — отдельный результат, а не
  правка истории. Журнал append-only, и прежний ответ не становится
  фактом задним числом.

Хеши берутся оттуда, где они записаны в момент получения данных: у сырых
выгрузок — из ``knowledge/raw/<дата>/manifest.jsonl`` (append-only,
поэтому читается последняя запись по имени файла), у материалов проекта —
из соседнего ``<файл>.meta.json``. Если хеш не записан, состояние честно
называется «хеш не записан»: сравнивать нечем, и это не то же самое, что
«не изменилось».
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pf.config import settings
from pf.journal import Journal, StepRecord, journal_path
from pf.knowledge_store import read_manifest, sha256_file
from pf.materials import KNOWLEDGE_SOURCE_PREFIX, METADATA_SUFFIX

__all__ = [
    "AffectedCommitment",
    "AffectedStep",
    "CheckedSource",
    "DataStatus",
    "ReviewResult",
    "review_markdown",
    "review_session",
]

#: Префикс материалов проекта в идентификаторах оснований.
PROJECT_PREFIX = "project/"


class DataStatus(StrEnum):
    """Состояние источника данных на момент пересмотра.

    Состояния различимы намеренно: «изменилось», «исчезло» и «хеш не
    записан» — три разные новости, и путать их нельзя. «Не изменилось»
    тоже говорится прямо, иначе молчание выглядит как проверка.
    """

    #: Хеш совпадает: вывод опирается на ту же версию данных.
    OK = "ok"
    #: Хеш другой: вывод может быть устаревшим.
    CHANGED = "changed"
    #: Файла нет: проверить нечего.
    MISSING = "missing"
    #: Хеш не записан: сравнивать не с чем.
    UNRECORDED = "unrecorded"

    @property
    def title_ru(self) -> str:
        """Как состояние называется по-русски."""
        return {
            DataStatus.OK: "не изменилось",
            DataStatus.CHANGED: "требует пересмотра",
            DataStatus.MISSING: "файл отсутствует",
            DataStatus.UNRECORDED: "хеш не записан",
        }[self]

    @property
    def affects(self) -> bool:
        """Затрагивает ли состояние выводы.

        «Хеш не записан» выводов не затрагивает: это пробел в
        происхождении данных, а не их изменение. Он называется отдельно.
        """
        return self in {DataStatus.CHANGED, DataStatus.MISSING}


@dataclass(frozen=True, slots=True)
class CheckedSource:
    """Источник, на который опирался вывод, и состояние его данных.

    Attributes:
        source_id: Идентификатор источника без якоря.
        status: Состояние данных.
        recorded: Хеш, записанный при получении данных.
        current: Текущий хеш файла (пустой, если файла нет).
        url: Адрес источника, если он записан.
        retrieved: Дата получения, если записана.
        detail: Пояснение для человека.
    """

    source_id: str
    status: DataStatus
    recorded: str = ""
    current: str = ""
    url: str = ""
    retrieved: str = ""
    detail: str = ""

    @property
    def short_hash(self) -> str:
        """Короткий вид записанного хеша для отчёта."""
        return self.recorded[:12] if self.recorded else "—"


@dataclass(frozen=True, slots=True)
class AffectedStep:
    """Шаг журнала, чьи основания опирались на изменившийся источник.

    Attributes:
        ordinal: Номер шага в сессии.
        role: Роль, выполнявшая шаг.
        phase: Фаза, в которой шаг выполнялся.
        citations: Ссылки, которые затронуты.
    """

    ordinal: int
    role: str
    phase: str
    citations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AffectedCommitment:
    """Обязательство, чьё основание опиралось на изменившийся источник.

    Attributes:
        ordinal: Номер шага, где обязательство принято.
        statement: Что обещано.
        owner: Кто отвечает.
        citations: Ссылки, которые затронуты.
    """

    ordinal: int
    statement: str
    owner: str = ""
    citations: tuple[str, ...] = ()


@dataclass(slots=True)
class ReviewResult:
    """Итог пересмотра сессии.

    Attributes:
        session_id: Сессия, которая пересматривается.
        checked: Проверенные источники — по одному на ссылку без якоря.
        affected_steps: Шаги, затронутые изменениями.
        affected_commitments: Обязательства, затронутые изменениями.
        outside: Ссылки, которые не пересматриваются (статьи корпуса).
        steps_total: Сколько шагов в журнале.
    """

    session_id: str
    checked: list[CheckedSource] = field(default_factory=list)
    affected_steps: list[AffectedStep] = field(default_factory=list)
    affected_commitments: list[AffectedCommitment] = field(default_factory=list)
    outside: list[str] = field(default_factory=list)
    steps_total: int = 0

    @property
    def changed(self) -> list[CheckedSource]:
        """Источники, чьи данные изменились или исчезли."""
        return [source for source in self.checked if source.status.affects]

    @property
    def unrecorded(self) -> list[CheckedSource]:
        """Источники без записанного хеша."""
        return [source for source in self.checked if source.status is DataStatus.UNRECORDED]

    @property
    def is_clean(self) -> bool:
        """Затронуто ли что-нибудь новыми данными."""
        return not self.changed


def _read_json(path: Path) -> dict[str, Any]:
    """Прочитать JSON-файл в словарь.

    Args:
        path: Путь к файлу.

    Returns:
        Словарь; пустой, если файла нет или он не разобрался.
    """
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _check_source(path: str, project_root: Path, knowledge_root: Path) -> CheckedSource:
    """Сравнить записанный хеш источника с текущим файлом.

    Args:
        path: Идентификатор источника без якоря (``project/...`` или
            ``knowledge/raw/<дата>/...``).
        project_root: Каталог материалов проекта.
        knowledge_root: Каталог хранилища знаний.

    Returns:
        Состояние данных источника.
    """
    if path.startswith(KNOWLEDGE_SOURCE_PREFIX):
        relative = path[len(KNOWLEDGE_SOURCE_PREFIX) :]
        snapshot, _, name = relative.partition("/")
        directory = knowledge_root / "raw" / snapshot
        records = [
            record for record in read_manifest(directory) if str(record.get("name", "")) == name
        ]
        # Манифест append-only: версий у файла может быть несколько.
        # Действующей считается последняя записанная.
        latest = records[-1] if records else {}
        recorded = str(latest.get("sha256", ""))
        url = str(latest.get("url", ""))
        retrieved = str(latest.get("retrieved", ""))
        file = directory / name
        origin = "выгрузка"
    else:
        relative = path[len(PROJECT_PREFIX) :]
        file = project_root / relative
        meta = _read_json(file.with_name(file.name + METADATA_SUFFIX))
        recorded = str(meta.get("sha256", ""))
        url = str(meta.get("url", ""))
        retrieved = str(meta.get("retrieved", ""))
        origin = "материал проекта"

    if not file.is_file():
        return CheckedSource(
            source_id=path,
            status=DataStatus.MISSING,
            recorded=recorded,
            url=url,
            retrieved=retrieved,
            detail=f"{origin} не найден по пути {file}",
        )

    current = sha256_file(file)
    if not recorded:
        return CheckedSource(
            source_id=path,
            status=DataStatus.UNRECORDED,
            current=current,
            url=url,
            retrieved=retrieved,
            detail=f"у {origin}а не записан хеш: сравнивать не с чем",
        )

    return CheckedSource(
        source_id=path,
        status=DataStatus.OK if recorded == current else DataStatus.CHANGED,
        recorded=recorded,
        current=current,
        url=url,
        retrieved=retrieved,
        detail="" if recorded == current else f"хеш {origin}а изменился с момента получения",
    )


def _citations_of(step: StepRecord) -> list[str]:
    """Ссылки шага: объявленные основания и основания обязательств.

    ``StepRecord.citations`` заполняет рабочий процесс, но обязательства
    ссылаются на источники ещё и через ``basis``. Обе дороги ведут к
    одному вопросу «какие данные затронуты», поэтому обходятся вместе.

    Args:
        step: Запись журнала.

    Returns:
        Ссылки в порядке появления, без повторов.
    """
    collected: list[str] = list(step.citations)
    for commitment in step.detail.get("commitments") or []:
        if not isinstance(commitment, dict):
            continue
        for item in commitment.get("basis") or []:
            if isinstance(item, dict):
                collected.append(str(item.get("source_id") or ""))

    unique: list[str] = []
    for citation in collected:
        if citation and citation not in unique:
            unique.append(citation)
    return unique


def review_session(
    session_id: str,
    *,
    journal_dir: Path | None = None,
    project_dir: Path | None = None,
    knowledge_dir: Path | None = None,
) -> ReviewResult:
    """Пересмотреть выводы сессии по изменившимся данным.

    Ничего не переписывает: журнал читается, результат возвращается
    отдельным объектом.

    Args:
        session_id: Идентификатор сессии.
        journal_dir: Каталог журналов. По умолчанию — из настроек.
        project_dir: Каталог материалов проекта. По умолчанию — из настроек.
        knowledge_dir: Каталог хранилища знаний. По умолчанию — из настроек.

    Returns:
        Итог пересмотра: проверенные источники и затронутые выводы.
    """
    path = journal_dir / f"session-{session_id}.jsonl" if journal_dir else journal_path(session_id)
    steps = Journal(path).read()
    project_root = project_dir or settings.project_dir
    knowledge_root = knowledge_dir or settings.knowledge_dir

    result = ReviewResult(session_id=session_id, steps_total=len(steps))
    checked: dict[str, CheckedSource] = {}
    affected: set[str] = set()

    for step in steps:
        step_citations: list[str] = []
        for citation in _citations_of(step):
            source_path = citation.partition("#")[0].strip()
            if not source_path:
                continue
            if not (
                source_path.startswith(PROJECT_PREFIX)
                or source_path.startswith(KNOWLEDGE_SOURCE_PREFIX)
            ):
                # Статьи корпуса — не выгрузки: их версии не хешируются.
                if source_path not in result.outside:
                    result.outside.append(source_path)
                continue
            if source_path not in checked:
                checked[source_path] = _check_source(source_path, project_root, knowledge_root)
            if checked[source_path].status.affects:
                affected.add(source_path)
                if citation not in step_citations:
                    step_citations.append(citation)

        if not step_citations:
            continue

        result.affected_steps.append(
            AffectedStep(
                ordinal=step.ordinal,
                role=step.role,
                phase=step.phase,
                citations=tuple(step_citations),
            )
        )
        for commitment in step.detail.get("commitments") or []:
            if not isinstance(commitment, dict):
                continue
            basis = [
                str(item.get("source_id") or "")
                for item in commitment.get("basis") or []
                if isinstance(item, dict)
            ]
            touched = tuple(
                citation for citation in basis if citation.partition("#")[0] in affected
            )
            if touched:
                result.affected_commitments.append(
                    AffectedCommitment(
                        ordinal=step.ordinal,
                        statement=str(commitment.get("statement", "")),
                        owner=str(commitment.get("owner", "")),
                        citations=touched,
                    )
                )

    result.checked = list(checked.values())
    return result


def review_markdown(result: ReviewResult) -> str:
    """Оформить итог пересмотра для человека.

    Три раздела обязательны. «Что проверено» — иначе молчание выглядит
    как проверка. «Что затронуто новыми данными» — ответ на вопрос брифа.
    «Чего пересмотр не показывает» — границы замера: без них «требует
    пересмотра» читается как «вывод неверен».

    Args:
        result: Итог пересмотра.

    Returns:
        Текст в разметке Markdown.
    """
    lines = [f"## Пересмотр выводов: сессия {result.session_id}", ""]
    lines.append(
        f"Шагов в журнале: {result.steps_total}. "
        f"Проверено источников данных: {len(result.checked)}."
    )
    if result.changed:
        lines.append(
            f"Затронуто новыми данными: источников {len(result.changed)}, "
            f"шагов {len(result.affected_steps)}, "
            f"обязательств {len(result.affected_commitments)}."
        )
    else:
        lines.append(
            "Затронутых выводов нет: данные, на которые опиралась работа, "
            "не изменились с момента получения."
        )

    lines.extend(
        ["", "### Что проверено", "", "| Источник | Записанный хеш | Состояние |", "|---|---|---|"]
    )
    if result.checked:
        for source in result.checked:
            lines.append(
                f"| `{source.source_id}` | {source.short_hash} | {source.status.title_ru} |"
            )
    else:
        lines.append("| — | — | ни одной проверяемой ссылки |")

    lines.extend(["", "### Что затронуто новыми данными", ""])
    if result.changed:
        lines.extend(["| Шаг | Роль | Фаза | Затронутые ссылки |", "|---|---|---|---|"])
        for step in result.affected_steps:
            touched = ", ".join(f"`{citation}`" for citation in step.citations)
            lines.append(f"| {step.ordinal} | {step.role} | {step.phase} | {touched} |")
        if result.affected_commitments:
            lines.extend(["", "| Обязательство | Владелец | Затронутые ссылки |", "|---|---|---|"])
            for commitment in result.affected_commitments:
                touched = ", ".join(f"`{citation}`" for citation in commitment.citations)
                lines.append(
                    f"| {commitment.statement or '—'} | {commitment.owner or '—'} | {touched} |"
                )
        lines.extend(
            [
                "",
                "Хеш меняется и при косметической правке файла, поэтому вывод помечен "
                "«требует пересмотра», а не «вывод неверен». Прежний ответ не становится "
                "фактом: пересмотр — это новый шаг с новым основанием, и в журнал он не "
                "переписывается.",
            ]
        )
    else:
        lines.append("Ничего: все источники на месте, записанные хеши совпадают с файлами.")

    lines.extend(["", "### Чего пересмотр не показывает", ""])
    lines.append(
        "- Данные заново не скачиваются: сравнение хешей отвечает на вопрос «устарел ли "
        "вывод», а не «обновите данные»."
    )
    lines.append(
        "- Совпадение хешей не доказывает, что вывод верен: оно говорит только, что файл "
        "не менялся с момента получения."
    )
    lines.append(
        "- Изменение байтов не доказывает ошибку: правка формата меняет хеш так же, как "
        "правка числа."
    )
    if result.unrecorded:
        names = ", ".join(f"`{source.source_id}`" for source in result.unrecorded)
        lines.append(
            f"- Без записанного хеша остались источники: {names}. Сравнивать нечем — это "
            "пробел в происхождении данных, а не «не изменилось»."
        )
    if result.outside:
        lines.append(
            f"- Статьи корпуса ({len(result.outside)}) не пересматриваются: у них нет "
            "версий с хешем."
        )
    lines.append(
        "- Журнал и состояние сессии не меняются: история остаётся append-only, а пересмотр "
        "— отдельный результат."
    )
    return "\n".join(lines)
