"""Лексикон: словарь близости и очередь распознания незнакомых слов.

Проверка оснований не должна ни падать, ни притворяться на незнакомом
слове. Если слова нет в словаре и морфология его не знает, это не повод
молча снизить покрытие: слово записывается в очередь распознания. Потом
человек или модель предлагают, что это за понятие, и словарь
расширяется — уже осознанно, а не «на глаз».

Два файла с разными правилами записи:

- ``synonyms.yaml`` — **словарь**, коммитится в git. Переписывается
  целиком: это данные с версией, а не журнал.
- ``oov.jsonl`` — **очередь**, append-only и не коммитится. Как журнал
  шагов: строки только дописываются, повторный прогон увеличивает
  счётчик, а не перезаписывает историю.

Ключевое ограничение: записи со статусом ``proposed`` в проверке
оснований **не участвуют**. Подтверждает словарь человек (``approved``),
иначе подтверждение основания снова стало бы самооценкой модели.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from pf import synonyms as synonyms_module
from pf.config import settings
from pf.corpus import tokenize
from pf.domain import subject_groups
from pf.domain import version as domain_stamp
from pf.lexical import TokenStatus, normalize_homoglyphs, recognize

#: Имя файла словаря.
SYNONYMS_NAME = "synonyms.yaml"
#: Имя файла очереди распознания.
OOV_NAME = "oov.jsonl"
#: Имя файла provenance словаря.
MANIFEST_NAME = "manifest.jsonl"

#: Статусы записи словаря. В проверке участвует только ``approved``.
APPROVED = "approved"
PROPOSED = "proposed"

#: Статусы записи очереди.
PENDING = "pending"
JUNK = "junk"


def _now() -> str:
    """Метка времени в ISO-формате."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def lexicon_dir() -> Path:
    """Каталог лексикона.

    Returns:
        Путь из настроек; в тестах подменяется на временный.
    """
    return settings.lexicon_dir


def synonyms_path() -> Path:
    """Путь к файлу словаря синонимов."""
    return lexicon_dir() / SYNONYMS_NAME


def oov_path() -> Path:
    """Путь к файлу очереди распознания."""
    return lexicon_dir() / OOV_NAME


@dataclass(frozen=True, slots=True)
class LexiconEntry:
    """Запись словаря: одно понятие и его названия.

    Attributes:
        canonical: Каноническая лемма понятия.
        aliases: Остальные названия (леммы), включая латиницу.
        status: ``approved`` либо ``proposed``.
        source: Откуда взялась запись: ``manual`` или ``llm``.
        added: Дата добавления.
        model: Модель, предложившая запись, если ``source == llm``.
    """

    canonical: str
    aliases: tuple[str, ...] = ()
    status: str = APPROVED
    source: str = "manual"
    added: str = ""
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Представление для файла словаря.

        Returns:
            Словарь с непустыми полями.
        """
        payload: dict[str, Any] = {
            "canonical": self.canonical,
            "aliases": list(self.aliases),
            "status": self.status,
            "source": self.source,
        }
        if self.added:
            payload["added"] = self.added
        if self.model:
            payload["model"] = self.model
        return payload


def read_entries(path: Path | None = None) -> list[LexiconEntry]:
    """Прочитать записи словаря.

    Args:
        path: Путь к файлу. По умолчанию — из настроек.

    Returns:
        Список записей; пустой, если файла нет или он испорчен.
    """
    target = path or synonyms_path()
    if not target.is_file():
        return []
    try:
        payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        # Испорченный словарь не должен ронять прогон: работаем на
        # встроенном наборе, а проблему видно в `pf lexicon stats`.
        return []
    entries: list[LexiconEntry] = []
    for item in payload.get("terms") or []:
        if not isinstance(item, dict):
            continue
        canonical = str(item.get("canonical", "")).strip().lower()
        if not canonical:
            continue
        aliases = tuple(
            str(alias).strip().lower() for alias in item.get("aliases") or [] if str(alias).strip()
        )
        entries.append(
            LexiconEntry(
                canonical=canonical,
                aliases=aliases,
                status=str(item.get("status", APPROVED)),
                source=str(item.get("source", "manual")),
                added=str(item.get("added", "")),
                model=str(item.get("model", "")),
            )
        )
    return entries


def write_entries(entries: list[LexiconEntry], path: Path | None = None) -> Path:
    """Записать словарь целиком (атомарно).

    Словарь — версионируемые данные, поэтому переписывается целиком: в
    отличие от журнала, здесь важна актуальная версия, а не история.

    Args:
        entries: Записи словаря.
        path: Путь к файлу. По умолчанию — из настроек.

    Returns:
        Путь записанного файла.
    """
    target = path or synonyms_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "metadata": {
            "description": "Словарь лексической близости: группы синонимов и точные термины",
            "updated": _now(),
            "terms": len(entries),
        },
        "terms": [entry.to_dict() for entry in sorted(entries, key=lambda e: e.canonical)],
    }
    temporary = target.with_suffix(".yaml.tmp")
    temporary.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(target)
    # Словарь изменился — сбрасываем кэш канонизации: иначе правка
    # подхватится только после перезапуска процесса.
    _groups_for.cache_clear()
    return target


def _groups_from_entries(entries: list[LexiconEntry]) -> dict[str, str]:
    """Собрать отображение синонимов из записей словаря.

    Args:
        entries: Записи словаря.

    Returns:
        Словарь «лемма → каноническая лемма». Встроенный набор берётся
        за основу, файл его дополняет.

    Raises:
        ValueError: Слово попало в две группы — канонизация неоднозначна.
    """
    groups = list(synonyms_module.BUILTIN_GROUPS)
    # Написания субъектов — из домена проекта: «МТС» и «MTS» это один
    # субъект, и разбирать их как разные понятия значило бы подтверждать
    # чужое имя (правило — в `pf.domain`).
    groups.extend(subject_groups())
    for entry in entries:
        if entry.status != APPROVED:
            # Предложенная моделью запись в проверке не участвует: иначе
            # подтверждение основания снова стало бы самооценкой модели.
            continue
        groups.append((entry.canonical, *entry.aliases))
    return synonyms_module.build_map(tuple(groups))


@lru_cache(maxsize=8)
def _groups_for(path: str, stamp: float, domain_stamp: int) -> dict[str, str]:
    """Собрать отображение синонимов и закэшировать по версии файлов.

    Args:
        path: Путь к файлу словаря.
        stamp: Время изменения файла словаря — часть ключа кэша.
        domain_stamp: Версия домена: написания субъектов приходят оттуда,
            и правка домена должна сбрасывать этот кэш тоже.

    Returns:
        Словарь «лемма → каноническая лемма».
    """
    return _groups_from_entries(read_entries(Path(path)))


def groups() -> dict[str, str]:
    """Отображение синонимов с учётом файла словаря и домена.

    Кэш привязан к пути и времени изменения: правка словаря человеком
    подхватывается без перезапуска процесса.

    Returns:
        Словарь «лемма → каноническая лемма».
    """
    path = synonyms_path()
    stamp = path.stat().st_mtime_ns if path.is_file() else 0
    return _groups_for(str(path), stamp, domain_stamp())


def token_resolution(token: str) -> Any:
    """Распознать слово с учётом файла словаря.

    Args:
        token: Слово в исходном написании.

    Returns:
        Результат распознавания (``pf.lexical.TokenResolution``).
    """
    return recognize(
        token,
        groups(),
        max_distance=settings.typo_max_distance,
        min_length=settings.oov_min_length,
    )


@dataclass(frozen=True, slots=True)
class Coverage:
    """Лексическое покрытие текста.

    Attributes:
        total: Сколько значимых слов проверено.
        known: Слова из словаря.
        morph: Слова, распознанные морфологией.
        heuristic: Слова, распознанные эвристикой.
        oov: Нераспознанные слова.
        oov_tokens: Сами нераспознанные слова — что расширять.
    """

    total: int = 0
    known: int = 0
    morph: int = 0
    heuristic: int = 0
    oov: int = 0
    oov_tokens: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        """Доля распознанных слов.

        Returns:
            Отношение распознанных к общему числу; ноль для пустого текста.
        """
        if not self.total:
            return 0.0
        return (self.total - self.oov) / self.total

    def to_dict(self) -> dict[str, Any]:
        """Представление для журнала и отчёта.

        Returns:
            Словарь с числами и топом нераспознанных слов.
        """
        return {
            "total": self.total,
            "known": self.known,
            "morph": self.morph,
            "heuristic": self.heuristic,
            "oov": self.oov,
            "rate": round(self.rate, 4),
            "oov_tokens": self.oov_tokens[:10],
        }

    @property
    def summary(self) -> str:
        """Краткая сводка для человека."""
        return (
            f"слов: {self.total}, распознано: {self.total - self.oov} "
            f"({self.rate:.0%}); в очереди распознания: {self.oov}"
        )


def coverage_report(texts: list[str] | tuple[str, ...], *, limit: int = 10) -> Coverage:
    """Посчитать лексическое покрытие набора текстов.

    Args:
        texts: Тексты утверждений (не источников: распознавать нужно то,
            что пришло от модели).
        limit: Сколько нераспознанных слов вернуть.

    Returns:
        Покрытие с топом нераспознанных слов.
    """
    table = groups()
    counters = {status: 0 for status in TokenStatus}
    oov: dict[str, int] = {}
    total = 0
    for text in texts:
        for word in tokenize(text):
            total += 1
            resolution = recognize(word, table)
            counters[resolution.status] += 1
            if resolution.status is TokenStatus.OOV:
                oov[resolution.token] = oov.get(resolution.token, 0) + 1
    ordered = sorted(oov, key=lambda token: (-oov[token], token))
    return Coverage(
        total=total,
        known=counters[TokenStatus.KNOWN],
        morph=counters[TokenStatus.RESOLVED_MORPH],
        heuristic=counters[TokenStatus.RESOLVED_HEURISTIC],
        oov=counters[TokenStatus.OOV],
        oov_tokens=ordered[:limit],
    )


@dataclass(slots=True)
class OovRecord:
    """Запись очереди распознания в свёрнутом виде.

    Attributes:
        token: Слово как оно встретилось.
        normalized: Слово после нормализации раскладки.
        reason: Почему не распознали (последняя причина).
        status: ``pending``, ``proposed``, ``approved`` или ``junk``.
        count: Сколько раз встретилось.
        first_seen: Когда встретилось впервые.
        last_seen: Когда встретилось в последний раз.
        sessions: Сессии, в которых встретилось (не больше пяти).
        contexts: Примеры контекста (не больше трёх, обрезаны).
        suggestions: Предложенные канонические леммы.
    """

    token: str
    normalized: str
    reason: str = ""
    status: str = PENDING
    count: int = 0
    first_seen: str = ""
    last_seen: str = ""
    sessions: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Представление для CLI и отчёта."""
        return {
            "token": self.token,
            "normalized": self.normalized,
            "reason": self.reason,
            "status": self.status,
            "count": self.count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "sessions": self.sessions,
            "suggestions": self.suggestions,
        }


def _append_event(payload: dict[str, Any], path: Path | None = None) -> None:
    """Дописать событие в очередь распознания.

    Очередь append-only: строки не переписываются, потому что по ним
    видно, когда слово встретилось и что с ним сделали.

    Args:
        payload: Событие без метки времени.
        path: Путь к файлу. По умолчанию — из настроек.
    """
    target = path or oov_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": _now(), **payload}
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _shorten(text: str, limit: int = 200) -> str:
    """Обрезать контекст, чтобы очередь не распухала.

    Args:
        text: Текст контекста.
        limit: Максимальная длина.

    Returns:
        Обрезанный текст в одну строку.
    """
    flat = " ".join(text.split())
    return flat[:limit]


def record_oov(
    token: str,
    *,
    reason: str = "",
    role: str = "",
    session: str = "",
    context: str = "",
    suggestions: tuple[str, ...] | list[str] = (),
    path: Path | None = None,
) -> None:
    """Записать нераспознанное слово в очередь.

    Args:
        token: Слово как оно встретилось.
        reason: Почему не распознали.
        role: Роль, в тексте которой встретилось слово.
        session: Идентификатор сессии.
        context: Короткий контекст для человека, решающего судьбу слова.
        suggestions: Слова словаря, похожие на это.
        path: Путь к файлу очереди. По умолчанию — из настроек.
    """
    cleaned = token.strip().lower()
    if not cleaned:
        return
    _append_event(
        {
            "event": "seen",
            "token": cleaned,
            "normalized": normalize_homoglyphs(cleaned),
            "reason": reason,
            "role": role,
            "session": session,
            "context": _shorten(context) if context else "",
            "suggestions": list(suggestions),
        },
        path,
    )


def read_oov(path: Path | None = None) -> list[OovRecord]:
    """Прочитать и свернуть очередь распознания.

    Args:
        path: Путь к файлу очереди. По умолчанию — из настроек.

    Returns:
        Записи в порядке убывания частоты. Битая строка пропускается:
        очередь дописывается, обрыв на границе — ожидаемое состояние.
    """
    target = path or oov_path()
    if not target.is_file():
        return []

    records: dict[str, OovRecord] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        token = str(event.get("token", "")).strip().lower()
        if not token:
            continue
        record = records.setdefault(
            token,
            OovRecord(token=token, normalized=str(event.get("normalized") or token)),
        )
        stamp = str(event.get("ts", ""))
        kind = str(event.get("event", "seen"))
        if kind == "seen":
            record.count += 1
            record.reason = str(event.get("reason", "")) or record.reason
            record.first_seen = record.first_seen or stamp
            record.last_seen = stamp or record.last_seen
            session = str(event.get("session", ""))
            if session and session not in record.sessions and len(record.sessions) < 5:
                record.sessions.append(session)
            context = str(event.get("context", ""))
            if (
                context
                and context not in record.contexts
                and len(record.contexts) < settings.oov_max_contexts
            ):
                record.contexts.append(context)
            for suggestion in event.get("suggestions") or []:
                text = str(suggestion)
                if text and text not in record.suggestions:
                    record.suggestions.append(text)
        elif kind == "proposed":
            record.status = PROPOSED
            canonical = str(event.get("canonical", "")).strip().lower()
            if canonical and canonical not in record.suggestions:
                record.suggestions.append(canonical)
        elif kind == "approved":
            record.status = APPROVED
        elif kind == "junk":
            record.status = JUNK

    return sorted(records.values(), key=lambda item: (-item.count, item.token))


def pending_oov(*, min_count: int = 1, path: Path | None = None) -> list[OovRecord]:
    """Слова, которые ещё ждут распознания.

    Args:
        min_count: Минимальная частота: редкие слова не тратят бюджет
            модели, пока не встретятся несколько раз.
        path: Путь к файлу очереди.

    Returns:
        Записи со статусом ``pending``, частые первыми.
    """
    return [
        record
        for record in read_oov(path)
        if record.status == PENDING and record.count >= min_count
    ]


def mark_junk(token: str, *, path: Path | None = None) -> None:
    """Отметить слово как мусор (не понятие).

    Args:
        token: Слово.
        path: Путь к файлу очереди.
    """
    _append_event({"event": "junk", "token": token.strip().lower()}, path)


def propose(
    token: str,
    canonical: str,
    *,
    source: str = "llm",
    model: str = "",
    path: Path | None = None,
    lexicon_path: Path | None = None,
) -> bool:
    """Записать предложение в словарь со статусом ``proposed``.

    Такая запись видна человеку, но в проверке не участвует: подтверждает
    словарь человек, а не модель.

    Args:
        token: Незнакомое слово.
        canonical: Лемма понятия, к которому оно относится.
        source: Откуда предложение: ``llm`` или ``manual``.
        model: Модель, если предложение от неё.
        path: Путь к файлу очереди.
        lexicon_path: Путь к файлу словаря.

    Returns:
        True, если словарь изменился.
    """
    cleaned = token.strip().lower()
    target = canonical.strip().lower()
    if not cleaned or not target:
        return False

    entries = read_entries(lexicon_path)
    changed = False
    for index, entry in enumerate(entries):
        if cleaned in entry.aliases:
            break
        if entry.canonical == target:
            entries[index] = LexiconEntry(
                canonical=entry.canonical,
                aliases=(*entry.aliases, cleaned),
                status=entry.status,
                source=entry.source,
                added=entry.added,
                model=entry.model or model,
            )
            changed = True
            break
    else:
        entries.append(
            LexiconEntry(
                canonical=target,
                aliases=(cleaned,) if cleaned != target else (),
                status=PROPOSED,
                source=source,
                added=_now()[:10],
                model=model,
            )
        )
        changed = True

    if changed:
        write_entries(entries, lexicon_path)
    _append_event(
        {"event": "proposed", "token": cleaned, "canonical": target, "source": source},
        path,
    )
    return changed


def approve(
    token: str,
    canonical: str = "",
    *,
    path: Path | None = None,
    lexicon_path: Path | None = None,
) -> bool:
    """Подтвердить запись словаря.

    Args:
        token: Слово.
        canonical: Каноническая лемма; если пусто — берётся из записи.
        path: Путь к файлу очереди.
        lexicon_path: Путь к файлу словаря.

    Returns:
        True, если запись найдена и подтверждена.
    """
    cleaned = token.strip().lower()
    target = canonical.strip().lower()
    entries = read_entries(lexicon_path)
    changed = False
    for index, entry in enumerate(entries):
        matches = (
            cleaned in entry.aliases
            or entry.canonical == cleaned
            or bool(target and entry.canonical == target)
        )
        if not matches:
            continue
        entries[index] = LexiconEntry(
            canonical=entry.canonical,
            aliases=entry.aliases,
            status=APPROVED,
            source=entry.source,
            added=entry.added,
            model=entry.model,
        )
        changed = True
        break
    if not changed:
        # Подтверждение без предварительного предложения — обычный способ
        # добавить слово вручную: человек знает понятие, модель не нужна.
        canonical_lemma = target or cleaned
        if not canonical_lemma:
            return False
        entries.append(
            LexiconEntry(
                canonical=canonical_lemma,
                aliases=(cleaned,) if cleaned != canonical_lemma else (),
                status=APPROVED,
                source="manual",
                added=_now()[:10],
            )
        )
        changed = True
    if changed:
        write_entries(entries, lexicon_path)
        _append_event({"event": "approved", "token": cleaned}, path)
    return changed


#: Сколько известных лемм показывать модели. Длинный список вытесняет
#: само задание, и локальная модель отвечает не по форме.
MAX_KNOWN_IN_PROMPT = 150

#: Действия распознания. Строки нужны для журнала и CLI.
ACTION_SYNONYM = "synonym"
ACTION_TERM = "term"
ACTION_JUNK = "junk"
ACTION_UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class EnrichProposal:
    """Предложение по одному незнакомому слову.

    Attributes:
        token: Незнакомое слово.
        action: Что предложено: синоним, термин, мусор или «не понял».
        canonical: Каноническая лемма для синонима и термина.
        answer: Сырой ответ модели — для разбора спорных случаев.
    """

    token: str
    action: str
    canonical: str = ""
    answer: str = ""


@dataclass(slots=True)
class EnrichResult:
    """Итог распознания пачки слов.

    Attributes:
        proposals: Предложения по каждому слову.
        tokens: Сколько токенов израсходовано.
    """

    proposals: list[EnrichProposal] = field(default_factory=list)
    tokens: int = 0

    @property
    def summary(self) -> str:
        """Краткая сводка для человека."""
        counts: dict[str, int] = {}
        for proposal in self.proposals:
            counts[proposal.action] = counts.get(proposal.action, 0) + 1
        parts = ", ".join(f"{action}: {count}" for action, count in sorted(counts.items()))
        return f"разобрано слов: {len(self.proposals)} ({parts}); токенов: {self.tokens}"


def build_enrich_prompt(token: str, known: list[str]) -> tuple[str, str]:
    """Собрать запрос на распознание одного слова.

    Args:
        token: Незнакомое слово.
        known: Известные канонические леммы.

    Returns:
        Пара ``(системная инструкция, текст задачи)``.
    """
    system = (
        "Ты пополняешь словарь понятий для проверки оснований. "
        "Тебе дают незнакомое слово и список уже известных понятий. "
        "Отвечай ОДНОЙ СТРОКОЙ и больше ничем."
    )
    listing = ", ".join(known[:MAX_KNOWN_IN_PROMPT]) or "список пуст"
    user = (
        f"Известные понятия: {listing}\n\n"
        f"Незнакомое слово: {token}\n\n"
        "Что это? Ответь одной строкой:\n"
        "- лемма из списка, если слово — её синоним;\n"
        "- слово «термин», если это отдельное понятие, которого нет в списке;\n"
        "- слово «мусор», если это опечатка, междометие или не понятие."
    )
    return system, user


def parse_enrich_answer(answer: str, known: set[str]) -> tuple[str, str]:
    """Разобрать ответ модели на распознание слова.

    Ответ модели свободный, поэтому разбор терпимый: ищем ключевые слова
    и известные леммы, а не надеемся на точный формат.

    Args:
        answer: Сырой ответ модели.
        known: Известные канонические леммы.

    Returns:
        Пара ``(действие, каноническая лемма)``.
    """
    text = answer.strip().lower()
    if not text:
        return ACTION_UNRESOLVED, ""
    if "мусор" in text or "junk" in text:
        return ACTION_JUNK, ""
    if "термин" in text or "term" in text:
        return ACTION_TERM, ""
    for lemma in sorted(known, key=len, reverse=True):
        if lemma and lemma in text:
            return ACTION_SYNONYM, lemma
    return ACTION_UNRESOLVED, ""


def enrich(
    *,
    limit: int = 50,
    min_count: int = 1,
    client: Any = None,
    dry_run: bool = True,
    auto_approve: bool = False,
    path: Path | None = None,
    lexicon_path: Path | None = None,
) -> EnrichResult:
    """Распознать накопленные незнакомые слова и пополнить словарь.

    Модель предлагает понятие, но словарь подтверждает человек: записи
    создаются со статусом ``proposed`` и в проверке не участвуют. Флаг
    ``auto_approve`` существует для демонстрации и требует явного
    согласия того, кто запускает команду.

    Args:
        limit: Сколько слов разобрать за один вызов.
        min_count: Минимальная частота слова.
        client: Клиент модели. По умолчанию — из настроек.
        dry_run: Только показать предложения, ничего не записывая.
        auto_approve: Сразу подтверждать записи (по умолчанию нет).
        path: Путь к файлу очереди.
        lexicon_path: Путь к файлу словаря.

    Returns:
        Предложения и расход токенов.
    """
    if client is None:
        from pf.llm import build_client

        client = build_client()
    inner = getattr(client, "inner", client)
    config = getattr(inner, "config", None)
    model_name = getattr(config, "effective_model", "") if config is not None else ""

    records = pending_oov(min_count=min_count, path=path)[:limit]
    # Список понятий — и встроенный словарь, и файл: иначе модель не
    # узнает базовые группы и всё сведётся к «термин».
    known = sorted(
        set(synonyms_module.build_map().values())
        | {entry.canonical for entry in read_entries(lexicon_path)}
    )
    result = EnrichResult()

    for record in records:
        system, user = build_enrich_prompt(record.token, known)
        try:
            response = client.complete(system, user)
        except Exception as exc:  # noqa: BLE001 — сбой модели не должен ломать разбор
            result.proposals.append(
                EnrichProposal(token=record.token, action=ACTION_UNRESOLVED, answer=str(exc))
            )
            continue

        result.tokens += response.usage.total_tokens
        action, canonical = parse_enrich_answer(response.content, set(known))
        result.proposals.append(
            EnrichProposal(
                token=record.token,
                action=action,
                canonical=canonical or record.normalized,
                answer=response.content.strip()[:200],
            )
        )
        if dry_run:
            continue
        if action == ACTION_SYNONYM and canonical:
            propose(
                record.token,
                canonical,
                source="llm",
                model=model_name,
                path=path,
                lexicon_path=lexicon_path,
            )
            if auto_approve:
                approve(record.token, canonical, path=path, lexicon_path=lexicon_path)
        elif action == ACTION_TERM:
            propose(
                record.token,
                record.normalized,
                source="llm",
                model=model_name,
                path=path,
                lexicon_path=lexicon_path,
            )
            if auto_approve:
                approve(record.token, record.normalized, path=path, lexicon_path=lexicon_path)
        elif action == ACTION_JUNK:
            mark_junk(record.token, path=path)
        else:
            _append_event(
                {"event": "unresolved", "token": record.token, "answer": response.content[:200]},
                path,
            )

    return result


def stats(*, path: Path | None = None, lexicon_path: Path | None = None) -> dict[str, Any]:
    """Сводка по лексикону и очереди распознания.

    Args:
        path: Путь к файлу очереди.
        lexicon_path: Путь к файлу словаря.

    Returns:
        Числа для `pf lexicon stats` и отчёта.
    """
    entries = read_entries(lexicon_path)
    records = read_oov(path)
    return {
        "entries": len(entries),
        "approved": sum(1 for entry in entries if entry.status == APPROVED),
        "proposed": sum(1 for entry in entries if entry.status == PROPOSED),
        "oov_total": len(records),
        "oov_pending": sum(1 for record in records if record.status == PENDING),
        "top": [record.to_dict() for record in records[:10]],
    }
