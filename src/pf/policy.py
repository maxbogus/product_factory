"""Политика расширения: когда добор оснований прекращается.

Расширение — не бесплатное действие. Сетевая ошибка, пустой результат,
дубль и исчерпанный бюджет означают одно: новый раунд не даст знания.
Без явного правила агент ходит по кругу — в прогоне `demo-v5` шесть
отказов подряд выглядели как «проверка работает», хотя часть из них
была одним и тем же источником.

Модуль ничего не знает про модель и сеть: он превращает код ответа в
исход попытки и исход — в решение «повторить / закрыть узел / больше не
ходить в этот класс источников».

Разделение важное: **исход** — это факт (что произошло), **решение** —
политика (что из этого следует). Смешивать нельзя: иначе «404» начнёт
означать «нет данных по теме», а это разные утверждения.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum

#: Сколько раз пробуем временный отказ. Две попытки — компромисс между
#: «сеть моргнула» и «источник лежит»: третья не добавляет информации.
MAX_TRANSIENT_ATTEMPTS = 2

#: Сколько пустых раундов подряд закрывают узел. Один пустой результат —
#: случайность, два — закономерность.
NO_PROGRESS_LIMIT = 2

_SPACES = re.compile(r"\s+")


class AttemptOutcome(StrEnum):
    """Исход одной попытки добрать основание."""

    #: Транспорт в порядке, содержание ещё не оценено.
    OK = "ok"
    #: Появилось новое основание или новый абзац.
    NEW = "new"
    #: Всё найденное уже известно (по ссылке или по тексту).
    DUPLICATE = "duplicate"
    #: Запрос удался, релевантного нет.
    EMPTY = "empty"
    #: Временный отказ: таймаут, сеть, 429, 5xx.
    TRANSIENT = "transient"
    #: Постоянный отказ: 4xx, robots.txt, битый ответ.
    PERMANENT = "permanent"
    #: Класс источника исчерпан: попытки кончились.
    UNAVAILABLE = "unavailable"
    #: Исчерпан бюджет расширения.
    BUDGET = "budget"

    @property
    def is_progress(self) -> bool:
        """Даёт ли исход новое знание."""
        return self is AttemptOutcome.NEW

    @property
    def is_stop(self) -> bool:
        """Останавливает ли исход работу с классом источника."""
        return self in {AttemptOutcome.BUDGET, AttemptOutcome.UNAVAILABLE}


def classify_http(status: int | None, *, robots: bool = False) -> AttemptOutcome:
    """Классифицировать ответ транспорта.

    Args:
        status: Код HTTP либо ``None``, если ответа не было (таймаут, сеть).
        robots: Запрет обхода, объявленный в robots.txt.

    Returns:
        Исход попытки на уровне транспорта.
    """
    if robots:
        return AttemptOutcome.PERMANENT
    if status is None:
        return AttemptOutcome.TRANSIENT
    if 200 <= status < 300:
        return AttemptOutcome.OK
    if status == 429 or 500 <= status < 600:
        return AttemptOutcome.TRANSIENT
    if 400 <= status < 500:
        return AttemptOutcome.PERMANENT
    return AttemptOutcome.TRANSIENT


def classify_code(code: str) -> AttemptOutcome:
    """Классифицировать машинный код разбора страницы.

    Код — не текст для человека: ``http_404`` и ``http_503`` требуют
    разных решений. Постоянный отказ не повторяют, временный — повторяют.

    Args:
        code: Код разбора (``ok``, ``robots``, ``http_404``, ``timeout``,
            ``network``, ``empty``).

    Returns:
        Исход попытки.
    """
    text = code.strip().lower()
    if text == "ok":
        return AttemptOutcome.OK
    if text in {"robots", "forbidden"}:
        return AttemptOutcome.PERMANENT
    if text == "empty":
        return AttemptOutcome.EMPTY
    if text.startswith("http_"):
        try:
            status = int(text[len("http_") :])
        except ValueError:
            return AttemptOutcome.TRANSIENT
        return classify_http(status)
    return AttemptOutcome.TRANSIENT


def classify_content(*, new: int, duplicate: int) -> AttemptOutcome:
    """Классифицировать найденное содержание.

    Args:
        new: Сколько новых абзацев или оснований найдено.
        duplicate: Сколько найденного оказалось уже известным.

    Returns:
        Исход попытки на уровне содержания.
    """
    if new > 0:
        return AttemptOutcome.NEW
    if duplicate > 0:
        return AttemptOutcome.DUPLICATE
    return AttemptOutcome.EMPTY


def content_key(text: str) -> str:
    """Ключ текста для поиска дублей.

    Разные адреса могут отдать один и тот же абзац. Сравнение по ссылке
    этого не видит, и «новый источник» создаёт иллюзию прогресса.

    Args:
        text: Текст абзаца.

    Returns:
        Короткий хеш нормализованного текста.
    """
    normalized = _SPACES.sub(" ", text.strip().lower().replace("ё", "е"))
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class SourceHealth:
    """Состояние класса источников: сколько попыток и чем кончились.

    Attributes:
        attempts: Всего попыток.
        transient_failures: Временных отказов подряд.
        unavailable: Класс исчерпан — больше не ходим.
        stopped: Работа остановлена бюджетом.
    """

    attempts: int = 0
    transient_failures: int = 0
    unavailable: bool = False
    stopped: bool = False

    def record(self, outcome: AttemptOutcome) -> None:
        """Учесть исход попытки.

        Args:
            outcome: Исход попытки.
        """
        self.attempts += 1
        if outcome is AttemptOutcome.PERMANENT:
            self.unavailable = True
        elif outcome is AttemptOutcome.TRANSIENT:
            self.transient_failures += 1
            if self.transient_failures >= MAX_TRANSIENT_ATTEMPTS:
                self.unavailable = True
        elif outcome is AttemptOutcome.UNAVAILABLE:
            self.unavailable = True
        elif outcome is AttemptOutcome.BUDGET:
            self.stopped = True
        elif outcome.is_progress:
            self.transient_failures = 0

    def should_retry(self) -> bool:
        """Есть ли смысл повторять попытку.

        Returns:
            Класс источника доступен и работа не остановлена.
        """
        return not self.unavailable and not self.stopped


@dataclass(slots=True)
class NodeProgress:
    """Состояние одного узла поиска (вопроса или пробела).

    Attributes:
        empty_hits: Пустых раундов подряд.
        closed_reason: Причина закрытия; пусто — узел открыт.
    """

    empty_hits: int = 0
    closed_reason: str = ""

    def record(self, outcome: AttemptOutcome) -> None:
        """Учесть исход попытки по узлу.

        Args:
            outcome: Исход попытки.
        """
        if self.closed_reason:
            return
        if outcome is AttemptOutcome.NEW:
            self.empty_hits = 0
            return
        if outcome is AttemptOutcome.UNAVAILABLE:
            self.closed_reason = "unavailable"
            return
        if outcome is AttemptOutcome.BUDGET:
            self.closed_reason = "budget"
            return
        if outcome in {AttemptOutcome.EMPTY, AttemptOutcome.DUPLICATE}:
            self.empty_hits += 1
            if self.empty_hits >= NO_PROGRESS_LIMIT:
                self.closed_reason = "no_progress"

    @property
    def is_closed(self) -> bool:
        """Закрыт ли узел."""
        return bool(self.closed_reason)

    def is_closed_with_reason(self, reason: str) -> bool:
        """Закрыт ли узел по конкретной причине.

        Args:
            reason: Причина закрытия.

        Returns:
            Узел закрыт именно по этой причине.
        """
        return self.closed_reason == reason
