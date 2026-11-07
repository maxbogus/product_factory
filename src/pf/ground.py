"""Проверка оснований: цитата должна дословно существовать в источнике.

Главный риск агента, который работает с языковой моделью, — правдоподобная
выдумка. Здесь это закрывается проверкой, а не просьбой к модели «не выдумывай».

Правило простое и проверяемое:

1. цитата из ответа модели должна **дословно** встречаться в исходном тексте
   статьи (сравнение по нормализованному тексту);
2. если цитата не найдена — основание отбрасывается, и поле получает статус
   «нет данных» вместо правдоподобного значения.

Смысл: модель может придумать *связный* текст, но не может придумать
*цитату из конкретной статьи*, не совпадающую с её содержимым. Поэтому
проверка идёт по цитате, а не по смыслу ответа.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum

from pf.config import settings
from pf.lexical import (
    Overlap,
    language,
    numbers_in,
    numbers_match,
    shares_words,
)
from pf.lexical import (
    overlap as lexical_overlap,
)

#: Нормализация текста перед сравнением: приводим к NFC, схлопываем пробелы
#: и убираем мягкие переносы. Регистр НЕ меняем — регистр может быть частью
#: смысла термина, но для поиска цитаты он мешает, поэтому сравниваем
#: в нижнем регистре.
_WHITESPACE = re.compile(r"\s+")
_HYPHEN = re.compile(r"[-‐‑‒–—―]")


class SupportStatus(StrEnum):
    """Статус подтверждения основания.

    Три состояния принципиально различаются: «нашли», «нашли, но сомневаемся»
    и «не нашли». Без этого «нет данных» невозможно отличить от нуля.
    """

    #: Цитата найдена в источнике дословно.
    CONFIRMED = "confirmed"
    #: Цитата близка к источнику, но не совпала точно — нужен просмотр.
    NEEDS_REVIEW = "needs_review"
    #: Цитата не найдена вовсе — основание отброшено.
    NO_DATA = "no_data"

    @property
    def title_ru(self) -> str:
        """Человекочитаемое название статуса."""
        return {
            SupportStatus.CONFIRMED: "подтверждено",
            SupportStatus.NEEDS_REVIEW: "требует проверки",
            SupportStatus.NO_DATA: "нет данных",
        }[self]


@dataclass(frozen=True, slots=True)
class Support:
    """Основание утверждения: цитата и её происхождение.

    Attributes:
        quote: Цитата из ответа модели.
        source_id: Идентификатор источника (``раздел/Заголовок``).
        paragraph_index: Номер абзаца в источнике, если найден.
        status: Статус подтверждения.
        detail: Пояснение для человека — что именно не сошлось.
        oov: Нераспознанные слова утверждения — их место в очереди
            распознания, а не в подозрениях на выдумку.
    """

    quote: str
    source_id: str
    status: SupportStatus
    paragraph_index: int | None = None
    detail: str = ""
    oov: tuple[str, ...] = ()

    @property
    def is_usable(self) -> bool:
        """Можно ли опереться на это основание."""
        return self.status is SupportStatus.CONFIRMED

    @property
    def citation(self) -> str:
        """Человекочитаемая ссылка на источник.

        Якорь один: если в идентификаторе уже есть ``#rowN`` или
        ``#pN``, второй не добавляется. Иначе ссылка выходила бы
        «project/файл.csv#row3#p3» — такой якорь не открывается.
        """
        if self.paragraph_index is None:
            return self.source_id
        if "#" in self.source_id:
            return self.source_id
        return f"{self.source_id}#p{self.paragraph_index}"


def mark_external(support: Support) -> Support:
    """Ограничить статус основания внешним источником.

    Внешний источник нельзя подтвердить: неизвестно, что за страницу
    отдала выдача. Дословность цитаты при этом проверяется, поэтому
    «нашли дословно» превращается в «требует проверки», а не в отказ.

    Args:
        support: Результат обычной проверки цитаты.

    Returns:
        То же основание со статусом не выше ``needs_review``.
    """
    if support.status is not SupportStatus.CONFIRMED:
        return support
    return replace(
        support,
        status=SupportStatus.NEEDS_REVIEW,
        detail="внешний источник: цитата найдена дословно, требует проверки",
    )


#: Насколько вердикт «хуже» для основания. Худший вердикт побеждает: если
#: на одну строку источника сослались и подтверждённой цитатой, и выдумкой,
#: основание считается отклонённым — иначе повтор верной цитаты прятал бы
#: рядом стоящую выдумку.
_SEVERITY: dict[SupportStatus, int] = {
    SupportStatus.CONFIRMED: 0,
    SupportStatus.NEEDS_REVIEW: 1,
    SupportStatus.NO_DATA: 2,
}

#: Имена групп вердиктов — так они попадают в журнал шага.
GROUND_BUCKETS: tuple[str, ...] = ("confirmed", "rejected", "needs_review")


def citation_key(source_id: str) -> str:
    """Ключ основания: путь и первый якорь, без повторов.

    Ссылка приходит как ``приложение/файл.csv#row3#p3`` — якорь может
    продублироваться. Для счёта оснований это одна строка таблицы, а не
    две, поэтому ключ у них общий. Без общего ключа отчёт печатал
    «уникальных оснований 2» и «проверено 8» в одном документе.

    Args:
        source_id: Ссылка на источник из ответа модели.

    Returns:
        Нормализованный ключ основания.
    """
    path, _, fragment = source_id.partition("#")
    parts = [part for part in fragment.split("#") if part]
    return f"{path}#{parts[0]}" if parts else path


@dataclass(frozen=True, slots=True)
class GroundTally:
    """Счёт оснований по уникальным ключам.

    Attributes:
        checked: Сколько уникальных оснований проверялось.
        confirmed: Сколько из них подтверждено.
        rejected: Сколько отклонено.
        needs_review: Сколько ждёт проверки человеком.
    """

    checked: int = 0
    confirmed: int = 0
    rejected: int = 0
    needs_review: int = 0


def _worst_by_key(verdicts: Iterable[tuple[str, SupportStatus]]) -> dict[str, SupportStatus]:
    """Худший вердикт по каждому ключу основания.

    Args:
        verdicts: Пары «ключ основания — вердикт».

    Returns:
        Словарь «ключ — худший вердикт» в порядке появления ключей.
    """
    worst: dict[str, SupportStatus] = {}
    for key, status in verdicts:
        current = worst.get(key)
        if current is None or _SEVERITY[status] > _SEVERITY[current]:
            worst[key] = status
    return worst


def ground_buckets(verdicts: Iterable[tuple[str, SupportStatus]]) -> dict[str, list[str]]:
    """Ключи оснований, разложенные по вердиктам.

    Нужно, чтобы отчёт мог объединить шаги: у стратегии и фиксации решения
    одни и те же строки источника, и складывать их счётчики нельзя.

    Args:
        verdicts: Пары «ключ основания — вердикт».

    Returns:
        Словарь ``confirmed`` / ``rejected`` / ``needs_review`` со ключами.
    """
    names = {
        SupportStatus.CONFIRMED: "confirmed",
        SupportStatus.NO_DATA: "rejected",
        SupportStatus.NEEDS_REVIEW: "needs_review",
    }
    buckets: dict[str, list[str]] = {name: [] for name in GROUND_BUCKETS}
    for key, status in _worst_by_key(verdicts).items():
        buckets[names[status]].append(key)
    return buckets


def tally_grounds(verdicts: Iterable[tuple[str, SupportStatus]]) -> GroundTally:
    """Посчитать основания по уникальным ключам.

    Повторная ссылка на ту же строку источника — одно основание. Именно
    этого не хватало: в прогоне ``demo-v9`` стратегия и решение объявляли
    по четыре основания из двух строк, и документ называл их и «8 из 8»,
    и «уникальных 2» одновременно.

    Args:
        verdicts: Пары «ключ основания — вердикт».

    Returns:
        Счёт по уникальным ключам, худший вердикт побеждает.
    """
    buckets = ground_buckets(verdicts)
    return GroundTally(
        checked=sum(len(keys) for keys in buckets.values()),
        confirmed=len(buckets["confirmed"]),
        rejected=len(buckets["rejected"]),
        needs_review=len(buckets["needs_review"]),
    )


def conclusion_status(
    checked: int,
    confirmed: int,
    rejected: int,
    needs_review: int = 0,
) -> str:
    """Статус оснований вывода — единственное место с этим правилом.

    Три состояния различаются принципиально: «подкреплено» — хотя бы одно
    основание подтверждено и ни одно не отклонено; «подкреплено частично» —
    часть отклонена или ждёт проверки; «неподкреплено» — ни одного
    подтверждённого. Молчать нельзя: вывод без основания выглядит в
    документе так же, как вывод с основанием.

    Правило было записано дважды — в рабочем процессе и в отчёте, — и
    версии разошлись: отчёт не знал про «ждёт проверки» и называл такой
    вывод подкреплённым. Теперь правило одно.

    Args:
        checked: Сколько было проверено.
        confirmed: Сколько подтверждено.
        rejected: Сколько отклонено.
        needs_review: Сколько ждёт проверки человеком.

    Returns:
        Статус оснований вывода.
    """
    if checked <= 0:
        return "не проверялось"
    if confirmed <= 0 and needs_review > 0:
        return "ждёт проверки"
    if confirmed <= 0:
        return "неподкреплено"
    if rejected > 0 or needs_review > 0:
        return "подкреплено частично"
    return "подкреплено"


@dataclass(frozen=True, slots=True)
class GroundEntry:
    """Одно основание в отчёте о качестве заземления.

    Attributes:
        citation: Ссылка на источник.
        status: Статус подтверждения.
        kind: Что это — факт или метрика.
    """

    citation: str
    status: str
    kind: str = "факт"


@dataclass(frozen=True, slots=True)
class GroundReport:
    """Отчёт о качестве оснований: какая доля подтверждена кодом.

    Самооценка «оснований достаточно» ничего не значит. Значима
    доля подтверждённых оснований — и отдельно по фактам и по
    метрикам: у них разный способ проверки.

    Метрики намеренно разделены: «факты подтверждены на 90 %» и
    «метрики воспроизводятся на 60 %» говорят разные вещи, а одно
    среднее число скрывает разницу.
    """

    entries: list[GroundEntry] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Сколько всего оснований."""
        return len(self.entries)

    @property
    def confirmed(self) -> int:
        """Сколько оснований подтверждено."""
        return sum(1 for entry in self.entries if entry.status == "confirmed")

    @property
    def grounding_rate(self) -> float:
        """Доля подтверждённых оснований от всех.

        Returns:
            Доля от 0 до 1; ноль для пустого отчёта.
        """
        if not self.entries:
            return 0.0
        return self.confirmed / self.total

    def _rate(self, kind: str) -> float:
        """Доля подтверждённых оснований одного вида.

        Args:
            kind: Вид основания.

        Returns:
            Доля от 0 до 1; ноль, если таких оснований нет.
        """
        selected = [entry for entry in self.entries if entry.kind == kind]
        if not selected:
            return 0.0
        confirmed = sum(1 for entry in selected if entry.status == "confirmed")
        return confirmed / len(selected)

    @property
    def fact_grounding_rate(self) -> float:
        """Доля подтверждённых фактов."""
        return self._rate("факт")

    @property
    def metric_grounding_rate(self) -> float:
        """Доля подтверждённых метрик."""
        return self._rate("метрика")

    @property
    def summary(self) -> str:
        """Краткая сводка для вывода.

        Returns:
            Текст сводки.
        """
        return (
            f"оснований: {self.total}, подтверждено: {self.confirmed} "
            f"({self.grounding_rate:.0%}); факты: {self.fact_grounding_rate:.0%}, "
            f"метрики: {self.metric_grounding_rate:.0%}"
        )


def normalize(text: str) -> str:
    """Нормализовать текст для сравнения.

    Args:
        text: Исходный текст.

    Returns:
        Нормализованный текст: нижний регистр, единые пробелы, без мягких
        переносов и тире.
    """
    normalized = unicodedata.normalize("NFC", text).lower()
    normalized = _HYPHEN.sub("-", normalized)
    normalized = _WHITESPACE.sub(" ", normalized)
    return normalized.strip()


def find_quote(source_text: str, quote: str) -> tuple[bool, int | None]:
    """Найти цитату в исходном тексте.

    Args:
        source_text: Исходный текст источника.
        quote: Искомая цитата.

    Returns:
        Пара ``(найдена, номер_абзаца)``. Номер абзаца — ``None``, если
        цитата найдена, но не внутри абзаца (например, в заголовке).
    """
    normalized_quote = normalize(quote)
    if not normalized_quote:
        return False, None

    paragraphs = source_text.split("\n")
    for index, paragraph in enumerate(paragraphs):
        if normalized_quote in normalize(paragraph):
            return True, index

    if normalized_quote in normalize(source_text):
        return True, None
    return False, None


def split_source_id(source_id: str) -> tuple[str, int | None]:
    """Разделить идентификатор источника на путь и номер строки.

    Модель часто склеивает их в одну строку. Якоря бывают двух видов:

    - ``#p126`` — номер абзаца статьи;
    - ``#row3`` — номер строки таблицы (так ссылаются метрики).

    Разбор делается здесь, чтобы остальной код работал с чистыми
    полями. Оба вида якоря дают номер строки: у абзацев он тоже
    нужен — по нему ищется фрагмент для проверки цитаты.

    Args:
        source_id: Идентификатор как прислала модель.

    Returns:
        Пара ``(source_id, номер строки)``.
    """
    text = source_id.strip()
    for anchor in ("#p", "#row"):
        if anchor in text:
            path, _, suffix = text.rpartition(anchor)
            if path and suffix.isdigit():
                return path, int(suffix)
    return text, None


def _groups() -> dict[str, str]:
    """Отображение синонимов из словаря лексикона.

    Импорт ленивый: ``pf.lexicon`` читает файлы, а ``pf.ground`` должен
    импортироваться быстро и без побочных эффектов.

    Returns:
        Словарь «лемма → каноническая лемма».
    """
    from pf import lexicon

    return lexicon.groups()


def _tolerances(
    abs_tolerance: float | None,
    rel_tolerance: float | None,
) -> tuple[float, float]:
    """Допуски для сравнения чисел.

    Args:
        abs_tolerance: Абсолютный допуск либо ``None`` — взять из настроек.
        rel_tolerance: Относительный допуск либо ``None``.

    Returns:
        Пара ``(абсолютный, относительный)``.
    """
    absolute = settings.metric_abs_tolerance if abs_tolerance is None else abs_tolerance
    relative = settings.metric_rel_tolerance if rel_tolerance is None else rel_tolerance
    return absolute, relative


def token_overlap(claim: str, paragraph: str) -> float:
    """Доля значимых слов утверждения, покрытых абзацем.

    Сравнение идёт по леммам и с учётом словаря синонимов: «абоненты» в
    источнике покрывают «клиентов» в утверждении. Это не ослабляет
    проверку: точные термины и числа сверяются отдельно и без синонимов.

    Args:
        claim: Текст утверждения.
        paragraph: Текст абзаца-источника.

    Returns:
        Доля от 0.0 до 1.0.
    """
    result = lexical_overlap(claim, paragraph, groups=_groups())
    return result.coverage


def _overlap_reason(result: Overlap, min_overlap: float) -> str:
    """Назвать причину, по которой утверждение не покрыто абзацем.

    Глухое «покрытие 0 %» читается как выдумка, хотя причина может быть
    другой: подменён термин, разошлось число, появилось отрицание.

    Args:
        result: Результат сравнения.
        min_overlap: Порог покрытия.

    Returns:
        Текст причины для журнала.
    """
    if result.missing_terms:
        return f"термин «{result.missing_terms[0]}» не найден в источнике"
    if not result.numbers_ok:
        return result.numbers_detail
    if not result.negation_ok:
        return "в утверждении появилось отрицание, которого нет в источнике"
    return (
        f"слова утверждения не подтверждаются абзацем "
        f"(покрытие {result.coverage:.0%} при пороге {min_overlap:.0%})"
    )


def verify_claim(
    source_text: str,
    claim_text: str,
    source_id: str,
    paragraph_index: int | None = None,
    quote: str = "",
    *,
    min_overlap: float | None = None,
    abs_tolerance: float | None = None,
    rel_tolerance: float | None = None,
) -> Support:
    """Подтвердить утверждение по источнику.

    Работают два пути:

    1. **Цитата.** Если модель прислала цитату, она проверяется дословно,
       а при расхождении — с допуском на округление чисел.
    2. **Покрытие абзацем.** Если цитаты нет, проверяется, что утверждение
       собрано из слов указанного абзаца: по леммам и синонимам, но с
       обязательным совпадением точных терминов, чисел и отрицания.

    Args:
        source_text: Полный текст источника.
        claim_text: Текст утверждения.
        source_id: Идентификатор источника.
        paragraph_index: Номер абзаца, если указан.
        quote: Дословная цитата, если прислана.
        min_overlap: Порог покрытия. По умолчанию — из настроек.
        abs_tolerance: Абсолютный допуск чисел. По умолчанию — из настроек.
        rel_tolerance: Относительный допуск чисел.

    Returns:
        Основание со статусом подтверждения.
    """
    if quote.strip():
        return verify_quote(
            source_text,
            quote,
            source_id,
            abs_tolerance=abs_tolerance,
            rel_tolerance=rel_tolerance,
        )

    threshold = settings.claim_min_overlap if min_overlap is None else min_overlap
    absolute, relative = _tolerances(abs_tolerance, rel_tolerance)
    groups = _groups()
    paragraphs = source_text.split("\n")

    def compare(paragraph: str) -> Overlap:
        """Сравнить утверждение с одним абзацем."""
        return lexical_overlap(
            claim_text,
            paragraph,
            groups=groups,
            abs_tolerance=absolute,
            rel_tolerance=relative,
            min_overlap=threshold,
        )

    if paragraph_index is not None and 0 <= paragraph_index < len(paragraphs):
        paragraph = paragraphs[paragraph_index].strip()
        result = compare(paragraph)
        if result.matched:
            return Support(
                quote=paragraph[:400],
                source_id=source_id,
                status=SupportStatus.CONFIRMED,
                paragraph_index=paragraph_index,
                detail=f"утверждение покрыто абзацем на {result.coverage:.0%}",
                oov=tuple(result.oov_tokens),
            )
        return Support(
            quote=claim_text.strip()[:400],
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail=_overlap_reason(result, threshold),
            oov=tuple(result.oov_tokens),
        )

    # Абзац не указан — ищем, в каком абзаце утверждение подтверждается.
    claim_numbers = numbers_in(claim_text)
    best: Overlap | None = None
    best_paragraph = ""
    for index, paragraph in enumerate(paragraphs):
        if len(paragraph.strip()) < 40:
            continue
        # Дешёвый предфильтр: у статьи тысячи абзацев, и морфологический
        # разбор каждого — самая дорогая часть проверки. Пропускаем абзац,
        # только если у него нет ни общих слов, ни чисел утверждения.
        paragraph_numbers_ok, _ = numbers_match(
            claim_numbers,
            numbers_in(paragraph),
            abs_tolerance=absolute,
            rel_tolerance=relative,
        )
        if not paragraph_numbers_ok and not shares_words(claim_text, paragraph):
            continue
        result = compare(paragraph)
        if best is None or result.coverage > best.coverage:
            best, best_paragraph = result, paragraph
        if result.matched:
            return Support(
                quote=paragraph.strip()[:400],
                source_id=source_id,
                status=SupportStatus.CONFIRMED,
                paragraph_index=index,
                detail=f"утверждение подтверждено абзацем на {result.coverage:.0%}",
                oov=tuple(result.oov_tokens),
            )

    if best is None:
        return Support(
            quote=claim_text.strip()[:400],
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail=(
                "в источнике нет абзаца, пригодного для сверки: "
                "ни общих слов, ни чисел с утверждением"
            ),
        )

    return Support(
        quote=best_paragraph.strip()[:400],
        source_id=source_id,
        status=SupportStatus.NO_DATA,
        detail=_overlap_reason(best, threshold),
        oov=tuple(best.oov_tokens),
    )


def verify_quote(
    source_text: str,
    quote: str,
    source_id: str,
    *,
    min_length: int = 12,
    abs_tolerance: float | None = None,
    rel_tolerance: float | None = None,
) -> Support:
    """Проверить, подтверждается ли цитата исходным текстом.

    Три исхода, и все честные:

    1. **Дословно** — подтверждено.
    2. **С допуском на округление.** Если в цитате есть числа и все они
       нашлись в одном абзаце с допуском, а слова совпали по леммам,
       цитата отличается от источника только точностью записи числа.
    3. **Другой язык.** Дословный перевод не проверяется: сверяются
       только числа, а статус остаётся «требует проверки».

    Args:
        source_text: Исходный текст источника.
        quote: Цитата из ответа модели.
        source_id: Идентификатор источника.
        min_length: Минимальная длина цитаты. Короткие совпадения вроде
            «да» или «и» находятся почти в любом тексте и ничего не доказывают.
        abs_tolerance: Абсолютный допуск чисел. По умолчанию — из настроек.
        rel_tolerance: Относительный допуск чисел.

    Returns:
        Основание со статусом подтверждения.
    """
    stripped = quote.strip()
    if len(stripped) < min_length:
        return Support(
            quote=stripped,
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail=f"цитата короче {min_length} символов — подтвердить нечего",
        )

    found, paragraph_index = find_quote(source_text, stripped)
    if found:
        return Support(
            quote=stripped,
            source_id=source_id,
            status=SupportStatus.CONFIRMED,
            paragraph_index=paragraph_index,
            detail="цитата найдена в источнике",
        )

    absolute, relative = _tolerances(abs_tolerance, rel_tolerance)

    # Другой язык источника: дословность недостижима. Сверяем числа —
    # единственное, что не зависит от перевода, — и не выдаём цитату за
    # подтверждённую.
    quote_language = language(stripped)
    source_language = language(source_text)
    if "other" not in {quote_language, source_language} and quote_language != source_language:
        numbers_ok, numbers_detail = numbers_match(
            numbers_in(stripped),
            numbers_in(source_text),
            abs_tolerance=absolute,
            rel_tolerance=relative,
        )
        if not numbers_ok:
            return Support(
                quote=stripped,
                source_id=source_id,
                status=SupportStatus.NO_DATA,
                detail=numbers_detail,
            )
        return Support(
            quote=stripped,
            source_id=source_id,
            status=SupportStatus.NEEDS_REVIEW,
            detail=(
                "иностранный источник: дословный перевод не проверяется, "
                "сверены только числа — нужен взгляд человека"
            ),
        )

    # Округление: цитата может отличаться от источника точностью записи
    # числа. Проверяем числа с допуском и слова по леммам.
    if not numbers_in(stripped):
        return Support(
            quote=stripped,
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail="цитата не найдена в источнике дословно",
        )

    groups = _groups()
    quote_numbers = numbers_in(stripped)
    best: Overlap | None = None
    for index, paragraph in enumerate(source_text.split("\n")):
        # Абзац, где не нашлось ни одного числа цитаты и ни одного общего
        # слова, не может быть её источником — не тратим на него разбор.
        paragraph_numbers_ok, _ = numbers_match(
            quote_numbers,
            numbers_in(paragraph),
            abs_tolerance=absolute,
            rel_tolerance=relative,
        )
        if not paragraph_numbers_ok and not shares_words(stripped, paragraph):
            continue
        result = lexical_overlap(
            stripped,
            paragraph,
            groups=groups,
            abs_tolerance=absolute,
            rel_tolerance=relative,
            min_overlap=settings.claim_min_overlap,
        )
        if best is None or result.coverage > best.coverage:
            best = result
        if result.matched:
            return Support(
                quote=stripped,
                source_id=source_id,
                status=SupportStatus.CONFIRMED,
                paragraph_index=index,
                detail="цитата совпала с источником с точностью до допуска чисел",
                oov=tuple(result.oov_tokens),
            )

    detail = "цитата не найдена в источнике дословно"
    if best is not None and best.coverage >= settings.claim_min_overlap:
        # Слова совпали, а цитата всё равно не подтвердилась — значит,
        # расхождение в числе или термине. Назвать это точнее полезнее:
        # «не найдена» читается как выдумка, хотя это округление или
        # подмена термина.
        detail = _overlap_reason(best, settings.claim_min_overlap)
    return Support(
        quote=stripped,
        source_id=source_id,
        status=SupportStatus.NO_DATA,
        detail=detail,
    )


def verify_quote_as_external(
    source_text: str,
    quote: str,
    source_id: str,
) -> Support:
    """Проверить цитату во внешнем источнике.

    Дословность проверяется как обычно, но подтвердить внешний источник
    нечем: никто не знает, что за страницу отдала выдача. Поэтому статус
    не выше «требует проверки».

    Args:
        source_text: Текст страницы.
        quote: Цитата из ответа модели.
        source_id: Идентификатор внешнего источника.

    Returns:
        Основание со статусом не выше ``needs_review``.
    """
    return mark_external(verify_quote(source_text, quote, source_id))
