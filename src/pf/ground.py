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
from dataclasses import dataclass
from enum import StrEnum

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
    """

    quote: str
    source_id: str
    status: SupportStatus
    paragraph_index: int | None = None
    detail: str = ""

    @property
    def is_usable(self) -> bool:
        """Можно ли опереться на это основание."""
        return self.status is SupportStatus.CONFIRMED

    @property
    def citation(self) -> str:
        """Человекочитаемая ссылка на источник."""
        if self.paragraph_index is None:
            return self.source_id
        return f"{self.source_id}#p{self.paragraph_index}"


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
    """Разделить идентификатор источника на путь и номер абзаца.

    Модель часто склеивает их в одну строку вида
    ``security/Мегафон (компания)#p126``. Разбор делается здесь, чтобы
    остальной код работал с чистыми полями.

    Args:
        source_id: Идентификатор как прислала модель.

    Returns:
        Пара ``(source_id, номер_абзаца)``.
    """
    text = source_id.strip()
    if "#p" in text:
        path, _, suffix = text.rpartition("#p")
        if path and suffix.isdigit():
            return path, int(suffix)
    return text, None


def token_overlap(claim: str, paragraph: str) -> float:
    """Доля значимых слов утверждения, покрытых абзацем.

    Используется, когда модель не прислала дословную цитату: если
    утверждение почти целиком собрано из слов абзаца, значит, оно
    действительно взято из этого места. Если же в утверждении есть слова,
    которых в абзаце нет, — это признак выдумки.

    Args:
        claim: Текст утверждения.
        paragraph: Текст абзаца-источника.

    Returns:
        Доля от 0.0 до 1.0.
    """
    from pf.corpus import tokenize

    claim_words = set(tokenize(claim))
    if not claim_words:
        return 0.0
    paragraph_words = set(tokenize(paragraph))
    return len(claim_words & paragraph_words) / len(claim_words)


def verify_claim(
    source_text: str,
    claim_text: str,
    source_id: str,
    paragraph_index: int | None = None,
    quote: str = "",
    *,
    min_overlap: float = 0.8,
) -> Support:
    """Подтвердить утверждение по источнику.

    Работают два пути:

    1. **Строгая цитата.** Если модель прислала цитату, она должна
       дословно встречаться в источнике.
    2. **Покрытие абзацем.** Если цитаты нет, проверяется, что утверждение
       собрано из слов указанного абзаца. Порог высокий: почти все
       значимые слова утверждения должны найтись в абзаце.

    Args:
        source_text: Полный текст источника.
        claim_text: Текст утверждения.
        source_id: Идентификатор источника.
        paragraph_index: Номер абзаца, если указан.
        quote: Дословная цитата, если прислана.
        min_overlap: Порог покрытия для второго пути.

    Returns:
        Основание со статусом подтверждения.
    """
    if quote.strip():
        return verify_quote(source_text, quote, source_id)

    paragraphs = source_text.split("\n")
    if paragraph_index is not None and 0 <= paragraph_index < len(paragraphs):
        paragraph = paragraphs[paragraph_index].strip()
        overlap = token_overlap(claim_text, paragraph)
        if overlap >= min_overlap:
            return Support(
                quote=paragraph[:400],
                source_id=source_id,
                status=SupportStatus.CONFIRMED,
                paragraph_index=paragraph_index,
                detail=f"утверждение покрыто абзацем на {overlap:.0%}",
            )
        return Support(
            quote=claim_text.strip()[:400],
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail=(
                f"слова утверждения не подтверждаются абзацем "
                f"(покрытие {overlap:.0%} при пороге {min_overlap:.0%})"
            ),
        )

    # Абзац не указан — ищем, в каком абзаце утверждение подтверждается.
    best = 0.0
    best_index: int | None = None
    for index, paragraph in enumerate(paragraphs):
        if len(paragraph.strip()) < 40:
            continue
        overlap = token_overlap(claim_text, paragraph)
        if overlap > best:
            best = overlap
            best_index = index

    if best >= min_overlap and best_index is not None:
        return Support(
            quote=paragraphs[best_index].strip()[:400],
            source_id=source_id,
            status=SupportStatus.CONFIRMED,
            paragraph_index=best_index,
            detail=f"утверждение подтверждено абзацем на {best:.0%}",
        )

    return Support(
        quote=claim_text.strip()[:400],
        source_id=source_id,
        status=SupportStatus.NO_DATA,
        detail=f"утверждение не подтверждается источником (покрытие {best:.0%})",
    )


def verify_quote(
    source_text: str,
    quote: str,
    source_id: str,
    *,
    min_length: int = 12,
) -> Support:
    """Проверить, подтверждается ли цитата исходным текстом.

    Args:
        source_text: Исходный текст источника.
        quote: Цитата из ответа модели.
        source_id: Идентификатор источника.
        min_length: Минимальная длина цитаты. Короткие совпадения вроде
            «да» или «и» находятся почти в любом тексте и ничего не доказывают.

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
    if not found:
        return Support(
            quote=stripped,
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail="цитата не найдена в источнике дословно",
        )

    return Support(
        quote=stripped,
        source_id=source_id,
        status=SupportStatus.CONFIRMED,
        paragraph_index=paragraph_index,
        detail="цитата найдена в источнике",
    )
