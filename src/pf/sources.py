"""Сбор источников: отбор статей и нарезка на цитатуемые абзацы.

Задача модуля — превратить корпус в **перечислимый набор оснований**.
Каждый абзац, попавший в контекст модели, должен:

- помнить, из какой статьи он взят (``source_id``);
- иметь свой номер (чтобы в ответе можно было сослаться на абзац);
- быть достаточно длинным, чтобы его можно было процитировать дословно.

Абзацы короче порога отбрасываются: из них нечего цитировать, а в выдаче
они только занимают место и мешают модели.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pf.corpus import DOMAIN_TERMS, Article, Corpus, get_corpus, tokenize
from pf.ground import Support, SupportStatus, split_source_id, verify_claim, verify_quote

#: Минимальная длина абзаца, пригодного для цитирования.
MIN_PARAGRAPH_CHARS = 60

#: Максимальная длина абзаца в контексте. Длинные абзацы режутся, иначе
#: контекст модели переполняется одним материалом.
MAX_PARAGRAPH_CHARS = 1200

#: Служебные заголовки статей («См. также», «Литература», «Примечания») —
#: они не содержат содержательных утверждений и в основания не идут.
_SKIP_HEADING = re.compile(
    r"^(См\.?\s*также|Литература|Примечания|Сноски|Ссылки|Примечания и источники|"
    r"Bibliography|References|See also|Notes|External links)\s*[:.]?$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Passage:
    """Абзац статьи, пригодный для цитирования.

    Attributes:
        source_id: Идентификатор источника.
        title: Заголовок статьи.
        section: Раздел корпуса.
        paragraph_index: Номер абзаца в статье (нумерация с нуля).
        text: Текст абзаца.
        relevance: Оценка релевантности абзаца запросу.
    """

    source_id: str
    title: str
    section: str
    paragraph_index: int
    text: str
    relevance: float

    @property
    def citation(self) -> str:
        """Ссылка на абзац для ответа агента."""
        return f"{self.source_id}#p{self.paragraph_index}"


@dataclass(frozen=True, slots=True)
class SourceBundle:
    """Набор источников, собранный под одну задачу.

    Attributes:
        passages: Абзацы, отобранные под запрос.
        articles: Статьи, из которых взяты абзацы.
        query: Исходный запрос.
    """

    passages: list[Passage]
    articles: list[Article]
    query: str

    @property
    def source_ids(self) -> list[str]:
        """Идентификаторы задействованных источников."""
        seen: list[str] = []
        for passage in self.passages:
            if passage.source_id not in seen:
                seen.append(passage.source_id)
        return seen

    def context(self, limit: int | None = None) -> str:
        """Собрать контекст для модели.

        Args:
            limit: Сколько абзацев включить. По умолчанию — все.

        Returns:
            Текст с абзацами, перед каждым — его идентификатор для цитирования.
        """
        selected = self.passages[:limit] if limit else self.passages
        blocks = []
        for passage in selected:
            blocks.append(f"[{passage.citation}] {passage.text}")
        return "\n\n".join(blocks)

    def is_relevant(self, source_id: str) -> bool:
        """Относится ли источник к текущей задаче.

        Цитата может быть дословной из настоящей статьи — и всё равно
        не иметь отношения к вопросу. Например, агент может честно
        процитировать статью «Испытание (техника)», обсуждающую методы
        проверки гипотез. Цитата настоящая, но к удержанию абонентов
        отношения не имеет.

        Поэтому источник принимается, только если он попал в отобранный
        для этой задачи срез: именно там отобраны релевантные абзацы.

        Args:
            source_id: Идентификатор источника (может содержать ``#pN``).

        Returns:
            Источник релевантен задаче.
        """
        clean_id, _ = split_source_id(source_id)
        return any(a.source_id == clean_id or a.title == clean_id for a in self.articles)

    def _resolve(self, clean_id: str) -> list[Article]:
        """Найти статью по идентификатору.

        Сначала ищем среди отобранных, затем — во всём корпусе. Модель
        может сослаться на статью, которая не попала в срез, но при этом
        существует: основание проверяем по всей статье, а не только по
        отобранному куску.

        Args:
            clean_id: Идентификатор без номера абзаца.

        Returns:
            Список подходящих статей (пустой, если статья не найдена).
        """
        if not clean_id:
            return []

        for pool in (self.articles, get_corpus().articles):
            found = [a for a in pool if a.source_id == clean_id]
            if found:
                return found
            found = [a for a in pool if a.title == clean_id]
            if found:
                return found
        return []

    def verify(self, quote: str, source_id: str | None = None) -> Support:
        """Проверить цитату против собранных источников.

        Args:
            quote: Цитата из ответа модели.
            source_id: Явный источник, если модель его назвала. Если не задан,
                цитата проверяется по всем собранным статьям.

        Returns:
            Основание со статусом подтверждения.
        """
        if source_id:
            # Модель часто склеивает путь и номер абзаца: «раздел/Заголовок#p126».
            clean_id, paragraph = split_source_id(source_id)
            candidates = self._resolve(clean_id)
        else:
            clean_id = ""
            candidates = list(self.articles)

        if not candidates:
            return Support(
                quote=quote.strip(),
                source_id=source_id or "",
                status=SupportStatus.NO_DATA,
                detail="источник не найден среди собранных",
            )

        best = Support(
            quote=quote.strip(),
            source_id=source_id or "",
            status=SupportStatus.NO_DATA,
            detail="цитата не найдена в источнике дословно",
        )
        for article in candidates:
            result = verify_quote(article.content, quote, article.source_id)
            if result.is_usable:
                return result
            if result.status is SupportStatus.NEEDS_REVIEW:
                best = result
        return best

    def verify_claim(
        self,
        claim_text: str,
        source_id: str = "",
        paragraph: int | None = None,
        quote: str = "",
    ) -> Support:
        """Подтвердить утверждение по собранным источникам.

        Args:
            claim_text: Текст утверждения.
            source_id: Идентификатор источника (может быть со склеенным ``#pN``).
            paragraph: Номер абзаца.
            quote: Дословная цитата, если модель её прислала.

        Returns:
            Основание со статусом подтверждения.
        """
        if source_id:
            clean_id, parsed_paragraph = split_source_id(source_id)
            if paragraph is None:
                paragraph = parsed_paragraph
            candidates = self._resolve(clean_id)
        else:
            candidates = list(self.articles)

        if not candidates:
            return Support(
                quote=claim_text[:400],
                source_id=source_id,
                status=SupportStatus.NO_DATA,
                detail="источник не найден среди собранных",
            )

        best = Support(
            quote=claim_text[:400],
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail="утверждение не подтверждается источником",
        )
        for article in candidates:
            result = verify_claim(
                article.content,
                claim_text,
                article.source_id,
                paragraph_index=paragraph,
                quote=quote,
            )
            if result.is_usable:
                return result
            best = result
        return best


def score_passages(
    article: Article,
    query_terms: set[str],
    max_paragraphs: int,
) -> list[Passage]:
    """Отобрать абзацы статьи, релевантные запросу.

    Args:
        article: Статья-источник.
        query_terms: Значимые слова запроса.
        max_paragraphs: Сколько абзацев оставить.

    Returns:
        Отсортированные по релевантности абзацы.
    """
    scored: list[tuple[float, int, str]] = []
    for index, raw in enumerate(article.content.split("\n")):
        text = raw.strip()
        if len(text) < MIN_PARAGRAPH_CHARS or _SKIP_HEADING.match(text):
            continue
        if len(text) > MAX_PARAGRAPH_CHARS:
            text = text[:MAX_PARAGRAPH_CHARS].rsplit(" ", 1)[0]

        words = tokenize(text)
        if not words:
            continue
        hits = sum(1 for word in words if word in query_terms)
        if hits == 0:
            continue
        # Доменные слова запроса весят больше общего словаря.
        weighted_hits = sum(
            2.0 if word in DOMAIN_TERMS else 1.0 for word in words if word in query_terms
        )
        # Релевантность — доля совпавших слов, с поправкой на длину абзаца:
        # короткий абзац с двумя совпадениями информативнее длинного с двумя.
        score = weighted_hits / (len(words) ** 0.5)
        scored.append((score, index, text))

    scored.sort(reverse=True)
    return [
        Passage(
            source_id=article.source_id,
            title=article.title,
            section=article.section,
            paragraph_index=index,
            text=text,
            relevance=score,
        )
        for score, index, text in scored[:max_paragraphs]
    ]


def collect(
    query: str,
    *,
    corpus: Corpus | None = None,
    top_k: int | None = None,
    max_paragraphs: int | None = None,
) -> SourceBundle:
    """Собрать источники под запрос.

    Args:
        query: Запрос агента.
        corpus: Корпус. По умолчанию — общий для процесса.
        top_k: Сколько статей взять.
        max_paragraphs: Сколько абзацев из каждой статьи.

    Returns:
        Набор источников с абзацами, пригодными для цитирования.
    """
    from pf.config import settings

    corpus = corpus or get_corpus()
    top_k = top_k or settings.source_top_k
    max_paragraphs = max_paragraphs or settings.source_max_paragraphs
    query_terms = set(tokenize(query))

    articles: list[Article] = []
    passages: list[Passage] = []

    for article, _score in corpus.search(query, top_k=top_k):
        selected = score_passages(article, query_terms, max_paragraphs)
        if not selected:
            continue
        articles.append(article)
        passages.extend(selected)

    passages.sort(key=lambda passage: passage.relevance, reverse=True)
    return SourceBundle(passages=passages, articles=articles, query=query)
