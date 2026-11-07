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
from typing import Any

from pf.config import settings
from pf.corpus import (
    DOMAIN_TERMS,
    STRONG_DOMAIN_TERMS,
    Article,
    Corpus,
    domain_lemmas,
    get_corpus,
    lemmatize,
    strong_domain_lemmas,
    tokenize,
)
from pf.ground import (
    Support,
    SupportStatus,
    find_quote,
    split_source_id,
    verify_claim,
    verify_quote,
)
from pf.materials import Material

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
        if any(a.source_id == clean_id or a.title == clean_id for a in self.articles):
            return True
        # Модель часто присылает короткое имя файла вместо полного
        # пути: «telecom_open_metrics» вместо
        # «project/data/telecom_open_metrics.csv». Источник тот же, а
        # строгий поиск его не находил — и все ссылки отклонялись.
        tail = clean_id.rsplit("/", 1)[-1]
        stem = tail.rsplit(".", 1)[0]
        return any(
            a.source_id.rsplit("/", 1)[-1] == tail
            or a.source_id.rsplit("/", 1)[-1].rsplit(".", 1)[0] == stem
            for a in self.articles
        )

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

        # Короткое имя файла вместо полного пути: «telecom_open_metrics»
        # вместо «project/data/telecom_open_metrics.csv». Модель так
        # отвечает, и цитата из своего материала иначе не находилась.
        tail = clean_id.rsplit("/", 1)[-1]
        stem = tail.rsplit(".", 1)[0]
        for pool in (self.articles, get_corpus().articles):
            found = [
                a
                for a in pool
                if a.source_id.rsplit("/", 1)[-1] == tail
                or a.source_id.rsplit("/", 1)[-1].rsplit(".", 1)[0] == stem
            ]
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
            # Ссылка на строку таблицы (#rowN) — это не абзац, а строка.
            # Цитата из неё ищется в самой строке, а не в склеенном тексте
            # материала: иначе цитата вида «...: среднее: 82» (это
            # посчитанная величина) никогда не нашлась бы.
            if "#row" in source_id and candidates:
                row = parsed_paragraph
                article = candidates[0]
                rows = _article_rows(article)
                if row is not None and 0 <= row < len(rows):
                    row_text = rows[row]
                    found, _ = find_quote(row_text, quote) if quote else (False, None)
                    if found:
                        return Support(
                            quote=quote,
                            source_id=source_id,
                            status=SupportStatus.CONFIRMED,
                            paragraph_index=row,
                            detail="цитата найдена в строке таблицы",
                        )
                    # Число из расчёта: проверяем, что оно есть в строке.
                    number = _first_number(quote)
                    if number is not None and _row_has_number(row_text, number):
                        return Support(
                            quote=quote,
                            source_id=source_id,
                            status=SupportStatus.CONFIRMED,
                            paragraph_index=row,
                            detail="значение из расчёта найдено в строке таблицы",
                        )
                    return Support(
                        quote=quote,
                        source_id=source_id,
                        status=SupportStatus.NO_DATA,
                        detail="цитата не найдена в строке таблицы",
                    )
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


def verify_metric(
    metric: Any,
    materials: list[Material],
    *,
    tolerance: float = 1e-6,
) -> Support:
    """Проверить расчётную метрику пересчётом по строке источника.

    Факт подтверждается дословной цитатой. Метрика — иначе: её нужно
    **пересчитать** по указанной строке и сравнить с заявленным
    значением. Это единственная проверка, которая отличает расчёт от
    выдумки: число нельзя «процитировать».

    Проверяются оба основания расчёта:

    1. значение метрики совпадает со значением в строке источника —
       либо с заявленными входами ``inputs``;
    2. если входы заданы, пересчёт по ним даёт то же значение.

    Args:
        metric: Метрика с provenance.
        materials: Материалы проекта.
        tolerance: Допустимое расхождение при сравнении.

    Returns:
        Результат проверки: подтверждено, требует проверки или нет
        данных.
    """
    from pf.analysis import Metric, parse_number

    if not isinstance(metric, Metric):
        return Support(
            quote="",
            source_id="",
            status=SupportStatus.NO_DATA,
            detail="метрика не имеет структуры с provenance",
        )

    source_id = metric.source_id
    if not source_id:
        return Support(
            quote=metric.name,
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail="у метрики нет источника — пересчитать нельзя",
        )

    material = next((m for m in materials if m.source_id == source_id), None)
    if material is None:
        return Support(
            quote=metric.name,
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail=f"источник {source_id} не найден среди материалов",
        )

    # 1. Значение из указанной строки источника.
    row_values: list[float] = []
    if metric.row is not None and 0 <= metric.row < len(material.rows):
        for value in material.rows[metric.row].values():
            parsed = parse_number(str(value))
            if parsed is not None:
                row_values.append(parsed)
    else:
        for record in material.rows:
            for value in record.values():
                parsed = parse_number(str(value))
                if parsed is not None:
                    row_values.append(parsed)

    if row_values and not _matches_any(metric.value, row_values, tolerance):
        return Support(
            quote=metric.formatted,
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail=(
                f"значение {metric.formatted} не совпадает со строкой "
                f"{metric.row if metric.row is not None else '?'} источника"
            ),
        )

    # 2. Пересчёт по заявленным входам.
    if metric.inputs:
        numbers = [
            parsed
            for parsed in (parse_number(str(v)) for v in metric.inputs.values())
            if parsed is not None
        ]
        if len(numbers) >= 2:
            base, new = numbers[0], numbers[1]
            recomputed = (new - base) / abs(base) * 100 if base else 0.0
            if metric.unit == "%" and not _matches_any(recomputed, [metric.value], tolerance):
                return Support(
                    quote=metric.formatted,
                    source_id=source_id,
                    status=SupportStatus.NO_DATA,
                    detail=(
                        f"пересчёт по входам даёт {recomputed:.4g} %, а заявлено {metric.formatted}"
                    ),
                )

    return Support(
        quote=metric.formatted,
        source_id=source_id,
        status=SupportStatus.CONFIRMED,
        paragraph_index=metric.row,
        detail=(
            "метрика пересчитана по строке источника"
            if metric.row is not None
            else "метрика подтверждена по материалу"
        ),
    )


def _article_rows(article: Article) -> list[str]:
    """Строки таблицы, если материал — таблица.

    Материал проекта хранит исходные строки таблицы; у статьи корпуса
    таких строк нет, и тогда проверка идёт обычным путём.

    Args:
        article: Статья или материал.

    Returns:
        Список строк таблицы (пустой, если это не таблица).
    """
    if article.section != "project":
        return []
    from pf.materials import load_materials

    for material in load_materials():
        if material.source_id == article.source_id:
            return [
                " ".join(f"{key}: {value}" for key, value in row.items()) for row in material.rows
            ]
    return []


def _first_number(text: str) -> float | None:
    """Первое число в тексте.

    Args:
        text: Текст.

    Returns:
        Число либо ``None``.
    """
    from pf.analysis import parse_number

    for token in re.findall(r"-?\d+(?:[.,]\d+)?", text):
        value = parse_number(token)
        if value is not None:
            return value
    return None


def _row_has_number(row_text: str, number: float) -> bool:
    """Есть ли число в строке таблицы.

    Args:
        row_text: Текст строки.
        number: Искомое число.

    Returns:
        Есть ли число.
    """
    from pf.analysis import parse_number

    for token in re.findall(r"-?\d+(?:[.,]\d+)?", row_text):
        value = parse_number(token)
        if value is not None and abs(value - number) < 1e-9:
            return True
    return False


def _matches_any(value: float, candidates: list[float], tolerance: float) -> bool:
    """Совпадает ли значение с одним из кандидатов.

    Args:
        value: Проверяемое значение.
        candidates: Кандидаты.
        tolerance: Допустимое расхождение.

    Returns:
        Есть ли подходящий кандидат.
    """
    return any(abs(value - candidate) <= tolerance for candidate in candidates)


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
    materials: list[Material] | None = None,
    include_materials: bool = True,
) -> SourceBundle:
    """Собрать источники под запрос.

    В срез попадают и статьи корпуса, и материалы проекта: это третий
    слой знаний — свои документы и таблицы. Материалы проходят ту же
    проверку дословности, что и статьи, поэтому смешивать их можно
    свободно.

    Args:
        query: Запрос агента.
        corpus: Корпус. По умолчанию — общий для процесса.
        top_k: Сколько статей взять.
        max_paragraphs: Сколько абзацев из каждой статьи.
        materials: Материалы проекта. По умолчанию — из каталога ``project/``.
        include_materials: Включать ли материалы проекта.

    Returns:
        Набор источников с абзацами, пригодными для цитирования.
    """
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

    if not include_materials:
        return SourceBundle(passages=passages, articles=articles, query=query)

    if materials is None:
        from pf.materials import load_materials

        materials = load_materials()

    project_passages, project_articles = _material_passages(materials, query_terms, max_paragraphs)
    if not project_passages:
        return SourceBundle(passages=passages, articles=articles, query=query)

    # Материалы проекта идут перед корпусом: они относятся к задаче
    # непосредственно, а корпус даёт рыночный контекст.
    return SourceBundle(
        passages=project_passages + passages,
        articles=project_articles + articles,
        query=query,
    )


def _material_passages(
    materials: list[Material],
    query_terms: set[str],
    max_paragraphs: int,
) -> tuple[list[Passage], list[Article]]:
    """Отобрать абзацы материалов проекта, релевантные запросу.

    Args:
        materials: Материалы проекта.
        query_terms: Значимые слова запроса.
        max_paragraphs: Сколько абзацев оставить из каждого материала.

    Returns:
        Пара ``(абзацы, материалы как статьи)``.
    """
    project_passages: list[Passage] = []
    project_articles: list[Article] = []

    for material in materials:
        selected = _score_material(material, query_terms, max_paragraphs)
        if not selected:
            continue
        project_articles.append(
            Article(
                source_id=material.source_id,
                title=material.title,
                section="project",
                content=material.content,
            )
        )
        project_passages.extend(selected)

    project_passages.sort(key=lambda passage: passage.relevance, reverse=True)
    return project_passages, project_articles


def _score_material(
    material: Material,
    query_terms: set[str],
    max_paragraphs: int,
) -> list[Passage]:
    """Отобрать абзацы материала, релевантные запросу.

    Отбор двухчастный, и это не украшение:

    1. **Прямое совпадение слов.** Абзац содержит слово из запроса —
       он точно по теме.
    2. **Совпадение по предметной области.** Материал может быть по
       теме, но называть её не теми словами, что запрос: задача
       «почему падает удержание» спрашивает про абонентов, а таблица
       про них говорит «абонентская база». Требовать буквального
       совпадения здесь нельзя — тогда собственные материалы агента
       никогда не попадали бы в срез, и они были бы видны только
       при дословном повторении формулировки задачи.

    Материал не по теме (рецепты при запросе про отток) не проходит
    ни по одному признаку.

    Args:
        material: Материал проекта.
        query_terms: Значимые слова запроса.
        max_paragraphs: Сколько абзацев оставить.

    Returns:
        Отобранные абзацы.
    """
    material_words: set[str] = set()
    for record in material.rows:
        # Заголовки колонок — тоже часть материала: в них часто
        # написано, что именно измеряется («абонентская база»).
        for key in record:
            material_words.update(tokenize(str(key)))
        for value in record.values():
            material_words.update(tokenize(str(value)))
    for passage in material.passages:
        material_words.update(tokenize(passage.text))

    material_lemmas = {lemmatize(word) for word in material_words}
    query_lemmas = {lemmatize(word) for word in query_terms}

    # Материал по предметной области: в нём есть сильный доменный
    # признак, а запрос — тоже про предметную область. Сравнение идёт
    # по леммам: словоформы («абонентская», «абонентов») у разных
    # слов разные, а лемма одна.
    strong_in_material = material_lemmas & strong_domain_lemmas()
    domain_in_query = query_lemmas & domain_lemmas()
    same_domain = bool(strong_in_material) and bool(domain_in_query)

    scored: list[tuple[float, int, str]] = []
    for passage in material.passages:
        text = passage.text
        words = tokenize(text)
        if not words:
            continue
        hits = sum(1 for word in words if word in query_terms)
        if hits == 0 and not same_domain:
            continue
        weighted = sum(
            2.0 if word in DOMAIN_TERMS else 1.0 for word in words if word in query_terms
        )
        if hits == 0:
            # Совпадение по предметной области: вес даёт сам факт, что
            # материал про ту же область, что и запрос.
            weighted = 1.0
        score = weighted / (len(words) ** 0.5)
        scored.append((score, passage.index, text))

    scored.sort(reverse=True)
    return [
        Passage(
            source_id=material.source_id,
            title=material.title,
            section="project",
            paragraph_index=index,
            text=text,
            relevance=score,
        )
        for score, index, text in scored[:max_paragraphs]
    ]
