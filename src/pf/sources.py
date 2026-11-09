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

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pf import domain
from pf.config import settings
from pf.corpus import (
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
    mark_external,
    split_source_id,
    verify_claim,
    verify_quote,
)
from pf.lexical import number_tolerance
from pf.lexical import significant_words as _significant_words
from pf.lexical import stems as _stems
from pf.materials import KNOWLEDGE_SOURCE_PREFIX, Material

#: Подписи, под которыми ввод пользователя попадает в ответ как «источник».
#: Модель видит его в контексте и ссылается на него наравне с документами.
#: Это не запрещено: ввод пользователя — законная опора, у неё просто другой
#: канал подтверждения (`D-49`).
HUMAN_INPUT_LABELS: frozenset[str] = frozenset(
    {
        "ответ человека",
        "выбор человека",
        "ввод человека",
        "решение человека",
        "ответ пользователя",
        "ввод пользователя",
        "выбор пользователя",
        "пользовательский ввод",
    }
)

#: Причины, по которым ссылка не годится как основание. Формулировки
#: стабильны: по ним `pf.report.rejection_kind` считает класс отказа, а
#: человек видит, что именно случилось, а не «похоже на другое».
REASON_NO_LINK = "ссылки нет — цитировать нечего"
REASON_HUMAN_INPUT = "не источник — это ввод человека, а не документ"
REASON_OUTSIDE_SLICE = "источник вне среза задачи"
REASON_UNKNOWN_SOURCE = "источник не найден среди материалов"


def human_input_stage(source_id: str) -> str | None:
    """Стадия из ссылки на ввод пользователя.

    Ссылка называет канал и стадию: ``ввод пользователя: analyzing``.
    Старая форма — просто подпись («Ответ человека») — тоже узнаётся, но
    стадии в ней нет: пустая строка означает «не названа».

    Args:
        source_id: Ссылка из ответа модели, возможно с якорем ``#pN``.

    Returns:
        Название стадии как его написала модель; пустая строка — ссылка на
        ввод без стадии; ``None`` — ссылка не про ввод пользователя.
    """
    clean, _ = split_source_id(source_id)
    label = clean.strip().lower().replace("ё", "е")
    if not label:
        return None
    head, _, tail = label.partition(":")
    if head.strip() in HUMAN_INPUT_LABELS:
        return tail.strip()
    return None


#: Машинные виды отказа: по ним считается корзина в отчёте и замере.
ABSENCE_KINDS: dict[str, str] = {
    REASON_NO_LINK: "source_not_named",
    REASON_HUMAN_INPUT: "human_input",
    REASON_OUTSIDE_SLICE: "outside_slice",
    REASON_UNKNOWN_SOURCE: "source_unknown",
}

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

#: Признаки посчитанной величины. В строке таблицы лежат исходные
#: значения, а «изменение за период» — результат расчёта: такого текста
#: в источнике нет. Отличает неверный канал оформления от выдумки.
_COMPUTED_MARKERS = re.compile(
    r"(изменени|средн|медиан|сумм|разниц|рост|падени|темп|доля|cagr|average|mean|delta)",
    re.IGNORECASE,
)

#: Строка блока РАСЧЁТЫ: ``имя: значение [путь#rowN] (формула)``.
#: Разметка приходит из ``Workflow._metric_line``, а модель копирует её в
#: основание как есть — поэтому она распознаётся, но текстом источника не
#: считается.
_CALC_LINE_RE = re.compile(
    r"^(?P<body>.+?)\s*\[\s*(?P<source>[^\[\]\n]+?)#row(?P<row>\d+)\s*\]"
    r"\s*(?:\((?P<markup>.+)\))?\s*$",
    re.DOTALL,
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


#: Пояснение для основания из сырой выгрузки. Своё, хотя правило то же, что у
#: внешнего источника: человеку важно знать, что ограничение не про цитату.
_KNOWLEDGE_REVIEW_REASON = (
    "выгрузка из хранилища знаний: цитата найдена дословно, происхождение "
    "не проверено — нужен взгляд человека"
)


def _limit_knowledge_source(support: Support, source_id: str) -> Support:
    """Ограничить статус основания из сырой выгрузки.

    Дословность у выгрузки проверяема так же, как у материала проекта, а
    происхождение — нет: файл скачан из открытого источника и с тех пор мог
    измениться. Поэтому «подтверждено» превращается в «требует проверки», а не
    в отказ: знание не выбрасывается, но и за проверенный факт не выдаётся.

    Args:
        support: Результат обычной проверки.
        source_id: Идентификатор источника цитаты.

    Returns:
        Основание со статусом не выше «требует проверки».
    """
    clean_id, _ = split_source_id(source_id)
    if not clean_id.startswith(KNOWLEDGE_SOURCE_PREFIX):
        return support
    return mark_external(support, reason=_KNOWLEDGE_REVIEW_REASON)


#: Префиксы идентификаторов своих материалов: документы проекта и сырые
#: выгрузки. Всё остальное — корпус статей.
LOCAL_SOURCE_PREFIXES = ("project/", KNOWLEDGE_SOURCE_PREFIX)


def _is_local_source(source_id: str) -> bool:
    """Свой материал это или статья корпуса.

    Различие нужно окну контекста: свои материалы относятся к задаче
    непосредственно и идут первыми, а корпус даёт рыночный контекст. Если
    не различать их, строки таблиц вытесняют статьи — и агент отвечает на
    вопрос про отток данными про размер базы.

    Args:
        source_id: Идентификатор источника.

    Returns:
        ``True``, если источник — материал проекта или выгрузка.
    """
    return source_id.startswith(LOCAL_SOURCE_PREFIXES)


def _take(
    passages: list[Passage],
    chosen: list[Passage],
    taken: dict[str, int],
    limit: int,
    max_per_source: int,
) -> None:
    """Добрать абзацы из группы до предела окна.

    Args:
        passages: Абзацы группы в порядке релевантности.
        chosen: Уже выбранные абзацы — пополняется на месте.
        taken: Сколько абзацев взято от каждого источника — пополняется на месте.
        limit: Общий предел окна.
        max_per_source: Предел абзацев от одного источника.
    """
    for passage in passages:
        if len(chosen) >= limit:
            return
        count = taken.get(passage.source_id, 0)
        if count >= max_per_source:
            continue
        chosen.append(passage)
        taken[passage.source_id] = count + 1


@dataclass(frozen=True, slots=True)
class SourceBundle:
    """Набор источников, собранный под одну задачу.

    Attributes:
        passages: Абзацы, отобранные под запрос.
        articles: Статьи, из которых взяты абзацы.
        query: Исходный запрос.
        metrics: Расчёты агента по этим материалам. Нужны, потому что
            агрегат (среднее, медиана, изменение) не лежит ни в одной
            строке таблицы: подтвердить его можно только сверкой с
            расчётом по имени и пересчётом по его входам.
    """

    passages: list[Passage]
    articles: list[Article]
    query: str
    metrics: tuple[Any, ...] = ()

    #: Сколько абзацев одного источника попадает в окно при ограничении.
    #: Одна таблица даёт по абзацу на строку, и без предела строки вытесняют
    #: весь корпус: в прогоне `demo-v9` окно целиком состояло из строк
    #: таблицы, а абзацев по теме вопроса модель не видела.
    MAX_PASSAGES_PER_SOURCE = 2

    @property
    def source_ids(self) -> list[str]:
        """Идентификаторы задействованных источников."""
        seen: list[str] = []
        for passage in self.passages:
            if passage.source_id not in seen:
                seen.append(passage.source_id)
        return seen

    def context(
        self,
        limit: int | None = None,
        *,
        max_per_source: int = MAX_PASSAGES_PER_SOURCE,
    ) -> str:
        """Собрать контекст для модели.

        При ограничении окна абзацы берутся с разнообразием: не больше
        ``max_per_source`` от одного источника, остаток — по релевантности.
        Набор документов из одной темы не должен занимать всё окно: модель
        отвечает тем, что видит, а не тем, что есть в корпусе.

        Если источников мало, окно всё равно заполняется — добором из того,
        что осталось. Иначе ограничение превратилось бы в обрезание.

        Args:
            limit: Сколько абзацев включить. По умолчанию — все.
            max_per_source: Предел абзацев от одного источника.

        Returns:
            Текст с абзацами, перед каждым — его идентификатор для цитирования.
        """
        selected = self._window(limit, max_per_source) if limit else self.passages
        blocks = []
        for passage in selected:
            blocks.append(f"[{passage.citation}] {passage.text}")
        return "\n\n".join(blocks)

    def _window(self, limit: int, max_per_source: int) -> list[Passage]:
        """Выбрать абзацы окна: сначала с разнообразием, затем добор.

        Свои материалы идут первыми, но всё окно не забирают. Когда файлов
        данных становится больше, чем слотов в окне, строки таблиц вытесняют
        корпус целиком — и модель, отвечая на вопрос про отток, не видит ни
        одной статьи про отток. Поэтому один слот остаётся корпусу, если в
        сборе есть его абзацы.

        Args:
            limit: Сколько абзацев нужно.
            max_per_source: Предел абзацев от одного источника.

        Returns:
            Абзацы в порядке релевантности.
        """
        local = [passage for passage in self.passages if _is_local_source(passage.source_id)]
        external = [passage for passage in self.passages if not _is_local_source(passage.source_id)]

        chosen: list[Passage] = []
        taken: dict[str, int] = {}
        # Резерв — ровно один слот: корпус не должен вытесняться, но и
        # главенство своих материалов (они относятся к задаче) не теряется.
        limit_local = max(limit - (1 if external else 0), 1)
        _take(local, chosen, taken, limit_local, max_per_source)
        _take(external, chosen, taken, limit, max_per_source)
        if len(chosen) < limit:
            for passage in self.passages:
                if len(chosen) >= limit:
                    break
                if passage not in chosen:
                    chosen.append(passage)
        return chosen

    def absence_reason(self, source_id: str) -> str:
        """Почему ссылка не годится как основание — словами.

        Три разных случая различаются, потому что чинятся по-разному: ссылки
        нет вовсе (промпт и схема ответа), указан ввод человека (промпт),
        источник есть, но не попал в срез задачи (поиск и контекст).
        Прежняя формулировка «источник не относится к задаче» говорила за
        читателя, что мы проверили тему источника, — а мы её не проверяли.

        Args:
            source_id: Ссылка из ответа модели, возможно с якорем ``#pN``.

        Returns:
            Причина словами; пустая строка — ссылка годится.
        """
        clean, _ = split_source_id(source_id)
        label = clean.strip().lower().replace("ё", "е")
        if not label:
            return REASON_NO_LINK
        if human_input_stage(source_id) is not None:
            return REASON_HUMAN_INPUT
        if "/" not in label and "." not in label:
            # Имя поля ответа («verification», «query») вместо пути.
            return REASON_NO_LINK
        if self.is_relevant(clean):
            return ""
        if self._resolve(clean) or self._in_project(clean):
            return REASON_OUTSIDE_SLICE
        return REASON_UNKNOWN_SOURCE

    def absence_kind(self, source_id: str) -> str:
        """Машинный вид отказа — для отчёта и замера.

        Args:
            source_id: Ссылка из ответа модели.

        Returns:
            ``source_not_named`` / ``human_input`` / ``outside_slice`` /
            ``source_unknown`` либо пустая строка, если ссылка годится.
        """
        return ABSENCE_KINDS.get(self.absence_reason(source_id), "")

    @staticmethod
    def _in_project(clean_id: str) -> bool:
        """Есть ли такой материал в проекте — пусть и вне среза задачи.

        Читается только для ссылок, которых нет ни в срезе, ни в корпусе:
        на обычном пути эта ветка не выполняется.

        Args:
            clean_id: Идентификатор без якоря.

        Returns:
            Материал существует в проекте.
        """
        from pf.materials import load_materials

        tail = clean_id.rsplit("/", 1)[-1]
        stem = tail.rsplit(".", 1)[0]
        for material in load_materials():
            item_tail = material.source_id.rsplit("/", 1)[-1]
            if (
                material.source_id == clean_id
                or item_tail == tail
                or item_tail.rsplit(".", 1)[0] == stem
            ):
                return True
        return False

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
            Основание со статусом подтверждения. У сырой выгрузки статус не
            выше «требует проверки»: дословность проверена, происхождение — нет.
        """
        support = self._verify_claim(claim_text, source_id, paragraph, quote)
        return _limit_knowledge_source(support, source_id or support.source_id)

    def _verify_claim(
        self,
        claim_text: str,
        source_id: str = "",
        paragraph: int | None = None,
        quote: str = "",
    ) -> Support:
        """Проверить утверждение, не ограничивая статус происхождением.

        Отдельный метод, потому что ограничение статуса — одно правило для
        цитаты и расчёта, и жить оно должно в одном месте, а не в каждой
        ветке проверки.

        Args:
            claim_text: Текст утверждения.
            source_id: Идентификатор источника (может быть со склеенным ``#pN``).
            paragraph: Номер абзаца.
            quote: Дословная цитата, если модель её прислала.

        Returns:
            Основание со статусом подтверждения.
        """
        # Строка расчёта, скопированная из блока РАСЧЁТЫ, несёт собственный
        # якорь: `имя: значение [путь#rowN] (формула)`. Путь и номер строки
        # берутся из цитаты — модель не обязана повторять их в ссылке, — а
        # разметка блока текстом источника не считается. Без этого верное
        # основание отклонялось всегда: разметку искали в материале, где её
        # нет и быть не может (`demo-v12`, T-34).
        calc_line = _calc_line_anchor(quote)
        if calc_line is not None:
            quote, quoted_id, quoted_row = calc_line
            if source_id:
                given_id, given_row = split_source_id(source_id)
                if not same_material(given_id, quoted_id) or (
                    "#row" in source_id and given_row != quoted_row
                ):
                    # Ссылка и разметка указывают в разные места. Выбирать
                    # молча нельзя: подтверждение привязалось бы не к тому
                    # основанию, которое назвал человек.
                    return Support(
                        quote=quote,
                        source_id=source_id,
                        status=SupportStatus.NO_DATA,
                        detail=(
                            "ссылка и строка расчёта расходятся: в ссылке "
                            f"«{given_id}#row{given_row}», в цитате "
                            f"«{quoted_id}#row{quoted_row}»"
                        ),
                    )
            source_id = f"{quoted_id}#row{quoted_row}"
            paragraph = quoted_row

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
                    # Строку расчёта блок печатает в `%.4g`: 78,7913 в таблице
                    # показано как 78,79. Для неё сверка идёт с допуском
                    # показанной точности, для обычной цитаты — точно.
                    tolerance = (
                        _display_tolerance(number)
                        if calc_line is not None and number is not None
                        else None
                    )
                    if number is not None and _row_has_number(
                        row_text, number, tolerance=tolerance
                    ):
                        # Строка объявляет свой показатель, и это утверждение
                        # сильнее цитаты. Без сверки «удержание 87 %»
                        # подтверждалось строкой «абонентская база: 87»:
                        # число в строке есть, а показатель другой — и
                        # основание выглядело подтверждённым.
                        indicator = _row_indicator_at(article, row)
                        if indicator and not _name_matches_row(claim_text, indicator):
                            return Support(
                                quote=quote,
                                source_id=source_id,
                                status=SupportStatus.NO_DATA,
                                paragraph_index=row,
                                detail=(
                                    f"в строке показатель «{indicator}», а утверждение "
                                    "говорит о другом: число в строке есть, но "
                                    "подтверждает оно другой показатель"
                                ),
                            )
                        return Support(
                            quote=quote,
                            source_id=source_id,
                            status=SupportStatus.CONFIRMED,
                            paragraph_index=row,
                            detail="значение из расчёта найдено в строке таблицы",
                        )
                    # Число могло быть посчитано, а не взято из строки:
                    # «изменение за период: 0 %» в таблице не лежит. Это
                    # не выдумка, но и не цитата — называем причину, чтобы
                    # модель переложила величину в metrics.
                    # Агрегат мог быть посчитан агентом: тогда он
                    # подтверждается расчётом, а не строкой.
                    aggregate = _aggregate_by_name(self.metrics, claim_text, quote)
                    if aggregate is not None:
                        return aggregate
                    if _looks_computed(quote):
                        return Support(
                            quote=quote,
                            source_id=source_id,
                            status=SupportStatus.NEEDS_REVIEW,
                            paragraph_index=row,
                            detail=(
                                "значение посчитано, а не процитировано — "
                                "оформи как метрику в metrics: с формулой и входами"
                            ),
                        )
                    return Support(
                        quote=quote,
                        source_id=source_id,
                        status=SupportStatus.NO_DATA,
                        detail="цитата не найдена в строке таблицы",
                    )
        else:
            candidates = list(self.articles)

        # Ссылки без якоря строки — самый частый вид основания вывода: модель
        # называет файл и цитирует сводку из блока РАСЧЁТЫ. Такой цитаты в
        # тексте материала нет и быть не может, поэтому агрегат проверяется
        # до поиска по статьям (иначе основание отклоняется всегда).
        aggregate = _aggregate_by_name(self.metrics, claim_text, quote)
        if aggregate is not None:
            return aggregate

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


def _metric_fields(
    metric: Any,
) -> tuple[str, float, str, int | None, str, str, dict[str, Any]] | None:
    """Снять поля метрики с любого носителя provenance.

    Метрика приходит из ответа агента по схеме ``pf.schemas.MetricRef``,
    а внутри расчётов живёт ``pf.analysis.Metric``. Это разные классы
    с одинаковыми полями, поэтому проверка по ``isinstance`` отбрасывала
    все метрики из ответа агента, а тест, проверявший внутренний класс,
    оставался зелёным.

    Поэтому здесь не тип, а поля: метрика проверяется по тем данным,
    которые в ней действительно есть.

    Args:
        metric: Объект метрики.

    Returns:
        Кортеж ``(name, value, source_id, row, unit, formula, inputs)``
        либо ``None``, если нужных полей нет.
    """
    name = getattr(metric, "name", None)
    value = getattr(metric, "value", None)
    source_id = getattr(metric, "source_id", None)
    if name is None or value is None or source_id is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    row = getattr(metric, "row", None)
    if row is not None:
        try:
            row = int(row)
        except (TypeError, ValueError):
            row = None
    inputs = getattr(metric, "inputs", None)
    return (
        str(name),
        number,
        str(source_id),
        row,
        str(getattr(metric, "unit", "") or ""),
        str(getattr(metric, "formula", "") or ""),
        dict(inputs) if isinstance(inputs, dict) else {},
    )


def _format_metric(value: float, unit: str) -> str:
    """Значение с единицей измерения.

    Args:
        value: Значение метрики.
        unit: Единица измерения.

    Returns:
        Текст значения.
    """
    if not unit:
        return f"{value:.4g}"
    return f"{value:.4g} {unit}"


def _aggregate_from_inputs(
    value: float,
    inputs: dict[str, Any],
    formula: str,
) -> float | None:
    """Пересчитать агрегатную метрику по заявленным входам.

    Среднее, медиана, размах и изменение — величины, которых нет ни в
    одной строке таблицы: они считаются по нескольким наблюдениям.
    Проверка по одной строке их справедливо отвергала, и в
    демо-прогоне так отбрасывались все сводные метрики.

    Пересчёт идёт по формуле: формула в метрике не выдумывается, она
    приходит из блока РАСЧЁТЫ, где её составил код.

    Args:
        value: Заявленное значение метрики.
        inputs: Входные значения.
        formula: Формула расчёта.

    Returns:
        Пересчитанное значение либо ``None``, если пересчёт невозможен.
    """
    from pf.analysis import change_percent, mean, median, parse_number

    numbers = [
        parsed
        for parsed in (parse_number(str(item)) for item in _input_numbers(inputs))
        if parsed is not None
    ]
    if not numbers:
        return None

    # Порядок проверок важен: у каждой формулы своё правило, и правило
    # изменения общее («последнее» и «первое») не должно перехватывать
    # размах или среднее.
    text = formula.lower()
    if "размах" in text or "максимум" in text:
        return max(numbers) - min(numbers)
    if "медиан" in text:
        return median(numbers)
    # «Сумма / количество» называет входы явно: sum и count. Складывать
    # их как наблюдения нельзя — 87 и 1 дали бы среднее 44 вместо 87.
    total = _named_input(inputs, ("sum", "сумма"))
    count = _named_input(inputs, ("count", "количество"))
    if "сумм" in text and total is not None and count:
        return total / count
    if "средн" in text or "сумма" in text:
        return mean(numbers)
    if "изменение" in text or "процент" in text or "первое" in text:
        if len(numbers) >= 2:
            return change_percent(numbers[0], numbers[-1])
        return None
    if len(numbers) >= 2:
        return change_percent(numbers[0], numbers[-1])
    return None


def _named_input(inputs: dict[str, Any], names: tuple[str, ...]) -> float | None:
    """Число из входа с одним из ожидаемых имён.

    Входы называют по-разному: `{"sum": 87, "count": 1}` или
    `{"сумма": 87, "количество": 1}`. Имя — часть смысла, поэтому
    ищем по ключу, а не по порядку.

    Args:
        inputs: Входные значения метрики.
        names: Допустимые имена в нижнем регистре.

    Returns:
        Число либо ``None``, если входа с таким именем нет.
    """
    from pf.analysis import parse_number

    for key, raw in inputs.items():
        if str(key).strip().lower() not in names:
            continue
        if isinstance(raw, (list, tuple)):
            continue
        parsed = parse_number(str(raw))
        if parsed is not None:
            return parsed
    return None


def _input_numbers(inputs: dict[str, Any]) -> list[Any]:
    """Числовые значения из входов метрики.

    Входы бывают разными: скаляр («база»: «80»), список
    («значения»: [87, 82]) или строка со списком. Последнее — не
    редкость: блок РАСЧЁТЫ печатает наблюдения через запятую, модель
    копирует строку как есть, и без разбора весь ряд выглядел одним
    нечисловым значением. В `demo-v10` из-за этого отбрасывались все
    сводные метрики, и вывод оставался «не подкреплён» при верных данных.

    Args:
        inputs: Входные значения метрики.

    Returns:
        Список значений — числа развёрнуты из списков и строк.
    """
    result: list[Any] = []
    for value in inputs.values():
        if isinstance(value, (list, tuple)):
            result.extend(value)
        elif isinstance(value, str):
            result.extend(_split_numbers(value))
        else:
            result.append(value)
    return result


def _split_numbers(value: str) -> list[str]:
    """Разделить строку со списком чисел на отдельные значения.

    Разделители — только вместе с пробелом («, » и «; »): запятая без
    пробела в русских данных означает десятичный разделитель, и резать
    по ней нельзя — «78,5» иначе стало бы двумя числами.

    Args:
        value: Строка со значением или их списком.

    Returns:
        Части строки, если каждая из них — число; иначе исходная строка.
    """
    from pf.analysis import parse_number

    text = value.strip()
    if not text:
        return []
    parts = [part.strip() for part in re.split(r";\s*|,\s+", text) if part.strip()]
    if len(parts) >= 2 and all(parse_number(part) is not None for part in parts):
        return parts
    return [text]


def _aggregate_by_name(
    metrics: tuple[Any, ...] | list[Any],
    claim_text: str,
    quote: str,
) -> Support | None:
    """Подтвердить агрегат в основании по расчёту агента.

    Среднее, медиана, размах и изменение не лежат ни в одной строке
    таблицы: их значение получается пересчётом по наблюдениям. В
    `demo-v10` модель цитировала средние из блока РАСЧЁТЫ как дословные
    строки источника, проверка искала фразу в тексте материала, не
    находила — и оба основания вывода отклонялись при верных данных.

    Агрегат ищется **по имени**, а не по совпадению числа: имя расчёта
    состоит из показателя и вида сводки («Абонентская база оператора А:
    среднее»), и оба куска должны быть названы в основании. Затем значение
    сверяется со значением расчёта и пересчитывается по его входам.

    Args:
        metrics: Расчёты агента по материалам.
        claim_text: Текст утверждения.
        quote: Цитата из ответа модели.

    Returns:
        Подтверждённое основание либо ``None``, если среди расчётов нет
        подходящего. ``None`` — не отказ, а «это правило не подошло»:
        проверка продолжается обычным путём.
    """
    if not metrics:
        return None
    text = f"{claim_text} {quote}".strip()
    numbers = _numbers_in(quote) or _numbers_in(claim_text)
    if not numbers:
        return None
    absolute = settings.metric_abs_tolerance
    relative = settings.metric_rel_tolerance
    # Последнее число идёт первым: в сводке «ARPU ... : среднее: 322.2»
    # величина стоит в конце, а числа показателя — в начале.
    for number in reversed(numbers):
        for metric in metrics:
            if not getattr(metric, "is_aggregate", False):
                continue
            name = str(getattr(metric, "name", ""))
            base, _, summary = name.rpartition(":")
            if not base.strip() or not summary.strip():
                continue
            if not (_name_matches_row(base, text) and _name_matches_row(summary, text)):
                continue
            value = float(getattr(metric, "value", 0.0))
            allowed = max(
                _display_tolerance(value),
                number_tolerance(value, abs_tolerance=absolute, rel_tolerance=relative),
            )
            if not _matches_any(number, [value], allowed):
                continue
            inputs = dict(getattr(metric, "inputs", {}) or {})
            formula = str(getattr(metric, "formula", ""))
            if _is_identity_calculation(inputs, formula):
                continue
            recomputed = _aggregate_from_inputs(value, inputs, formula)
            if recomputed is None or not _matches_any(value, [recomputed], allowed):
                continue
            support = Support(
                quote=quote or claim_text[:400],
                source_id=str(getattr(metric, "citation", "")),
                status=SupportStatus.CONFIRMED,
                detail=(
                    "агрегат совпал с расчётом агента: значение пересчитано "
                    f"по входам метрики «{name}»"
                ),
            )
            # Статус задаёт происхождение расчёта, а не форма ссылки в
            # ответе: среднее по сырой выгрузке подтверждено быть не может.
            return _limit_knowledge_source(support, support.source_id)
    return None


#: Колонки таблицы, которые называют показатель: что строка измеряет.
_INDICATOR_COLUMNS: frozenset[str] = frozenset(
    {"показатель", "показатели", "метрика", "metric", "indicator"}
)

#: Колонки таблицы, которые называют субъекта: о ком строка.
_SUBJECT_COLUMNS: frozenset[str] = frozenset(
    {"оператор", "операторы", "компания", "игрок", "operator", "company", "player"}
)

#: Колонки провенанса: откуда взято. Это не утверждение строки, поэтому в
#: оценку она не входит. Пока входила, порядок строк задавала длина URL:
#: в прогоне `demo-v9` «МТС Беларусь 2026» стояла первой просто потому,
#: что адрес её статьи в Википедии короче остальных.
_PROVENANCE_COLUMNS: frozenset[str] = frozenset(
    {"источник", "источники", "source", "sources", "url", "ссылка", "ссылки", "link"}
)

#: Балл строки, попавшей в срез по предметной области, а не по словам
#: запроса. Меньше любого совпадения: строка, отвечающая на вопрос,
#: всегда важнее строки, которая просто про ту же отрасль.
_DOMAIN_FALLBACK_SCORE = 0.25

#: Веса полей строки при отборе. Показатель — главное, что строка о себе
#: говорит; субъект уточняет, о ком она; остальное содержание — контекст.
_INDICATOR_WEIGHT = 2.0
_SUBJECT_WEIGHT = 1.0
_CONTENT_WEIGHT = 0.5


def _column_name(key: Any) -> str:
    """Имя колонки в сравнимом виде: «Показатель » и «показатель» — одно.

    Args:
        key: Заголовок колонки.

    Returns:
        Имя без лишних пробелов и подчёркиваний, в нижнем регистре.
    """
    return re.sub(r"[\s_]+", " ", str(key).strip().lower())


def _row_indicator(row: dict[str, Any]) -> str:
    """Показатель, который объявляет сама строка таблицы.

    Колонка «показатель» — это утверждение источника о том, что за число в
    строке. Если её нет, строка говорит только «значение», и сравнивать
    название метрики не с чем: имя показателя берётся из заголовка файла.

    Args:
        row: Строка таблицы.

    Returns:
        Значение колонки-показателя либо пустая строка.
    """
    for key, value in row.items():
        if _column_name(key) in _INDICATOR_COLUMNS:
            return str(value)
    return ""


def _row_subject(row: dict[str, Any]) -> str:
    """Субъект строки: о ком она.

    Args:
        row: Строка таблицы.

    Returns:
        Значение колонки-субъекта либо пустая строка.
    """
    for key, value in row.items():
        if _column_name(key) in _SUBJECT_COLUMNS:
            return str(value)
    return ""


def _row_content(row: dict[str, Any]) -> str:
    """Содержание строки без провенанса.

    Args:
        row: Строка таблицы.

    Returns:
        Пары «колонка: значение» через запятую.
    """
    parts = [
        f"{key}: {value}"
        for key, value in row.items()
        if _column_name(key) not in _PROVENANCE_COLUMNS
    ]
    return ", ".join(parts)


def _row_score(row: dict[str, Any], query_terms: set[str]) -> float:
    """Балл строки таблицы: по её показателю, а не по словам вообще.

    Три уровня, и они различаются намеренно:

    1. **Совпадение по показателю** — строка отвечает на заданный вопрос.
    2. **Совпадение по субъекту и содержанию** — строка про того же
       оператора или год, но показатель другой.
    3. **Предметная область** — совпадений нет, но показатель строки
       называет объект отрасли («абонентская база»). Такая строка остаётся
       контекстом, и её балл фиксирован: длиной текста он не меряется.

    Строка, у которой нет ни одного из трёх признаков, в срез не идёт.
    Так «доля на рынке спутникового ТВ» перестала попадать в окно: это
    другой показатель и другая отрасль, а не уточнение к вопросу.

    Args:
        row: Строка таблицы.
        query_terms: Значимые слова запроса.

    Returns:
        Балл релевантности; ноль — строку не брать.
    """
    indicator = _row_indicator(row)
    subject = _row_subject(row)
    content = _row_content(row)
    tokens = tokenize(content)
    if not tokens:
        return 0.0

    indicator_tokens = tokenize(indicator)
    subject_tokens = tokenize(subject)
    weighted = 0.0
    for word in indicator_tokens:
        if word in query_terms:
            weighted += _INDICATOR_WEIGHT
    for word in subject_tokens:
        if word in query_terms:
            weighted += _SUBJECT_WEIGHT
    for word in tokens:
        if word in query_terms and word not in indicator_tokens and word not in subject_tokens:
            weighted += _CONTENT_WEIGHT
    if weighted:
        return weighted / math.sqrt(len(tokens))

    anchors = {lemmatize(word) for word in tokenize(indicator or content)}
    if anchors & strong_domain_lemmas():
        return _DOMAIN_FALLBACK_SCORE
    return 0.0


def _name_matches_row(name: str, row_text: str) -> bool:
    """Имя показателя относится к тому, что стоит в строке.

    Сначала сравниваются слова как есть, затем — основы: «число
    абонентов» и показатель «абонентская база» говорят об одном, хотя
    словоформы разные.

    Args:
        name: Название метрики или утверждение.
        row_text: Текст строки таблицы.

    Returns:
        Хотя бы одно значимое слово названия встречается в строке.
    """
    words = _significant_words(name)
    if not words:
        return True
    if words & _significant_words(row_text):
        return True
    return bool(_stems(name) & _stems(row_text))


def _is_identity_calculation(inputs: dict[str, Any], formula: str) -> bool:
    """Формула вырождается в копирование значения.

    «Сумма / количество» по одному наблюдению (87 / 1) и «среднее» по
    одному значению — это не расчёт: результат равен входу. Такой формулой
    модель закрывает требование «покажи, как считал», и переписанное из
    строки число выглядит посчитанным.

    Args:
        inputs: Входные значения метрики.
        formula: Формула расчёта.

    Returns:
        Формула есть, но считать по ней нечего.
    """
    text = formula.lower()
    if not any(marker in text for marker in ("сумм", "средн", "колич")):
        return False
    count = _named_input(inputs, ("count", "количество"))
    return count == 1


def verify_metric(
    metric: Any,
    materials: list[Material],
    *,
    tolerance: float | None = None,
    rel_tolerance: float | None = None,
) -> Support:
    """Проверить расчётную метрику, ограничив статус происхождением источника.

    Args:
        metric: Метрика с provenance.
        materials: Материалы проекта.
        tolerance: Абсолютный допуск. По умолчанию — из настроек.
        rel_tolerance: Относительный допуск. По умолчанию — из настроек.

    Returns:
        Результат проверки. У сырой выгрузки статус не выше «требует
        проверки»: пересчёт верен, а происхождение данных — нет.
    """
    support = _verify_metric(
        metric,
        materials,
        tolerance=tolerance,
        rel_tolerance=rel_tolerance,
    )
    fields = _metric_fields(metric)
    return _limit_knowledge_source(support, fields[2] if fields else support.source_id)


def _verify_metric(
    metric: Any,
    materials: list[Material],
    *,
    tolerance: float | None = None,
    rel_tolerance: float | None = None,
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
        tolerance: Абсолютный допуск. По умолчанию — из настроек:
            «77,3» подтверждает «77,26», потому что это округление, а не
            подмена числа.
        rel_tolerance: Относительный допуск. По умолчанию — из настроек.

    Returns:
        Результат проверки: подтверждено, требует проверки или нет
        данных.
    """
    from pf.analysis import parse_number

    fields = _metric_fields(metric)
    if fields is None:
        return Support(
            quote="",
            source_id="",
            status=SupportStatus.NO_DATA,
            detail="метрика не имеет структуры с provenance",
        )
    name, value, source_id, row, unit, formula, inputs = fields

    if not source_id:
        return Support(
            quote=name,
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail="у метрики нет источника — пересчитать нельзя",
        )

    # Якорь живёт внутри source_id: модель копирует его из блока
    # РАСЧЁТЫ вместе с путём. Материал лежит под чистым путём, поэтому
    # искать по строке с якорем нельзя — якорь отделяется, искомым
    # остаётся путь.
    clean_id, anchor_row = split_source_id(source_id)
    material = _find_material(clean_id, materials)
    if material is None:
        return Support(
            quote=name,
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail=f"источник {source_id} не найден среди материалов",
        )

    # Якорь и поле row должны говорить о одной строке. Если говорят о
    # разных, непонятно, какая строка имелась в виду, и подтверждать
    # такую метрику нельзя: проверка шла бы по счастливой строке.
    if anchor_row is not None and row is not None and anchor_row != row:
        return Support(
            quote=name,
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail=(
                f"якорь {source_id} указывает на строку {anchor_row}, "
                f"а в метрике указана строка {row}"
            ),
        )
    if row is None:
        row = anchor_row

    # 1. Значение из указанной строки источника.
    row_values: list[float] = []
    if row is not None and 0 <= row < len(material.rows):
        for cell in material.rows[row].values():
            parsed = parse_number(str(cell))
            if parsed is not None:
                row_values.append(parsed)
    else:
        for record in material.rows:
            for cell in record.values():
                parsed = parse_number(str(cell))
                if parsed is not None:
                    row_values.append(parsed)

    formatted = _format_metric(value, unit)
    absolute = settings.metric_abs_tolerance if tolerance is None else tolerance
    relative = settings.metric_rel_tolerance if rel_tolerance is None else rel_tolerance
    # Допуск — максимум из трёх: точность показа числа, округление до
    # десятой доли и относительная погрешность. Для малых величин
    # абсолютная часть не включается: 0,1 — половина от 0,2.
    allowed = max(
        _display_tolerance(value),
        number_tolerance(value, abs_tolerance=absolute, rel_tolerance=relative),
    )

    # 2. Пересчёт по заявленным входам. Агрегат (среднее, медиана,
    # размах, изменение) вычислен по нескольким наблюдениям, и его
    # значения нет ни в одной строке таблицы — сверять его со строкой
    # бессмысленно. Поэтому пересчёт идёт первым: если значение
    # выводится из входов, основание подтверждено.
    recomputed = _aggregate_from_inputs(value, inputs, formula)
    if recomputed is not None:
        # Формула-тождество: 87 / 1 = 87. Число переписано из строки, а не
        # посчитано, — и подтверждать его как расчёт нельзя.
        if _is_identity_calculation(inputs, formula):
            return Support(
                quote=formatted,
                source_id=source_id,
                status=SupportStatus.NO_DATA,
                detail=(
                    "формула — тождество: значение не пересчитано, а переписано из строки источника"
                ),
            )
        if not _matches_any(value, [recomputed], allowed):
            return Support(
                quote=formatted,
                source_id=source_id,
                status=SupportStatus.NO_DATA,
                detail=(f"пересчёт по входам даёт {recomputed:.6g}, а заявлено {formatted}"),
            )
        return Support(
            quote=formatted,
            source_id=source_id,
            status=SupportStatus.CONFIRMED,
            paragraph_index=row,
            detail="метрика подтверждена пересчётом по заявленным входам",
        )

    # 1. Значение из указанной строки источника.
    if row is not None and 0 <= row < len(material.rows):
        # Строка сама говорит, что за показатель в ней, — и это утверждение
        # сильнее названия метрики. Если строка называет абонентскую базу,
        # а метрика — удержание, число в строке есть, а показатель другой.
        indicator = _row_indicator(material.rows[row])
        if indicator and not _name_matches_row(name, indicator):
            return Support(
                quote=formatted,
                source_id=source_id,
                status=SupportStatus.NO_DATA,
                detail=(
                    "имя показателя не совпало с колонкой строки: в строке "
                    f"показатель «{indicator}», а метрика говорит о другом"
                ),
            )
    if row_values and not _matches_any(value, row_values, allowed):
        return Support(
            quote=formatted,
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail=(
                f"значение {formatted} не совпадает со строкой "
                f"{row if row is not None else '?'} источника"
            ),
        )

    if inputs:
        numbers = [
            parsed
            for parsed in (parse_number(str(v)) for v in inputs.values())
            if parsed is not None
        ]
        if len(numbers) >= 2:
            base, new = numbers[0], numbers[1]
            recomputed = (new - base) / abs(base) * 100 if base else 0.0
            if unit == "%" and not _matches_any(recomputed, [value], allowed):
                return Support(
                    quote=formatted,
                    source_id=source_id,
                    status=SupportStatus.NO_DATA,
                    detail=(f"пересчёт по входам даёт {recomputed:.4g} %, а заявлено {formatted}"),
                )

    return Support(
        quote=formatted,
        source_id=source_id,
        status=SupportStatus.CONFIRMED,
        paragraph_index=row,
        detail=(
            "метрика пересчитана по строке источника"
            if row is not None
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


def _row_indicator_at(article: Article, row: int) -> str:
    """Показатель строки таблицы по её номеру.

    ``_article_rows`` отдаёт строки уже текстом, поэтому колонку-показатель
    из них не достать. А проверке числа в строке она нужна: строка сама
    объявляет, что за показатель в ней, и утверждение о другом показателе
    подтверждать нельзя — иначе 87 млн абонентов подтверждают «87 %
    удержания».

    Args:
        article: Материал проекта.
        row: Номер строки (нумерация с нуля).

    Returns:
        Показатель строки либо пустая строка, если колонки нет.
    """
    if article.section != "project":
        return ""
    from pf.materials import load_materials

    for material in load_materials():
        if material.source_id != article.source_id:
            continue
        if 0 <= row < len(material.rows):
            return _row_indicator(material.rows[row])
        return ""
    return ""


def indicator_of_citation(citation: str, materials: list[Material] | None = None) -> str:
    """Показатель строки, на которую ссылается расчёт.

    Расчёт несёт номер строки (``#rowN``), а не название показателя: имя
    метрики — это заголовок материала, и оно не говорит, что именно
    посчитано. Показатель берётся из самой строки — тем же правилом, что
    в проверке оснований. Иначе «87» из строки «абонентская база»
    подтверждало бы «удержание 87 %» уже на уровне расчёта.

    Args:
        citation: Ссылка вида ``project/data/file.csv#row0``.
        materials: Материалы проекта. По умолчанию — из каталога ``project/``.

    Returns:
        Показатель строки либо пустая строка, если строки или колонки нет.
    """
    clean_id, row = split_source_id(citation)
    if row is None or not clean_id:
        return ""
    if materials is None:
        from pf.materials import load_materials

        materials = load_materials()
    material = _find_material(clean_id, materials)
    if material is None or not 0 <= row < len(material.rows):
        return ""
    return _row_indicator(material.rows[row])


def _find_material(clean_id: str, materials: list[Material]) -> Material | None:
    """Найти материал по идентификатору, допуская короткое имя файла.

    Модель часто присылает «numbers.csv» вместо «project/numbers.csv».
    Для фактов это учтено в ``SourceBundle._resolve``, а для метрик —
    нет: проверка отклоняла верный расчёт с «источник не найден среди
    материалов». В прогоне `demo-v5` так потерялись все посчитанные
    метрики.

    Args:
        clean_id: Идентификатор источника без якоря.
        materials: Материалы проекта.

    Returns:
        Материал либо ``None``, если он не найден.
    """
    found = next((item for item in materials if item.source_id == clean_id), None)
    if found is not None:
        return found
    tail = clean_id.rsplit("/", 1)[-1]
    stem = tail.rsplit(".", 1)[0]
    return next(
        (
            item
            for item in materials
            if item.source_id.rsplit("/", 1)[-1] == tail
            or item.source_id.rsplit("/", 1)[-1].rsplit(".", 1)[0] == stem
        ),
        None,
    )


def _calc_line_anchor(quote: str) -> tuple[str, str, int] | None:
    """Разобрать строку расчёта, скопированную из блока РАСЧЁТЫ.

    Блок печатает расчёт как ``- {имя}: {значение} [{путь}#rowN] ({формула})``
    (``Workflow._metric_line``). Модель копирует эту строку в основание
    целиком, и до этой правки проверка искала разметку блока в тексте
    материала: ``[...]`` и ``(формула)`` там, разумеется, не встречаются, и
    верное основание отклонялось. В прогоне `demo-v12` так были потеряны
    скопированный ARPU (302) и ключевая ставка (78.79).

    Разметка — подпись блока, а не текст источника: якорь берётся из неё, а
    сравнивается с материалом только ``body``.

    Args:
        quote: Цитата из ответа модели.

    Returns:
        Тройка ``(body, путь, номер строки)`` либо ``None``, если цитата не
        размечена как строка расчёта.
    """
    match = _CALC_LINE_RE.match(quote.strip())
    if match is None:
        return None
    body = match.group("body").strip()
    source = match.group("source").strip()
    if not body or not source:
        return None
    return body, source, int(match.group("row"))


def same_material(left: str, right: str) -> bool:
    """Указывают ли два идентификатора на один материал.

    Модель называет файл то полным путём, то коротким именем
    («telecom_arpu.csv» вместо «project/data/telecom_arpu.csv»). Для сверки
    ссылки с якорем цитаты достаточно совпадения имени файла без расширения.

    Правило публичное: им пользуется и сверка чисел плана
    (``Workflow._check_plan_numbers``), когда проверяет, что число названо
    вместе со ссылкой на свою строку расчёта. Вторая копия сравнения имён
    разошлась бы с первой.

    Args:
        left: Первый идентификатор.
        right: Второй идентификатор.

    Returns:
        Один и тот же материал.
    """
    left = left.strip().lower()
    right = right.strip().lower()
    if not left or not right:
        return False
    left_tail = left.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    right_tail = right.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return left_tail == right_tail


def _looks_computed(quote: str) -> bool:
    """Похожа ли цитата на посчитанную величину, а не на текст источника.

    Args:
        quote: Цитата из ответа модели.

    Returns:
        В цитате есть слово-признак расчёта.
    """
    return bool(_COMPUTED_MARKERS.search(quote))


def _numbers_in(text: str) -> list[float]:
    """Все числа текста в порядке появления.

    Args:
        text: Текст.

    Returns:
        Числа, разобранные с учётом десятичной запятой.
    """
    from pf.analysis import parse_number

    found: list[float] = []
    for token in re.findall(r"-?\d+(?:[.,]\d+)?", text):
        value = parse_number(token)
        if value is not None:
            found.append(value)
    return found


def _first_number(text: str) -> float | None:
    """Первое число в тексте.

    Args:
        text: Текст.

    Returns:
        Число либо ``None``.
    """
    numbers = _numbers_in(text)
    return numbers[0] if numbers else None


def _row_has_number(
    row_text: str,
    number: float,
    *,
    tolerance: float | None = None,
) -> bool:
    """Есть ли число в строке таблицы.

    Args:
        row_text: Текст строки.
        number: Искомое число.
        tolerance: Допуск расхождения. ``None`` — точное сравнение:
            обычная цитата обязана называть то же число. Допуск передаётся
            для строки расчёта, где значение показано в ``%.4g`` и поэтому
            заведомо округлено.

    Returns:
        Есть ли число.
    """
    from pf.analysis import parse_number

    for token in re.findall(r"-?\d+(?:[.,]\d+)?", row_text):
        value = parse_number(token)
        if value is None:
            continue
        limit = 1e-9 if tolerance is None else max(tolerance, 1e-9)
        if abs(value - number) <= limit:
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


def _display_tolerance(value: float) -> float:
    """Допуск, соответствующий точности показа числа.

    Блок РАСЧЁТЫ показывает значения в формате ``%.4g``: из 78,7913
    получается 78,79. Модель копирует в метрику именно округлённое
    число, поэтому сверка с точной строкой источника обязана принимать
    расхождение в пределах показанной точности. Иначе верный расчёт
    отбрасывается с текстом «значение не совпадает со строкой».

    Args:
        value: Проверяемое значение.

    Returns:
        Допуск для сверки.
    """
    magnitude = abs(value)
    if magnitude == 0:
        return 1e-6
    # 4 значащие цифры: погрешность округления не превосходит половины
    # единицы последнего разряда.
    return max(1e-6, magnitude * 5e-4)


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
        domain_words = domain.domain_terms()
        weighted_hits = sum(
            2.0 if word in domain_words else 1.0 for word in words if word in query_terms
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
    metrics: Sequence[Any] | None = None,
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
        metrics: Расчёты агента по материалам — ими подтверждаются агрегаты
            в основаниях вывода.

    Returns:
        Набор источников с абзацами, пригодными для цитирования.
    """
    corpus = corpus or get_corpus()
    top_k = top_k or settings.source_top_k
    max_paragraphs = max_paragraphs or settings.source_max_paragraphs
    query_terms = set(tokenize(query))
    computed: tuple[Any, ...] = tuple(metrics or ())

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
        return SourceBundle(passages=passages, articles=articles, query=query, metrics=computed)

    if materials is None:
        from pf.materials import load_materials

        materials = load_materials()

    project_passages, project_articles = _material_passages(materials, query_terms, max_paragraphs)
    if not project_passages:
        return SourceBundle(passages=passages, articles=articles, query=query, metrics=computed)

    # Материалы проекта идут перед корпусом: они относятся к задаче
    # непосредственно, а корпус даёт рыночный контекст.
    return SourceBundle(
        passages=project_passages + passages,
        articles=project_articles + articles,
        query=query,
        metrics=computed,
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

    Строка таблицы оценивается иначе, чем абзац документа: по своим
    полям (показатель, субъект, остальное содержание), а не по тексту
    строки целиком. Провенанс в оценку не входит. Иначе порядок строк
    задавала их длина, то есть длина URL, и в окно попадали строки о
    другом показателе — «Дочерняя компания 2026» и «доля на рынке
    спутникового ТВ».

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

    rows_by_index = {index: row for index, row in enumerate(material.rows)}
    domain_words = domain.domain_terms()

    scored: list[tuple[float, int, str]] = []
    for passage in material.passages:
        text = passage.text
        row = rows_by_index.get(passage.index)
        if row is not None:
            row_score = _row_score(row, query_terms)
            if row_score > 0.0:
                scored.append((row_score, passage.index, text))
            continue
        words = tokenize(text)
        if not words:
            continue
        hits = sum(1 for word in words if word in query_terms)
        if hits == 0 and not same_domain:
            continue
        weighted = sum(
            2.0 if word in domain_words else 1.0 for word in words if word in query_terms
        )
        if hits == 0:
            # Совпадение по предметной области: вес даёт сам факт, что
            # материал про ту же область, что и запрос.
            weighted = 1.0
        score = weighted / (len(words) ** 0.5)
        scored.append((score, passage.index, text))

    # Ничьи разрешает порядок источника, а не номер строки: сортировка
    # кортежа по убыванию ставила последнюю строку таблицы первой.
    scored.sort(key=lambda item: (-item[0], item[1]))
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
