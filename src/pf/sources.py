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

#: Признаки посчитанной величины. В строке таблицы лежат исходные
#: значения, а «изменение за период» — результат расчёта: такого текста
#: в источнике нет. Отличает неверный канал оформления от выдумки.
_COMPUTED_MARKERS = re.compile(
    r"(изменени|средн|медиан|сумм|разниц|рост|падени|темп|доля|cagr|average|mean|delta)",
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
                    # Число могло быть посчитано, а не взято из строки:
                    # «изменение за период: 0 %» в таблице не лежит. Это
                    # не выдумка, но и не цитата — называем причину, чтобы
                    # модель переложила величину в metrics.
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

    Входы бывают разными: скаляр («база»: «80») или списком
    («значения»: [87, 82]). И то и другое надо увидеть числами.

    Args:
        inputs: Входные значения метрики.

    Returns:
        Список значений — числа развёрнуты из списков.
    """
    result: list[Any] = []
    for value in inputs.values():
        if isinstance(value, (list, tuple)):
            result.extend(value)
        else:
            result.append(value)
    return result


def _significant_words(text: str) -> set[str]:
    """Значимые слова текста: без скобок, цифр и коротких служебных.

    Скобки отбрасываются: «удержание клиентов (МТС, абонентская база)» —
    это метрика про удержание, а база в скобках лишь уточняет, откуда
    взято число. Если оставить скобки, подмена показателя не видна.

    Args:
        text: Название метрики или текст строки.

    Returns:
        Множество слов длиной от четырёх букв.
    """
    without_notes = re.sub(r"\([^)]*\)", " ", text)
    return set(re.findall(r"[a-zа-яё]{4,}", without_notes.lower().replace("ё", "е")))


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
        name = str(key).strip().lower()
        if name in {"показатель", "показатели", "метрика", "metric", "indicator"}:
            return str(value)
    return ""


def _name_matches_row(name: str, row_text: str) -> bool:
    """Имя показателя относится к тому, что стоит в строке.

    Args:
        name: Название метрики.
        row_text: Текст строки таблицы.

    Returns:
        Хотя бы одно значимое слово названия встречается в строке.
    """
    words = _significant_words(name)
    if not words:
        return True
    return bool(words & _significant_words(row_text))


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
    allowed = max(tolerance, _display_tolerance(value))

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


def _looks_computed(quote: str) -> bool:
    """Похожа ли цитата на посчитанную величину, а не на текст источника.

    Args:
        quote: Цитата из ответа модели.

    Returns:
        В цитате есть слово-признак расчёта.
    """
    return bool(_COMPUTED_MARKERS.search(quote))


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
