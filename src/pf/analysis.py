"""Расчёты по данным: числа вместо соображений.

Требование брифа к плану проверки: «ожидаемое изменение, с чем
сравниваем, срок наблюдения». Обещание «повышение на 5–10 %» без
baseline — не план, а мечта. Чтобы план был проверяемым, агент должен
**считать** по данным, а не полагаться на оценку модели.

Поэтому здесь всё устроено как в надёжном анализе:

- числа извлекаются из строк данных, а не из текста модели;
- каждая метрика несёт **provenance**: источник, номер строки, формулу
  и входные значения, — чтобы её можно было пересчитать;
- метрика без источника считается невоспроизводимой и в отчёт не идёт.

Результат расчёта — не текст, а структура: её можно показать в отчёте,
сложить с другими метриками и проверить тестом.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from pf.periods import key_of_row

__all__ = [
    "Metric",
    "cagr",
    "change_percent",
    "Series",
    "PERIOD_COLUMNS",
    "compute_from_rows",
    "group_series",
    "label_column_for",
    "mean",
    "median",
    "parse_number",
    "percentile",
    "summarize_groups",
    "summarize_series",
]


def parse_number(text: str) -> float | None:
    """Разобрать число из ячейки таблицы.

    В данных встречаются русские форматы: запятая как десятичный
    разделитель, неразрывный пробел, проценты, знак «минус» и
    тире. Нечисловое значение («н/д», «—») возвращает ``None``:
    пропуск данных не должен ломать расчёт.

    Args:
        text: Значение ячейки.

    Returns:
        Число либо ``None``, если значение не число.
    """
    cleaned = (
        text.replace(" ", "")
        .replace(" ", "")
        .replace("%", "")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .strip()
    )
    if not cleaned:
        return None
    # Запятая в русских данных — десятичный разделитель. Но если в
    # строке есть и запятая, и точка, это разделитель разрядов.
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")
    else:
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class Metric:
    """Расчётная метрика с provenance.

    Attributes:
        name: Название метрики.
        value: Значение.
        unit: Единица измерения.
        source_id: Идентификатор источника данных.
        row: Номер строки в источнике (нумерация с нуля).
        formula: Формула расчёта — словами или выражением.
        inputs: Входные значения, из которых получено значение.
        note: Пояснение для человека.
    """

    name: str
    value: float
    unit: str = ""
    source_id: str = ""
    row: int | None = None
    formula: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    @property
    def citation(self) -> str:
        """Ссылка на источник значения.

        Returns:
            Ссылка вида ``источник#rowN`` либо пустая строка.
        """
        if not self.source_id:
            return ""
        if self.row is None:
            return self.source_id
        return f"{self.source_id}#row{self.row}"

    @property
    def is_reproducible(self) -> bool:
        """Можно ли пересчитать метрику по её же данным.

        Метрика без источника — это утверждение, а не расчёт: пересчитать
        её нельзя, а значит, проверить тоже нельзя.
        """
        return bool(self.source_id)

    @property
    def is_aggregate(self) -> bool:
        """Агрегат ли это — величина, посчитанная по нескольким строкам.

        Среднее, медиана, размах и изменение вычисляются по нескольким
        наблюдениям, и их значения нет ни в одной строке таблицы. Такие
        метрики проверяются не сверкой со строкой, а пересчётом по
        входам — поэтому входы должны быть видны.

        Returns:
            True, если метрика агрегатная.
        """
        observations = self.inputs.get("наблюдения")
        if isinstance(observations, (list, tuple)):
            return True
        # Признак агрегата может стоять в названии («изменение за
        # период») или в формуле («сумма / количество»). Формула
        # изменения выглядит как «(последнее - первое) / |первое| * 100»
        # — слова «изменение» в ней нет, поэтому смотрим оба места.
        text = f"{self.name.lower()} {self.formula.lower()}"
        return any(
            marker in text for marker in ("средн", "медиан", "размах", "изменение", "процент")
        )

    @property
    def formatted(self) -> str:
        """Значение с единицей измерения."""
        if not self.unit:
            return f"{self.value:.4g}"
        return f"{self.value:.4g} {self.unit}"


def mean(values: list[float]) -> float:
    """Среднее арифметическое.

    Args:
        values: Значения.

    Returns:
        Среднее; ноль для пустого списка.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values: list[float]) -> float:
    """Медиана.

    Args:
        values: Значения.

    Returns:
        Медиана; ноль для пустого списка.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def percentile(values: list[float], p: float) -> float:
    """Перцентиль с линейной интерполяцией.

    Args:
        values: Значения.
        p: Перцентиль от 0 до 100.

    Returns:
        Значение перцентиля; ноль для пустого списка.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (p / 100)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def change_percent(base: float, new: float) -> float:
    """Изменение в процентах относительно базы.

    Args:
        base: Базовое значение.
        new: Новое значение.

    Returns:
        Изменение в процентах; ноль, если база нулевая.
    """
    if base == 0:
        return 0.0
    return (new - base) / abs(base) * 100


def cagr(base: float, new: float, *, years: float) -> float:
    """Среднегодовой темп изменения.

    Args:
        base: Значение на начало периода.
        new: Значение на конец периода.
        years: Число лет.

    Returns:
        Темп в процентах в год; ноль, если значения неположительны или
        период нулевой.
    """
    if base <= 0 or new <= 0 or years <= 0:
        return 0.0
    ratio: float = new / base
    exponent: float = 1 / years
    growth: float = ratio**exponent
    return (growth - 1) * 100


def compute_from_rows(
    rows: list[dict[str, str]],
    *,
    value_column: str,
    source_id: str,
    name: str,
    unit: str = "",
    label_column: str = "",
    row_indices: list[int] | None = None,
) -> list[Metric]:
    """Посчитать метрики по строкам таблицы.

    Для каждой числовой строки получается метрика со своим
    provenance. Нечисловые строки пропускаются.

    Args:
        rows: Строки таблицы.
        value_column: Колонка со значением.
        source_id: Идентификатор источника данных.
        name: Базовое название метрики.
        unit: Единица измерения.
        label_column: Колонка с подписью строки (период, категория).
        row_indices: Номера строк **исходной таблицы** для каждой строки
            из ``rows``. Нужен, когда ``rows`` — часть таблицы, а не вся
            таблица: иначе номер строки считается внутри переданного
            списка и указывает на чужую строку.

    Returns:
        Список метрик — по одной на числовую строку.
    """
    metrics: list[Metric] = []
    for index, row in enumerate(rows):
        raw = row.get(value_column, "")
        value = parse_number(str(raw))
        if value is None:
            continue
        label = str(row.get(label_column, "")) if label_column else ""
        full_name = f"{name}, {label}" if label else name
        table_index = row_indices[index] if row_indices else index
        metrics.append(
            Metric(
                name=full_name,
                value=value,
                unit=unit,
                source_id=source_id,
                row=table_index,
                formula=f"значение из колонки «{value_column}», строка {table_index + 1}",
                inputs={value_column: str(raw)},
            )
        )
    return metrics


@dataclass(frozen=True, slots=True)
class Series:
    """Ряд наблюдений одной величины во времени.

    Ряд несёт не только строки, но и номера этих строк в исходной
    таблице. Связь обязана быть вместе с рядом: если её передавать
    отдельно, её легко не передать — и метрика получит номер строки
    внутри ряда вместо номера строки таблицы. Тогда «82 млн» укажет на
    строку, где лежит 87, и проверка справедливо отвергнет верный
    расчёт.

    Attributes:
        rows: Строки ряда.
        row_indices: Номера этих строк в исходной таблице.
    """

    rows: list[dict[str, str]]
    row_indices: list[int]


#: Колонки, которые задают период, а не различие величин. Значения
#: «2017», «2018», «конец июля» — это точки одной величины во времени.
#: Такая колонка не может быть подписью ряда: иначе каждая дата
#: образовала бы свой ряд из одной строки, и изменение всегда было бы
#: нулевым.
PERIOD_COLUMNS: frozenset[str] = frozenset(
    {"период", "дата", "period", "date", "месяц", "квартал", "год", "time", "day"}
)


def label_column_for(columns: list[str], *, value_column: str) -> str:
    """Выбрать колонку, по которой ряд различается.

    Подпись ряда отвечает на вопрос «что за величина» — оператор,
    показатель, категория. Колонка с периодом отвечает на другой
    вопрос («когда»), и выбирать её подписью нельзя: тогда каждый
    период становится отдельным рядом из одной строки.

    Args:
        columns: Колонки таблицы.
        value_column: Колонка со значением.

    Returns:
        Имя колонки-подписи либо пустая строка, если подписи нет.
    """
    for name in columns:
        if name == value_column:
            continue
        if name.strip().lower() in PERIOD_COLUMNS:
            continue
        return name
    return ""


def group_series(
    rows: list[dict[str, str]],
    *,
    label_column: str,
    group_columns: tuple[str, ...] = (),
) -> list[Series]:
    """Разделить строки на ряды по подписи.

    Таблица может содержать разные операторы, показатели и годы.
    Считать изменение по всем строкам сразу бессмысленно: из 87 млн
    абонентов МТС и 5,7 млн абонентов МТС Беларусь получается
    «-16,67 %», что не означает ничего. Ряды разделяются по подписи,
    и изменение считается внутри каждого.

    Ряды разделяются и по показателю: у одного оператора бывают
    разные показатели (абонентская база — 87 млн, доля на рынке —
    6,2 %), и складывать их в один ряд тоже нельзя.

    Args:
        rows: Строки таблицы.
        label_column: Колонка с подписью ряда.
        group_columns: Дополнительные колонки, по которым ряд должен
            быть один и тот же.

    Returns:
        Список рядов — каждый со своим набором строк и номерами этих
        строк в исходной таблице.
    """
    groups: dict[tuple[str, ...], list[int]] = {}
    for table_index, row in enumerate(rows):
        label = str(row.get(label_column, ""))
        extra = tuple(str(row.get(column, "")) for column in group_columns)
        groups.setdefault((label, *extra), []).append(table_index)
    return [
        Series(rows=[rows[index] for index in indices], row_indices=list(indices))
        for indices in groups.values()
    ]


def summarize_groups(
    groups: list[Series],
    *,
    value_column: str,
    source_id: str,
    name: str,
    unit: str = "",
) -> list[Metric]:
    """Сводные метрики по каждому ряду отдельно.

    Args:
        groups: Ряды строк.
        value_column: Колонка со значением.
        source_id: Идентификатор источника данных.
        name: Базовое название метрики.
        unit: Единица измерения.
    Returns:
        Список метрик: по каждому ряду — изменение, среднее, медиана
        и размах.
    """
    summaries: list[Metric] = []
    for series in groups:
        # Порядок наблюдений важен: изменение считается от первого к
        # последнему. Если строки лежат в таблице не по времени, ряд
        # надо упорядочить по периоду, иначе «изменение» окажется
        # обратным. Порядок задаётся индексами, а не перестановкой
        # строк: так номер строки остаётся номером строки таблицы.
        order = sorted(
            range(len(series.rows)),
            key=lambda position: _period_key(series.rows[position]),
        )
        ordered = [series.rows[position] for position in order]
        indices = [series.row_indices[position] for position in order]
        metrics = compute_from_rows(
            ordered,
            value_column=value_column,
            source_id=source_id,
            name=name,
            unit=unit,
            row_indices=indices,
        )
        if not metrics:
            continue
        # Название ряда: оператор и показатель — иначе в отчёте
        # «изменение» без указания, что именно менялось.
        parts = [
            str(series.rows[0].get(column, ""))
            for column in ("оператор", "показатель")
            if series.rows[0].get(column)
        ]
        label = ", ".join(parts) if parts else name
        series_name = f"{name} ({label})" if label else name
        summaries.extend(summarize_series(metrics, name=series_name, unit=unit))
    return summaries


#: Разбор периода (год, квартал, дата, интервал) живёт в `pf.periods`:
#: то же правило нужно проверке оснований, а вторая его копия уже приводила к
#: тому, что «1 кв. 2017» и «2017-Q1» считались разными периодами (`D-34`).


def _period_key(row: dict[str, str]) -> tuple[int, float, str]:
    """Ключ упорядочивания строк по периоду.

    Периоды бывают разными: датами («13.01.2026»), годами («2018»),
    кварталами («2017-Q1», «1 кв. 2017», «I квартал 2017»), текстом
    («конец июля»). Для сравнения нужен числовой ключ, а текстовый
    остаётся последним — тогда порядок сохраняется хотя бы стабильным.

    Квартал живёт **в шкале года**: ключ — ``год + квартал / 5``. Поэтому
    год идёт раньше своих кварталов, четвёртый квартал — раньше
    следующего года, а ничьи между годом и первым кварталом не возникает
    (``docs/decisions.md``, D-34).

    Разбор делегируется `pf.periods`: то же правило нужно проверке
    оснований, и второй копии быть не должно.

    Args:
        row: Строка таблицы.

    Returns:
        Кортеж для сортировки.
    """
    return key_of_row(row)


def summarize_series(
    metrics: list[Metric],
    *,
    name: str,
    unit: str = "",
) -> list[Metric]:
    """Сводные метрики по ряду: изменение, среднее, медиана, размах.

    Args:
        metrics: Метрики одного ряда, в порядке наблюдений.
        name: Название сводки.
        unit: Единица измерения.

    Returns:
        Список сводных метрик. Каждая переносит provenance первого и
        последнего наблюдения — по ним видно, из чего считалось.
    """
    if not metrics:
        return []

    values = [metric.value for metric in metrics]
    first = metrics[0]
    last = metrics[-1]
    # Наблюдения ряда — не их количество. Среднее по ряду из 6, 7, 10, 12
    # равно 8,75, но пересчитать его можно только по самим числам: если
    # в inputs лежит «наблюдений: 4», сложить нечего, и метрику нечем
    # подтвердить. Поэтому в inputs кладутся наблюдения.
    observations = [f"{value:.6g}" for value in values]
    return [
        Metric(
            name=f"{name}: изменение за период",
            value=change_percent(first.value, last.value),
            unit="%",
            source_id=last.source_id,
            row=last.row,
            formula="(последнее - первое) / |первое| * 100",
            inputs={
                "база": f"{first.value:.6g}",
                "новое": f"{last.value:.6g}",
            },
        ),
        Metric(
            name=f"{name}: среднее",
            value=mean(values),
            unit=unit,
            source_id=first.source_id,
            row=first.row,
            formula="сумма / количество",
            inputs={"наблюдения": observations},
        ),
        Metric(
            name=f"{name}: медиана",
            value=median(values),
            unit=unit,
            source_id=first.source_id,
            row=first.row,
            formula="медиана ряда",
            inputs={"наблюдения": observations},
        ),
        Metric(
            name=f"{name}: размах",
            value=max(values) - min(values),
            unit=unit,
            source_id=first.source_id,
            row=first.row,
            formula="максимум - минимум",
            inputs={"наблюдения": observations},
        ),
    ]
