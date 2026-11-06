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

__all__ = [
    "Metric",
    "cagr",
    "change_percent",
    "compute_from_rows",
    "mean",
    "median",
    "parse_number",
    "percentile",
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
        metrics.append(
            Metric(
                name=full_name,
                value=value,
                unit=unit,
                source_id=source_id,
                row=index,
                formula=f"значение из колонки «{value_column}», строка {index + 1}",
                inputs={value_column: str(raw)},
            )
        )
    return metrics


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
            inputs={"наблюдений": len(values)},
        ),
        Metric(
            name=f"{name}: медиана",
            value=median(values),
            unit=unit,
            source_id=first.source_id,
            row=first.row,
            formula="медиана ряда",
            inputs={"наблюдений": len(values)},
        ),
        Metric(
            name=f"{name}: размах",
            value=max(values) - min(values),
            unit=unit,
            source_id=first.source_id,
            row=first.row,
            formula="максимум - минимум",
            inputs={"наблюдений": len(values)},
        ),
    ]
