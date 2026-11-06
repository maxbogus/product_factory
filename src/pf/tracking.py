"""Трекинг прогонов: накопление измерений в MLflow.

Бриф требует «сравнение с исходным способом»: измерить, насколько
агент быстрее, дешевле или точнее, чем обычный чат с ИИ. Для этого
нужен **накопитель** измерений, а не разовые замеры: один прогон
ничего не доказывает, особенно когда модель нестабильна.

MLflow выбран потому, что он умеет хранить прогоны рядом и сравнивать
их. Но он не обязателен:

- если пакет не установлен — трекинг молча выключается;
- если ``MLFLOW_DISABLED=1`` — выключается;
- если MLflow падает — прогон не ломается, пишется предупреждение.

Наблюдаемость не должна ломать то, что наблюдают.
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

#: URI трекинга по умолчанию: файловая база рядом с проектом. Сервер
#: не нужен — демонстрация должна запускаться без инфраструктуры.
DEFAULT_TRACKING_URI = "file:./mlruns"

#: Эксперимент по умолчанию.
DEFAULT_EXPERIMENT = "product-factory-strategist"

#: Значения, при которых трекинг выключен явно.
_OFF_VALUES = frozenset({"0", "false", "no", "off"})


@dataclass
class RunMetrics:
    """Измерение одного прогона.

    Attributes:
        task: Название задачи.
        model: Модель, которая отвечала.
        provider: Провайдер модели.
        mode: Режим прогона: ``agent`` или ``baseline``.
        steps: Сколько шагов прошло агент.
        citations: Сколько подтверждённых оснований найдено.
        rejected: Сколько оснований отклонено проверкой.
        facts: Сколько фактов в разборе.
        variants: Сколько вариантов предложено.
        total_tokens: Всего токенов.
        duration_sec: Длительность прогона в секундах.
        cost_units: Расход в условных единицах.
        errors: Ошибки, если были.
    """

    task: str = ""
    model: str = ""
    provider: str = ""
    mode: str = "agent"
    steps: int = 0
    citations: int = 0
    rejected: int = 0
    facts: int = 0
    variants: int = 0
    total_tokens: int = 0
    duration_sec: float = 0.0
    cost_units: float = 0.0
    errors: str = ""

    def as_metrics(self) -> dict[str, float]:
        """Числовые метрики для MLflow.

        Returns:
            Метрики без нечисловых значений.
        """
        raw: dict[str, Any] = {
            "steps": self.steps,
            "citations": self.citations,
            "rejected_citations": self.rejected,
            "facts": self.facts,
            "variants": self.variants,
            "total_tokens": self.total_tokens,
            "duration_sec": round(self.duration_sec, 3),
            "cost_units": round(self.cost_units, 8),
        }
        clean: dict[str, float] = {}
        for key, value in raw.items():
            if isinstance(value, bool):
                value = int(value)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                clean[key] = number
        return clean

    def as_params(self) -> dict[str, str]:
        """Параметры прогона для MLflow.

        Returns:
            Строковые параметры без пустых значений.
        """
        raw = {
            "task": self.task,
            "model": self.model,
            "provider": self.provider,
            "mode": self.mode,
            "errors": self.errors,
        }
        return {str(key): str(value) for key, value in raw.items() if value}


def _mlflow() -> Any | None:
    """Ленивый импорт MLflow.

    Импорт идёт через ``importlib``: пакет опциональный, и прямой
    ``import mlflow`` ломал бы проверку типов, когда его нет.

    Returns:
        Модуль MLflow либо ``None``, если он не установлен.
    """
    import importlib

    try:
        return importlib.import_module("mlflow")
    except ImportError:
        return None


def is_enabled() -> bool:
    """Включён ли трекинг.

    Returns:
        Трекинг включён.
    """
    flag = os.environ.get("MLFLOW_DISABLED", "").lower()
    if flag in _OFF_VALUES or flag == "1":
        return False
    return _mlflow() is not None


def log_run(metrics: RunMetrics, *, tags: dict[str, str] | None = None) -> str:
    """Записать прогон в MLflow.

    Args:
        metrics: Измерение прогона.
        tags: Дополнительные метки (например, номер повтора).

    Returns:
        Идентификатор прогона; пустая строка, если трекинг выключен.
    """
    mlflow = _mlflow()
    if mlflow is None or not is_enabled():
        return ""
    try:
        # MLflow 3.x держит файловую базу в «режиме обслуживания»:
        # мы сознательно остаёмся на файле, сервер не нужен.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT))
        name = f"{metrics.mode}-{metrics.model or 'model'}-{time.strftime('%H%M%S')}"
        with mlflow.start_run(run_name=name) as run:
            params = metrics.as_params()
            if params:
                mlflow.log_params(params)
            values = metrics.as_metrics()
            if values:
                mlflow.log_metrics(values)
            if tags:
                mlflow.set_tags(tags)
            return str(run.info.run_id)
    except Exception as exc:  # noqa: BLE001 — наблюдаемость не должна ломать прогон
        print(f"WARN: MLflow не записал прогон: {type(exc).__name__}: {exc}", file=sys.stderr)
        return ""


def log_runs(runs: list[RunMetrics], *, tags: dict[str, str] | None = None) -> list[str]:
    """Записать несколько прогонов.

    Args:
        runs: Измерения прогонов.
        tags: Дополнительные метки.

    Returns:
        Список идентификаторов прогонов.
    """
    ids: list[str] = []
    for item in runs:
        run_id = log_run(item, tags=tags)
        if run_id:
            ids.append(run_id)
    return ids


@dataclass
class Accumulator:
    """Накопитель измерений: пишет их и в файл, и в MLflow.

    Файл нужен, потому что MLflow может быть не установлен, а сравнение
    требуется всегда. Формат — JSONL, одна строка на прогон.

    Attributes:
        path: Путь к файлу измерений.
        runs: Накопленные измерения.
    """

    path: str = "reports/bench/runs.jsonl"
    runs: list[RunMetrics] = field(default_factory=list)

    def add(self, metrics: RunMetrics) -> str:
        """Записать измерение.

        Args:
            metrics: Измерение прогона.

        Returns:
            Идентификатор прогона в MLflow (пустая строка, если трекинг выключен).
        """
        self.runs.append(metrics)
        run_id = log_run(metrics)
        _append_jsonl(self.path, metrics, run_id)
        return run_id

    def summary(self) -> dict[str, Any]:
        """Сводка по накопленным прогонам.

        Returns:
            Средние, минимальные и максимальные значения по метрикам.
        """
        if not self.runs:
            return {"runs": 0}
        keys = ("citations", "total_tokens", "duration_sec", "cost_units", "steps", "facts")
        summary: dict[str, Any] = {"runs": len(self.runs)}
        for key in keys:
            values = [
                float(item.as_metrics().get(key, 0.0))
                for item in self.runs
                if key in item.as_metrics()
            ]
            if values:
                summary[f"{key}_avg"] = round(sum(values) / len(values), 4)
                summary[f"{key}_min"] = round(min(values), 4)
                summary[f"{key}_max"] = round(max(values), 4)
        return summary


def _append_jsonl(path: str, metrics: RunMetrics, run_id: str) -> None:
    """Дописать измерение в файл.

    Args:
        path: Путь к файлу.
        metrics: Измерение.
        run_id: Идентификатор прогона в MLflow.
    """
    import json
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        **metrics.as_params(),
        **metrics.as_metrics(),
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


