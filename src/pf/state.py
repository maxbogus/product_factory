"""Состояние сессии: сохранение и продолжение после перерыва.

Работа агента можно прервать в любой момент — в том числе закрыв чат. Чтобы
продолжить с того же места, состояние лежит на диске в **текстовом** виде:
его можно посмотреть глазами, исправить руками и перенести на другую машину.

В состоянии хранится не «что думает модель», а факты о ходе работы:
текущая фаза, результаты завершённых шагов, заданные вопросы человеку
и накопленная стоимость. Это делает продолжение предсказуемым: агент
начинает с той же фазы и с теми же данными.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pf.config import settings


class Phase(StrEnum):
    """Фазы рабочего процесса агента."""

    #: Постановка задачи: уточняем цель, получателя, ограничения.
    INTAKE = "intake"
    #: План исследования: какие вопросы могут изменить решение.
    PLANNING = "planning"
    #: Маршрутизация: проверка, что работа идёт верным путём.
    ROUTING = "routing"
    #: Сбор источников.
    COLLECTING = "collecting"
    #: Разбор: факты, гипотезы, пробелы.
    ANALYZING = "analyzing"
    #: Генерация вариантов.
    OPTIONS = "options"
    #: Критическая проверка вариантов.
    REDTEAM = "redteam"
    #: Разработка стратегии.
    STRATEGY = "strategy"
    #: Фиксация решения и обязательств.
    KEEPER = "keeper"
    #: Работа завершена.
    DONE = "done"

    @property
    def title_ru(self) -> str:
        """Человекочитаемое название фазы."""
        return {
            Phase.INTAKE: "Постановка задачи",
            Phase.PLANNING: "План исследования",
            Phase.ROUTING: "Маршрутизация",
            Phase.COLLECTING: "Сбор источников",
            Phase.ANALYZING: "Разбор источников",
            Phase.OPTIONS: "Варианты решения",
            Phase.REDTEAM: "Критическая проверка",
            Phase.STRATEGY: "Разработка стратегии",
            Phase.KEEPER: "Фиксация решения",
            Phase.DONE: "Завершено",
        }[self]

    @property
    def next(self) -> Phase:
        """Следующая фаза по порядку работы."""
        order = list(Phase)
        if self is Phase.DONE:
            return Phase.DONE
        return order[order.index(self) + 1]


#: Как фазу называют в ответе модели и в ссылке на ввод пользователя.
#: Ключи — уже приведённые к нижнему регистру без «ё»: «Разбор источников»
#: должно найтись и по «разбор», и по «analyzing». Одного названия мало:
#: пока правило было только по `Phase.title_ru`, ссылка «разбор» не узнавалась.
_PHASE_ALIASES: dict[str, Phase] = {
    "постановка": Phase.INTAKE,
    "постановка задачи": Phase.INTAKE,
    "прием": Phase.INTAKE,
    "вводный": Phase.INTAKE,
    "план": Phase.PLANNING,
    "планирование": Phase.PLANNING,
    "план исследования": Phase.PLANNING,
    "маршрут": Phase.ROUTING,
    "маршрутизация": Phase.ROUTING,
    "сбор": Phase.COLLECTING,
    "сбор источников": Phase.COLLECTING,
    "сбор материала": Phase.COLLECTING,
    "разбор": Phase.ANALYZING,
    "разбор источников": Phase.ANALYZING,
    "анализ": Phase.ANALYZING,
    "аналитика": Phase.ANALYZING,
    "варианты": Phase.OPTIONS,
    "варианты решения": Phase.OPTIONS,
    "критика": Phase.REDTEAM,
    "критическая проверка": Phase.REDTEAM,
    "красная команда": Phase.REDTEAM,
    "стратегия": Phase.STRATEGY,
    "разработка стратегии": Phase.STRATEGY,
    "решение": Phase.KEEPER,
    "фиксация решения": Phase.KEEPER,
    "завершено": Phase.DONE,
}


def _plain(text: str) -> str:
    """Привести название к виду для сравнения: нижний регистр, без «ё».

    Args:
        text: Название стадии из ответа модели или из ссылки.

    Returns:
        Приведённый текст.
    """
    return " ".join(str(text).lower().replace("ё", "е").split())


def phase_of(text: str) -> Phase | None:
    """Назвать стадию по её названию — по-русски или по-английски.

    Стадия вводится пользователем, и её называют словами: «разбор»,
    «Анализ», «analyzing», «Разбор источников». Пока правило было одно
    (только точное имя фазы), ссылка на ввод не узнавалась, и законная
    опора отклонялась по форме слова.

    Args:
        text: Название стадии.

    Returns:
        Фаза либо ``None``, если название не распознано.
    """
    key = _plain(text)
    if not key:
        return None
    for phase in Phase:
        if key == _plain(phase.value) or key == _plain(phase.title_ru):
            return phase
    return _PHASE_ALIASES.get(key)


@dataclass(frozen=True, slots=True)
class HumanInput:
    """Ввод пользователя: что сказано и на какой стадии.

    Ответ человека — не документ, но и не пустая ссылка. Это опора: её
    называют каналом и стадией, а решение «верить или нет» принимает
    человек. Стадия обязательна: по ней проверяется ссылка и отчёт говорит,
    где именно пользователь вмешался.

    Attributes:
        text: Что сказал пользователь.
        phase: Фаза, на которой он это сказал (пусто у записей до правки).
        question: Вопрос, на который он отвечал.
        at: Когда записано.
    """

    text: str
    phase: str = ""
    question: str = ""
    at: str = ""

    @property
    def phase_title(self) -> str:
        """Название стадии для человека.

        Returns:
            Название фазы; «стадия не записана» — для записи до правки.
        """
        phase = phase_of(self.phase)
        return phase.title_ru if phase is not None else "стадия не записана"

    def as_dict(self) -> dict[str, str]:
        """Представить запись словарём — так она лежит в состоянии сессии.

        Returns:
            Словарь с ключами ``text``, ``phase``, ``question``, ``at``.
        """
        return {
            "text": self.text,
            "phase": self.phase,
            "question": self.question,
            "at": self.at,
        }


def record_human_input(state: SessionState, text: str, question: str = "") -> HumanInput:
    """Записать ответ пользователя вместе со стадией.

    Стадия берётся из состояния **до** очистки вопроса: ответ дан там, где
    работа стояла, и это единственное место, где стадия ещё известна.

    Args:
        state: Состояние сессии.
        text: Ответ пользователя.
        question: Вопрос, на который он отвечает.

    Returns:
        Записанный ввод.
    """
    record = HumanInput(
        text=str(text),
        phase=state.phase.value,
        question=str(question or state.pending_question or ""),
        at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    state.results.setdefault("human_answers", []).append(record.as_dict())
    return record


def human_inputs(state: SessionState | None) -> list[HumanInput]:
    """Все ответы пользователя в сессии.

    Читаются и старые записи: до появления стадии в состоянии лежали голые
    строки. Стадию у них не выдумываем — пустая строка означает «не
    записана».

    Args:
        state: Состояние сессии либо ``None``.

    Returns:
        Записи в порядке появления.
    """
    if state is None:
        return []
    raw = state.results.get("human_answers") or []
    records: list[HumanInput] = []
    for item in raw:
        if isinstance(item, dict):
            records.append(
                HumanInput(
                    text=str(item.get("text") or ""),
                    phase=str(item.get("phase") or ""),
                    question=str(item.get("question") or ""),
                    at=str(item.get("at") or ""),
                )
            )
        else:
            records.append(HumanInput(text=str(item)))
    return [record for record in records if record.text.strip()]


@dataclass(slots=True)
class SessionState:
    """Состояние сессии агента.

    Attributes:
        session_id: Идентификатор сессии.
        phase: Текущая фаза.
        question: Исходный вопрос руководителя.
        results: Результаты завершённых шагов по фазам.
        pending_question: Вопрос, ждущий ответа человека.
        cost_units: Накопленная стоимость.
        steps_done: Сколько шагов выполнено.
        updated_at: Когда состояние обновлялось.
    """

    session_id: str
    phase: Phase = Phase.INTAKE
    question: str = ""
    results: dict[str, Any] = field(default_factory=dict)
    pending_question: str = ""
    cost_units: float = 0.0
    steps_done: int = 0
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def to_json(self) -> str:
        """Сериализовать состояние в JSON."""
        payload = asdict(self)
        payload["phase"] = self.phase.value
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> SessionState:
        """Восстановить состояние из JSON.

        Args:
            raw: Текст JSON.

        Returns:
            Восстановленное состояние.

        Raises:
            ValueError: JSON не разбирается.
        """
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"состояние повреждено: {exc}") from exc
        payload["phase"] = Phase(payload.get("phase", Phase.INTAKE.value))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


def state_path(session_id: str) -> Path:
    """Путь к файлу состояния сессии.

    Args:
        session_id: Идентификатор сессии.

    Returns:
        Путь к файлу состояния.
    """
    return settings.state_dir / f"session-{session_id}.json"


def save(state: SessionState) -> Path:
    """Сохранить состояние на диск.

    Args:
        state: Состояние сессии.

    Returns:
        Путь, куда записано состояние.
    """
    path = state_path(state.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.to_json(), encoding="utf-8")
    return path


def load(session_id: str) -> SessionState | None:
    """Загрузить состояние сессии.

    Args:
        session_id: Идентификатор сессии.

    Returns:
        Состояние или ``None``, если сессии ещё не было.
    """
    path = state_path(session_id)
    if not path.exists():
        return None
    return SessionState.from_json(path.read_text(encoding="utf-8"))


def list_sessions() -> list[str]:
    """Идентификаторы сохранённых сессий, свежие первыми.

    Returns:
        Список идентификаторов.
    """
    directory = settings.state_dir
    if not directory.is_dir():
        return []
    sessions = []
    for path in directory.glob("session-*.json"):
        sessions.append(path.stem.removeprefix("session-"))
    return sorted(sessions, key=lambda sid: state_path(sid).stat().st_mtime, reverse=True)
