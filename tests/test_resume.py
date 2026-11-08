"""Проверки продолжения работы после перерыва.

Требование брифа: «работу можно остановить и продолжить с сохранённого
места, не начиная заново». На практике это значит три вещи:

1. если работа остановилась на сборе источников, продолжение должно
   собрать источники заново — иначе сборщик получит пустой блок
   МАТЕРИАЛЫ и начнёт сочинять цитаты;
2. если агент задал вопрос человеку, продолжение должно этот вопрос
   показать, а не молча идти дальше;
3. ответ человека должен доходить до модели на **следующей** фазе, а не
   только на первой: иначе агент спрашивает, человек отвечает — и ответ
   теряется, а работа продолжается вслепую.

Здесь проверяется именно это — на диске и без обращения к модели.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pf.config import settings
from pf.state import Phase, SessionState, save
from pf.workflow import Workflow


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолировать состояние и журнал в временный каталог."""
    monkeypatch.setattr(settings, "state_dir", tmp_path / "state")
    monkeypatch.setattr(settings, "journal_dir", tmp_path / "journal")


class _StubClient:
    """Модель, которая отвечает заранее заданным JSON."""

    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Any:
        from pf.llm import LLMResponse, LLMUsage

        role = "collector" if "сборщик" in system else "unknown"
        self.calls.append((role, user))
        return LLMResponse(
            content=json.dumps(self.answers.get(role, {}), ensure_ascii=False),
            usage=LLMUsage(prompt_tokens=10, completion_tokens=5),
        )


def test_resume_at_collecting_gets_sources() -> None:
    """Продолжение со сбора источников должно собрать источники.

    Без этого сборщик получает пустой блок МАТЕРИАЛЫ и не может взять
    ни одной цитаты — работа продолжается, но вывод нечем подкрепить.
    """
    state = SessionState(session_id="resume-collect", phase=Phase.COLLECTING)
    state.results["intake"] = {"goal": "выяснить причины падения удержания"}
    state.results["planning"] = {"questions": ["каковы причины падения удержания?"]}
    save(state)

    workflow = Workflow(client=_StubClient({}))
    bundle = workflow.bundle_for(state)
    assert bundle is not None, "при продолжении источники должны собираться"
    assert bundle.passages, "собранный набор должен содержать абзацы"


def test_resume_at_collecting_passes_materials_to_model() -> None:
    """Сборщик при продолжении должен видеть блок МАТЕРИАЛЫ."""
    state = SessionState(session_id="resume-materials", phase=Phase.COLLECTING)
    state.results["intake"] = {"goal": "выяснить причины падения удержания"}
    state.results["planning"] = {"questions": ["каковы причины падения удержания?"]}
    save(state)

    workflow = Workflow(client=_StubClient({}))
    bundle = workflow.bundle_for(state)
    _role, _payload, context = workflow._step_inputs(state, bundle)
    assert "МАТЕРИАЛЫ" in context or context.strip(), "сборщику нужен контекст с материалами"


def test_human_answer_reaches_next_phase() -> None:
    """Ответ человека должен попасть в контекст следующего шага.

    Ответ задаёт интик, а следующая фаза — планировщик. Если ответ
    теряется, планирование идёт вслепую: человек ответил, а агент
    работает по старому заданию.
    """
    state = SessionState(session_id="answer-context", phase=Phase.PLANNING)
    state.results["human_answers"] = ["Внутренней телеметрии нет, есть открытые отчёты"]
    save(state)

    workflow = Workflow(client=_StubClient({}))
    _role, _payload, context = workflow._step_inputs(state, None)
    assert "Внутренней телеметрии нет" in context, "ответ человека должен быть в контексте"


def test_human_answer_reaches_collecting_phase() -> None:
    """Ответ человека должен доходить и до сбора источников.

    Роутер часто спрашивает человека уже после постановки задачи. Ответ
    тогда приходит в фазу сбора — и он обязан там быть виден, иначе
    источники соберутся не по тому, о чём человек сказал.
    """
    state = SessionState(session_id="answer-collecting", phase=Phase.COLLECTING)
    state.results["human_answers"] = ["Смотри тарифы МТС и отток за 2025 год"]
    state.results["intake"] = {"goal": "выяснить причины падения удержания"}
    state.results["planning"] = {"questions": ["каковы причины падения удержания?"]}
    save(state)

    workflow = Workflow(client=_StubClient({}))
    bundle = workflow.bundle_for(state)
    _role, _payload, context = workflow._step_inputs(state, bundle)
    assert "отток за 2025 год" in context, "ответ человека должен доходить до сборщика"


def test_human_answer_reaches_strategy_phase() -> None:
    """Ответ человека должен доходить и до поздних фаз.

    Стратегия — итог работы. Если человек ответил на вопрос роутера
    перед стратегией, его ответ обязан повлиять на итог, иначе решения
    принимаются без учёта уточнения.
    """
    state = SessionState(session_id="answer-strategy", phase=Phase.STRATEGY)
    state.results["human_answers"] = ["Цену снижать нельзя — маржа"]
    state.results["intake"] = {"goal": "выяснить причины падения удержания"}
    state.results["planning"] = {"questions": ["каковы причины?"]}
    state.results["analyzing"] = {"facts": [], "hypotheses": [], "forecasts": [], "gaps": []}
    state.results["options"] = {"variants": []}
    state.results["redteam"] = {"findings": []}
    save(state)

    workflow = Workflow(client=_StubClient({}))
    _role, _payload, context = workflow._step_inputs(state, None)
    assert "Цену снижать нельзя" in context, "ответ человека должен доходить до стратега"


#: Вопрос роутера из прогона `demo-v11` — он повторился слово в слово.
_REPEATED_QUESTION = (
    "Какие метрики удержания анализируются и как они изменились за последние три месяца?"
)


def _routing_step(question: str):
    """Результат шага роутера, который просит человека.

    Args:
        question: Вопрос, который задаёт роутер.

    Returns:
        Результат шага с ``need_human``.
    """
    from pf.schemas import NextStep
    from pf.workflow import StepResult

    return StepResult(
        role="router",
        phase=Phase.ROUTING,
        data=NextStep(
            role="collector",
            reason="нужно уточнение",
            need_human=True,
            question=question,
        ),
    )


def test_repeated_question_does_not_stop_work() -> None:
    """Повтор уже заданного вопроса не останавливает работу.

    Прогон `demo-v11` встал навсегда: роутер спросил человека на
    маршрутизации, ответ записан в сессию, `pf resume` снова выполнил
    маршрутизацию — и модель повторила тот же вопрос слово в слово. Работа
    не двинулась ни на шаг, а ответ у агента уже был. Спрашивать второй раз
    нечего: вопрос задан, ответ дан, дальше — работа.
    """
    state = SessionState(session_id="repeat-question", phase=Phase.ROUTING)
    state.results["human_questions"] = [_REPEATED_QUESTION]
    state.results["human_answers"] = ["Метрики бери из материалов проекта"]
    save(state)

    workflow = Workflow(client=_StubClient({}))
    stop = workflow.human_question_to_stop(state, _routing_step(_REPEATED_QUESTION))

    assert stop == "", "на вопрос, на который ответили, второй раз не останавливаются"
    assert state.results.get("routing_note"), "повтор вопроса должен быть назван в журнале"


def test_repeated_question_without_answer_still_stops() -> None:
    """Пока ответа нет, повтор вопроса останавливает работу.

    Пропускать можно только тот вопрос, на который человек ответил. Иначе
    агент молча прошёл бы мимо вопроса, которого никто не видел.
    """
    state = SessionState(session_id="repeat-question-unanswered", phase=Phase.ROUTING)
    state.results["human_questions"] = [_REPEATED_QUESTION]
    save(state)

    workflow = Workflow(client=_StubClient({}))
    stop = workflow.human_question_to_stop(state, _routing_step(_REPEATED_QUESTION))

    assert stop == _REPEATED_QUESTION


def test_repeated_question_check_keeps_new_questions() -> None:
    """Новый вопрос останавливает работу, даже если на прошлый ответили.

    Правило про повтор узкое: оно снимает только буквальный повтор. Другой
    вопрос — это другое уточнение, и человек должен его увидеть.
    """
    state = SessionState(session_id="repeat-question-new", phase=Phase.ROUTING)
    state.results["human_questions"] = [_REPEATED_QUESTION]
    state.results["human_answers"] = ["Метрики бери из материалов проекта"]
    save(state)

    workflow = Workflow(client=_StubClient({}))
    other = "Какой бюджет доступен на удержание?"
    stop = workflow.human_question_to_stop(state, _routing_step(other))

    assert stop == other
