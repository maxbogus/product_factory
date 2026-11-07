"""Проверки переходов между фазами рабочего процесса.

Отдельные роли проверяются в других файлах. Здесь проверяется другое:
**связность цепочки**. Агент — это не набор функций, а последовательность,
где результат одной фазы становится входом следующей. Если связь рвётся,
шаги формально проходят, а работа бессмысленна: стратегия строится не по
тому разбору, который был сделан.

Поэтому проверяются четыре вещи:
1. данные передаются по цепочке (intake → planner → …);
2. неподтверждённое основание не проходит дальше как факт;
3. работа останавливается там, где нужен человек, и продолжается после
   ответа;
4. каждый переход пишет ровно одну запись в журнал и правильно считает
   стоимость.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pf.config import settings
from pf.journal import Journal, journal_path
from pf.llm import LLMResponse, LLMUsage
from pf.sources import Passage, SourceBundle
from pf.state import Phase, SessionState, load, save
from pf.workflow import Workflow


class FakeClient:
    """Модель, которая отвечает заранее заданными ответами.

    Ответы выдаются по порядку ролей. Это позволяет прогнать цепочку
    целиком без обращения к сети — и увидеть, что именно передаётся
    между шагами.
    """

    def __init__(self, answers: dict[str, Any]) -> None:
        """Запомнить ответы.

        Args:
            answers: Ответ по имени роли.
        """
        self.answers = answers
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> LLMResponse:
        """Ответить заранее подготовленным JSON.

        Args:
            system: Системная инструкция.
            user: Текст задачи.

        Returns:
            Ответ модели.

        Raises:
            AssertionError: Для роли нет подготовленного ответа.
        """
        role = self._role_from_system(system)
        self.calls.append((role, user))
        if role not in self.answers:
            raise AssertionError(f"для роли {role!r} нет подготовленного ответа")
        return LLMResponse(
            content=json.dumps(self.answers[role], ensure_ascii=False),
            usage=LLMUsage(prompt_tokens=100, completion_tokens=50),
        )

    @staticmethod
    def _role_from_system(system: str) -> str:
        """Определить роль по тексту инструкции.

        В промптах роль названа по-русски («Роль: сборщик»), а
        идентификатор — по-английски (``collector``). Поэтому ищем по
        русским названиям, которые уникальны.

        Args:
            system: Системная инструкция.

        Returns:
            Имя роли.
        """
        markers = {
            "intake": "интик",
            "planner": "планировщик",
            "collector": "сборщик",
            "analyst": "аналитик",
            "options": "генератор",
            "redteam": "критик",
            "strategy": "стратег",
            "keeper": "decision-keeper",
        }
        for role, marker in markers.items():
            if marker in system:
                return role
        return "unknown"


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолировать состояние и журнал в временный каталог."""
    monkeypatch.setattr(settings, "state_dir", tmp_path / "state")
    monkeypatch.setattr(settings, "journal_dir", tmp_path / "journal")


def _bundle() -> SourceBundle:
    """Собрать источник с одним абзацем, пригодным для цитирования."""
    passage = Passage(
        source_id="telecom/Мегафон (компания)",
        title="Мегафон (компания)",
        section="telecom",
        paragraph_index=154,
        text=(
            "Ввела специальный тариф «Транзитный», главная особенность которого — "
            "бесплатные входящие вызовы при нахождении в зоне действия сети «Мегафон» "
            "в Сибири и на Дальнем Востоке, единая стоимость исходящих вызовов."
        ),
        relevance=1.0,
    )
    article = type(
        "A",
        (),
        {
            "source_id": passage.source_id,
            "title": passage.title,
            "section": passage.section,
            "content": passage.text,
        },
    )()
    return SourceBundle(passages=[passage], articles=[article], query="удержание")


def _answers() -> dict[str, Any]:
    """Ответы ролей для сквозного прогона."""
    return {
        "intake": {
            "question": "Почему падает удержание?",
            "goal": "Понять причины и предложить действия",
            "audience": "руководитель продукта",
            "constraints": ["цену снижать нельзя"],
        },
        "planner": {
            "questions": ["Какие данные по удержанию есть?", "Какие тарифы у операторов?"],
            "priority_order": [1, 2],
            "gaps_expected": ["причины оттока по причинам"],
        },
        "collector": {
            "facts": [
                {
                    "kind": "факт",
                    "text": "Введён тариф «Транзитный»",
                    "sources": [
                        {
                            "source_id": "telecom/Мегафон (компания)",
                            "paragraph": 154,
                            "quote": (
                                "Ввела специальный тариф «Транзитный», главная особенность "
                                "которого — бесплатные входящие вызовы"
                            ),
                        }
                    ],
                }
            ],
            "hypotheses": [],
            "forecasts": [],
            "gaps": ["нет телеметрии"],
        },
        "analyst": {
            "facts": [
                {
                    "kind": "факт",
                    "text": "Введён тариф «Транзитный»",
                    "sources": [
                        {
                            "source_id": "telecom/Мегафон (компания)",
                            "paragraph": 154,
                            "quote": (
                                "Ввела специальный тариф «Транзитный», главная особенность "
                                "которого — бесплатные входящие вызовы"
                            ),
                        }
                    ],
                }
            ],
            "hypotheses": [],
            "forecasts": [],
            "gaps": ["нет телеметрии"],
        },
        "options": {
            "variants": [
                {"title": "Оставить как есть", "summary": "с", "objections": ["о"]},
                {"title": "Закрыть направление", "summary": "с", "objections": ["о"]},
                {"title": "Изменить тариф", "summary": "с", "objections": ["о"]},
            ],
            "keep_current_marker": "Оставить как есть",
            "close_direction_marker": "Закрыть направление",
        },
        "redteam": {
            "findings": ["тариф не решает отток"],
            "strongest_objection": "нет телеметрии",
            "unresolved": [],
        },
        "strategy": {
            "product": {"audience": "абонент", "chosen_direction": "изменить тариф"},
            "communication": {"barrier": "не понимает ценности", "key_message": "сообщение"},
            "verification": {"expected_change": "рост удержания", "compare_against": "база"},
        },
        "keeper": {
            "decision": "изменить тариф",
            "commitments": [
                {
                    "statement": "изменить тариф",
                    "owner": "не назначен",
                    "metric": "удержание",
                    "review_condition": "если не сработает",
                    "review_by": "через месяц",
                    "basis": [],
                }
            ],
            "superseded_claims": [],
        },
    }


def _workflow(answers: dict[str, Any] | None = None) -> Workflow:
    """Собрать рабочий процесс с поддельной моделью.

    Args:
        answers: Ответы ролей.

    Returns:
        Готовый исполнитель.
    """
    return Workflow(client=FakeClient(answers or _answers()), config=settings)


def _state() -> SessionState:
    """Собрать состояние на фазе постановки задачи."""
    return SessionState(session_id="t1", phase=Phase.INTAKE, question="Почему падает удержание?")


# ── 1. Данные передаются по цепочке ────────────────────────────────────────


def test_intake_result_reaches_planner() -> None:
    """Цель, поставленная на интике, доходит до планировщика."""
    workflow = _workflow()
    state = _state()
    bundle = _bundle()

    workflow.run_step(state, bundle)
    assert state.results["intake"]["goal"] == "Понять причины и предложить действия"

    state.phase = Phase.PLANNING
    _role, payload, _context = workflow._step_inputs(state, bundle)
    assert payload["task"]["goal"] == "Понять причины и предложить действия"


def test_plan_questions_reach_collector() -> None:
    """Вопросы плана доходят до сборщика — по ним ищутся источники."""
    workflow = _workflow()
    state = _state()
    bundle = _bundle()

    workflow.run_step(state, bundle)
    state.phase = Phase.PLANNING
    workflow.run_step(state, bundle)
    assert state.results["planning"]["questions"]

    state.phase = Phase.COLLECTING
    _role, payload, _context = workflow._step_inputs(state, bundle)
    assert "удержание" in payload["query"].lower()


def test_collector_receives_source_passages() -> None:
    """Сборщик получает абзацы источников.

    Без этого он не может выбрать основание и сочиняет цитату.
    """
    workflow = _workflow()
    state = _state()
    bundle = _bundle()

    state.phase = Phase.COLLECTING
    _role, _payload, context = workflow._step_inputs(state, bundle)
    assert "Транзитный" in context
    assert "telecom/Мегафон (компания)#p154" in context


def test_analyst_receives_previous_facts() -> None:
    """Разбор получает то, что собрал сборщик."""
    workflow = _workflow()
    state = _state()
    bundle = _bundle()

    state.phase = Phase.COLLECTING
    workflow.run_step(state, bundle)
    state.phase = Phase.ANALYZING
    _role, payload, _context = workflow._step_inputs(state, bundle)
    assert payload["collected"]["facts"]


def test_redteam_reaches_strategy() -> None:
    """Критика вариантов доходит до стратегии."""
    workflow = _workflow()
    state = _state()
    bundle = _bundle()

    for phase in (Phase.INTAKE, Phase.PLANNING, Phase.COLLECTING, Phase.ANALYZING, Phase.OPTIONS):
        state.phase = phase
        workflow.run_step(state, bundle)
    state.phase = Phase.REDTEAM
    workflow.run_step(state, bundle)
    assert state.results["redteam"]["findings"]

    state.phase = Phase.STRATEGY
    _role, payload, _context = workflow._step_inputs(state, bundle)
    assert payload["redteam"]["findings"]


# ── 2. Неподтверждённое основание не проходит как факт ────────────────────


def test_fabricated_quote_does_not_become_confirmed_fact() -> None:
    """Выдуманная цитата отбрасывается, и факт остаётся без подтверждения.

    Это главная проверка цепочки: если основание не подтверждено, оно не
    должно тихо утечь дальше как проверенный факт.
    """
    answers = _answers()
    answers["collector"] = {
        "facts": [
            {
                "kind": "факт",
                "text": "Удержание упало на 8 процентных пунктов",
                "sources": [
                    {
                        "source_id": "telecom/Мегафон (компания)",
                        "paragraph": 154,
                        "quote": "Удержание упало на 8 процентных пунктов",
                    }
                ],
            }
        ],
        "hypotheses": [],
        "forecasts": [],
        "gaps": [],
    }
    workflow = _workflow(answers)
    state = _state()
    bundle = _bundle()

    state.phase = Phase.COLLECTING
    result = workflow.run_step(state, bundle)

    assert result.citations == []
    assert result.rejected == 1
    assert result.rejected_details


# ── 3. Остановка и продолжение ─────────────────────────────────────────────


def test_human_answer_reaches_role_context() -> None:
    """Ответ человека попадает в контекст следующего шага.

    Иначе агент задаёт вопрос, человек отвечает, а работа продолжается
    вслепую.
    """
    workflow = _workflow()
    state = _state()
    bundle = _bundle()

    state.results.setdefault("human_answers", []).append("цену снижать нельзя")
    _role, _payload, context = workflow._step_inputs(state, bundle)

    assert "цену снижать нельзя" in context
    assert "Ответ человека" in context


def test_web_context_reaches_role_context() -> None:
    """Внешние источники попадают в контекст роли.

    Раньше агент выходил в интернет, складывал материалы в состояние — и
    модель их не видела. Теперь видят, но отдельным блоком: их статус
    «требует проверки», и путать их с подтверждёнными основаниями нельзя.
    """
    workflow = _workflow()
    state = _state()
    bundle = _bundle()

    state.phase = Phase.COLLECTING
    state.results["web_context"] = "[web/Новости | https://example.com] Текст статьи."
    _role, _payload, context = workflow._step_inputs(state, bundle)

    assert "Новости" in context
    assert "требует проверки" in context
    # Локальные материалы не теряются.
    assert "Транзитный" in context


def test_local_and_web_materials_are_separated() -> None:
    """Локальные и внешние материалы не смешиваются в один список."""
    workflow = _workflow()
    state = _state()
    bundle = _bundle()

    state.phase = Phase.COLLECTING
    state.results["web_context"] = "ВНЕШНИЙ ТЕКСТ"
    _role, _payload, context = workflow._step_inputs(state, bundle)

    local_part, _, web_part = context.partition("ВНЕШНИЕ ИСТОЧНИКИ")
    assert "Транзитный" in local_part
    assert "ВНЕШНИЙ ТЕКСТ" in web_part


def test_repair_prompt_explains_missing_source() -> None:
    """Ошибка «факт без источника» объясняется моделью явно.

    Текст ошибки для модели абстрактен: она не понимает, что нужно не
    «дописать ссылку», а перенести утверждение в гипотезы. Без
    пояснения модель повторяет ту же ошибку три раза.
    """
    error = "факт должен ссылаться хотя бы на один источник"
    prompt = Workflow._repair_prompt("ЗАДАЧА", error)

    assert "sources" in prompt
    assert "hypotheses" in prompt
    assert "Не выдумывай цитату" in prompt


def test_repair_prompt_without_special_hint() -> None:
    """Для прочих ошибок подсказка остаётся общей."""
    prompt = Workflow._repair_prompt("ЗАДАЧА", "не хватает поля")
    assert "не хватает поля" in prompt
    assert "Не выдумывай цитату" not in prompt


def test_state_survives_process_restart() -> None:
    """Работа продолжается после перерыва: состояние читается с диска."""
    workflow = _workflow()
    state = _state()
    bundle = _bundle()

    state.phase = Phase.COLLECTING
    workflow.run_step(state, bundle)
    save(state)

    reloaded = load("t1")
    assert reloaded is not None
    assert reloaded.phase is Phase.COLLECTING
    assert reloaded.results["collecting"]["facts"]


# ── 4. Журнал и стоимость на каждом переходе ───────────────────────────────


def test_each_transition_writes_one_journal_record() -> None:
    """Каждый шаг пишет ровно одну запись в журнал."""
    workflow = _workflow()
    state = _state()
    bundle = _bundle()

    for phase in (Phase.INTAKE, Phase.PLANNING, Phase.COLLECTING):
        state.phase = phase
        workflow.run_step(state, bundle)

    records = Journal(journal_path("t1")).read()
    assert len(records) == 3
    assert [r.phase for r in records] == ["intake", "planning", "collecting"]


def test_cost_accumulates_across_transitions() -> None:
    """Стоимость накапливается по шагам и совпадает с журналом."""
    workflow = _workflow()
    state = _state()
    bundle = _bundle()

    for phase in (Phase.INTAKE, Phase.PLANNING):
        state.phase = phase
        workflow.run_step(state, bundle)

    journal = Journal(journal_path("t1"))
    assert journal.total_tokens == 300
    assert state.cost_units == pytest.approx(journal.total_cost_units)


def test_confirmed_citation_is_recorded_in_journal() -> None:
    """Подтверждённое основание попадает в журнал со ссылкой."""
    workflow = _workflow()
    state = _state()
    bundle = _bundle()

    state.phase = Phase.COLLECTING
    workflow.run_step(state, bundle)

    records = Journal(journal_path("t1")).read()
    assert records
    assert records[0].citations
    assert records[0].citations[0].startswith("telecom/Мегафон (компания)#p")


def test_step_without_sources_says_grounds_were_not_checked() -> None:
    """Шаг без собранных источников говорит, что проверять было нечем.

    Роутер может повести вперёд, перескочив сбор источников. Тогда
    аналитик работает без материалов, и пустой список отказов читается
    как «всё в порядке». Молчать нельзя: шаг обязан сказать, что
    основания не проверялись.
    """
    workflow = _workflow()
    state = _state()
    state.phase = Phase.ANALYZING

    result = workflow.run_step(state, None)

    assert any("источники не собраны" in detail for detail in result.rejected_details)
