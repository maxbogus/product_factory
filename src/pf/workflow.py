"""Рабочий процесс агента: конечный автомат от запроса до стратегии.

Автомат проходит фазы по порядку, но умеет **остановиться и спросить
человека** там, где без ответа человека работа потеряет смысл. После
ответа работа продолжается с сохранённого места — это и есть требование
«продолжение после перерыва».

Каждый шаг автомата:

1. собирает контекст из того, что уже есть (результаты прошлых фаз);
2. вызывает роль — модель возвращает JSON;
3. **проверяет** результат: схема, а для фактов — дословность цитат;
4. записывает шаг в журнал и сохраняет состояние.

Проверка на шаге 3 — обязательная. Без неё агент записал бы в журнал
правдоподобный, но ничем не подкреплённый вывод.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pf import roles
from pf.config import Settings, settings
from pf.journal import Journal, StepRecord, journal_path
from pf.llm import LLMClient, LLMError, build_client, extract_json
from pf.schemas import (
    Analysis,
    DecisionRecord,
    NextStep,
    RedTeamReport,
    ResearchPlan,
    Strategy,
    TaskFrame,
    Variants,
)
from pf.sources import SourceBundle, collect
from pf.state import Phase, SessionState, save


@dataclass(slots=True)
class StepResult:
    """Результат одного шага.

    Attributes:
        role: Роль, выполнявшая шаг.
        phase: Фаза шага.
        data: Валидированный результат роли.
        citations: Подтверждённые ссылки на источники.
        rejected: Сколько цитат отброшено проверкой.
        cost_units: Стоимость шага.
        tokens: Токены на шаге.
    """

    role: str
    phase: Phase
    data: BaseModelType
    citations: list[str] = field(default_factory=list)
    rejected: int = 0
    rejected_details: list[str] = field(default_factory=list)
    cost_units: float = 0.0
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: int = 0


#: Тип результата роли — любой валидированный объект схемы.
BaseModelType = Any


class Workflow:
    """Исполнитель рабочего процесса агента.

    Attributes:
        client: Клиент модели.
        config: Настройки.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        config: Settings | None = None,
    ) -> None:
        self.client = client or build_client()
        self.config = config or settings

    # ── Вызов роли ──────────────────────────────────────────────────────────

    def _call_role(
        self,
        role_name: str,
        payload: dict[str, Any],
        context: str = "",
    ) -> tuple[BaseModelType, Any]:
        """Вызвать роль и проверить результат.

        Args:
            role_name: Идентификатор роли.
            payload: Данные роли (задача, разбор, варианты и т. п.).
            context: Текстовый контекст (источники, предыдущие результаты).

        Returns:
            Пара ``(валидированный результат, расход)``.

        Raises:
            LLMError: Модель не ответила или ответ не прошёл проверку.
        """
        system = roles.load_prompt(role_name)
        user = self._build_user(payload, context)

        last_error = ""
        attempts = 3
        for _attempt in range(1, attempts + 1):
            response = self.client.complete(system, user)

            try:
                raw = extract_json(response.content)
            except LLMError as exc:
                last_error = f"не удалось разобрать JSON: {exc}"
                user = self._repair_prompt(user, last_error)
                continue

            try:
                validated = roles.validate(role_name, raw)
            except Exception as exc:
                last_error = str(exc)
                user = self._repair_prompt(user, last_error)
                continue

            return validated, response.usage

        raise LLMError(
            f"роль «{role_name}» не дала пригодного ответа за {attempts} попытки: {last_error}"
        )

    @staticmethod
    def _repair_prompt(user: str, error: str) -> str:
        """Добавить к задаче указание, что именно исправить.

        Модель справляется с форматом гораздо лучше, когда видит, **что**
        не так. Поэтому в подсказку попадает текст ошибки валидации.
        """
        return (
            f"{user}\n\n"
            "ПРЕДЫДУЩИЙ ОТВЕТ НЕ ПРИНЯТ. Причина:\n"
            f"{error[:600]}\n\n"
            "Верни ТОЛЬКО JSON строго по схеме. Обязательные поля: "
            "заполни все — пустые строки будут отклонены."
        )

    @staticmethod
    def _build_user(payload: dict[str, Any], context: str) -> str:
        """Собрать текст задачи для модели.

        Args:
            payload: Данные роли.
            context: Текстовый контекст.

        Returns:
            Текст запроса.
        """
        parts = []
        if context:
            parts.append(f"МАТЕРИАЛЫ:\n{context}")
        parts.append("ДАННЫЕ:\n" + json.dumps(payload, ensure_ascii=False, indent=2))
        parts.append("Верни ТОЛЬКО JSON по схеме из инструкции, без пояснений.")
        return "\n\n".join(parts)

    # ── Проверка оснований ──────────────────────────────────────────────────

    @staticmethod
    def _verify_claims(
        data: Any,
        bundle: SourceBundle | None,
    ) -> tuple[list[str], int, list[str]]:
        """Проверить цитаты в результате роли.

        Args:
            data: Результат роли.
            bundle: Собранные источники.

        Returns:
            Пара ``(подтверждённые ссылки, число отброшенных, пояснения)``.
        """
        if bundle is None:
            return [], 0, []

        claims = getattr(data, "facts", None) or []
        confirmed: list[str] = []
        rejected = 0
        details: list[str] = []

        for claim in claims:
            sources = getattr(claim, "sources", None) or []
            claim_text = getattr(claim, "text", "")
            for source in sources:
                support = bundle.verify_claim(
                    claim_text,
                    source_id=getattr(source, "source_id", ""),
                    paragraph=getattr(source, "paragraph", None),
                    quote=getattr(source, "quote", ""),
                )
                named = getattr(source, "source_id", "")
                if not bundle.is_relevant(named):
                    rejected += 1
                    details.append(
                        f"{named or 'источник не назван'}: источник не относится к задаче"
                    )
                    continue
                if support.is_usable:
                    confirmed.append(support.citation)
                else:
                    rejected += 1
                    details.append(f"{named or 'источник не назван'}: {support.detail}")

        return confirmed, rejected, details

    # ── Шаги ────────────────────────────────────────────────────────────────

    def run_step(self, state: SessionState, bundle: SourceBundle | None = None) -> StepResult:
        """Выполнить один шаг автомата по текущей фазе.

        Args:
            state: Состояние сессии.
            bundle: Собранные источники, если фаза их использует.

        Returns:
            Результат шага.

        Raises:
            LLMError: Шаг не удался выполнить.
            ValueError: Для фазы нет роли.
        """
        phase = state.phase
        role_name, payload, context = self._step_inputs(state, bundle)

        data, usage = self._call_role(role_name, payload, context)
        citations, rejected, details = self._verify_claims(data, bundle)

        result = StepResult(
            role=role_name,
            phase=phase,
            data=data,
            citations=citations,
            rejected=rejected,
            rejected_details=details,
            cost_units=usage.cost_units,
            tokens=usage.total_tokens,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            duration_ms=usage.duration_ms,
        )

        state.results[phase.value] = self._serialize(data)
        state.steps_done += 1
        state.cost_units += usage.cost_units
        self._record(state, result)
        return result

    def _step_inputs(
        self,
        state: SessionState,
        bundle: SourceBundle | None,
    ) -> tuple[str, dict[str, Any], str]:
        """Определить роль, данные и контекст для текущей фазы.

        Args:
            state: Состояние сессии.
            bundle: Собранные источники.

        Returns:
            Пара ``(роль, данные, контекст)``.

        Raises:
            ValueError: Для фазы нет роли.
        """
        phase = state.phase

        if phase is Phase.INTAKE:
            return "intake", {"question": state.question}, ""

        if phase is Phase.PLANNING:
            frame = state.results.get(Phase.INTAKE.value, {})
            return "planner", {"task": frame}, ""

        if phase is Phase.COLLECTING:
            plan = state.results.get(Phase.PLANNING.value, {})
            frame = state.results.get(Phase.INTAKE.value, {})
            query = " ".join(
                [
                    state.question,
                    str(frame.get("goal", "")),
                    " ".join(plan.get("questions", [])[:5]),
                ]
            )
            return "collector", {"query": query, "plan": plan}, ""

        if phase is Phase.ANALYZING:
            collector = state.results.get(Phase.COLLECTING.value, {})
            context = self._bundle_context(bundle)
            return "analyst", {"collected": collector}, context

        if phase is Phase.OPTIONS:
            analysis = state.results.get(Phase.ANALYZING.value, {})
            frame = state.results.get(Phase.INTAKE.value, {})
            return "options", {"task": frame, "analysis": analysis}, ""

        if phase is Phase.REDTEAM:
            options = state.results.get(Phase.OPTIONS.value, {})
            analysis = state.results.get(Phase.ANALYZING.value, {})
            return "redteam", {"variants": options, "analysis": analysis}, ""

        if phase is Phase.STRATEGY:
            options = state.results.get(Phase.OPTIONS.value, {})
            redteam = state.results.get(Phase.REDTEAM.value, {})
            analysis = state.results.get(Phase.ANALYZING.value, {})
            frame = state.results.get(Phase.INTAKE.value, {})
            context = self._bundle_context(bundle)
            return (
                "strategy",
                {
                    "task": frame,
                    "analysis": analysis,
                    "variants": options,
                    "redteam": redteam,
                },
                context,
            )

        if phase is Phase.KEEPER:
            strategy = state.results.get(Phase.STRATEGY.value, {})
            redteam = state.results.get(Phase.REDTEAM.value, {})
            return "keeper", {"strategy": strategy, "redteam": redteam}, ""

        raise ValueError(f"для фазы «{phase.value}» нет роли")

    #: Сколько абзацев источников уходит в контекст модели. Ограничение
    #: существенно: на длинном контексте локальная модель малого размера
    #: начинает терять формат ответа и возвращать «историю диалога» вместо
    #: JSON. Пять абзацев — это и качество, и управляемость.
    CONTEXT_PASSAGES = 5

    @classmethod
    def _bundle_context(cls, bundle: SourceBundle | None) -> str:
        """Собрать контекст из источников."""
        if bundle is None:
            return ""
        return bundle.context(limit=cls.CONTEXT_PASSAGES)

    @staticmethod
    def _serialize(data: Any) -> dict[str, Any]:
        """Сериализовать результат роли для состояния."""
        if hasattr(data, "model_dump"):
            dumped: dict[str, Any] = data.model_dump(mode="json")
            return dumped
        if isinstance(data, dict):
            return dict(data)
        return {"value": str(data)}

    def _record(self, state: SessionState, result: StepResult) -> None:
        """Записать шаг в журнал и сохранить состояние."""
        journal = Journal(journal_path(state.session_id))
        record = StepRecord(
            ordinal=state.steps_done,
            role=result.role,
            phase=result.phase.value,
            summary=self._summarize(result),
            detail=self._serialize(result.data),
            citations=result.citations,
            rejected_details=result.rejected_details,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_units=result.cost_units,
            duration_ms=result.duration_ms,
        )
        journal.append(record)
        save(state)

    @staticmethod
    def _summarize(result: StepResult) -> str:
        """Короткое описание шага для журнала."""
        data = result.data
        if isinstance(data, TaskFrame):
            return f"задача: {data.question[:80]}"
        if isinstance(data, ResearchPlan):
            return f"план: {len(data.questions)} вопросов"
        if isinstance(data, Analysis):
            return (
                f"фактов: {len(data.facts)}, гипотез: {len(data.hypotheses)}, "
                f"пробелов: {len(data.gaps)}"
            )
        if isinstance(data, Variants):
            return f"вариантов: {len(data.variants)}"
        if isinstance(data, RedTeamReport):
            return f"находок: {len(data.findings)}"
        if isinstance(data, Strategy):
            return "стратегия готова"
        if isinstance(data, DecisionRecord):
            return f"обязательств: {len(data.commitments)}"
        if isinstance(data, NextStep):
            return f"маршрут: {data.role}"
        return result.role

    # ── Сбор источников ─────────────────────────────────────────────────────

    def collect_sources(self, state: SessionState) -> SourceBundle:
        """Собрать источники под текущую задачу.

        Args:
            state: Состояние сессии.

        Returns:
            Набор источников.
        """
        plan = state.results.get(Phase.PLANNING.value, {})
        frame = state.results.get(Phase.INTAKE.value, {})
        questions = plan.get("questions", [])[:5]
        query = " ".join([state.question, str(frame.get("goal", "")), *questions])
        return collect(query)


def run_full_workflow(
    question: str,
    session_id: str,
    *,
    client: LLMClient | None = None,
    config: Settings | None = None,
    max_steps: int = 12,
    on_step: Any = None,
) -> SessionState:
    """Пройти весь рабочий процесс до стратегии.

    Args:
        question: Вопрос руководителя продукта.
        session_id: Идентификатор сессии.
        client: Клиент модели.
        config: Настройки.
        max_steps: Предохранитель от бесконечного цикла.
        on_step: Обработчик шага ``(state, result) -> None`` — для вывода
            прогресса и поддержки интерактивной остановки.

    Returns:
        Итоговое состояние сессии.

    Raises:
        LLMError: Шаг не удался выполнить.
    """
    workflow = Workflow(client=client, config=config)
    state = SessionState(session_id=session_id, phase=Phase.INTAKE, question=question)
    save(state)

    bundle: SourceBundle | None = None
    steps = 0

    while state.phase is not Phase.DONE and steps < max_steps:
        if state.phase is Phase.COLLECTING:
            bundle = workflow.collect_sources(state)

        result = workflow.run_step(state, bundle)
        steps += 1

        if on_step is not None:
            on_step(state, result)

        if state.phase is Phase.KEEPER:
            state.phase = Phase.DONE
            save(state)
            break

        state.phase = state.phase.next
        save(state)

    return state
