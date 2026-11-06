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


def _describe_fields(
    spec: dict[str, Any],
    defs: dict[str, Any],
    indent: int,
) -> list[str]:
    """Описать поля объекта схемы, развернув ссылки на модели.

    Args:
        spec: Описание поля или объекта.
        defs: Определения моделей из ``$defs``.
        indent: Отступ для выравнивания.

    Returns:
        Строки описания полей.
    """
    pad = " " * indent
    ref = spec.get("$ref")
    if isinstance(ref, str):
        name = ref.rsplit("/", 1)[-1]
        resolved = defs.get(name)
        if resolved is None:
            return [f"{pad}(модель {name})"]
        spec = resolved

    properties = spec.get("properties")
    if not properties:
        # Массив: описываем его элемент.
        items = spec.get("items")
        if isinstance(items, dict) and _nested_properties(items, defs):
            return _describe_fields(items, defs, indent)
        return [f"{pad}{_type_name(spec)}"]

    required = set(spec.get("required", []))
    lines: list[str] = []
    for name, name_spec in properties.items():
        mark = "" if name in required else " (необязательное)"
        nested = _nested_properties(name_spec, defs)
        if nested:
            lines.append(f"{pad}{name}:")
            lines.extend(_describe_fields(name_spec, defs, indent + 2))
            continue
        # Массив объектов: показываем поля элемента, иначе модель не знает,
        # что класть в список, и отвечает пустым списком.
        items = name_spec.get("items") if name_spec.get("type") == "array" else None
        if items is not None:
            item_props = _nested_properties(items, defs)
            if item_props:
                lines.append(f"{pad}{name}: list of")
                lines.extend(_describe_fields(items, defs, indent + 2))
                continue
        lines.append(f"{pad}{name}: {_type_name(name_spec)}{mark}")
    return lines


def _nested_properties(spec: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any] | None:
    """Свойства объекта, если схема на него ссылается или описывает напрямую.

    Args:
        spec: Описание поля.
        defs: Определения моделей из ``$defs``.

    Returns:
        Описание свойств объекта либо ``None``.
    """
    direct = spec.get("properties")
    if direct:
        return dict(direct)
    ref = spec.get("$ref")
    if isinstance(ref, str):
        resolved = defs.get(ref.rsplit("/", 1)[-1], {})
        properties = resolved.get("properties")
        if properties:
            return dict(properties)
    return None


def _as_text(payload: Any, indent: int = 0) -> str:
    """Подать входные данные текстом, а не JSON.

    Это измерено: при JSON во входе модель отвечает его эхом — на
    материалах по удержанию возвращала пустые списки фактов (за три
    попытки: 1, 0, 0), а на текстовой подаче — факт каждый раз (1, 1, 1).
    Причина в том, что JSON задаёт форму ответа, и модель заполняет её
    структурой входа вместо того, чтобы извлечь содержимое.

    Args:
        payload: Данные для подачи.
        indent: Текущий отступ.

    Returns:
        Текстовое представление данных.
    """
    pad = " " * indent
    if isinstance(payload, dict):
        lines = []
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(_as_text(value, indent + 2))
            else:
                lines.append(f"{pad}{key}: {value}")
        return "\n".join(lines)
    if isinstance(payload, list):
        lines = []
        for item in payload:
            if isinstance(item, (dict, list)):
                lines.append(_as_text(item, indent + 2))
            else:
                lines.append(f"{pad}- {item}")
        return "\n".join(lines)
    return f"{pad}{payload}"


def _type_name(spec: dict[str, Any]) -> str:
    """Описать тип поля схемы в человекочитаемом виде.

    Args:
        spec: Описание поля из JSON-схемы.

    Returns:
        Название типа.
    """
    kind = spec.get("type", "any")
    if kind == "array":
        return f"list[{_type_name(spec.get('items', {}))}]"
    return {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "object": "object",
        "null": "null",
    }.get(str(kind), str(kind))


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

#: Соответствие «роль → фаза». По нему решение роутера превращается в
#: следующий шаг рабочего процесса.
#: Фазы, которым нужны собранные источники. Для остальных набор не
#: собирается: это экономит время и не засоряет контекст модели.
_PHASES_WITH_SOURCES: frozenset[Phase] = frozenset(
    {Phase.COLLECTING, Phase.ANALYZING, Phase.STRATEGY}
)

_PHASE_BY_ROLE: dict[str, Phase] = {
    "intake": Phase.INTAKE,
    "planner": Phase.PLANNING,
    "router": Phase.ROUTING,
    "collector": Phase.COLLECTING,
    "analyst": Phase.ANALYZING,
    "options": Phase.OPTIONS,
    "redteam": Phase.REDTEAM,
    "strategy": Phase.STRATEGY,
    "keeper": Phase.KEEPER,
}


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
        user = self._build_user(payload, context, role_name)

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

        Ошибка «факт должен ссылаться хотя бы на один источник» для
        модели абстрактна: она не понимает, что нужно не «дописать
        ссылку», а перенести утверждение из фактов в гипотезы. Поэтому
        для этой ошибки добавляется явное объяснение.
        """
        hint = ""
        if "ссылаться хотя бы на один источник" in error:
            hint = (
                "\n\nВАЖНО: у факта в поле `sources` пустой массив. Это значит, "
                "что утверждение не подтверждено цитатой из предоставленных "
                "абзацев. Перенеси его в `hypotheses` (там sources не нужен) "
                "либо укажи в `sources` реальный `source_id` из блока "
                "МАТЕРИАЛЫ и дословную `quote` из него. Не выдумывай цитату."
            )
        elif "чужой бренд" in error:
            hint = (
                "\n\nВАЖНО: в `key_message` назван чужой бренд. "
                "Коммуникация обращается к аудитории НАШЕГО продукта: "
                "напиши сообщение от своего имени, без названия конкурента. "
                "Конкурент может остаться в `barrier` — это контекст рынка."
            )
        return (
            f"{user}\n\n"
            "ПРЕДЫДУЩИЙ ОТВЕТ НЕ ПРИНЯТ. Причина:\n"
            f"{error[:600]}{hint}\n\n"
            "Верни ТОЛЬКО JSON строго по схеме. Обязательные поля: "
            "заполни все — пустые строки будут отклонены."
        )

    @staticmethod
    def _build_user(
        payload: dict[str, Any],
        context: str,
        role_name: str = "",
    ) -> str:
        """Собрать текст задачи для модели.

        В запрос входит **схема ответа** — из той же схемы, по которой
        потом проверяется результат. Это не украшение: при длинном входе
        модель возвращает структуру того, что видела (ключи «variants»,
        «analysis»), вместо требуемой. Схема в промпте при этом
        вытесняется объёмом данных и перестаёт работать.

        Args:
            payload: Данные роли.
            context: Текстовый контекст.
            role_name: Роль, для которой строится запрос.

        Returns:
            Текст запроса.
        """
        parts = []
        if context:
            parts.append(f"МАТЕРИАЛЫ:\n{context}")
        parts.append("ДАННЫЕ:\n" + _as_text(payload))
        if role_name:
            parts.append(Workflow._schema_hint(role_name))
        parts.append("Верни ТОЛЬКО JSON по схеме выше, без пояснений.")
        return "\n\n".join(parts)

    @staticmethod
    def _schema_hint(role_name: str) -> str:
        """Собрать подсказку со схемой ответа роли.

        Вложенные модели разворачиваются: в JSON-схеме они лежат в
        ``$defs`` под именем класса и подставляются ссылкой ``$ref``.
        Без развёртки подсказка показывала бы пустые блоки — ровно то,
        из-за чего модель возвращала не те поля.

        Args:
            role_name: Идентификатор роли.

        Returns:
            Текст с перечнем полей ответа.
        """
        try:
            schema = roles.schema_for(role_name).model_json_schema()
        except KeyError:
            return ""
        defs = schema.get("$defs", {})
        lines = [
            "СХЕМА ОТВЕТА (строго эти ключи, других не возвращай):",
        ]
        for name, spec in schema.get("properties", {}).items():
            lines.append(f"  {name}:")
            lines.extend(_describe_fields(spec, defs, indent=2))
        lines.append(
            "Ключи из блока ДАННЫЕ — это входные данные, их не нужно "
            "возвращать. Верни только поля своей схемы."
        )
        return "\n".join(lines)

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

        confirmed_metrics, rejected_metrics, metric_details = Workflow._verify_metrics(data)
        confirmed.extend(confirmed_metrics)
        rejected += rejected_metrics
        details.extend(metric_details)

        return confirmed, rejected, details

    @staticmethod
    def _verify_metrics(data: Any) -> tuple[list[str], int, list[str]]:
        """Проверить расчётные метрики пересчётом по источнику.

        Факт подтверждается дословной цитатой. Метрику подтверждает
        только пересчёт: число нельзя «процитировать», его можно только
        посчитать заново. Метрика без источника или с неверным
        значением отбрасывается — как и выдуманная цитата.

        Args:
            data: Результат роли.

        Returns:
            Пара ``(подтверждённые ссылки, число отброшенных, пояснения)``.
        """
        metrics = getattr(data, "metrics", None) or []
        if not metrics:
            return [], 0, []

        from pf.materials import load_materials
        from pf.sources import verify_metric

        materials = load_materials()
        confirmed: list[str] = []
        rejected = 0
        details: list[str] = []
        for metric in metrics:
            support = verify_metric(metric, materials)
            if support.is_usable:
                confirmed.append(support.citation)
            else:
                rejected += 1
                name = getattr(metric, "name", "метрика")
                details.append(f"{name}: {support.detail}")
        return confirmed, rejected, details

    def _run_fact_check(
        self,
        state: SessionState,
        bundle: SourceBundle | None,
        claims: list[Any],
    ) -> Any:
        """Позвать субагента fact-check по собранным основаниям.

        Субагент отвечает на другой вопрос, чем основной проверяющий:
        не «найдена ли цитата», а «относится ли источник к задаче». Он
        идёт от факта к источнику, а не от цитаты к тексту.

        Args:
            state: Состояние сессии.
            bundle: Локальные источники.
            claims: Утверждения для проверки.

        Returns:
            Отчёт субагента либо ``None``, если проверять нечего.
        """
        from pf.subagents import check_facts

        if not claims or bundle is None:
            return None

        def verify(
            text: str,
            source_id: str,
            paragraph: int | None,
            quote: str,
        ) -> Any:
            support = bundle.verify_claim(
                text,
                source_id=source_id,
                paragraph=paragraph,
                quote=quote,
            )
            named = source_id or "источник не назван"
            if not bundle.is_relevant(named):
                from pf.ground import Support, SupportStatus

                return Support(
                    quote=quote,
                    source_id=source_id,
                    status=SupportStatus.NO_DATA,
                    detail="источник не относится к задаче",
                )
            return support

        return check_facts(claims, verify)

    def route_next(self, state: SessionState, result: StepResult) -> Phase:
        """Определить следующую фазу по решению роутера.

        Роутер выбирает, кто работает следующим. Решение может быть
        ошибочным — и тогда его надо исправить, а не выполнять:

        - роль, которой нет в реестре, — берётся обычный порядок;
        - фаза, которая уже позади, — это откат назад: маршрут
          отбрасывается, работа продолжается по порядку;
        - та же фаза, что и текущая, — зацикливание: берётся следующая.

        Args:
            state: Состояние сессии.
            result: Результат шага роутера.

        Returns:
            Фаза, которая должна выполняться следующей.
        """
        from pf.roles import ROLES

        default = state.phase.next
        data = result.data
        if not isinstance(data, NextStep):
            return default

        chosen = data.role.strip().lower()
        if chosen not in ROLES:
            state.results["routing_note"] = (
                f"маршрут «{data.role}» неизвестен — идём по порядку: {default.title_ru}"
            )
            return default

        target = _PHASE_BY_ROLE.get(chosen)
        if target is None:
            return default

        order = list(Phase)
        if order.index(target) <= order.index(state.phase):
            state.results["routing_note"] = (
                f"маршрут «{data.role}» ведёт назад — исправлено, "
                f"идём по порядку: {default.title_ru}"
            )
            return default

        state.results["routing_note"] = (
            f"маршрут: {data.role} — {target.title_ru}. Причина: {data.reason or 'не указана'}"
        )
        return target

    # ── Шаги ─────────────────────────────────────────────────────────────────

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
        self._apply_hooks(state, result)

        # Субагент включается после разбора: он перепроверяет основания
        # с другой стороны — не «найдена ли цитата», а «относится ли
        # источник к задаче». Его отчёт виден в журнале.
        if phase is Phase.ANALYZING:
            claims = getattr(data, "facts", None) or []
            report = self._run_fact_check(state, bundle, list(claims))
            if report is not None:
                state.results["fact_check"] = {
                    "summary": report.summary(),
                    "confirmed": report.confirmed,
                    "needs_review": report.needs_review,
                    "no_data": report.no_data,
                    "findings": [
                        {
                            "source_id": item.source_id,
                            "paragraph": item.paragraph,
                            "status": item.status,
                            "detail": item.detail,
                        }
                        for item in report.findings
                    ],
                }
        return result

    def _search_web(self, state: SessionState) -> list[Any]:
        """Выйти в интернет, если локальных оснований не хватает.

        Внешние источники никогда не становятся подтверждёнными: их
        статус — «требует проверки». Поэтому в журнале видно, что
        основание внешнее и ждёт человека.

        Args:
            state: Состояние сессии.

        Returns:
            Собранные внешние источники.
        """
        if not self.config.enable_web:
            return []
        from pf import webfetch
        from pf.web import as_context, collect_web, search_web, summarize
        from pf.webfetch import fetch_pages

        query = state.question
        sources = collect_web(query)
        pages = fetch_pages(search_web(query, limit=self.config.web_search_limit))
        if not sources and not pages:
            return []

        state.results["web_sources"] = summarize(sources) + webfetch.summarize(pages)
        # Веб-контекст попадает в модель вместе с локальными абзацами:
        # иначе агент выходит в интернет, но не видит, что там нашёл.
        state.results["web_context"] = as_context(sources) + "\n\n" + webfetch.as_context(pages)
        return pages

    def _apply_hooks(self, state: SessionState, result: StepResult) -> None:
        """Применить хуки к результату шага.

        Два хука:

        - ``sources.collected`` — подтверждённых оснований мало, значит
          вывод пока нечем подкрепить: план расширяется, чтобы поиск шёл
          дальше;
        - ``budget.exhausted`` — расход достиг предела: работа
          останавливается, чтобы не жечь токены впустую.

        Args:
            state: Состояние сессии.
            result: Результат шага.
        """
        from pf.hooks import expand_plan, fire

        collected = fire("sources.collected", float(len(result.citations)))
        if collected.fired and result.phase is Phase.COLLECTING:
            added = expand_plan(state)
            if added:
                state.results["plan_expanded"] = added
            self._search_web(state)

        limit = self.config.max_cost_units
        if limit <= 0:
            return
        exhausted = fire("budget.exhausted", state.cost_units)
        if exhausted.fired:
            state.results["stopped_at"] = state.phase.value
            state.pending_question = (
                f"Расход достиг предела ({state.cost_units:.6f} из {limit:.6f}). "
                "Продолжить или снять предел?"
            )

    def _step_inputs(
        self,
        state: SessionState,
        bundle: SourceBundle | None,
    ) -> tuple[str, dict[str, Any], str]:
        """Определить роль, данные и контекст для текущей фазы.

        К контексту добавляются ответы человека: они должны видеть любая
        следующая фаза, а не только первая. Иначе агент спрашивает,
        человек отвечает — и ответ теряется: работа продолжается вслепую.

        Args:
            state: Состояние сессии.
            bundle: Собранные источники.

        Returns:
            Пара ``(роль, данные, контекст)``.

        Raises:
            ValueError: Для фазы нет роли.
        """
        role, payload, context = self._phase_inputs(state, bundle)
        human = self._human_context(state)
        if human:
            context = f"{human}\n\n{context}" if context else human
        return role, payload, context

    def _phase_inputs(
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

        if phase is Phase.ROUTING:
            frame = state.results.get(Phase.INTAKE.value, {})
            plan = state.results.get(Phase.PLANNING.value, {})
            return "router", {"task": frame, "plan": plan}, ""

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
            # Сборщику не передаются вопросы плана. Это измерено: с
            # вопросами («каковы текущие показатели удержания») модель
            # отвечала «данных нет» и не извлекала ничего, хотя в
            # материалах было что цитировать. Вопросы плана нужны для
            # **поиска** источников — они уже учтены в запросе, — но не
            # для **извлечения** фактов: перед сборщиком уже отобранные
            # абзацы, и его дело — взять из них то, что есть.
            # Сборщик обязан видеть материалы: без них он не может выбрать
            # основания и сочиняет цитаты. Контекст — те же абзацы, что
            # потом получит аналитик. Внешние источники идут отдельным
            # блоком: их статус — «требует проверки», и смешивать их с
            # локальными основаниями нельзя.
            context = self._materials(bundle, state)
            return "collector", {"query": query}, context

        if phase is Phase.ANALYZING:
            collector = state.results.get(Phase.COLLECTING.value, {})
            context = self._materials(bundle, state)
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
            context = self._materials(bundle, state)
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
    def _web_context(state: SessionState) -> str:
        """Собрать контекст из внешних источников.

        Веб-материалы лежат отдельно от локальных: у них другой статус
        подтверждения, и смешивать их нельзя — иначе агент перестаёт
        различать «подтверждено» и «требует проверки».

        Args:
            state: Состояние сессии.

        Returns:
            Текст внешних источников либо пустая строка.
        """
        context = state.results.get("web_context")
        return context if isinstance(context, str) else ""

    @staticmethod
    def _computed_metrics(state: SessionState) -> str:
        """Готовые расчёты по материалам проекта.

        Считать должен код, а не модель: числа из таблиц — это то, что
        можно пересчитать и проверить. Агенту отдаются уже посчитанные
        значения с указанием строки и формулы, чтобы он опирался на них
        при построении плана проверки.

        Args:
            state: Состояние сессии.

        Returns:
            Текст блока с расчётами либо пустая строка.
        """
        from pf.analysis import compute_from_rows, summarize_series
        from pf.materials import load_materials

        materials = load_materials()
        blocks: list[str] = []
        for material in materials:
            if not material.rows:
                continue
            columns = list(material.rows[0].keys())
            value_column = next(
                (name for name in columns if "знач" in name.lower() or "value" in name.lower()),
                None,
            )
            if value_column is None:
                continue
            label_column = next(
                (name for name in columns if name != value_column),
                None,
            )
            metrics = compute_from_rows(
                material.rows,
                value_column=value_column,
                source_id=material.source_id,
                name=material.title,
                label_column=label_column or "",
            )
            if not metrics:
                continue
            lines = [
                f"- {metric.name}: {metric.formatted} [{metric.citation}] ({metric.formula})"
                for metric in metrics[:10]
            ]
            summary = summarize_series(metrics, name=material.title, unit=metrics[0].unit)
            lines.extend(
                f"- {metric.name}: {metric.formatted} [{metric.citation}] ({metric.formula})"
                for metric in summary
            )
            blocks.append(f"РАСЧЁТЫ по {material.source_id}:\n" + "\n".join(lines))
        if not blocks:
            return ""
        return "\n\n".join(blocks)

    @classmethod
    def _materials(cls, bundle: SourceBundle | None, state: SessionState) -> str:
        """Собрать все материалы для роли: локальные и внешние.

        Разделены намеренно: подтверждённые основания приходят из
        локального среза, внешние материалы требуют проверки человеком.
        Если смешать их в один список, агент перестанет различать, где
        правда, а где гипотеза.

        Args:
            bundle: Локальные источники.
            state: Состояние сессии.

        Returns:
            Текст материалов.
        """
        local = cls._bundle_context(bundle)
        web = cls._web_context(state)
        computed = cls._computed_metrics(state)
        parts = []
        if local:
            parts.append(local)
        if computed:
            parts.append(computed)
        if web:
            parts.append(
                "ВНЕШНИЕ ИСТОЧНИКИ (статус: требует проверки человеком, не подтверждено):\n" + web
            )
        return "\n\n".join(parts)

    @staticmethod
    def _human_context(state: SessionState) -> str:
        """Собрать ответы человека для роли.

        Ответ, который человек дал на вопрос агента, должен попасть в
        контекст следующего шага. Иначе агент спрашивает, человек
        отвечает — и ответ теряется: работа продолжается вслепую.

        Args:
            state: Состояние сессии.

        Returns:
            Текст ответов либо пустая строка.
        """
        answers = state.results.get("human_answers") or []
        if not answers:
            return ""
        lines = [f"Ответ человека: {text}" for text in answers]
        return "\n".join(lines)

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

    def bundle_for(self, state: SessionState) -> SourceBundle | None:
        """Источники, нужные текущей фазе.

        Фазы, которые работают с материалами, получают собранный набор
        источников; остальные — ``None``. Вынесено отдельно, потому что
        продолжение после перерыва должно собрать источники заново: если
        остановились на сборе и продолжить без источников, сборщик
        получит пустой блок МАТЕРИАЛЫ и начнёт сочинять цитаты.

        Args:
            state: Состояние сессии.

        Returns:
            Набор источников либо ``None``, если фаза их не использует.
        """
        if state.phase in _PHASES_WITH_SOURCES:
            return self.collect_sources(state)
        return None


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

        # Человек подключается там, где без его решения работа потеряла
        # бы смысл. Это требование брифа: агент сам ведёт работу, но
        # подключает человека при существенном выборе.
        if isinstance(result.data, NextStep) and result.data.need_human:
            question = result.data.question or "Нужно ваше решение, чтобы продолжить."
            state.pending_question = question
            state.results.setdefault("human_questions", []).append(question)
            save(state)
            break

        if state.phase is Phase.KEEPER:
            state.phase = Phase.DONE
            save(state)
            break

        if state.phase is Phase.ROUTING:
            state.phase = workflow.route_next(state, result)
        else:
            state.phase = state.phase.next
        save(state)

    return state
