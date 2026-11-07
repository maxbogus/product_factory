"""Отчёт по прогону: числа вместо пересказа.

Демонстрация должна показывать не «агент работал», а измеримые
результаты: сколько оснований подтверждено, сколько отклонено, сколько
стоит прогон и что именно замер **не** показывает.

Поэтому отчёт строится по журналу шагов (append-only, значит, цифры
воспроизводимы) и содержит:

- числа по основаниям: подтверждено, отклонено, доля подтверждённых;
- расход: итог и по ролям;
- раздел «чего этот замер не показывает» — обязательный. Метрика без
  оговорок о границах выглядит точнее, чем она есть.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from pf.journal import Journal, journal_path
from pf.state import load

#: Подпись расчёта, у которого не указана формула. Непересчитываемое
#: число нельзя показывать так же, как проверенное.
NO_FORMULA = "формула не указана — пересчитать нечем"


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """Расчётная метрика в отчёте.

    У каждой метрики есть статус: посчитанное и проверенное значение
    и отклонённое — это разные вещи, и путать их нельзя. Раньше
    отклонённые расчёты попадали в документ как «посчитанные кодом по
    строкам таблицы».

    Attributes:
        name: Название метрики.
        value: Значение.
        unit: Единица измерения.
        citation: Ссылка на строку источника.
        formula: Как считалось.
        inputs: Входные значения.
        status: Статус проверки: «подтверждено» или «отклонено».
        detail: Пояснение для человека.
    """

    name: str
    value: float
    unit: str = ""
    citation: str = ""
    formula: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    status: str = "подтверждено"
    detail: str = ""

    @property
    def formatted(self) -> str:
        """Значение с единицей измерения."""
        if not self.unit:
            return f"{self.value:.6g}"
        return f"{self.value:.6g} {self.unit}"

    @property
    def formula_note(self) -> str:
        """Формула расчёта или явная пометка, что её нет.

        Пустая колонка «Как считалось» читалась бы как «всё в порядке»,
        хотя означает обратное: число нечем пересчитать. Поэтому
        отсутствие формулы называется словами.
        """
        return self.formula.strip() or NO_FORMULA


@dataclass(frozen=True, slots=True)
class ConclusionRecord:
    """Вывод прогона: что решили и на чём это держится.

    Отчёт о замере отвечает на вопрос «сколько оснований подтверждено», а
    руководителю нужен ответ на заданный вопрос. Поэтому вывод собирается
    из журнала так же, как расчёты: направление и обязательства — из шагов
    стратегии и решения, статус — из проверки их оснований.

    Attributes:
        direction: Выбранное направление — что делать.
        value_to_cost: Почему это выгоднее альтернатив.
        deliberate_omissions: Что сознательно не учтено.
        decision: Зафиксированное решение.
        commitments: Принятые обязательства.
        expected_change: Ожидаемое изменение.
        compare_against: С чем сравниваем.
        observation_period: Срок наблюдения.
        adverse_effects: Нежелательные последствия.
        status: Статус оснований вывода.
        checked: Сколько оснований вывода проверено.
        confirmed: Сколько подтверждено.
        rejected: Сколько отклонено.
        grounds: Подтверждённые ссылки вывода.
        rejected_grounds: Отклонённые ссылки вывода.
    """

    direction: str = ""
    value_to_cost: str = ""
    deliberate_omissions: list[str] = field(default_factory=list)
    decision: str = ""
    commitments: list[dict[str, Any]] = field(default_factory=list)
    expected_change: str = ""
    compare_against: str = ""
    observation_period: str = ""
    adverse_effects: list[str] = field(default_factory=list)
    status: str = ""
    checked: int = 0
    confirmed: int = 0
    rejected: int = 0
    grounds: list[str] = field(default_factory=list)
    rejected_grounds: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SessionReport:
    """Отчёт по одной сессии.

    Attributes:
        session_id: Идентификатор сессии.
        steps: Сколько шагов выполнено.
        confirmed_citations: Сколько оснований подтверждено.
        rejected_citations: Сколько оснований отклонено.
        total_cost_units: Расход в условных единицах.
        cost_by_role: Расход по ролям.
        tokens: Токены по ролям.
        rejected_details: Что и почему отклонено.
        question: Исходный вопрос.
        phase: Фаза, на которой остановилась сессии.
        conclusion: Вывод прогона и статус его оснований.
        needs_review_details: Основания, ожидающие проверки человеком.
        lexical: Лексическое покрытие прогона и топ незнакомых слов.
        context: Окно контекста: сколько шагов, сколько упёрлось в предел.
    """

    session_id: str
    steps: int = 0
    confirmed_citations: int = 0
    rejected_citations: int = 0
    total_cost_units: float = 0.0
    cost_by_role: dict[str, float] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    rejected_details: list[str] = field(default_factory=list)
    question: str = ""
    phase: str = ""
    metrics: list[MetricRecord] = field(default_factory=list)
    unique_citations: int = 0
    unique_sources: int = 0
    routing_notes: list[str] = field(default_factory=list)
    fact_check: dict[str, Any] | None = None
    rejection_kinds: dict[str, int] = field(default_factory=dict)
    metrics_with_formula: int = 0
    conclusion: ConclusionRecord | None = None
    contradictions: list[str] = field(default_factory=list)
    needs_review_details: list[str] = field(default_factory=list)
    lexical: dict[str, Any] | None = None
    context: dict[str, Any] | None = None

    @property
    def total_citations(self) -> int:
        """Всего оснований: подтверждённых и отклонённых."""
        return self.confirmed_citations + self.rejected_citations

    @property
    def grounding_rate(self) -> float:
        """Доля подтверждённых оснований.

        Returns:
            Доля от 0 до 1; ноль, если оснований не было.
        """
        if not self.total_citations:
            return 0.0
        return self.confirmed_citations / self.total_citations

    @property
    def limitations(self) -> list[str]:
        """Чего этот замер не показывает.

        Returns:
            Список оговорок.
        """
        items = [
            "Подтверждение основания — дословное совпадение цитаты с "
            "источником. Оно не гарантирует, что цитата поддерживает "
            "утждение по смыслу: честная цитата из настоящей статьи "
            "проходит проверку.",
            "Доля подтверждённых оснований не измеряет полноту: "
            "несколько подтверждённых оснований не означают, что "
            "пропущенных не было.",
            "Расход в условных единицах — это объём токенов. Деньги "
            "зависят от тарифа модели, а время — от её скорости.",
            "Замер сделан на одной модели и одной задаче: разброс между "
            "прогонами локальных моделей велик, поэтому сравнения "
            "считаются по нескольким прогонам.",
        ]
        # Оговорка об окне контекста появляется только тогда, когда обрезка
        # действительно была: постоянная оговорка перестаёт читаться.
        if self.context and int(self.context.get("truncated") or 0) > 0:
            truncated = int(self.context["truncated"])
            steps = int(self.context.get("steps") or 0)
            limit = int(self.context.get("limit") or 0)
            items.append(
                f"Промпт упирался в окно контекста ({limit} токенов) на "
                f"{truncated} шагах из {steps}: начало запроса отброшено "
                "моделью, поэтому часть материалов она не видела. "
                "Поднимать окно — по замеру (`docs/decisions.md`, D-20)."
            )
        return items

    def as_markdown(self) -> str:
        """Оформить отчёт в markdown.

        Returns:
            Текст отчёта.
        """
        lines: list[str] = []
        lines.append(f"# Отчёт по прогону: {self.session_id}")
        lines.append("")
        if self.question:
            lines.append(f"**Задача:** {self.question}")
            lines.append("")

        lines.append("## Основы")
        lines.append("")
        lines.append("| Показатель | Значение |")
        lines.append("|---|---:|")
        lines.append(f"| Шагов выполнено | {self.steps} |")
        lines.append(f"| Подтверждённых записей | {self.confirmed_citations} |")
        lines.append(f"| Подтверждённых оснований (уникальных) | {self.unique_citations} |")
        lines.append(f"| Использовано источников | {self.unique_sources} |")
        lines.append(f"| Оснований отклонено | {self.rejected_citations} |")
        lines.append(f"| Ожидают проверки человеком | {len(self.needs_review_details)} |")
        lines.append(f"| Доля подтверждённых | {self.grounding_rate:.0%} |")
        lines.append("")

        if self.metrics:
            lines.append("## Расчёты по данным")
            lines.append("")
            lines.append("| Метрика | Значение | Строка источника | Как считалось | Статус |")
            lines.append("|---|---:|---|---|---|")
            for metric in self.metrics:
                lines.append(
                    f"| {metric.name} | {metric.formatted} | "
                    f"`{metric.citation}` | {metric.formula_note} | {metric.status} |"
                )
            lines.append("")

        lines.append("## Контроль качества")
        lines.append("")
        lines.append("| Что проверено | Значение |")
        lines.append("|---|---:|")
        lines.append(
            f"| Расчётов с формулой | {self.metrics_with_formula} из {len(self.metrics)} |"
        )
        if self.context:
            lines.append(f"| Окно контекста | {int(self.context.get('limit') or 0)} |")
            if int(self.context.get("truncated") or 0) > 0:
                lines.append(
                    f"| Промпт упёрся в окно | {int(self.context['truncated'])} "
                    f"из {int(self.context.get('steps') or 0)} шагов |"
                )
        for kind, count in sorted(
            self.rejection_kinds.items(), key=lambda item: (-item[1], item[0])
        ):
            lines.append(f"| Отклонено: {kind} | {count} |")
        if self.fact_check:
            summary = self.fact_check.get("summary", "—")
            lines.append(f"| Повторная проверка (fact-check) | {summary} |")
        lines.append("")
        if self.routing_notes:
            lines.append("**Решения роутера.**")
            lines.append("")
            for note in self.routing_notes:
                lines.append(f"- {note}")
            lines.append("")

        if self.contradictions:
            lines.append("**Противоречия в фактах.**")
            lines.append("")
            for item in self.contradictions:
                lines.append(f"- {item}")
            lines.append("")

        lines.extend(conclusion_markdown(self.conclusion))

        lines.append("## Расход")
        lines.append("")
        lines.append("| Роль | Расход, усл. ед. | Токенов |")
        lines.append("|---|---:|---:|")
        for role, cost in sorted(self.cost_by_role.items(), key=lambda item: -item[1]):
            lines.append(f"| {role} | {cost:.6f} | {self.tokens.get(role, 0)} |")
        lines.append(f"| **Итого** | **{self.total_cost_units:.6f}** | |")
        lines.append("")

        if self.lexical:
            lines.append("**Лексическое покрытие.**")
            lines.append("")
            lines.append("| Слов проверено | Распознано | В очереди распознания | Доля |")
            lines.append("|---|---:|---:|---:|")
            total = int(self.lexical.get("total", 0))
            oov = int(self.lexical.get("oov", 0))
            rate = float(self.lexical.get("rate", 0.0))
            lines.append(f"| {total} | {total - oov} | {oov} | {rate:.0%} |")
            unknown = self.lexical.get("oov_tokens") or []
            if unknown:
                lines.append("")
                lines.append(
                    "Нераспознанные слова (пополняются через `pf lexicon`): "
                    + ", ".join(f"`{token}`" for token in unknown)
                )
            lines.append("")

        if self.rejected_details:
            lines.append("## Отклонённые основания")
            lines.append("")
            for detail in self.rejected_details:
                lines.append(f"- {detail}")
            lines.append("")

        if self.needs_review_details:
            # Не отказ и не подтверждение: внешний источник или иностранная
            # страница. Отдельный раздел нужен, чтобы «ждёт проверки» не
            # читалось как «выдумка».
            lines.append("## Ожидают проверки человеком")
            lines.append("")
            for detail in self.needs_review_details:
                lines.append(f"- {detail}")
            lines.append("")

        lines.append("## Чего этот замер не показывает")
        lines.append("")
        for item in self.limitations:
            lines.append(f"1. {item}")
        lines.append("")
        return "\n".join(lines)


def conclusion_markdown(conclusion: ConclusionRecord | None) -> list[str]:
    """Строки раздела «Вывод» — один текст для отчёта и документа демонстрации.

    Два разных рендера разошлись бы в том, что именно решили: отчёт и
    поставляемый документ обязаны говорить одно и то же.

    Args:
        conclusion: Вывод прогона либо ``None``, если работа не дошла до
            стратегии.

    Returns:
        Строки раздела в markdown.
    """
    lines: list[str] = ["## Вывод", ""]
    if conclusion is None:
        lines.append("Вывод не получен: работа не дошла до фазы стратегии.")
        lines.append(
            "Ниже — то, что успели проверить; ответа на заданный вопрос в этом прогоне нет."
        )
        lines.append("")
        return lines

    if conclusion.direction:
        lines.append(f"**Что делать:** {conclusion.direction}")
        lines.append("")
    if conclusion.value_to_cost:
        lines.append(f"**Почему:** {conclusion.value_to_cost}")
        lines.append("")
    if conclusion.decision:
        lines.append(f"**Решение:** {conclusion.decision}")
        lines.append("")
    if lines[-1] != "":
        lines.append("")

    if conclusion.deliberate_omissions:
        lines.append("Осознанные отказы:")
        lines.append("")
        for item in conclusion.deliberate_omissions:
            lines.append(f"- {item}")
        lines.append("")

    if conclusion.commitments:
        lines.append("| Обязательство | Владелец | Показатель | Условие пересмотра |")
        lines.append("|---|---|---|---|")
        for commitment in conclusion.commitments:
            lines.append(
                f"| {commitment.get('statement', '—')} | {commitment.get('owner', '—')} | "
                f"{commitment.get('metric', '—')} | "
                f"{commitment.get('review_condition', '—')} |"
            )
        lines.append("")

    if conclusion.expected_change or conclusion.adverse_effects:
        adverse = "; ".join(conclusion.adverse_effects) or "не указаны"
        lines.append("| Что ожидаем | С чем сравниваем | Срок | Нежелательные последствия |")
        lines.append("|---|---|---|---|")
        lines.append(
            f"| {conclusion.expected_change or '—'} | {conclusion.compare_against or '—'} | "
            f"{conclusion.observation_period or '—'} | {adverse} |"
        )
        lines.append("")

    lines.extend(_grounds_markdown(conclusion))
    return lines


def _grounds_markdown(conclusion: ConclusionRecord) -> list[str]:
    """Строки о том, на чём держится вывод.

    Args:
        conclusion: Вывод прогона.

    Returns:
        Строки раздела про основания вывода.
    """
    lines: list[str] = []
    numbers = (
        f"проверено {conclusion.checked}, подтверждено {conclusion.confirmed}, "
        f"отклонено {conclusion.rejected}"
    )
    if conclusion.status == "неподкреплено":
        lines.append(
            f"**Основания вывода: вывод не подкреплён** — ни одно основание не "
            f"подтверждено ({numbers})."
        )
    elif conclusion.status == "подкреплено частично":
        lines.append(f"**Основания вывода: подкреплено частично** ({numbers}).")
    elif conclusion.status == "подкреплено":
        lines.append(f"**Основания вывода: подкреплено** ({numbers}).")
    else:
        lines.append(
            "**Основания вывода не проверялись:** журнал этого прогона старше "
            "правила проверки оснований вывода. Пересоберите документ после "
            "нового прогона — иначе статус вывода неизвестен."
        )
    lines.append("")

    if conclusion.grounds:
        lines.append("На чём держится:")
        lines.append("")
        for citation in conclusion.grounds:
            lines.append(f"- `{citation}`")
        lines.append("")
    if conclusion.rejected_grounds:
        lines.append("Что отклонено:")
        lines.append("")
        for detail in conclusion.rejected_grounds:
            lines.append(f"- {detail}")
        lines.append("")
    return lines


def _citation_key(citation: str) -> str:
    """Ключ основания: путь и якорь, без повторов.

    Ссылка может выглядеть как ``проект/файл.csv#row3#p3`` — с
    продублированным якорем. Для счёта уникальных оснований это один
    и тот же факт, а не два.

    Args:
        citation: Ссылка на основание.

    Returns:
        Нормализованный ключ.
    """
    path, _, fragment = citation.partition("#")
    parts = [part for part in fragment.split("#") if part]
    if not parts:
        return path
    return f"{path}#{parts[0]}"


def rejection_kind(detail: str) -> str:
    """Класс отказа по тексту причины.

    Группировка нужна, чтобы видеть, **что чинить**: промпт (значение
    посчитано, а выдано за цитату), контекст (источник не по теме) или
    проверку (у расчёта нет provenance). Это подпись для человека, а не
    решение о статусе: статус ставит проверка, а не эта функция.

    Args:
        detail: Строка отклонения из журнала.

    Returns:
        Название класса отказа.
    """
    text = detail.lower()
    if "посчитано, а не процитировано" in text:
        return "значение посчитано как цитата"
    if "не относится к задаче" in text:
        return "источник нерелевантен"
    if "не найден среди материалов" in text:
        return "фантомный источник"
    if "не совпало с колонкой" in text:
        return "имя метрики не совпало с колонкой"
    if "формула — тождество" in text:
        return "формула-тождество"
    if "критерий не относится к показателю" in text:
        return "критерий не про показатель"
    if "источники не собраны" in text:
        return "источники не собраны"
    if "не имеет структуры" in text:
        return "у расчёта нет provenance"
    if "не совпадает со строкой" in text or "пересчёт по входам даёт" in text:
        return "значение не сошлось с источником"
    if "не найдена в строке" in text:
        return "цитата не найдена в строке"
    if "цитата не найдена" in text:
        return "цитата не найдена"
    return "прочее"


def _collect_metrics(records: list[Any]) -> list[MetricRecord]:
    """Собрать расчёты из журнала со статусом проверки.

    Расчёт попадает в отчёт только вместе со статусом: подтверждённый
    и отклонённый нельзя показывать одинаково. Статус берётся из
    строк отклонения того же шага.

    Args:
        records: Записи журнала.

    Returns:
        Список расчётов.
    """
    merged: dict[tuple[str, float, str, str], MetricRecord] = {}
    for record in records:
        detail = record.detail if isinstance(record.detail, dict) else {}
        raw_metrics = detail.get("metrics")
        if not isinstance(raw_metrics, list):
            continue
        rejected_names = {line.split(":", 1)[0].strip() for line in record.rejected_details}
        for raw in raw_metrics:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip()
            status = "подтверждено"
            detail_text = ""
            if name in rejected_names:
                status = "отклонено"
                detail_text = next(
                    (
                        line
                        for line in record.rejected_details
                        if line.split(":", 1)[0].strip() == name
                    ),
                    "",
                )
            source_id = str(raw.get("source_id", "")).strip()
            row = raw.get("row")
            citation = source_id
            if row is not None and "#" not in source_id:
                citation = f"{source_id}#row{row}"
            inputs = raw.get("inputs")
            metric = MetricRecord(
                name=name,
                value=float(raw.get("value", 0.0)),
                unit=str(raw.get("unit", "") or ""),
                citation=citation,
                formula=str(raw.get("formula", "") or ""),
                inputs=inputs if isinstance(inputs, dict) else {},
                status=status,
                detail=detail_text,
            )
            # Коллектор и аналитик считают одни и те же метрики по тем же
            # строкам. В отчёте это один расчёт: иначе таблица выглядит
            # вдвое убедительнее, чем есть. Отклонение сильнее
            # подтверждения — если хоть одна проверка не прошла, расчёт
            # не выдаётся за посчитанный.
            key = (metric.name, metric.value, metric.citation, metric.formula)
            previous = merged.get(key)
            if previous is None or (
                previous.status == "подтверждено" and metric.status != "подтверждено"
            ):
                merged[key] = metric
    return list(merged.values())


def _mapping(value: Any) -> dict[str, Any]:
    """Значение как словарь — либо пустой словарь.

    Args:
        value: Значение из журнала.

    Returns:
        Словарь.
    """
    return value if isinstance(value, dict) else {}


def _text_list(value: Any) -> list[str]:
    """Значение как список строк.

    Args:
        value: Значение из журнала.

    Returns:
        Список непустых строк.
    """
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _conclusion_status(checked: int, confirmed: int, rejected: int) -> str:
    """Статус оснований вывода — то же правило, что в рабочем процессе.

    Args:
        checked: Сколько оснований вывода проверено.
        confirmed: Сколько подтверждено.
        rejected: Сколько отклонено.

    Returns:
        «подкреплено», «подкреплено частично», «неподкреплено» либо
        «не проверялось», если проверки в журнале нет.
    """
    if checked <= 0:
        return "не проверялось"
    if confirmed <= 0:
        return "неподкреплено"
    if rejected > 0:
        return "подкреплено частично"
    return "подкреплено"


def _collect_conclusion(records: list[Any]) -> ConclusionRecord | None:
    """Собрать вывод прогона из журнала.

    Args:
        records: Записи журнала.

    Returns:
        Вывод либо ``None``, если работа не дошла до стратегии.
    """
    strategy = next((r for r in records if str(getattr(r, "phase", "")) == "strategy"), None)
    keeper = next((r for r in records if str(getattr(r, "phase", "")) == "keeper"), None)
    if strategy is None and keeper is None:
        return None

    strategy_detail = _mapping(getattr(strategy, "detail", None))
    keeper_detail = _mapping(getattr(keeper, "detail", None))
    product = _mapping(strategy_detail.get("product"))
    verification = _mapping(strategy_detail.get("verification"))

    checks = [
        check
        for check in (
            strategy_detail.get("conclusion_check"),
            keeper_detail.get("conclusion_check"),
        )
        if isinstance(check, dict)
    ]
    checked = sum(int(check.get("checked") or 0) for check in checks)
    confirmed = sum(int(check.get("confirmed") or 0) for check in checks)
    rejected = sum(int(check.get("rejected") or 0) for check in checks)

    grounds: list[str] = []
    rejected_grounds: list[str] = []
    for record in (strategy, keeper):
        if record is None:
            continue
        for citation in getattr(record, "citations", []) or []:
            if citation not in grounds:
                grounds.append(citation)
        for detail in getattr(record, "rejected_details", []) or []:
            if detail not in rejected_grounds:
                rejected_grounds.append(detail)

    commitments = [
        item for item in (keeper_detail.get("commitments") or []) if isinstance(item, dict)
    ]
    return ConclusionRecord(
        direction=str(product.get("chosen_direction") or ""),
        value_to_cost=str(product.get("value_to_cost") or ""),
        deliberate_omissions=_text_list(product.get("deliberate_omissions")),
        decision=str(keeper_detail.get("decision") or ""),
        commitments=commitments,
        expected_change=str(verification.get("expected_change") or ""),
        compare_against=str(verification.get("compare_against") or ""),
        observation_period=str(verification.get("observation_period") or ""),
        adverse_effects=_text_list(verification.get("adverse_effects")),
        status=_conclusion_status(checked, confirmed, rejected),
        checked=checked,
        confirmed=confirmed,
        rejected=rejected,
        grounds=grounds,
        rejected_grounds=rejected_grounds,
    )


def build_session_report(session_id: str) -> SessionReport:
    """Собрать отчёт по сессии из журнала шагов.

    Args:
        session_id: Идентификатор сессии.

    Returns:
        Отчёт. Для несуществующей сессии — пустой, без ошибки.
    """
    records = Journal(journal_path(session_id)).read()
    cost_by_role: dict[str, float] = {}
    tokens: dict[str, int] = {}
    confirmed = 0
    rejected = 0
    rejected_details: list[str] = []
    rejection_kinds: dict[str, int] = {}
    routing_notes: list[str] = []
    contradictions: list[str] = []
    fact_check: dict[str, Any] | None = None
    total_cost = 0.0
    metrics = _collect_metrics(records)

    all_citations: list[str] = []
    needs_review_details: list[str] = []
    lexical_total = 0
    lexical_known = 0
    lexical_oov = 0
    oov_tokens: list[str] = []
    context_steps = 0
    context_truncated = 0
    context_limit = 0
    for record in records:
        confirmed += len(record.citations)
        rejected += len(record.rejected_details)
        rejected_details.extend(record.rejected_details)
        needs_review_details.extend(record.needs_review_details)
        for line in record.rejected_details:
            kind = rejection_kind(line)
            rejection_kinds[kind] = rejection_kinds.get(kind, 0) + 1
        detail = record.detail if isinstance(record.detail, dict) else {}
        lexical = detail.get("lexical")
        if isinstance(lexical, dict):
            lexical_total += int(lexical.get("total", 0))
            lexical_known += int(lexical.get("total", 0)) - int(lexical.get("oov", 0))
            lexical_oov += int(lexical.get("oov", 0))
            for token in lexical.get("oov_tokens") or []:
                text = str(token)
                if text and text not in oov_tokens:
                    oov_tokens.append(text)
        routing = detail.get("routing")
        if isinstance(routing, dict) and routing.get("note"):
            routing_notes.append(str(routing["note"]))
        window = detail.get("context")
        if isinstance(window, dict):
            context_steps += 1
            if window.get("truncated"):
                context_truncated += 1
            context_limit = max(context_limit, int(window.get("limit") or 0))
        found = detail.get("fact_check")
        if isinstance(found, dict):
            fact_check = found
        gaps = detail.get("gaps")
        if isinstance(gaps, list):
            contradictions.extend(str(gap) for gap in gaps if "противоречие" in str(gap).lower())
        all_citations.extend(record.citations)
        total_cost += record.cost_units
        cost_by_role[record.role] = cost_by_role.get(record.role, 0.0) + record.cost_units
        used = record.prompt_tokens + record.completion_tokens
        tokens[record.role] = tokens.get(record.role, 0) + used

    # Основания и записи — разные числа. Сборщик и аналитик цитируют
    # одни и те же абзацы: четыре записи — это два разных факта.
    # Называть их четырьмя основаниями неверно, так доля выглядит
    # лучше, чем она есть.
    unique_citations = {_citation_key(citation) for citation in all_citations}
    unique_sources = {citation.split("#", 1)[0] for citation in all_citations}

    state = load(session_id)
    return SessionReport(
        session_id=session_id,
        steps=len(records),
        confirmed_citations=confirmed,
        rejected_citations=rejected,
        total_cost_units=total_cost,
        cost_by_role=cost_by_role,
        tokens=tokens,
        rejected_details=rejected_details,
        question=state.question if state else "",
        phase=state.phase.value if state else "",
        metrics=metrics,
        unique_citations=len(unique_citations),
        unique_sources=len(unique_sources),
        routing_notes=routing_notes,
        fact_check=fact_check,
        rejection_kinds=rejection_kinds,
        metrics_with_formula=sum(1 for metric in metrics if metric.formula.strip()),
        conclusion=_collect_conclusion(records),
        contradictions=contradictions,
        needs_review_details=needs_review_details,
        lexical=(
            {
                "total": lexical_total,
                "known": lexical_known,
                "oov": lexical_oov,
                "rate": round((lexical_total - lexical_oov) / lexical_total, 4)
                if lexical_total
                else 0.0,
                "oov_tokens": oov_tokens[:10],
            }
            if lexical_total
            else None
        ),
        context=(
            {
                "steps": context_steps,
                "truncated": context_truncated,
                "limit": context_limit,
            }
            if context_steps
            else None
        ),
    )


def report_as_dict(report: SessionReport) -> dict[str, Any]:
    """Отчёт в виде словаря — для машинного чтения.

    Args:
        report: Отчёт.

    Returns:
        Словарь с показателями.
    """
    return {
        "session_id": report.session_id,
        "steps": report.steps,
        "confirmed_citations": report.confirmed_citations,
        "rejected_citations": report.rejected_citations,
        "needs_review": len(report.needs_review_details),
        "lexical": report.lexical,
        "context": report.context,
        "grounding_rate": report.grounding_rate,
        "total_cost_units": report.total_cost_units,
        "cost_by_role": report.cost_by_role,
        "tokens": report.tokens,
        "question": report.question,
        "phase": report.phase,
        "limitations": report.limitations,
        "metrics_with_formula": report.metrics_with_formula,
        "rejection_kinds": report.rejection_kinds,
        "routing_notes": report.routing_notes,
        "fact_check": report.fact_check,
        "conclusion": asdict(report.conclusion) if report.conclusion else None,
    }
