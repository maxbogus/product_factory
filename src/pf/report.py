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

from pf.ground import (
    CONCLUSION_RULE,
    GroundTally,
    SupportStatus,
    citation_key,
    conclusion_status,
    tally_grounds,
)
from pf.journal import Journal, journal_path
from pf.state import load
from pf.verdicts import classify_detail, label_of

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
        assumptions: Что должно быть верно, чтобы вывод работал.
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
        off_topic_grounds: Показатели подтверждённых оснований, когда они не
            совпали с показателем вывода.
        off_topic_conclusion: Показатель, который обещает изменить вывод.
        unsupported_numbers: Сколько величин-фактов плана осталось без своего
            расчёта. Понижает статус, поэтому называется рядом с ним.
        unique: Посчитаны ли счётчики по уникальным основаниям. Журнал до
            этого правила называет только записи; если в нём были отказы,
            распределить их по уникальным ключам нечем, и отчёт говорит
            «записей» вместо «оснований».
        rule: Имя правила, которым посчитан статус. Пусто — унаследованный
            журнал, который правила не записывал.
        contradicted: Ключи оснований, отклонённых в этом же журнале на других
            шагах. Такая строка не бывает подтверждённым основанием.
    """

    direction: str = ""
    value_to_cost: str = ""
    deliberate_omissions: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
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
    off_topic_grounds: list[str] = field(default_factory=list)
    off_topic_conclusion: list[str] = field(default_factory=list)
    unsupported_numbers: int = 0
    unique: bool = True
    rule: str = ""
    contradicted: list[str] = field(default_factory=list)
    #: Опоры вывода на ввод пользователя: стадия и слова. Это не отказ и не
    #: подтверждение — решение о доверии принимает человек (`D-49`).
    user_input: int = 0
    user_input_grounds: list[str] = field(default_factory=list)
    #: Переданные получателю утверждения: канонические поля и хеш источника
    #: (`pf.handoff`). Пусто у журнала, который передачу не записывал.
    transmission: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rule_is_current(self) -> bool:
        """Статус посчитан правилом, которое действует сейчас.

        Журнал до `D-44` правила не записывал, поэтому пустое имя — это
        «неизвестно», а не «совпало».

        Returns:
            ``True``, если журнал назвал текущее правило счёта.
        """
        return self.rule == CONCLUSION_RULE


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
        hypotheses: Сколько гипотез поддержано, опровергнуто и осталось
            неизвестными.
        collection_nodes: Сколько узлов добора объявил последний разбор.
        collection_found: Кандидаты добора — ссылки, заголовки и близость.
            Это **не** основания: статус у них не ставится.
        collection_outcomes: Исходы добора по узлам: ``new``, ``duplicate``,
            ``empty``, ``no_progress``.
        collection_not_collectable: Пробелы, которые поиском не закрываются.
        variants: Набор рассмотренных вариантов — с полюсами и возражениями.
        duplicate_steps: Номера шагов, записанных в журнал больше одного раза.
            Так выглядит унаследованный журнал, в который писали два запуска.
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
    hypotheses: dict[str, int] = field(default_factory=dict)
    collection_nodes: int = 0
    collection_found: list[dict[str, Any]] = field(default_factory=list)
    collection_outcomes: dict[str, int] = field(default_factory=dict)
    collection_not_collectable: list[str] = field(default_factory=list)
    variants: list[dict[str, Any]] = field(default_factory=list)
    duplicate_steps: list[str] = field(default_factory=list)
    #: Опоры на ввод пользователя по всем шагам (стадия и слова).
    user_input_grounds: list[str] = field(default_factory=list)
    #: Нарушения ограничений задачи: направление против запрета (`T-06b`).
    constraints: list[str] = field(default_factory=list)

    @property
    def collection_candidates(self) -> int:
        """Сколько абзацев нашёл добор.

        Считается по найденному, а не хранится отдельным числом: два
        счётчика одного и того же разошлись бы при первой правке.

        Returns:
            Число кандидатов.
        """
        return len(self.collection_found)

    @property
    def total_citations(self) -> int:
        """Всего оснований: подтверждённых и отклонённых."""
        return self.confirmed_citations + self.rejected_citations

    @property
    def grounding_rate(self) -> float:
        """Доля подтверждённых **записей**.

        Считается по записям, а не по уникальным основаниям: строка
        источника, процитированная двумя ролями, даёт две записи. Поэтому и
        подпись в отчёте говорит «записей»: прежняя «доля подтверждённых»
        читалась как доля оснований.

        Returns:
            Доля от 0 до 1; ноль, если записей не было.
        """
        if not self.total_citations:
            return 0.0
        return self.confirmed_citations / self.total_citations

    def readiness(self) -> str:
        """Готова ли работа к решению — одной строкой.

        Отчёт с числами, расходом и отклонениями читается как результат, даже
        если вывода в нём нет: `demo-final` и `demo-v6` выглядели
        равноправными участниками сравнения, хотя ответа на вопрос в них не
        было.

        Returns:
            Строка-вердикт либо пустая строка, если вывод подкреплён.
        """
        if self.conclusion is None:
            return (
                "**Прогон не даёт решения:** работа не дошла до вывода — это "
                "заготовка, а не ответ на заданный вопрос."
            )
        if self.conclusion.status == "неподкреплено":
            return (
                "**Прогон не даёт решения:** ни одно основание вывода не "
                "подтверждено — рекомендации держатся ни на чём."
            )
        if self.conclusion.status == "ждёт проверки":
            return (
                "**Прогон не даёт решения:** основания вывода требуют проверки "
                "человеком — подтвердить их нечем."
            )
        if self.conclusion.status == "опирается на ввод пользователя":
            return (
                "**Прогон даёт решение с оговоркой:** вывод держится на вводе "
                "пользователя — это его слова, а не данные. Верить им или нет "
                "решает он сам."
            )
        if self.conclusion.off_topic_grounds:
            return (
                "**Прогон даёт решение с оговоркой:** подтверждённые основания "
                "называют другой показатель — причина названа в разделе «Вывод»."
            )
        if not self.conclusion.rule_is_current and self.conclusion.status in (
            "подкреплено",
            "подкреплено частично",
        ):
            # Унаследованный статус не бывает зелёным светом: он посчитан
            # правилом, которого уже нет, и переносить его на сегодня нельзя.
            return (
                "**Прогон даёт решение по прежнему правилу:** статус оснований "
                "вывода записан правилом прогона и на сегодня не переносится — "
                "перепроверьте на сохранённых результатах (`make replay`)."
            )
        return ""

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
        # Оговорка об опорах на ввод пользователя появляется только тогда,
        # когда такие опоры есть: постоянная оговорка перестаёт читаться.
        if self.user_input_grounds:
            items.append(
                "Опоры на ввод пользователя не проверены документом: подтверждено "
                "только то, что человек это сказал и на названной стадии. Верить "
                "им или нет — его решение, и замер этого решения не принимает."
            )
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
        # Доля считается по записям, а не по уникальным основаниям: одна и та
        # же строка источника, процитированная двумя ролями, даёт две записи.
        # Прежняя подпись «Доля подтверждённых» читалась как доля оснований.
        lines.append(f"| Доля подтверждённых записей | {self.grounding_rate:.0%} |")
        if self.user_input_grounds:
            # Опоры на ввод пользователя стоят отдельной строкой: они не
            # основания, проверенные документом, и не отказы (`D-49`).
            lines.append(
                f"| Опор на ввод пользователя (не основания) | {len(self.user_input_grounds)} |"
            )
        lines.append("")

        readiness = self.readiness()
        if readiness:
            lines.append(readiness)
            lines.append("")

        if self.duplicate_steps:
            # Унаследованный журнал: в один файл писали два запуска. На HEAD
            # такой журнал уже не создать (`journal.DuplicateOrdinalError`),
            # но прочитать его можно — и молчать об этом нельзя.
            lines.append(
                "**Журнал писался дважды:** "
                + "; ".join(self.duplicate_steps)
                + ". Счётчики шага объединены, статус не удваивается."
            )
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
            # Отдельный раздел, а не абзац внутри «Контроля качества»:
            # противоречие — это про вывод, а не про метрику чистоты
            # (`D-52`). Ось названа в самом пробеле, поэтому читатель
            # видит, по чему источники разошлись, а не только что они
            # разошлись.
            lines.append("## Противоречия источников")
            lines.append("")
            lines.append(
                "Источники говорят разное об одном показателе. Ось названа — "
                "по ней видно, что обсуждать; какой из фактов верен, из данных "
                "не следует."
            )
            lines.append("")
            for item in self.contradictions:
                lines.append(f"- {item}")
            lines.append("")

        if self.constraints:
            # Направление, нарушающее запрет из постановки задачи, делает
            # стратегию непригодной. Это отдельная строка, а не отказ
            # основания: чинится не источник, а выбранная линия (`T-06b`).
            lines.append("**Ограничения задачи нарушены.**")
            lines.append("")
            for item in self.constraints:
                lines.append(f"- {item}")
            lines.append("")

        if self.hypotheses:
            # Вердикт ставит код, поэтому строка отчёта — результат проверки,
            # а не пересказ ответа модели. «Требует проверки» назван наравне с
            # остальными: неизвестность прятать нечем.
            lines.append("**Гипотезы.**")
            lines.append("")
            lines.append(
                f"поддержано {self.hypotheses.get('supported', 0)}, "
                f"опровергнуто {self.hypotheses.get('refuted', 0)}, "
                f"требует проверки {self.hypotheses.get('unknown', 0)}"
            )
            lines.append("")

        lines.extend(_collection_markdown(self))
        lines.extend(_variants_markdown(self.variants))
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
            for detail, count in _group_repeats(self.rejected_details):
                suffix = f" (×{count})" if count > 1 else ""
                lines.append(f"- {detail}{suffix}")
            lines.append("")

        if self.needs_review_details:
            # Не отказ и не подтверждение: внешний источник или иностранная
            # страница. Отдельный раздел нужен, чтобы «ждёт проверки» не
            # читалось как «выдумка».
            lines.append("## Ожидают проверки человеком")
            lines.append("")
            for detail, count in _group_repeats(self.needs_review_details):
                suffix = f" (×{count})" if count > 1 else ""
                lines.append(f"- {detail}{suffix}")
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

    # Допущения стоят рядом с выводом: пока они верны, вывод работает, и
    # читатель должен увидеть их до обязательств, а не в приложении.
    if conclusion.assumptions:
        lines.append("**Что должно быть верно, чтобы вывод работал:**")
        lines.append("")
        for item in conclusion.assumptions:
            lines.append(f"- {item}")
        lines.append("")
    elif conclusion.rule_is_current:
        # Поля не было до `D-44`, поэтому унаследованный журнал не судится.
        # Для нового прогона молчание недопустимо: отсутствие допущений
        # спряталось бы в «всё хорошо».
        lines.append(
            "**Допущения вывода не названы** — проверить, при каких условиях решение "
            "перестаёт работать, нечем. Назовите их в продуктовом блоке (`assumptions`)."
        )
        lines.append("")

    if conclusion.commitments:
        lines.append(
            "| Обязательство | Владелец | Показатель | База | Метод | "
            "Контрольная группа | Откат | Условие пересмотра |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for commitment in conclusion.commitments:
            lines.append(
                f"| {commitment.get('statement', '—')} | {commitment.get('owner', '—')} | "
                f"{commitment.get('metric', '—')} | "
                f"{commitment.get('baseline') or '—'} | "
                f"{commitment.get('measurement') or '—'} | "
                f"{commitment.get('control_group') or '—'} | "
                f"{commitment.get('rollback') or '—'} | "
                f"{commitment.get('review_condition', '—')} |"
            )
        lines.append("")
        ownerless = _ownerless_count(conclusion.commitments)
        if ownerless:
            # В `cloud-deepseek-2` так было у трёх обязательств из шести.
            # В таблице это видно, но как риск — нет.
            lines.append(
                f"**Без владельца: {ownerless} из {len(conclusion.commitments)}.** "
                "Обязательство без владельца некому исполнить — это риск, а не оформление."
            )
            lines.append("")
        baseline = _baseline_missing(conclusion.commitments)
        if baseline is not None and baseline[0]:
            # База — не оформление: без неё «вырастет на 5 %» не говорит,
            # от чего, и обязательство нельзя проверить (`T-50`).
            missing, total = baseline
            lines.append(
                f"**Без базы: {missing} из {total}.** Обязательство без базовой линии "
                "нельзя проверить: непонятно, от чего считаем изменение."
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
    lines.extend(_transmission_markdown(conclusion))
    return lines


def _transmission_markdown(conclusion: ConclusionRecord) -> list[str]:
    """Строки раздела «Передача»: поля утверждений и хеш источника.

    Запись передачи — контракт между шагами: получатель сверяет поля, а не
    прозу. Секция пустая, если журнал передачи не записывал (`T-46`).

    Args:
        conclusion: Вывод прогона.

    Returns:
        Строки раздела в markdown.
    """
    if not conclusion.transmission:
        return []
    from pf.handoff import handoff_markdown

    return handoff_markdown(conclusion.transmission)


def _grounds_markdown(conclusion: ConclusionRecord) -> list[str]:
    """Строки о том, на чём держится вывод.

    Args:
        conclusion: Вывод прогона.

    Returns:
        Строки раздела про основания вывода.
    """
    lines: list[str] = []
    # Счёт идёт по уникальным основаниям: две ссылки на одну строку
    # источника — одно основание. Старый журнал называет только записи, и
    # отчёт говорит это словом «записей», а не выдаёт их за основания.
    label = "уникальных оснований" if conclusion.unique else "записей"
    numbers = (
        f"{label}: проверено {conclusion.checked}, подтверждено {conclusion.confirmed}, "
        f"отклонено {conclusion.rejected}"
    )
    # Подпись «по правилу прогона» стоит там, где строка читается как
    # сегодняшняя оценка: унаследованный журнал своего правила не записывал,
    # и `demo-v9` печатал «подкреплено» рядом с «отклонено» по тем же строкам.
    run_rule = "" if conclusion.rule_is_current else " по правилу прогона"
    if conclusion.status == "неподкреплено":
        lines.append(
            f"**Основания вывода: вывод не подкреплён** — ни одно основание не "
            f"подтверждено ({numbers})."
        )
    elif conclusion.status == "ждёт проверки":
        lines.append(
            f"**Основания вывода: ждут проверки** — ни одно не подтверждено, "
            f"подтвердить их нечем ({numbers})."
        )
    elif conclusion.status == "подкреплено частично":
        lines.append(f"**Основания вывода: подкреплено частично{run_rule}** ({numbers}).")
    elif conclusion.status == "опирается на ввод пользователя":
        lines.append(
            f"**Основания вывода: опирается на ввод пользователя{run_rule}** ({numbers}). "
            "Подтверждённых документов нет: это слова человека, а не данные."
        )
    elif conclusion.status == "подкреплено":
        lines.append(f"**Основания вывода: подкреплено{run_rule}** ({numbers}).")
    else:
        lines.append(
            "**Основания вывода не проверялись:** журнал этого прогона старше "
            "правила проверки оснований вывода. Пересоберите документ после "
            "нового прогона — иначе статус вывода неизвестен."
        )
    lines.append("")

    if conclusion.user_input_grounds:
        # Опоры на ввод пользователя называются рядом со статусом: «верить
        # или нет» — решение человека, и он должен видеть, на какие именно
        # свои слова опирается вывод (`D-49`).
        lines.append(
            "**Опоры на ввод пользователя** (это не документы: подтверждается "
            "только то, что человек так сказал):"
        )
        lines.append("")
        for item in conclusion.user_input_grounds:
            lines.append(f"- {item}")
        lines.append("")

    if not conclusion.rule_is_current and conclusion.status in (
        "подкреплено",
        "подкреплено частично",
        "опирается на ввод пользователя",
    ):
        # Подпись нужна там, где статус **заявляет опору**. У «неподкреплено»
        # и «ждут проверки» переносить нечего: они и так ничего не обещают,
        # а лишняя строка в каждом унаследованном документе — шум.
        recorded = f"`{conclusion.rule}`" if conclusion.rule else "правило не записано"
        lines.append(
            f"**Правило счёта: {recorded}.** Сейчас основания считают правилом "
            f"`{CONCLUSION_RULE}`. Статус выше — запись прогона, а не проверка "
            "сегодняшним правилом: унаследованный журнал на глаз не "
            "пересчитывается, а перепроверка на сохранённых результатах — "
            "`make replay SESSIONS=<сессия>`."
        )
        lines.append("")

    if conclusion.contradicted:
        # Противоречие внутри одного журнала: строка отклонена на одном шаге и
        # подтверждена в выводе на другом. Молчать об этом нельзя — рядом в
        # документе стоят «отклонено» и «подкреплено» по одной строке.
        keys = ", ".join(f"`{key}`" for key in conclusion.contradicted)
        lines.append(
            "**Журнал себе противоречит:** эти строки отклонены на других шагах, "
            f"а основания вывода считают их подтверждёнными — {keys}. "
            "Повтор ссылки на отклонённую строку не делает её основанием."
        )
        lines.append("")

    if conclusion.off_topic_grounds:
        # Дословность цитаты не отвечает на вопрос, тот ли показатель она
        # подтверждает. Строка нужна, чтобы читатель видел причину понижения
        # статуса, а не считал его случайной строгостью.
        promised = _join_names(conclusion.off_topic_conclusion) or "другой показатель"
        confirmed = _join_names(conclusion.off_topic_grounds)
        lines.append(
            f"**Основания не про тот показатель:** вывод обещает изменение по «{promised}», "
            f"а основания называют «{confirmed}». Такое основание не считается "
            "подтверждённым: дословная цитата здесь ничего не доказывает."
        )
        lines.append("")

    if conclusion.unsupported_numbers:
        # Потолок без причины читается как случайная строгость: рядом стоят
        # «подкреплено частично» и подтверждённые основания, и почему не
        # «подкреплено» — не видно.
        lines.append(
            f"**Число без своего расчёта: {conclusion.unsupported_numbers}** — величина "
            "названа в плане или обязательстве, а расчёта того же показателя нет. "
            "Статус не выше «подкреплено частично»."
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


def rejection_kind(detail: str) -> str:
    """Класс отказа по тексту причины.

    Подпись берётся из таблицы видов (`pf.verdicts`), а не из списка
    подстрок отчёта: тот же вид, что назвала проверка, иначе считался бы
    вторым правилом, и строки «прочее» расходились бы с замером.

    Разбор текста остался потому, что журнал append-only: шаги, записанные до
    появления машинного вида, хранят фразу. Для новых записей вид приходит
    готовым (`StepRecord.rejections`), и эта функция не нужна.

    Args:
        detail: Строка отклонения из журнала.

    Returns:
        Название класса отказа для человека.
    """
    return label_of(classify_detail(detail))


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


#: Как в ответе модели называется отсутствующий владелец обязательства.
_NO_OWNER: frozenset[str] = frozenset({"", "нет", "—", "-", "не назначен", "не назначена"})


def _times(count: int) -> str:
    """Сколько раз — с правильным окончанием.

    Args:
        count: Число повторов.

    Returns:
        Строка вида «2 раза» или «5 раз».
    """
    tail = count % 100
    if 11 <= tail <= 14:
        return f"{count} раз"
    if count % 10 == 1:
        return f"{count} раз"
    if count % 10 in (2, 3, 4):
        return f"{count} раза"
    return f"{count} раз"


def _collection_markdown(report: SessionReport) -> list[str]:
    """Раздел «Добор оснований»: что искали и что нашли.

    Раздел нужен, потому что «нашли кандидатов» легко читается как «вывод
    подкреплён». Кандидат — абзац, который может ответить, а не основание:
    дословность проверяет `pf.ground`, вердикт гипотезы ставит код.

    Args:
        report: Отчёт по сессии.

    Returns:
        Строки раздела; пусто, если добора не было.
    """
    if report.collection_nodes <= 0:
        return []
    lines = ["**Добор оснований.**", ""]
    outcomes = ", ".join(
        f"{name} {count}" for name, count in sorted(report.collection_outcomes.items())
    )
    summary = f"узлов: {report.collection_nodes}, найдено абзацев: {report.collection_candidates}"
    if outcomes:
        summary += f" (исходы: {outcomes})"
    lines.append(summary)
    lines.append("")
    for item in report.collection_found[:5]:
        citation = str(item.get("citation") or "")
        title = str(item.get("title") or "")
        node = str(item.get("node") or "")
        lines.append(f"- `{citation}` {title} — под узел «{node}»")
    if report.collection_found:
        lines.append("")
    if report.collection_not_collectable:
        lines.append("Поиском не закрывается: " + "; ".join(report.collection_not_collectable))
        lines.append("")
    lines.append(
        "Кандидаты основаниями не считаются: цитату проверяет `pf.ground`, "
        "вердикт гипотезы ставит код по подтверждённым утверждениям."
    )
    lines.append("")
    return lines


def _variants_markdown(variants: list[dict[str, Any]]) -> list[str]:
    """Раздел «Рассмотренные варианты»: набор, полюса и возражения.

    Набор вариантов писался в журнал, но в отчёт не попадал: в
    `cloud-deepseek-2` четыре варианта, включая «Оставить как есть: работать
    только с коммуникацией», были невидимы, и читатель видел только
    «осознанные отказы». Полюса определяются тем же правилом, что и в схеме
    (`pf.schemas.pole_of`, D-23), — второй копии списков названий нет.

    Args:
        variants: Варианты из журнала шага «варианты решения».

    Returns:
        Строки раздела либо пустой список, если вариантов нет.
    """
    if not variants:
        return []

    from pf.schemas import (
        POLE_CLOSE_DIRECTION,
        POLE_KEEP_CURRENT,
        pole_of,
    )

    lines: list[str] = ["## Рассмотренные варианты", ""]
    lines.append("| Вариант | Полюс | Возражения |")
    lines.append("|---|---|---|")
    found_poles: set[str] = set()
    for variant in variants:
        title = str(variant.get("title") or variant.get("name") or "—").strip()
        pole = pole_of(title) or ""
        if pole:
            found_poles.add(pole)
        objections = "; ".join(
            str(item).strip() for item in (variant.get("objections") or []) if str(item).strip()
        )
        lines.append(f"| {title} | {pole or '—'} | {objections or '—'} |")
    lines.append("")
    missing = [
        label
        for pole, label in (
            (POLE_KEEP_CURRENT, "оставить как есть"),
            (POLE_CLOSE_DIRECTION, "закрыть направление"),
        )
        if pole not in found_poles
    ]
    if missing:
        lines.append(
            "**Полюса нет:** "
            + ", ".join(f"«{item}»" for item in missing)
            + " — без него набор читается как список пожеланий, а не как решение."
        )
    else:
        lines.append(
            "**Полюса набора:** «оставить как есть» и «закрыть направление» — оба присутствуют."
        )
    lines.append("")
    return lines


def _ownerless_count(commitments: list[dict[str, Any]]) -> int:
    """Сколько обязательств осталось без владельца.

    «Не назначен» — честный ответ модели, и он лучше выдуманной фамилии. Но
    обязательство без владельца некому исполнить, поэтому отчёт называет это
    числом: в `cloud-deepseek-2` так было у четырёх из шести.

    Args:
        commitments: Обязательства из журнала шага решения.

    Returns:
        Число обязательств без владельца.
    """
    count = 0
    for commitment in commitments:
        owner = str(commitment.get("owner") or "").strip().lower()
        if owner in _NO_OWNER or owner.startswith("не назначен"):
            count += 1
    return count


def _baseline_missing(commitments: list[dict[str, Any]]) -> tuple[int, int] | None:
    """Сколько обязательств осталось без базы сравнения.

    База — не оформление: без неё «удержание вырастет на 5 %» не говорит,
    от чего, и обязательство нельзя проверить (`T-50`). Поле появилось
    позже схемы, поэтому унаследованный журнал не судится: если ни одно
    обязательство ключа `baseline` не несёт, отчёт молчит — сравнивать
    нечем, и это не «всё заполнено» (`D-33`).

    Args:
        commitments: Обязательства из журнала шага решения.

    Returns:
        Пара «без базы — всего» либо ``None``, если поле не записывалось.
    """
    if not commitments:
        return None
    if not any("baseline" in commitment for commitment in commitments):
        return None
    missing = sum(
        1 for commitment in commitments if not str(commitment.get("baseline") or "").strip()
    )
    return missing, len(commitments)


def _group_repeats(items: list[str]) -> list[tuple[str, int]]:
    """Схлопнуть одинаковые строки, сохранив число повторов.

    Один и тот же отказ приходит с нескольких шагов: в `cloud-deepseek-2`
    строка «источник не назван — вместо ссылки указано поле „Ответ
    человека"» стояла трижды. Читается это как шесть разных отказов, хотя
    отказ один. Число повторов сохраняется — оно показывает, на скольких
    шагах модель повторила ошибку.

    Args:
        items: Строки в порядке появления.

    Returns:
        Пары «строка — сколько раз встретилась», тоже в порядке появления.
    """
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return list(counts.items())


def _join_names(items: list[str], limit: int = 3) -> str:
    """Назвать показатели по-человечески: без повторов и без полотна.

    Показатель обязательства бывает длинной фразой целиком: «наличие
    зафиксированных определения удержания, базового периода и источника
    замера». В строку отчёта такое название вставляется целиком, но не
    больше `limit` штук — иначе причина статуса занимает абзац.

    Args:
        items: Названия показателей.
        limit: Сколько назвать.

    Returns:
        Строка через запятую; пустая, если названий нет.
    """
    unique: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in unique:
            unique.append(text)
    shown = ", ".join(unique[:limit])
    return f"{shown}, …" if len(unique) > limit else shown


#: Как вердикт основания называется в журнале шага.
_STATUS_BY_BUCKET: dict[str, SupportStatus] = {
    "confirmed": SupportStatus.CONFIRMED,
    "rejected": SupportStatus.NO_DATA,
    "needs_review": SupportStatus.NEEDS_REVIEW,
}


def _declared_ground_keys(detail: dict[str, Any]) -> list[str]:
    """Ключи оснований вывода, объявленных на шаге.

    Основания вывода лежат в трёх местах результата: ``product.evidence``,
    ``communication.promise_grounding`` и ``commitments[].basis``. Журнал
    хранит их целиком, поэтому уникальные основания считаются и по
    старому журналу — в нём нет только раскладки по вердиктам.

    Args:
        detail: Результат шага из журнала.

    Returns:
        Ключи оснований в порядке объявления, без повторов.
    """
    declared: list[Any] = []
    declared.extend(_mapping(detail.get("product")).get("evidence") or [])
    declared.extend(_mapping(detail.get("communication")).get("promise_grounding") or [])
    for commitment in detail.get("commitments") or []:
        if isinstance(commitment, dict):
            declared.extend(commitment.get("basis") or [])

    keys: list[str] = []
    for item in declared:
        if not isinstance(item, dict):
            continue
        key = citation_key(str(item.get("source_id") or ""))
        if key and key not in keys:
            keys.append(key)
    return keys


def _journal_tally(checks: list[dict[str, Any]], declared: list[str]) -> GroundTally | None:
    """Счёт оснований вывода по тому, что журнал действительно знает.

    Три случая различаются, и путать их нельзя:

    1. в проверке есть ключи оснований — считаем по ним, объединяя шаги:
       стратегия и решение ссылаются на одни и те же строки;
    2. ключей нет, но и отказов нет — значит все объявленные основания
       подтверждены, и уникальных ровно столько, сколько объявлено;
    3. ключей нет и отказы есть — распределить их по уникальным ключам
       нечем (неизвестно, какая из повторных ссылок отклонена), поэтому
       отчёт назовёт записи, а не основания.

    Args:
        checks: Блоки ``conclusion_check`` из журнала.
        declared: Ключи объявленных оснований вывода.

    Returns:
        Счёт по уникальным основаниям либо ``None``, если журнал не
        позволяет его посчитать.
    """
    if all(isinstance(check.get("grounds"), dict) for check in checks):
        verdicts: list[tuple[str, SupportStatus]] = []
        for check in checks:
            grounds = _mapping(check.get("grounds"))
            for bucket, status in _STATUS_BY_BUCKET.items():
                for key in grounds.get(bucket) or []:
                    verdicts.append((citation_key(str(key)), status))
        return tally_grounds(verdicts)
    if declared and all(
        int(check.get("rejected") or 0) == 0 and int(check.get("needs_review") or 0) == 0
        for check in checks
    ):
        return GroundTally(checked=len(declared), confirmed=len(declared))
    return None


def _rejected_material_keys(records: list[Any]) -> set[str]:
    """Ключи оснований, которые журнал отклонил хотя бы на одном шаге.

    Строка отказа выглядит как ``путь#якорь: причина``. Отказы по имени
    расчёта («Удержание пользователей: пересчёт по входам даёт 0») к
    основаниям не относятся: у них нет ни пути, ни якоря, и попади они в
    этот набор, отчёт назвал бы противоречием то, чем оно не является.

    Args:
        records: Записи журнала.

    Returns:
        Множество ключей оснований.
    """
    keys: set[str] = set()
    for record in records:
        for line in getattr(record, "rejected_details", []) or []:
            prefix = str(line).split(":", 1)[0].strip()
            if "#" in prefix or "/" in prefix:
                keys.add(citation_key(prefix))
    return keys


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
    declared: list[str] = []
    for recorded in (strategy_detail, keeper_detail):
        for key in _declared_ground_keys(recorded):
            if key not in declared:
                declared.append(key)

    # Правило, которым считали. Журнал до `D-44` его не записывал, поэтому
    # пустое имя значит «неизвестно»: выдавать такой статус за текущую оценку
    # нельзя — именно так `demo-v9` печатал «подкреплено» рядом с «отклонено».
    rules = {str(check.get("rule") or "") for check in checks}
    rule = rules.pop() if len(rules) == 1 else ""

    # Подтверждённые основания по ключам: нужны, чтобы увидеть противоречие
    # внутри одного журнала — та же строка отклонена на одном шаге и
    # подтверждена в выводе на другом.
    if all(isinstance(check.get("grounds"), dict) for check in checks):
        confirmed_keys: list[str] = []
        for check in checks:
            for key in _mapping(check.get("grounds")).get("confirmed") or []:
                clean = citation_key(str(key))
                if clean not in confirmed_keys:
                    confirmed_keys.append(clean)
    elif checks and all(
        int(check.get("rejected") or 0) == 0 and int(check.get("needs_review") or 0) == 0
        for check in checks
    ):
        # Журнал без ключей, но и без отказов: подтверждено всё объявленное.
        confirmed_keys = list(declared)
    else:
        confirmed_keys = []
    rejected_keys = _rejected_material_keys(records)
    contradicted = [key for key in confirmed_keys if key in rejected_keys]

    tally = _journal_tally(checks, declared) if checks else GroundTally()
    if tally is None:
        # Журнал называет только записи: так было до правила уникальных
        # оснований, и распределить отказы по ключам нечем. Выдать записи
        # за уникальные основания значило бы соврать в удобную сторону.
        checked = sum(int(check.get("checked") or 0) for check in checks)
        confirmed = sum(int(check.get("confirmed") or 0) for check in checks)
        rejected = sum(int(check.get("rejected") or 0) for check in checks)
        needs_review = sum(int(check.get("needs_review") or 0) for check in checks)
        unique = False
    else:
        checked = tally.checked
        confirmed = tally.confirmed
        rejected = tally.rejected
        needs_review = tally.needs_review
        unique = True

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

    # Показатель основания против показателя вывода. Правило живёт в
    # рабочем процессе (`Workflow._verify_claims`); отчёт только переносит
    # его вывод в документ, а статус считает тем же `conclusion_status`.
    off_topic_grounds: list[str] = []
    off_topic_conclusion: list[str] = []
    for check in checks:
        block = _mapping(check.get("off_topic"))
        for key, target in (
            ("grounds", off_topic_grounds),
            ("conclusion", off_topic_conclusion),
        ):
            for item in block.get(key) or []:
                text = str(item).strip()
                if text and text not in target:
                    target.append(text)

    commitments = [
        item for item in (keeper_detail.get("commitments") or []) if isinstance(item, dict)
    ]
    # Замечания о числах живут на шагах, а понижают статус вывода. Складываем
    # по шагам: одно и то же число в плане и в обязательстве — два места, где
    # оно названо, и шаг считает свои. Статус от этого не зависит: понижает
    # любое положительное число (`T-48b`).
    unsupported_numbers = sum(int(check.get("unsupported_numbers") or 0) for check in checks)
    # Опоры на ввод пользователя: считаются по журналу шагов, а не по статусу
    # — иначе «опирается на ввод пользователя» не отличить от «нечем
    # подтвердить» (`D-49`).
    user_input = sum(int(check.get("user_input") or 0) for check in checks)
    user_input_grounds: list[str] = []
    for record in (strategy, keeper):
        if record is None:
            continue
        for item in getattr(record, "user_input_grounds", []) or []:
            if not isinstance(item, dict):
                continue
            stage = str(item.get("stage") or "").strip()
            words = str(item.get("quote") or item.get("claim") or "").strip()
            text = f"{stage}: «{words}»" if stage else f"«{words}»"
            if text not in user_input_grounds:
                user_input_grounds.append(text)
    # Передача вывода (`T-46`): канонические поля утверждений и хеш источника.
    # Берётся из тех же `conclusion_check`, что и статус: получатель сверяет
    # поля, а не прозу.
    transmission: list[dict[str, Any]] = []
    for check in checks:
        for item in check.get("transmission") or []:
            if isinstance(item, dict) and item not in transmission:
                transmission.append(item)
    return ConclusionRecord(
        direction=str(product.get("chosen_direction") or ""),
        value_to_cost=str(product.get("value_to_cost") or ""),
        deliberate_omissions=_text_list(product.get("deliberate_omissions")),
        assumptions=_text_list(product.get("assumptions")),
        decision=str(keeper_detail.get("decision") or ""),
        commitments=commitments,
        expected_change=str(verification.get("expected_change") or ""),
        compare_against=str(verification.get("compare_against") or ""),
        observation_period=str(verification.get("observation_period") or ""),
        adverse_effects=_text_list(verification.get("adverse_effects")),
        status=conclusion_status(
            checked,
            confirmed,
            rejected,
            needs_review,
            unsupported_numbers=unsupported_numbers,
            user_input=user_input,
        ),
        checked=checked,
        confirmed=confirmed,
        rejected=rejected,
        grounds=grounds,
        rejected_grounds=rejected_grounds,
        off_topic_grounds=off_topic_grounds,
        off_topic_conclusion=off_topic_conclusion,
        unsupported_numbers=unsupported_numbers,
        unique=unique,
        rule=rule,
        contradicted=contradicted,
        user_input=user_input,
        user_input_grounds=user_input_grounds,
        transmission=transmission,
    )


def _collect_hypotheses(records: list[Any]) -> dict[str, int]:
    """Счёт гипотез по вердиктам последнего разбора.

    Берётся последний шаг разбора: при возврате назад гипотезы формулируются
    заново, и складывать счётчики двух разборов значило бы удвоить работу.

    Запись без поля ``verdict`` (журнал до правила) считается ``unknown``:
    вердикта в ней нет, и выдавать её за проверенную гипотезу нельзя.

    Args:
        records: Записи журнала.

    Returns:
        Счётчик «вердикт → сколько гипотез». Пустой словарь, если гипотез не
        было: «гипотез нет» и «все неизвестны» — разные ответы.
    """
    latest: list[Any] = []
    for record in records:
        if record.phase != "analyzing":
            continue
        detail = record.detail if isinstance(record.detail, dict) else {}
        found = detail.get("hypotheses")
        if isinstance(found, list):
            latest = found
    if not latest:
        return {}
    counts = {"supported": 0, "refuted": 0, "unknown": 0}
    for item in latest:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict") or "unknown")
        counts[verdict if verdict in counts else "unknown"] += 1
    return counts


def _collect_collection(records: list[Any]) -> dict[str, Any]:
    """Добор оснований из последнего шага разбора.

    Берётся последний шаг разбора: при возврате назад узлы формулируются
    заново, и складывать кандидатов двух раундов значило бы выдать повтор за
    новое знание (`pf.policy.content_key` — тот же ответ на уровне текста).

    Args:
        records: Записи журнала.

    Returns:
        Словарь с числом узлов, найденными абзацами, исходами и
        неразбираемыми пробелами. Пустой, если добора не было.
    """
    latest: dict[str, Any] = {}
    for record in records:
        if record.phase != "analyzing":
            continue
        detail = record.detail if isinstance(record.detail, dict) else {}
        found = detail.get("collection")
        if isinstance(found, dict):
            latest = found
    if not latest:
        return {}

    nodes = latest.get("nodes")
    candidates = latest.get("candidates")
    node_items: list[Any] = nodes if isinstance(nodes, list) else []
    found_items: list[Any] = candidates if isinstance(candidates, list) else []
    outcomes: dict[str, int] = {}
    for item in latest.get("outcomes") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("outcome") or "unknown")
        outcomes[name] = outcomes.get(name, 0) + 1
    not_collectable = [
        str(item) for item in (latest.get("not_collectable") or []) if str(item).strip()
    ]
    return {
        "nodes": len(node_items),
        "found": [item for item in found_items if isinstance(item, dict)],
        "outcomes": outcomes,
        "not_collectable": not_collectable,
    }


def _record_rejection_kinds(record: Any) -> dict[str, int]:
    """Классы отказов одного шага: записанный вид сильнее разбора текста.

    Вид приходит из проверки (`Support.kind`, `ABSENCE_KINDS`,
    `PlanNumberNote.kind`) и лежит в журнале рядом с текстом. Пока его не
    было, класс зависел от формулировки: новый отказ попадал в «прочее».

    Args:
        record: Запись шага из журнала.

    Returns:
        Счёт по подписям классов.
    """
    kinds: dict[str, int] = {}
    recorded = [
        item for item in getattr(record, "rejections", None) or [] if isinstance(item, dict)
    ]
    if recorded:
        for item in recorded:
            kind = str(item.get("kind") or "")
            label = label_of(kind) if kind else rejection_kind(str(item.get("detail") or ""))
            kinds[label] = kinds.get(label, 0) + 1
        return kinds
    for line in getattr(record, "rejected_details", None) or []:
        label = rejection_kind(str(line))
        kinds[label] = kinds.get(label, 0) + 1
    return kinds


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
    constraints: list[str] = []
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
        for kind, count in _record_rejection_kinds(record).items():
            rejection_kinds[kind] = rejection_kinds.get(kind, 0) + count
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
        constraint_check = detail.get("constraints")
        if isinstance(constraint_check, dict):
            for reason in constraint_check.get("violations") or []:
                text = str(reason).strip()
                if text and text not in constraints:
                    constraints.append(text)
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
    unique_citations = {citation_key(citation) for citation in all_citations}
    unique_sources = {citation.split("#", 1)[0] for citation in all_citations}

    # Шаги считаются по номерам, а не по записям: в унаследованный журнал
    # писали два запуска, и номера 9 и 10 встречались дважды. Двенадцать
    # записей — это десять шагов, и отчёт обязан сказать это словом, иначе
    # два запуска выглядят как более долгая работа.
    ordinal_counts: dict[int, int] = {}
    for record in records:
        ordinal_counts[record.ordinal] = ordinal_counts.get(record.ordinal, 0) + 1
    duplicate_steps = [
        f"шаг {ordinal} записан {_times(count)}"
        for ordinal, count in ordinal_counts.items()
        if count > 1
    ]
    variants: list[dict[str, Any]] = []
    for record in records:
        if str(getattr(record, "phase", "")) != "options":
            continue
        detail = _mapping(getattr(record, "detail", None))
        found = detail.get("variants")
        if isinstance(found, list):
            variants = [item for item in found if isinstance(item, dict)]

    collection = _collect_collection(records)

    # Опоры на ввод пользователя по всем шагам: считаются отдельно от
    # оснований и отказов. Это не проверка документа — это слова человека,
    # и решение о доверии к ним принимает он (`D-49`).
    user_input_grounds: list[str] = []
    for record in records:
        for item in getattr(record, "user_input_grounds", []) or []:
            if not isinstance(item, dict):
                continue
            stage = str(item.get("stage") or "").strip()
            words = str(item.get("quote") or item.get("claim") or "").strip()
            text = f"{stage}: «{words}»" if stage else f"«{words}»"
            if text not in user_input_grounds:
                user_input_grounds.append(text)

    state = load(session_id)
    return SessionReport(
        session_id=session_id,
        steps=len(ordinal_counts),
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
        hypotheses=_collect_hypotheses(records),
        collection_nodes=int(collection.get("nodes") or 0),
        collection_found=list(collection.get("found") or []),
        collection_outcomes=dict(collection.get("outcomes") or {}),
        collection_not_collectable=list(collection.get("not_collectable") or []),
        variants=variants,
        duplicate_steps=duplicate_steps,
        needs_review_details=needs_review_details,
        user_input_grounds=user_input_grounds,
        constraints=constraints,
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
        "user_input_grounds": report.user_input_grounds,
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
        "hypotheses": report.hypotheses,
        "collection": {
            "nodes": report.collection_nodes,
            "candidates": report.collection_candidates,
            "outcomes": report.collection_outcomes,
            "not_collectable": report.collection_not_collectable,
        },
        "conclusion": asdict(report.conclusion) if report.conclusion else None,
    }
