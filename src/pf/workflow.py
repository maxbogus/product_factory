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

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pf import roles
from pf.config import Settings, settings
from pf.contradictions import find_contradictions
from pf.ground import (
    CONCLUSION_RULE,
    GroundTally,
    SupportStatus,
    citation_key,
    conclusion_status,
    ground_buckets,
    mark_external,
    split_source_id,
    tally_grounds,
    verify_quote,
    verify_quote_as_external,
)
from pf.hypotheses import CollectionPlan, plan_collection, verdict_of
from pf.journal import Journal, StepRecord, journal_path
from pf.lexical import indicator_stems, numbers_match, stems
from pf.llm import LLMClient, LLMError, build_client, extract_json
from pf.schemas import (
    CLOSE_DIRECTION_TITLES,
    KEEP_CURRENT_TITLES,
    Analysis,
    DecisionRecord,
    NextStep,
    RedTeamReport,
    ResearchPlan,
    Strategy,
    TaskFrame,
    Variants,
)
from pf.sources import (
    SourceBundle,
    collect,
    human_input_stage,
    indicator_of_citation,
    same_material,
)
from pf.state import Phase, SessionState, human_inputs, phase_of, save
from pf.verdicts import (
    COMMITMENT_TEMPLATE,
    MISSING_EVIDENCE,
    NO_RELEVANT_SOURCES,
    REFERENCE,
    Rejection,
    classify_detail,
    label_of,
)

if TYPE_CHECKING:
    from pf.analysis import Metric


@dataclass(frozen=True, slots=True)
class ClaimVerdict:
    """Вердикт по одному основанию.

    Счётчики говорят «сколько подтверждено», но не «что именно изменилось».
    Для сравнения двух версий проверки нужно сопоставлять вердикты по
    утверждению и источнику, иначе «58 против 58» не отвечает на вопрос,
    какие основания переехали из подтверждённых в отклонённые.

    Attributes:
        claim: Текст утверждения (факт, обещание, обязательство, вывод).
        source_id: Ссылка, на которую опирается утверждение.
        status: Чем кончилась проверка.
        detail: Пояснение для человека.
        is_conclusion: Относится ли основание к выводу, а не к факту.
        kind: Машинный вид отказа (``pf.sources.ABSENCE_KINDS``) либо вид
            проверки цитаты (``Support.kind``). Нужен, чтобы отчёт считал
            корзину по факту, а не по совпадению слов в пояснении.
    """

    claim: str
    source_id: str
    status: SupportStatus
    detail: str = ""
    is_conclusion: bool = False
    kind: str = ""


@dataclass(frozen=True, slots=True)
class HumanInputVerdict:
    """Итог проверки ссылки на ввод пользователя.

    Отличается от `Support` тем, что никогда не бывает подтверждённым:
    слова человека проверяются на то, что он их сказал, а решение о
    доверии принимает он сам (`D-49`). Поэтому у вердикта есть стадия —
    её называют и отчёт, и документ.

    Attributes:
        status: Итог: опора принята или отказ.
        detail: Пояснение для человека — что именно не сошлось.
        kind: Машинный вид (`user_input` или вид отказа).
        stage: Стадия, к которой отнесена опора.
    """

    status: SupportStatus
    detail: str
    kind: str
    stage: str = ""


@dataclass(frozen=True, slots=True)
class GroundingResult:
    """Итог проверки оснований одного шага.

    Attributes:
        citations: Подтверждённые ссылки.
        rejected: Сколько оснований отброшено.
        details: Что и почему отброшено.
        conclusion: Статус оснований вывода либо ``None``, если вывода
            в результате нет (факты, план, варианты — это не вывод).
        needs_review: Ссылки, которые нельзя подтвердить: внешний источник
            или другой язык. Это не отказ — это «ждёт человека», и
            смешивать его с отказом нельзя.
        needs_review_details: Что именно ждёт проверки и почему.
        user_input: Опоры на ввод пользователя — что сказано и на какой
            стадии. Не отказ и не подтверждение: доверие к своим словам
            определяет человек (`D-49`).
        user_input_grounds: Те же опоры с полями для журнала: стадия, слова,
            утверждение, пояснение.
        verdicts: Вердикт по каждому основанию — для замера и сверки версий.
    """

    citations: list[str] = field(default_factory=list)
    rejected: int = 0
    details: list[str] = field(default_factory=list)
    conclusion: dict[str, Any] | None = None
    needs_review: list[str] = field(default_factory=list)
    needs_review_details: list[str] = field(default_factory=list)
    user_input: list[str] = field(default_factory=list)
    user_input_grounds: list[dict[str, Any]] = field(default_factory=list)
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    #: Записи передачи вывода (`pf.handoff`): поля утверждения и хеш источника.
    #: Читает их отчёт через `conclusion_check`: получатель сверяет поля, а не
    #: прозу.
    handoffs: list[dict[str, Any]] = field(default_factory=list)


def _named_source(source_id: str) -> str:
    """Как назвать источник в пояснении отказа.

    Модель иногда ставит в ``source_id`` имя поля из входа — «query»,
    «verification». Это не источник, и в журнале должно быть видно, что
    вместо ссылки пришло поле, а не что «источник не относится к задаче».

    Args:
        source_id: Значение, которое модель указала как источник.

    Returns:
        Подпись для пояснения.
    """
    text = source_id.strip()
    if not text:
        return "источник не назван"
    if human_input_stage(text) is not None:
        # Модель видит ввод пользователя в контексте и ссылается на него.
        # Это не «нерелевантный источник» и не фантом: канал назван.
        return "ввод пользователя"
    if "/" in text or "." in text:
        return text
    return f"источник не назван — вместо ссылки указано поле «{text}»"


def _metric_line(metric: Any) -> str:
    """Строка блока расчётов для одной метрики.

    У агрегата показываются входы: его значения нет ни в одной строке
    таблицы, и без входов его нечем пересчитать, а значит — нечем
    подтвердить. Модель копирует в свою метрику ровно то, что видит
    здесь, поэтому входы обязаны быть в строке.

    Args:
        metric: Метрика.

    Returns:
        Строка блока расчётов.
    """
    if metric.is_aggregate:
        # Входы агрегата называются по-разному: у среднего и медианы —
        # «наблюдения», у изменения — «база» и «новое». Показываем
        # все: без них агрегат нечем пересчитать, а значит — нечем
        # подтвердить.
        parts = []
        for key, raw in metric.inputs.items():
            if isinstance(raw, (list, tuple)):
                parts.append(f"{key}: {', '.join(str(item) for item in raw)}")
            else:
                parts.append(f"{key}: {raw}")
        inputs = "; ".join(parts)
        return (
            f"- {metric.name}: {metric.formatted} [{metric.citation}] ({metric.formula}; {inputs})"
        )
    return f"- {metric.name}: {metric.formatted} [{metric.citation}] ({metric.formula})"


def _calculations_block(source_id: str, lines: list[str]) -> str:
    """Блок РАСЧЁТЫ по одному источнику.

    Args:
        source_id: Идентификатор материала.
        lines: Строки расчётов.

    Returns:
        Текст блока с заголовком.
    """
    header = f"РАСЧЁТЫ по {source_id} (числа посчитаны кодом, source_id копируется отсюда):"
    return header + "\n" + "\n".join(lines)


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


def _missing_fields(error: str) -> list[str]:
    """Имена полей, которых не хватает в ответе модели.

    pydantic v2 печатает ошибку блоками: имя поля отдельной строкой, ниже —
    ``Field required [type=missing, ...]``. Модели этот текст ничего не
    говорит, поэтому имена достаются здесь и называются в подсказке.

    Args:
        error: Текст ошибки ``ValidationError``.

    Returns:
        Имена пропущенных полей в порядке появления; пусто, если их нет.
    """
    names: list[str] = []
    lines = error.splitlines()
    for index, line in enumerate(lines):
        if index == 0 or "type=missing" not in line:
            continue
        name = lines[index - 1].strip()
        if name and " " not in name and name not in names:
            names.append(name)
    return names


@dataclass(slots=True)
class StepResult:
    """Результат одного шага.

    Attributes:
        role: Роль, выполнявшая шаг.
        phase: Фаза шага.
        data: Валидированный результат роли.
        citations: Подтверждённые ссылки на источники.
        rejected: Сколько цитат отброшено проверкой.
        rejected_details: Что и почему отброшено.
        rejections: Отказы с машинным видом: по нему считается класс отказа,
            а не по формулировке причины.
        needs_review: Ссылки, ожидающие проверки человеком.
        lexical: Лексическое покрытие шага (сколько слов распознано).
        cost_units: Стоимость шага.
        tokens: Токены на шаге.
    """

    role: str
    phase: Phase
    data: BaseModelType
    citations: list[str] = field(default_factory=list)
    rejected: int = 0
    rejected_details: list[str] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    needs_review: list[str] = field(default_factory=list)
    needs_review_details: list[str] = field(default_factory=list)
    #: Опоры на ввод пользователя: что сказано и на какой стадии. Это не
    #: отказ и не подтверждение — решение о доверии принимает человек
    #: (`D-49`).
    user_input_grounds: list[dict[str, Any]] = field(default_factory=list)
    lexical: dict[str, Any] | None = None
    cost_units: float = 0.0
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: int = 0
    #: Окно контекста, в котором считался промпт (``0`` — неизвестно).
    context_limit: int = 0

    @property
    def prompt_truncated(self) -> bool:
        """Упёрся ли промпт шага в окно контекста."""
        return self.context_limit > 0 and self.prompt_tokens >= self.context_limit


#: Тип результата роли — любой валидированный объект схемы.
BaseModelType = Any

#: Соответствие «роль → фаза». По нему решение роутера превращается в
#: следующий шаг рабочего процесса.
#: Сколько расчётных метрик одного материала попадает в контекст.
#: Без ограничения блок расчётов разрастался до 50 тысяч символов и
#: вытеснял сами источники: модель переставала видеть идентификаторы и
#: цитировала вопросы из блока ДАННЫЕ.
MAX_METRICS_IN_CONTEXT = 8

#: Сколько символов русского текста приходится примерно на один токен.
#: Нужно только для подсказки модели: лимит генерации считает токены, а
#: человек думает словами и символами.
CHARS_PER_TOKEN = 2.5

#: Явный лимит ответа в тексте задачи: «не более 200 слов», «максимум 5
#: пунктов», «лимит 1500 токенов». Такой лимит терялся: ограничения
#: собирались на входе и дальше не читались.
_LIMIT_PATTERN = re.compile(
    r"(?:не более|не больше|максимум|лимит[а-яё]*)\s+(\d+)\s*"
    r"(токен[а-яё]*|слов[а-яё]*|символ[а-яё]*|пункт[а-яё]*|строк[а-яё]*)",
    re.IGNORECASE,
)

#: Колонки, которые задают период, а не различие величин. Значения
#: «2017», «2018», «конец июля» — это точки одного ряда во времени.
_PERIOD_COLUMNS: frozenset[str] = frozenset(
    {"период", "дата", "period", "date", "месяц", "квартал", "год", "time", "day"}
)

#: Величина с единицей измерения. Счётчик, год и срок величиной не
#: считаются: «3 месяца» — это горизонт проверки, а не измеренное
#: значение, и требовать для него расчёта было бы шумом.
#:
#: Lookbehind от цифры и знаков числа: дефис в диапазоне «5-10 %» — не минус.
#: Без него проверка читала «−10 %», и замечание называло число, которого в
#: ответе нет: в `demo-v10` и `demo-v12` так появлялись «-7 %» и «-10 %»,
#: хотя план говорил «5-7 %» и «5-10 %».
_MAGNITUDE = re.compile(
    r"(?<![\d,.])"
    r"(?P<value>-?\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>%|процент\w*|п\.\s?п\.|млн|млрд|тыс\.?|руб\w*)",
    re.IGNORECASE,
)

#: Ссылка на строку расчёта внутри текста плана. Модель копирует её из блока
#: РАСЧЁТЫ вместе с числом: «7,39 млн … [project/data/mnp_stats.csv#row2]».
#: Такой якорь называет расчёт прямо, и сравнивать число нужно с ним, а не с
#: показателем, у которого случайно совпало служебное слово.
_CALC_ANCHOR = re.compile(r"\[(?P<path>[^\[\]]+?)#row(?P<row>\d+)\]")

#: Слова, которыми называют готовый агрегат. Если они есть в тексте числа,
#: подтверждать его строкой таблицы нельзя: строку нельзя назвать «средним».
#: Агрегат и его первая строка делят якорь, и без этой развилки значение
#: строки подтверждало сводку, которую посчитал код (`T-62`).
_AGGREGATE_MARKERS: tuple[str, ...] = ("средн", "медиан", "размах")


def _wants_aggregate(text: str) -> bool:
    """Называет ли текст готовый агрегат (среднее, медиану, размах).

    Args:
        text: Текст плана, критерия или обязательства.

    Returns:
        ``True``, если число относится к сводке, а не к строке таблицы.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in _AGGREGATE_MARKERS)


@dataclass(slots=True)
class _Calculation:
    """Расчёт в сверке чисел плана.

    Attributes:
        indicator: Показатель — из строки источника, если расчёт на неё
            ссылается, иначе из имени метрики. У агрегата — полное имя:
            сводку нельзя потерять, иначе строка подтверждает «среднее».
        stems: Основы слов показателя без служебных.
        numbers: Числа расчёта: значение и входы агрегата.
        citation: Ссылка на строку источника (``путь#rowN``) либо пусто.
        is_aggregate: Сводка ли это (среднее, медиана, размах, изменение).
    """

    indicator: str
    stems: set[str]
    numbers: list[float]
    citation: str
    is_aggregate: bool = False


#: Величина взята из утверждения о мире — с неё можно спрашивать расчёт.
PLAN_NUMBER_FACT = "факт"
#: Величина — выбранная цель («вырастет на 5 %»). Цель не утверждение о
#: мире: требовать для неё расчёт значило бы понижать каждый вывод с планом.
PLAN_NUMBER_GOAL = "цель"

#: Число расчёта того же показателя есть, а этого числа в нём нет.
NUMBER_MISMATCH = "number_mismatch"
#: Число есть, но в расчёте другого показателя — подмена.
NUMBER_FOREIGN = "foreign_indicator"
#: Расчёта с таким показателем нет вовсе.
NUMBER_WITHOUT_CALC = "no_calculation"

#: Причина, по которой вывод без оснований не выдаётся за «не проверялся».
#: Текст один: его читают тест, журнал шага и отчёт (`T-77`).
MISSING_EVIDENCE_DETAIL = (
    "вывод не назвал ни одного основания: evidence, promise_grounding и basis пусты"
)


@dataclass(frozen=True, slots=True)
class PlanNumberNote:
    """Одно замечание к числу плана или обязательства.

    Attributes:
        text: Строка для журнала — что и почему не сошлось.
        kind: Вид расхождения (``NUMBER_MISMATCH``, ``NUMBER_FOREIGN``,
            ``NUMBER_WITHOUT_CALC``) — по нему считается замер, а не по
            русскому тексту.
        basis: ``PLAN_NUMBER_FACT`` или ``PLAN_NUMBER_GOAL``. Статус вывода
            понижает только факт: цель — выбранное число.
        magnitude: Пара «значение, единица» — по ней замечания разных текстов
            об одной величине считаются одним.
    """

    text: str
    kind: str
    basis: str
    magnitude: tuple[float, str]


@dataclass(frozen=True, slots=True)
class PlanNumberCheck:
    """Итог сверки чисел плана и обязательств.

    Attributes:
        notes: Замечания для журнала — все, включая цели.
    """

    notes: tuple[PlanNumberNote, ...] = ()

    @property
    def unsupported(self) -> int:
        """Сколько величин-фактов осталось без расчёта своего показателя.

        Считаются уникальные величины: одно и то же число в плане и в
        обязательстве — одно утверждение. Цели сюда не входят — иначе
        «вырастет на 5 %» понижало бы каждый вывод с планом.
        """
        return len({note.magnitude for note in self.notes if note.basis == PLAN_NUMBER_FACT})


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

#: Заголовок блока справочника. Знание о применимости — не факт о кейсе:
#: статус «требует проверки», и это названо прямо (правило
#: `domain-reference-is-not-case-fact`, `T-85`).
REFERENCE_RULE = "СПРАВОЧНИК (знание о применимости, не факт о кейсе; статус: требует проверки):"


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
        *,
        extras: tuple[str, ...] = (),
    ) -> tuple[BaseModelType, Any]:
        """Вызвать роль и проверить результат.

        Args:
            role_name: Идентификатор роли.
            payload: Данные роли (задача, разбор, варианты и т. п.).
            context: Текстовый контекст (источники, предыдущие результаты).
            extras: Дополнительные блоки запроса: ограничения задачи и
                бюджет ответа.

        Returns:
            Пара ``(валидированный результат, расход)``.

        Raises:
            LLMError: Модель не ответила или ответ не прошёл проверку.
        """
        system = roles.load_prompt(role_name)
        user = self._build_user(payload, context, role_name, extras=extras)
        # Модель выбирается по роли: извлечение — дешёвая, синтез — сильная.
        model = roles.model_for(role_name, self.config)

        last_error = ""
        attempts = 3
        for _attempt in range(1, attempts + 1):
            response = self.client.complete(system, user, model=model)

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
        не так. Поэтому в подсказку попадает текст ошибки валидации и
        разбор частых причин (``_repair_hint``).

        Args:
            user: Текст задачи.
            error: Текст ошибки разбора или проверки схемы.

        Returns:
            Задача с пояснением, что исправить.
        """
        hint = Workflow._repair_hint(error)
        return (
            f"{user}\n\n"
            "ПРЕДЫДУЩИЙ ОТВЕТ НЕ ПРИНЯТ. Причина:\n"
            f"{error[:600]}{hint}\n\n"
            "Верни ТОЛЬКО JSON строго по схеме. Обязательные поля: "
            "заполни все — пустые строки будут отклонены."
        )

    @staticmethod
    def _repair_hint(error: str) -> str:
        """Пояснение к частым ошибкам ответа роли.

        Args:
            error: Текст ошибки разбора или проверки схемы.

        Returns:
            Дополнение к запросу; пустая строка, если особой подсказки нет.
        """
        missing = _missing_fields(error)
        if missing:
            names = ", ".join(f"`{name}`" for name in missing)
            return (
                "\n\nВАЖНО: не хватает обязательных полей: "
                f"{names}. Добавь их по схеме и сократи остальное: "
                "длинный ответ обрезается, и тогда не проходит ничего."
            )
        if "обрезан" in error or "JSON не закрыт" in error:
            return (
                "\n\nВАЖНО: ответ обрезан по длине — не хватает закрывающих "
                "скобок. Сократи ответ: убери пояснения и повторы, оставь "
                "только обязательные поля и пункты по схеме."
            )
        if "ссылаться хотя бы на один источник" in error:
            return (
                "\n\nВАЖНО: у факта в поле `sources` пустой массив. Это значит, "
                "что утверждение не подтверждено цитатой из предоставленных "
                "абзацев. Перенеси его в `hypotheses` (там sources не нужен) "
                "либо укажи в `sources` реальный `source_id` из блока "
                "МАТЕРИАЛЫ и дословную `quote` из него. Не выдумывай цитату."
            )
        if "чужой бренд" in error:
            return (
                "\n\nВАЖНО: в `key_message` назван чужой бренд. "
                "Коммуникация обращается к аудитории НАШЕГО продукта: "
                "напиши сообщение от своего имени, без названия конкурента. "
                "Конкурент может остаться в `barrier` — это контекст рынка."
            )
        if "источник не относится к задаче" in error or "source_id" in error:
            return (
                "\n\nВАЖНО: указаны источники, которых нет в блоке МАТЕРИАЛЫ. "
                "source_id копируется из материала ДОСЛОВНО, вместе со знаком #pN. "
                "Придумывать свои названия (Churn Rate, Retention Rate, метрика X) "
                "нельзя: их нет в источниках, и основание будет отброшено. "
                "Если нужного материала нет — запиши это в gaps."
            )
        if "нет полюса" in error:
            # Формулировки берутся из схемы, а не переписываются здесь:
            # иначе подсказка разойдётся с проверкой, как разошёлся
            # `.env.example` с умолчаниями кода.
            keep = "», «".join(KEEP_CURRENT_TITLES[:3])
            close = "», «".join(CLOSE_DIRECTION_TITLES[:3])
            return (
                "\n\nВАЖНО: в наборе нет обязательных вариантов-полюсов. "
                f"Честный набор содержит оба: «{keep}» и «{close}». "
                "Нужны сами варианты с такими названиями, а не маркеры: "
                "`keep_current_marker` и `close_direction_marker` выводятся "
                "из названий, заполнять их не требуется."
            )
        if "String should have at least 1 character" in error:
            return (
                "\n\nВАЖНО: обязательное поле осталось пустым. "
                "У стратегии нужны и аудитория, и выбранное направление — "
                "заполни их по разбору и вариантам. Если данных для "
                "аудитории нет, укажи, кто выигрывает от изменения, "
                "и отметь это как допущение в `consistency_note`."
            )
        return ""

    @classmethod
    def _turn_limit(cls, state: SessionState) -> str:
        """Явный лимит ответа из текста задачи.

        Args:
            state: Состояние сессии.

        Returns:
            Формулировка лимита («не более 200 слов») либо пустая строка.
        """
        frame = state.results.get(Phase.INTAKE.value, {})
        parts = [state.question]
        if isinstance(frame, dict):
            parts.extend(str(item) for item in frame.get("constraints") or [])
        match = _LIMIT_PATTERN.search(" ".join(parts))
        if not match:
            return ""
        return f"не более {match.group(1)} {match.group(2).lower()}"

    @staticmethod
    def _budget_hint(limit_text: str = "") -> str:
        """Подсказка модели о лимите ответа.

        Лимит уже режет генерацию (``num_predict``/``max_tokens``), но
        модель об этом не знает: она пишет длинный JSON и получает
        обрезанный ответ. Дешевле сказать ей предел заранее.

        Args:
            limit_text: Явный лимит из задачи, если он есть.

        Returns:
            Строка подсказки либо пустая строка, если подсказка выключена.
        """
        if not settings.answer_limit_hint:
            return ""
        tokens = settings.effective_max_tokens
        chars = int(tokens * CHARS_PER_TOKEN)
        hint = (
            f"БЮДЖЕТ ОТВЕТА: не более {tokens} токенов (~{chars} символов). "
            "Пиши компактно: без повторов и без пояснений вокруг JSON."
        )
        if limit_text:
            hint += f" Из задачи: {limit_text}."
        return hint

    @staticmethod
    def _constraints_block(state: SessionState) -> str:
        """Ограничения задачи — их нельзя нарушать.

        Ограничения собираются на входе и раньше дальше не читались: в
        демо-прогоне ограничение «цену снижать нельзя» потерялось, и
        выбранным направлением стало «снизить цену». Передача их в запрос
        не делает проверку полной, но убирает молчаливую потерю.

        Args:
            state: Состояние сессии.

        Returns:
            Блок ограничений либо пустая строка.
        """
        frame = state.results.get(Phase.INTAKE.value, {})
        if not isinstance(frame, dict):
            return ""
        items = [str(item).strip() for item in frame.get("constraints") or [] if str(item).strip()]
        if not items:
            return ""
        text = "; ".join(items)
        if len(text) > settings.max_constraints_chars:
            # Длинный блок вытесняет схему ответа: модель возвращает
            # структуру входа вместо своей.
            text = text[: settings.max_constraints_chars].rstrip() + "…"
        return "ОГРАНИЧЕНИЯ ЗАДАЧИ (нарушать нельзя):\n" + text

    @classmethod
    def _prompt_extras(cls, state: SessionState) -> tuple[str, ...]:
        """Дополнительные блоки запроса, общие для всех ролей.

        Язык ответа задаётся здесь, а не в каждом промпте роли: ролей девять, и
        требование, продублированное девять раз, расходится быстрее, чем одно.
        Пока контекст был русским, отсутствие требования было незаметно; на
        английском материале модель отвечала по-английски, и JSON приходил с
        русскими ключами и английскими значениями.

        Args:
            state: Состояние сессии.

        Returns:
            Кортеж непустых блоков: ограничения задачи, бюджет ответа и язык.
        """
        from pf.translate import LANGUAGE_BLOCK

        blocks = [
            cls._constraints_block(state),
            cls._budget_hint(cls._turn_limit(state)),
            LANGUAGE_BLOCK,
        ]
        return tuple(block for block in blocks if block)

    @staticmethod
    def _build_user(
        payload: dict[str, Any],
        context: str,
        role_name: str = "",
        *,
        extras: tuple[str, ...] = (),
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
            extras: Дополнительные блоки: ограничения задачи и бюджет
                ответа. Идут после данных и перед схемой, чтобы схема
                оставалась последним, что видит модель.

        Returns:
            Текст запроса.
        """
        parts = []
        if context:
            # Заголовок обязателен: без него модель не понимает, что
            # перед ней источники, и начинает цитировать вопросы из
            # блока ДАННЫЕ. Идентификаторы копируются только отсюда.
            parts.append(
                "МАТЕРИАЛЫ (источники для цитирования, source_id копируется отсюда):\n" + context
            )
        parts.append("ДАННЫЕ (это вход задачи, цитировать их нельзя):\n" + _as_text(payload))
        parts.extend(extra for extra in extras if extra)
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

    # ── Противоречия ────────────────────────────────────────────────────────
    # Ось расхождения вычисляет код (`pf.contradictions`), а не модель: скил
    # описывает процедуру, хук реагирует на событие. Списки слов о
    # направлении и предмете живут в `pf.lexical` — то же правило нужно
    # вердикту гипотез (`pf.hypotheses`), и две копии уже расходились.

    @classmethod
    def _find_contradictions(cls, facts: list[Any]) -> list[str]:
        """Найти противоречащие утверждения об одном показателе.

        Два факта могут быть подтверждены дословно и при этом говорить
        противоположное: «удержание падает» и «удержание не изменилось за
        последний квартал». Проверка цитат этого не ловит — она смотрит на
        дословность, а не на смысл. Молча стоять рядом такие факты не
        должны: вывод, собранный из них, опирается на противоречие.

        Ось расхождения называет :func:`pf.contradictions.find_contradictions`;
        здесь остаётся только текст пробела для разбора.

        Args:
            facts: Проверенные утверждения разбора.

        Returns:
            Пробелы вида «противоречие по оси …».
        """
        return [item.gap() for item in find_contradictions(facts)]

    # ── Вердикт гипотез ─────────────────────────────────────────────────────

    @staticmethod
    def _confirmed_grounds(grounding: GroundingResult) -> list[str]:
        """Тексты подтверждённых утверждений шага.

        Только ``CONFIRMED``: основание, ожидающее проверки, вердикта не даёт —
        иначе гипотеза выглядела бы проверенной по источнику, который никто
        не подтвердил.

        Args:
            grounding: Результат проверки оснований.

        Returns:
            Тексты подтверждённых утверждений в порядке проверки.
        """
        return [
            verdict.claim
            for verdict in grounding.verdicts
            if verdict.status is SupportStatus.CONFIRMED and verdict.claim.strip()
        ]

    @classmethod
    def _apply_hypothesis_verdicts(cls, data: Any, grounding: GroundingResult) -> None:
        """Поставить вердикт каждой гипотезе разбора.

        Вердикт ставит код: гипотеза — это утверждение модели, а «подтверждено»
        — вывод из проверенных оснований. Гипотеза без подтверждённого
        основания остаётся ``unknown`` и называется в отчёте: неизвестность —
        тоже результат.

        Args:
            data: Результат роли аналитика.
            grounding: Результат проверки оснований этого же шага.
        """
        hypotheses = list(getattr(data, "hypotheses", None) or [])
        if not hypotheses:
            return
        grounds = cls._confirmed_grounds(grounding)
        for hypothesis in hypotheses:
            text = str(getattr(hypothesis, "text", "")).strip()
            result = verdict_of(text, grounds)
            hypothesis.verdict = result.verdict
            hypothesis.verdict_basis = result.basis

    # ── Добор оснований по неизвестному ─────────────────────────────────────

    def _plan_collection(
        self,
        state: SessionState,
        data: Any,
        bundle: SourceBundle | None,
    ) -> CollectionPlan | None:
        """Спланировать добор по неизвестным гипотезам и пробелам.

        Добор детерминированный: корпус ищется по тексту узла, модель не
        вызывается. Кандидат — не основание: он не повышает статус и не
        меняет вердикт, а называется в отчёте отдельным разделом. Иначе
        поиск сам себе выдавал бы подтверждение.

        Без собранных источников добор не идёт: сравнивать найденное с уже
        известным нечем, и «новое» могло бы оказаться тем, что уже собрано.

        Args:
            state: Состояние сессии — в нём живёт прогресс узлов.
            data: Результат разбора с гипотезами и пробелами.
            bundle: Собранные источники.

        Returns:
            План добора либо ``None``, если добор не нужен или невозможен.
        """
        if bundle is None:
            return None
        hypotheses = list(getattr(data, "hypotheses", None) or [])
        gaps = [str(item) for item in (getattr(data, "gaps", None) or [])]
        if not hypotheses and not gaps:
            return None

        from pf.corpus import get_corpus

        recorded = state.results.get("collection_progress")
        plan = plan_collection(
            hypotheses,
            gaps,
            corpus=get_corpus(),
            known=bundle.passages,
            progress=recorded if isinstance(recorded, dict) else None,
        )
        # Состояние узлов переезжает в следующий раунд: два пустых раунда
        # закрывают узел правилом `pf.policy`, а не повторным поиском.
        state.results["collection_progress"] = plan.progress
        return plan

    # ── Проверка оснований ──────────────────────────────────────────────────

    @staticmethod
    def _stems(text: str) -> set[str]:
        """Основы слов текста — показатель сравнивается без словоформ.

        Здесь раньше была своя копия правила, и она не отбрасывала скобки:
        «удержание клиентов (оператор А, абонентская база)» проходило по слову
        «база» из скобок. Правило одно на репозиторий — ``pf.lexical.stems``.

        Args:
            text: Текст утверждения.

        Returns:
            Множество основ значимых слов.
        """
        return stems(text)

    @staticmethod
    def _check_success_criteria(data: Any) -> list[str]:
        """Критерии успеха обязаны говорить о показателе плана.

        План проверки отвечает на вопрос «как поймём, что сработало».
        Критерий про другой показатель («снижение текучести кадров» при
        плане про удержание) проверить нечем: он не следует из плана и не
        связан с решением.

        Args:
            data: Результат роли стратегии.

        Returns:
            Строки отклонений для журнала.
        """
        verification = getattr(data, "verification", None)
        if verification is None:
            return []
        expected = str(getattr(verification, "expected_change", "") or "")
        criteria = list(getattr(verification, "success_criteria", None) or [])
        if not expected or not criteria:
            return []
        expected_stems = Workflow._stems(expected)
        if not expected_stems:
            return []

        notes: list[str] = []
        for criterion in criteria:
            text = str(criterion).strip()
            if not text:
                continue
            if Workflow._stems(text) & expected_stems:
                continue
            notes.append(f"критерий успеха «{text}»: критерий не относится к показателю плана")
        return notes

    @staticmethod
    def _check_constraints(data: Any, state: SessionState) -> dict[str, Any]:
        """Сверить выбранное направление с запретами из постановки задачи.

        Ограничение — часть постановки задачи. Направление, которое его
        нарушает, делает стратегию непригодной, а выглядит она при этом
        связно. До `T-06` ограничения только подставлялись в промпт: в
        прогоне до `demo-v9` выбранным направлением стало «снизить цену»
        при запрете «цену снижать нельзя».

        Args:
            data: Результат роли стратегии.
            state: Состояние сессии — ограничения лежат в постановке задачи.

        Returns:
            Разбор для журнала: сколько ограничений проверено и какие
            нарушения названы. Пустой словарь, если проверять нечего.
        """
        from pf.constraints import check_direction, parse_constraints

        frame = state.results.get(Phase.INTAKE.value, {})
        if not isinstance(frame, dict):
            return {}
        constraints = parse_constraints(frame.get("constraints") or [])
        product = getattr(data, "product", None)
        direction = str(getattr(product, "chosen_direction", "") or "")
        violations = check_direction(direction, constraints)
        if not constraints and not violations:
            return {}
        return {
            "checked": len(constraints),
            "direction": direction,
            "violations": [violation.reason for violation in violations],
        }

    @staticmethod
    def _magnitudes(text: str) -> list[tuple[float, str]]:
        """Числа текста, заявленные как величина.

        Берутся только числа с единицей измерения. Счётчики и сроки сюда
        не попадают: «3 месяца» — горизонт проверки, а не измеренное
        значение, и требовать для него расчёта было бы шумом.

        Args:
            text: Текст плана, критерия или обязательства.

        Returns:
            Пары ``(значение, единица)`` в порядке появления.
        """
        found: list[tuple[float, str]] = []
        for match in _MAGNITUDE.finditer(text):
            try:
                value = float(match.group("value").replace(",", "."))
            except ValueError:
                continue
            found.append((value, match.group("unit")))
        return found

    @staticmethod
    def _calculation_pool(
        metrics: Sequence[Metric] | None = None,
    ) -> list[_Calculation]:
        """Показатели расчётов и числа, которые в них есть.

        Показатель расчёта берётся из строки источника, на которую расчёт
        ссылается: имя метрики — это заголовок материала, и оно не
        говорит, что посчитано. Правило то же, что в проверке оснований.

        Основы показателя берутся без служебных слов
        (:func:`pf.lexical.indicator_stems`): «изменение за период» есть в
        названии каждого второго расчёта и показателем не является.

        Args:
            metrics: Расчёты. По умолчанию — посчитанные по материалам.

        Returns:
            Список расчётов: показатель, его основы, числа и ссылка.
        """
        if metrics is None:
            metrics = Workflow._computed_metric_records()
        materials = []
        if any("#row" in metric.citation for metric in metrics):
            from pf.materials import load_materials

            materials = load_materials()

        pool: list[_Calculation] = []
        for metric in metrics:
            aggregate = metric.is_aggregate
            # У агрегата показатель — полное имя со сводкой: «среднее» должно
            # остаться в основах, иначе строка таблицы подтверждает сводку.
            indicator = (
                metric.name
                if aggregate
                else indicator_of_citation(metric.citation, materials) or metric.name
            )
            if not indicator:
                continue
            numbers = [float(metric.value)]
            numbers.extend(
                float(value)
                for value in metric.inputs.values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            )
            pool.append(
                _Calculation(
                    indicator=indicator,
                    stems=indicator_stems(indicator),
                    numbers=numbers,
                    citation=metric.citation,
                    is_aggregate=aggregate,
                )
            )
        return pool

    @staticmethod
    def _anchored_calculations(text: str, pool: Sequence[_Calculation]) -> list[_Calculation]:
        """Расчёты, на строки которых текст ссылается сам.

        Число, названное вместе со ссылкой ``[путь#rowN]``, подтверждается
        тем расчётом, на который ссылка указывает. Без этого правила база
        сравнения из `cloud-deepseek-2` («рыночный фон: 7,39 млн …
        [mnp_stats.csv#row2]») отклонялась расчётом по ключевой ставке.

        Args:
            text: Текст плана, критерия или обязательства.
            pool: Расчёты, среди которых ищется строка.

        Returns:
            Расчёты, чья строка названа в тексте.
        """
        found: list[_Calculation] = []
        for match in _CALC_ANCHOR.finditer(text):
            path = match.group("path")
            row = int(match.group("row"))
            for calc in pool:
                clean, calc_row = split_source_id(calc.citation)
                if calc_row != row or not same_material(clean, path):
                    continue
                if calc not in found:
                    found.append(calc)
        return found

    @staticmethod
    def _plan_number_texts(data: Any) -> list[tuple[str, str, str]]:
        """Тексты, в которых число — утверждение, а не оформление.

        План проверки называет базу сравнения и цели, решение —
        обязательства. Сроки (``observation_period``, ``review_by``) сюда
        не входят: это календарь, а не измерение.

        Вид текста решает, что делать с неподтверждённым числом. База
        сравнения — утверждение о мире («удержание на уровне 87 %»), и без
        расчёта она ни из чего не следует. Ожидаемое изменение, критерий,
        обязательство и условие пересмотра называют **цель**: «вырастет на
        5 %» — выбранное число, а не измеренное. Замечание о цели остаётся
        для человека, но статус вывода не понижает (`D-47`).

        Args:
            data: Результат роли стратегии или решения.

        Returns:
            Тройки «что это за текст — сам текст — факт или цель».
        """
        pairs: list[tuple[str, str, str]] = []
        verification = getattr(data, "verification", None)
        if verification is not None:
            for label, field_name, basis in (
                ("ожидаемое изменение", "expected_change", PLAN_NUMBER_GOAL),
                ("база сравнения", "compare_against", PLAN_NUMBER_FACT),
            ):
                text = str(getattr(verification, field_name, "") or "").strip()
                if text:
                    pairs.append((label, text, basis))
            for criterion in getattr(verification, "success_criteria", None) or []:
                text = str(criterion).strip()
                if text:
                    pairs.append(("критерий успеха", text, PLAN_NUMBER_GOAL))
        for commitment in getattr(data, "commitments", None) or []:
            for label, field_name, basis in (
                ("обязательство", "statement", PLAN_NUMBER_GOAL),
                ("показатель обязательства", "metric", PLAN_NUMBER_GOAL),
                # База — утверждение о мире, а не цель: число в ней требует
                # расчёта, как число в базе сравнения плана (`T-50`, `D-47`).
                ("база обязательства", "baseline", PLAN_NUMBER_FACT),
                ("метод измерения", "measurement", PLAN_NUMBER_GOAL),
                ("контрольная группа", "control_group", PLAN_NUMBER_GOAL),
                ("условие отката", "rollback", PLAN_NUMBER_GOAL),
                ("условие пересмотра", "review_condition", PLAN_NUMBER_GOAL),
            ):
                text = str(getattr(commitment, field_name, "") or "").strip()
                if text:
                    pairs.append((label, text, basis))
        return pairs

    @staticmethod
    def _check_commitment_template(data: Any) -> list[str]:
        """Замечания «обязательство без базы / без метода».

        Обязательство без базы нельзя ни проверить, ни исполнить: «удержание
        вырастет на 5 %» не говорит, от чего. Без метода неясно, как считать:
        «доля уведомлений до списания» считается по доставке, по открытию или
        по факту оплаты. Поля необязательны в схеме — прогон без них остаётся
        возможным, но названным (`T-50`). Замечание не останавливает работу:
        это повод заполнить шаблон, а не отказ.

        Args:
            data: Результат роли решения.

        Returns:
            Строки замечаний: одна — на обязательство, одна — на отсутствие
            базы или метода.
        """
        notes: list[str] = []
        for index, commitment in enumerate(getattr(data, "commitments", None) or [], start=1):
            statement = str(getattr(commitment, "statement", "") or "").strip()
            label = f"обязательство {index}" + (f" «{statement}»" if statement else "")
            baseline = str(getattr(commitment, "baseline", "") or "").strip()
            measurement = str(getattr(commitment, "measurement", "") or "").strip()
            if not baseline:
                notes.append(f"{label}: база сравнения не названа — от чего измеряем, неизвестно")
            if not measurement:
                notes.append(f"{label}: метод измерения не назван — как считаем, неизвестно")
        return notes

    @staticmethod
    def _check_plan_numbers(
        data: Any,
        metrics: Sequence[Metric] | None = None,
    ) -> PlanNumberCheck:
        """Числа плана и обязательств сверяются с расчётами.

        Прежняя проверка сравнивала только основы слов: критерий
        «увеличение удержания на 5 %» проходил по слову «удержание», а
        число не сверялось ни с чем. В прогоне `demo-v9` так появилось
        «удержание на уровне 87 %», где 87 — абонентская база оператора.

        Число подтверждено, только если расчёт **того же** показателя его
        содержит либо текст сам называет строку расчёта якорем. Четыре
        исхода различаются, потому что причина важнее отметки:

        - расчёт по этому показателю есть, но числа в нём нет — расхождение;
        - расчёт есть, но про другой показатель — подмена;
        - текст называет строку расчёта, и число в ней есть — подтверждено;
        - расчёта нет вовсе — величина ни из чего не следует.

        Служебные слова в показателе не участвуют: в прогоне
        `cloud-deepseek-2` из-за слова «период» расчёт по ключевой ставке
        считался «тем же показателем», что и рыночный фон, и верные числа
        отклонялись с чужим именем в причине.

        Args:
            data: Результат роли стратегии или решения.
            metrics: Расчёты. По умолчанию — посчитанные по материалам.

        Returns:
            Замечания и их вид. ``unsupported`` считает величины-факты без
            расчёта своего показателя — это число идёт в статус вывода.
        """
        pool = Workflow._calculation_pool(metrics)
        notes: list[PlanNumberNote] = []
        for label, text, basis in Workflow._plan_number_texts(data):
            text_stems = indicator_stems(text)
            same = [calc for calc in pool if calc.stems & text_stems]
            others = [calc for calc in pool if not calc.stems & text_stems]
            anchored = Workflow._anchored_calculations(text, pool)
            if _wants_aggregate(text):
                # Готовый агрегат подтверждает только расчёт-сводка: значение
                # строки с тем же якорем — это арифметика модели (`T-62`).
                same = [calc for calc in same if calc.is_aggregate]
                anchored = [calc for calc in anchored if calc.is_aggregate]
            for value, unit in Workflow._magnitudes(text):
                shown = f"{value:g} {unit}".strip()
                magnitude = (value, unit)
                if any(numbers_match([value], calc.numbers)[0] for calc in same):
                    continue
                if any(numbers_match([value], calc.numbers)[0] for calc in anchored):
                    continue
                if same:
                    notes.append(
                        PlanNumberNote(
                            text=(
                                f"{label} «{text}»: число {shown} не совпадает с расчётом "
                                f"по показателю «{same[0].indicator}»"
                            ),
                            kind=NUMBER_MISMATCH,
                            basis=basis,
                            magnitude=magnitude,
                        )
                    )
                    continue
                foreign = next(
                    (calc.indicator for calc in others if numbers_match([value], calc.numbers)[0]),
                    "",
                )
                if foreign:
                    notes.append(
                        PlanNumberNote(
                            text=(
                                f"{label} «{text}»: число {shown} взято из расчёта «{foreign}» — "
                                "другой показатель"
                            ),
                            kind=NUMBER_FOREIGN,
                            basis=basis,
                            magnitude=magnitude,
                        )
                    )
                    continue
                notes.append(
                    PlanNumberNote(
                        text=(
                            f"{label} «{text}»: число {shown} ничем не подтверждено — "
                            "расчёта с таким показателем нет"
                        ),
                        kind=NUMBER_WITHOUT_CALC,
                        basis=basis,
                        magnitude=magnitude,
                    )
                )
        return PlanNumberCheck(notes=tuple(notes))

    @staticmethod
    def _declares_grounds(data: Any) -> bool:
        """Результат роли — вывод, который обязан назвать основания.

        Стратегия обязана нести ``product.evidence`` и
        ``communication.promise_grounding``, решение — ``commitments[].basis``.
        Факты, план и варианты выводом не являются и здесь не судятся: у них
        нет полей, которые можно оставить пустыми (`T-77`).

        Args:
            data: Результат роли.

        Returns:
            ``True``, если результат — стратегия или запись решения.
        """
        return getattr(data, "product", None) is not None or hasattr(data, "commitments")

    @staticmethod
    def _sources_of(data: Any) -> list[tuple[str, Any, bool]]:
        """Все основания результата: утверждение, ссылка, это вывод или нет.

        Основание — не только факт. Стратегия ссылается на материалы через
        ``product.evidence`` и ``communication.promise_grounding``, решение —
        через ``commitments[].basis``. Раньше эти ссылки не проверялись
        вовсе, и вывод мог опираться на файл, которого нет в материалах.

        Args:
            data: Результат роли.

        Returns:
            Тройки «текст утверждения — основание — относится к выводу».
        """
        pairs: list[tuple[str, Any, bool]] = []

        for claim in getattr(data, "facts", None) or []:
            text = getattr(claim, "text", "")
            for source in getattr(claim, "sources", None) or []:
                pairs.append((text, source, False))

        product = getattr(data, "product", None)
        direction = getattr(product, "chosen_direction", "") if product is not None else ""
        for source in getattr(product, "evidence", None) or []:
            pairs.append((direction, source, True))

        communication = getattr(data, "communication", None)
        message = getattr(communication, "key_message", "") if communication is not None else ""
        for source in getattr(communication, "promise_grounding", None) or []:
            pairs.append((message, source, True))

        for commitment in getattr(data, "commitments", None) or []:
            statement = getattr(commitment, "statement", "")
            for source in getattr(commitment, "basis", None) or []:
                pairs.append((statement, source, True))

        return pairs

    @staticmethod
    def _conclusion_topic(data: Any) -> tuple[set[str], list[str]]:
        """Показатель вывода: что в нём обещано измерить.

        Берутся два места результата: ожидаемое изменение и показатель
        обязательства. Выбранное направление сюда **не** входит: это текст
        действия, и модель упоминает в нём любой показатель — тогда правило
        перестаёт различать подмену («направление: поднять абонентскую базу»
        пересекалось бы с основанием про базу, хотя вывод измеряет удержание).
        Вопрос задачи тоже не входит: «почему падает удержание» — вопрос, а
        не утверждение, и честный вывод может быть про другое.

        Args:
            data: Результат роли стратегии или решения.

        Returns:
            Пары «основы показателя вывода» и «как он назван словами».
        """
        texts: list[str] = []
        verification = getattr(data, "verification", None)
        if verification is not None:
            texts.append(str(getattr(verification, "expected_change", "") or ""))
        metrics = [
            str(getattr(commitment, "metric", "") or "")
            for commitment in getattr(data, "commitments", None) or []
        ]
        texts.extend(metrics)
        topic: set[str] = set()
        for text in texts:
            if text.strip():
                topic |= indicator_stems(text)
        named = [text.strip() for text in metrics if text.strip()]
        if not named:
            named = [text.strip()[:200] for text in texts if text.strip()][:1]
        return topic, named

    @staticmethod
    def _ground_indicators(anchors: Mapping[str, str]) -> dict[str, str]:
        """Показатели подтверждённых оснований вывода: ключ основания → показатель.

        Показатель известен только у основания-строки таблицы: у абзаца статьи
        его нет. Такие основания правило не судит — иначе оно срабатывало бы на
        каждом выводе, а не на подмене показателя.

        Словарь, а не список названий: понижение считается по каждому
        основанию отдельно. Список отвечал только «все чужие или нет» и
        молчал, когда из двух оснований чужое было одно.

        Args:
            anchors: Ключи оснований вывода и их якоря (``путь#якорь``).

        Returns:
            Словарь «ключ основания — название показателя»; пустых нет.
        """
        indicators: dict[str, str] = {}
        materials = None
        for key, anchor in anchors.items():
            clean, row = split_source_id(anchor)
            if row is None or not clean:
                continue
            if materials is None:
                from pf.materials import load_materials

                materials = load_materials()
            indicator = indicator_of_citation(anchor, materials)
            if indicator:
                indicators[key] = indicator
        return indicators

    @staticmethod
    def _conclusion_verdict(
        tally: GroundTally,
        grounds: dict[str, list[str]],
        off_topic: dict[str, list[str]] | None = None,
        unsupported_numbers: int = 0,
        user_input: int = 0,
        declared: bool = False,
    ) -> dict[str, Any] | None:
        """Статус оснований вывода.

        Основания считаются по уникальным ключам, а не по записям:
        стратегия и решение нередко объявляют одну и ту же строку
        источника дважды. В прогоне `demo-v9` четыре объявленных основания
        вели на две строки таблицы, и документ называл их одновременно
        «8 из 8» и «уникальных 2» — доля выглядела лучше, чем была.

        Три состояния различаются принципиально: «подкреплено» — хотя бы
        одно основание подтверждено и ни одно не отклонено; «подкреплено
        частично» — часть отклонена; «неподкреплено» — ни одного. Молчать
        нельзя: вывод без основания выглядит в документе так же, как вывод
        с основанием.

        Args:
            tally: Счёт по уникальным ключам (худший вердикт побеждает).
            grounds: Ключи оснований по вердиктам — по ним отчёт
                объединяет шаги, не складывая повторы.
            off_topic: Показатель вывода и показатели подтверждённых
                оснований, когда они не совпали.
            unsupported_numbers: Сколько величин плана осталось без своего
                расчёта. Ключ пишется всегда, включая ноль: пропуск
                читался бы как «не проверяли».
            user_input: Сколько опор держится на вводе пользователя. Они
                входят в ``needs_review``, но называются отдельно: доверие
                к своим словам человек выставляет сам.
            declared: Результат — вывод, который мог назвать основания.
                Тогда ``checked=0`` значит «оснований не названо», и это
                неподкреплённый вывод, а не отсутствие вывода (`T-77`).

        Returns:
            Статус для журнала и отчёта либо ``None``, если вывода нет.
        """
        if tally.checked <= 0 and not declared:
            return None
        status = conclusion_status(
            tally.checked,
            tally.confirmed,
            tally.rejected,
            tally.needs_review,
            unsupported_numbers=unsupported_numbers,
            user_input=user_input,
            declared=declared,
        )
        payload: dict[str, Any] = {
            "checked": tally.checked,
            "confirmed": tally.confirmed,
            "rejected": tally.rejected,
            "needs_review": tally.needs_review,
            "status": status,
            "grounds": grounds,
            # Число без расчёта понижает статус — причина едет вместе с ним,
            # иначе читатель видит потолок и не знает, за что.
            "unsupported_numbers": unsupported_numbers,
            # Сколько опор держится на словах человека: статус «опирается на
            # ввод пользователя» без этого числа не отличить от «нечем
            # подтвердить».
            "user_input": user_input,
            # Имя правила едет вместе со статусом. Статус без него читается
            # как текущая оценка, оставаясь записью прогона, — так `demo-v9`
            # печатал «подкреплено» рядом с «отклонено» по тем же строкам.
            "rule": CONCLUSION_RULE,
        }
        if off_topic is not None:
            payload["off_topic"] = off_topic
        return payload

    @staticmethod
    def _external_text(named: str, state: SessionState | None) -> str:
        """Текст внешнего источника по идентификатору.

        Внешние источники не попадают в ``SourceBundle``: у них другой
        статус доверия. Но проверить дословность цитаты по ним можно —
        страница уже скачана и лежит в состоянии сессии.

        Args:
            named: Идентификатор из ответа модели.
            state: Состояние сессии либо ``None``.

        Returns:
            Текст страницы либо пустая строка, если источник не внешний.
        """
        if state is None or not named:
            return ""
        for page in state.results.get("web_pages") or []:
            title = str(getattr(page, "title", ""))
            url = str(getattr(page, "url", ""))
            candidates = {f"web/{title}", f"web/{url}", f"web/{title or url}", url}
            if named in candidates:
                return str(getattr(page, "text", "") or "")
        for source in state.results.get("web_source_objects") or []:
            candidates = {
                str(getattr(source, "source_id", "")),
                str(getattr(source, "url", "")),
            }
            if named in candidates:
                return str(getattr(source, "text", "") or "")
        return ""

    @staticmethod
    def _queue_oov(
        tokens: set[str],
        *,
        context: str = "",
        role: str = "",
        session: str = "",
    ) -> None:
        """Положить нераспознанные слова в очередь распознания.

        Очередь — след работы, а не часть проверки: она ничего не
        подтверждает и не отменяет. Выключить можно настройкой
        ``PF_LEXICON_ENABLED=false`` — это нужно для чистых замеров.

        Args:
            tokens: Нераспознанные слова.
            context: Контекст, в котором они встретились.
            role: Роль шага.
            session: Идентификатор сессии.
        """
        if not tokens or not settings.lexicon_enabled:
            return
        from pf import lexicon

        for token in sorted(tokens):
            lexicon.record_oov(
                token,
                reason="слово не распознано при проверке оснований",
                role=role,
                session=session,
                context=context,
            )

    @staticmethod
    def _lexical_report(data: Any) -> dict[str, Any] | None:
        """Посчитать лексическое покрытие утверждений шага.

        Покрытие — измеримый след работы с незнакомыми словами: видно,
        сколько слов проверка поняла, а сколько ушло в очередь. Это не
        замена доле подтверждённых оснований, а соседняя метрика.

        Args:
            data: Результат роли.

        Returns:
            Словарь покрытия либо ``None``, если считать нечего.
        """
        if not settings.lexicon_enabled:
            return None
        from pf.lexicon import coverage_report

        texts = list(dict.fromkeys(text for text, _source, _kind in Workflow._sources_of(data)))
        if not texts:
            return None
        return coverage_report(texts).to_dict()

    @staticmethod
    def _verify_human_input(
        quote: str,
        stage: str,
        state: SessionState | None,
    ) -> HumanInputVerdict:
        """Проверить ссылку на ввод пользователя.

        Канал подтверждения у слов человека свой: не «встречается ли цитата в
        документе», а «говорил ли он это и на той ли стадии». Три отказа
        различаются, потому что чинятся по-разному: ввода не было вовсе
        (модель приписала человеку слова), стадия названа не та, слова
        пересказаны.

        Принятая опора остаётся «требует проверки человеком»: код не имеет
        права объявить слова пользователя подтверждённым фактом — иначе
        доверие к ним выставил бы он, а не человек (`D-49`).

        Args:
            quote: Что модель приписывает пользователю.
            stage: Стадия из ссылки — может быть пустой у старых записей.
            state: Состояние сессии с записанными ответами.

        Returns:
            Вердикт: опора принята либо отказ с видом.
        """
        records = human_inputs(state)
        if not records:
            return HumanInputVerdict(
                status=SupportStatus.NO_DATA,
                detail="в сессии не записано ни одного ответа пользователя — ссылаться не на что",
                kind="human_input_unrecorded",
            )

        wanted = phase_of(stage)
        if wanted is not None:
            matched = [record for record in records if record.phase == wanted.value]
            if not matched:
                return HumanInputVerdict(
                    status=SupportStatus.NO_DATA,
                    detail=(
                        f"на стадии «{wanted.title_ru}» пользователь ничего не вводил — "
                        "стадия названа не та"
                    ),
                    kind="human_input_phase_mismatch",
                )
            where = wanted.title_ru
        elif stage:
            return HumanInputVerdict(
                status=SupportStatus.NO_DATA,
                detail=f"стадия «{stage}» не распознана — проверять ссылку нечем",
                kind="human_input_phase_mismatch",
            )
        elif len(records) == 1:
            # Один ввод в сессии — стадия неоднозначна, и это сказано. Так
            # выглядят сохранённые прогоны: ссылка «Ответ человека» без
            # стадии, а других вводов нет.
            matched = records
            where = f"{records[0].phase_title} — единственный ввод в сессии"
        elif all(not record.phase for record in records):
            matched = records
            where = "стадия не записана (сессия до правки)"
        else:
            return HumanInputVerdict(
                status=SupportStatus.NO_DATA,
                detail=(
                    f"вводов в сессии несколько ({len(records)}), а стадия не названа — "
                    "непонятно, на каком из них держится утверждение"
                ),
                kind="human_input_phase_mismatch",
            )

        if quote and not any(Workflow._said_before(quote, record.text) for record in matched):
            return HumanInputVerdict(
                status=SupportStatus.NO_DATA,
                detail=(
                    "таких слов в записанном вводе пользователя нет: ссылка выдаёт "
                    "за сказанное то, чего он не говорил"
                ),
                kind="human_input_quote_mismatch",
            )
        note = (
            "слова подтверждены записью"
            if quote
            else "цитата не приведена — проверено только, что ввод был"
        )
        return HumanInputVerdict(
            status=SupportStatus.NEEDS_REVIEW,
            detail=(
                f"ввод пользователя ({where}): {note}; "
                "верить этим словам или нет — решение пользователя"
            ),
            kind="user_input",
            stage=where,
        )

    @staticmethod
    def _said_before(quote: str, said: str) -> bool:
        """Сказаны ли эти слова в записанном вводе.

        Сравниваются основы слов, а не дословная подстрока: цитата может быть
        фрагментом ответа в другой словоформе, и это не подмена. Человек — не
        документ, требовать от него буквального совпадения значило бы
        отклонять честную ссылку по форме.

        Args:
            quote: Что модель приписывает пользователю.
            said: Что он действительно сказал.

        Returns:
            Все значимые слова цитаты есть в записи.
        """
        wanted = stems(quote)
        return not wanted or wanted <= stems(said)

    @staticmethod
    def _verify_claims(
        data: Any,
        bundle: SourceBundle | None,
        *,
        state: SessionState | None = None,
        role_name: str = "",
        session_id: str = "",
        unsupported_numbers: int = 0,
    ) -> GroundingResult:
        """Проверить основания в результате роли.

        Проверка одна для всех: дословность цитаты и релевантность
        источника задаче. Факт, обещание коммуникации и основание
        обязательства проверяются одинаково — иначе вывод оказывается
        единственным утверждением без проверки.

        Три исхода различаются: подтверждено, отклонено и «требует
        проверки». Третий — для внешних источников: подтвердить их нечем,
        но и отказом это не является.

        Args:
            data: Результат роли.
            bundle: Собранные источники.
            state: Состояние сессии — нужно для доступа к внешним страницам.
            role_name: Роль шага — попадает в очередь распознания.
            session_id: Идентификатор сессии — попадает туда же.
            unsupported_numbers: Сколько величин-фактов плана осталось без
                расчёта своего показателя. Считается до проверки оснований:
                статус вывода зависит от чисел, и считать его дважды
                (сначала без чисел, потом с ними) значило бы иметь два
                статуса одного шага (`T-48b`).

        Returns:
            Подтверждённые ссылки, отказы, пояснения, ожидающие проверки
            основания и статус оснований вывода, если вывод есть.
        """
        if bundle is None:
            return GroundingResult()

        from pf import reference

        confirmed: list[str] = []
        rejected = 0
        details: list[str] = []
        needs_review: list[str] = []
        needs_review_details: list[str] = []
        # Вердикты вывода собираются парами «ключ основания — вердикт»:
        # счёт по записям завышал число, когда на одну строку источника
        # ссылались дважды.
        conclusion_verdicts: list[tuple[str, SupportStatus]] = []
        # Якорь основания с номером строки — если модель положила номер в
        # поле ``paragraph``, а не в ссылку. Нужен, чтобы понять, о каком
        # показателе строка; счёт оснований он не меняет.
        conclusion_anchors: dict[str, str] = {}
        oov: set[str] = set()
        oov_context = ""
        verdicts: list[ClaimVerdict] = []
        # Опоры на ввод пользователя: ключи оснований вывода, которые держатся
        # на словах человека. Считаются отдельно от подтверждённых и
        # отклонённых: «верить или нет» решает человек, а не проверка.
        user_input_grounds: list[dict[str, str]] = []
        user_input_titles: list[str] = []
        user_input_keys: set[str] = set()
        # Передача вывода (`T-46`): поля утверждения и хеш источника. Только
        # для подтверждённых опор вывода — у отклонённого передавать нечего.
        handoffs: list[dict[str, Any]] = []

        for text, source, is_conclusion in Workflow._sources_of(data):
            named = getattr(source, "source_id", "") or ""
            quote = getattr(source, "quote", "") or ""
            # Ключ основания — строка источника. Безымянное основание
            # (модель прислала имя поля вместо ссылки) различается по
            # цитате: два разных утверждения — это два основания.
            key = citation_key(named) or f"без источника: {quote}"
            # Строка таблицы не всегда в ссылке: модель кладёт её номер в
            # поле ``paragraph``, и тогда в журнале остаётся только путь.
            # Для сверки показателя основания якорь нужен, а счёт оснований
            # он менять не должен — поэтому якорь живёт отдельной картой.
            paragraph = getattr(source, "paragraph", None)
            anchor = named if "#" in named or paragraph is None else f"{named}#p{paragraph}"

            # Канал «ввод пользователя» проверяется своим правилом: слова
            # человека подтверждаются записью, а не документом. Проверка идёт
            # до внешних источников и до причины отказа: ссылка на ввод — не
            # «фантомный источник» и не «ссылки нет» (`D-49`).
            stage = human_input_stage(named)
            if stage is None and getattr(source, "basis_kind", "факт") == "ввод пользователя":
                stage = str(getattr(source, "basis_phase", "") or "")
            if stage is not None:
                input_support = Workflow._verify_human_input(quote, stage, state)
                verdicts.append(
                    ClaimVerdict(
                        claim=text,
                        source_id=named,
                        status=input_support.status,
                        detail=input_support.detail,
                        is_conclusion=is_conclusion,
                        kind=input_support.kind,
                    )
                )
                if input_support.status is SupportStatus.NEEDS_REVIEW:
                    # Опора принята: слова были сказаны. Доверие к ним — не
                    # дело проверки, поэтому основание ждёт решения человека.
                    user_input_grounds.append(
                        {
                            "stage": input_support.stage,
                            "quote": quote,
                            "claim": text,
                            "source": input_support.detail,
                        }
                    )
                    user_input_titles.append(f"{input_support.stage}: «{quote or text}»")
                    if is_conclusion:
                        user_input_keys.add(key)
                        conclusion_verdicts.append((key, SupportStatus.NEEDS_REVIEW))
                else:
                    rejected += 1
                    details.append(f"{_named_source(named)}: {input_support.detail}")
                    if is_conclusion:
                        conclusion_verdicts.append((key, SupportStatus.NO_DATA))
                continue

            external = Workflow._external_text(named, state)
            if external:
                external_support = verify_quote_as_external(external, quote, named)
                verdicts.append(
                    ClaimVerdict(
                        claim=text,
                        source_id=named,
                        status=external_support.status,
                        detail=external_support.detail,
                        is_conclusion=is_conclusion,
                    )
                )
                if external_support.status is SupportStatus.NEEDS_REVIEW:
                    needs_review.append(external_support.citation)
                    needs_review_details.append(
                        f"{_named_source(named)}: {external_support.detail}"
                    )
                    if is_conclusion:
                        conclusion_verdicts.append((key, SupportStatus.NEEDS_REVIEW))
                else:
                    rejected += 1
                    details.append(f"{_named_source(named)}: {external_support.detail}")
                    if is_conclusion:
                        conclusion_verdicts.append((key, SupportStatus.NO_DATA))
                continue

            # Тематический справочник — отдельный канал: знание о
            # применимости, а не факт о кейсе. Дословность проверяется, но
            # статус не поднимается выше «требует проверки»: правило
            # `domain-reference-is-not-case-fact` (`T-85`).
            if reference.is_reference(named):
                reference_text = reference.text_of(named)
                if reference_text:
                    reference_support = mark_external(
                        verify_quote(reference_text, quote, named),
                        reason=(
                            "справочник: знание о применимости, не факт о кейсе "
                            "— требует проверки человеком"
                        ),
                    )
                    verdicts.append(
                        ClaimVerdict(
                            claim=text,
                            source_id=named,
                            status=reference_support.status,
                            detail=reference_support.detail,
                            is_conclusion=is_conclusion,
                            kind=REFERENCE,
                        )
                    )
                    if reference_support.status is SupportStatus.NEEDS_REVIEW:
                        needs_review.append(reference_support.citation)
                        needs_review_details.append(
                            f"{_named_source(named)}: {reference_support.detail}"
                        )
                        if is_conclusion:
                            conclusion_verdicts.append((key, SupportStatus.NEEDS_REVIEW))
                    else:
                        rejected += 1
                        details.append(f"{_named_source(named)}: {reference_support.detail}")
                        if is_conclusion:
                            conclusion_verdicts.append((key, SupportStatus.NO_DATA))
                else:
                    rejected += 1
                    details.append(f"{_named_source(named)}: понятие справочника не найдено")
                    verdicts.append(
                        ClaimVerdict(
                            claim=text,
                            source_id=named,
                            status=SupportStatus.NO_DATA,
                            detail="понятие справочника не найдено",
                            is_conclusion=is_conclusion,
                            kind=REFERENCE,
                        )
                    )
                    if is_conclusion:
                        conclusion_verdicts.append((key, SupportStatus.NO_DATA))
                continue

            # Причина отказа проверяется **до** сверки: если ссылки нет, то
            # и сверять нечего, а прежняя формулировка «источник не относится
            # к задаче» называла чужим отказ, которого не было (`D-45`).
            absence = bundle.absence_reason(named)
            if absence:
                rejected += 1
                details.append(f"{_named_source(named)}: {absence}")
                verdicts.append(
                    ClaimVerdict(
                        claim=text,
                        source_id=named,
                        status=SupportStatus.NO_DATA,
                        detail=absence,
                        is_conclusion=is_conclusion,
                        kind=bundle.absence_kind(named),
                    )
                )
                if is_conclusion:
                    conclusion_verdicts.append((key, SupportStatus.NO_DATA))
                continue

            support = bundle.verify_claim(
                text,
                source_id=named,
                paragraph=getattr(source, "paragraph", None),
                quote=quote,
            )
            if support.oov:
                oov.update(support.oov)
                oov_context = oov_context or text
            verdicts.append(
                ClaimVerdict(
                    claim=text,
                    source_id=named,
                    status=support.status,
                    detail=support.detail,
                    is_conclusion=is_conclusion,
                    kind=support.kind,
                )
            )
            if support.is_usable:
                confirmed.append(support.citation)
                if is_conclusion:
                    conclusion_verdicts.append((key, SupportStatus.CONFIRMED))
                    if anchor and anchor != key:
                        conclusion_anchors[key] = anchor
                    if support.source_hash and support.canonical is not None:
                        handoffs.append(
                            {
                                "claim": text,
                                "source_id": support.source_id or named,
                                "source_hash": support.source_hash,
                                "fields": dict(support.canonical),
                            }
                        )
            elif support.status is SupportStatus.NEEDS_REVIEW:
                needs_review.append(support.citation)
                needs_review_details.append(f"{_named_source(named)}: {support.detail}")
                if is_conclusion:
                    conclusion_verdicts.append((key, SupportStatus.NEEDS_REVIEW))
            else:
                rejected += 1
                details.append(f"{_named_source(named)}: {support.detail}")
                if is_conclusion:
                    conclusion_verdicts.append((key, SupportStatus.NO_DATA))

        confirmed_metrics, rejected_metrics, metric_details, metric_verdicts = (
            Workflow._verify_metrics(data)
        )
        confirmed.extend(confirmed_metrics)
        rejected += rejected_metrics
        details.extend(metric_details)
        verdicts.extend(metric_verdicts)

        # Якорь, на котором расчёт отклонён, не подкрепляет вывод. В прогоне
        # demo-v9 обе метрики были отклонены («пересчёт по входам даёт 0, а
        # заявлено 87 тыс.»), а вывод назывался «подкреплено»: он опирался
        # на те же две строки таблицы. Один якорь не может быть одновременно
        # непригодным для расчёта и достаточным для решения.
        failed_keys = {
            citation_key(verdict.source_id)
            for verdict in metric_verdicts
            if verdict.status is not SupportStatus.CONFIRMED and verdict.source_id
        }
        if state is not None:
            if metric_verdicts:
                # Решение «этому якорю верить нельзя» переезжает в следующую
                # фазу: вывод проверяется уже после разбора.
                state.results["rejected_metric_keys"] = sorted(failed_keys)
            failed_keys |= set(state.results.get("rejected_metric_keys") or [])
        if failed_keys and conclusion_verdicts:
            conclusion_verdicts = [
                (key, SupportStatus.NO_DATA if key in failed_keys else status)
                for key, status in conclusion_verdicts
            ]

        # Дословная цитата ничего не говорит о том, **тот ли** показатель
        # подтверждён. В `cloud-deepseek-2` вывод про удержание держался на
        # уровне ARPU, в `cloud-deepseek-baseline-1` — на абонентской базе.
        # Правило узкое: судит только те основания, у которых показатель
        # назван (строка таблицы с колонкой «показатель»); про абзацы статей
        # молчит.
        #
        # Понижается каждое чужое основание отдельно, а не «все или ничего»:
        # в `demo-v10` одно основание из одного было про ARPU, и потолок
        # «подкреплено частично» оставлял его в подтверждённых. Дословная
        # цитата про другую величину вывод не подтверждает — счётчик
        # подтверждённых не должен её считать (`T-57`, `D-46`).
        off_topic: dict[str, list[str]] | None = None
        if conclusion_verdicts:
            topic, named = Workflow._conclusion_topic(data)
            confirmed_keys = {
                key for key, status in conclusion_verdicts if status is SupportStatus.CONFIRMED
            }
            anchors = {key: conclusion_anchors.get(key, key) for key in confirmed_keys}
            indicators = Workflow._ground_indicators(anchors)
            foreign: dict[str, str] = {}
            if topic:
                foreign = {
                    key: indicator
                    for key, indicator in indicators.items()
                    if indicator_stems(indicator) and not indicator_stems(indicator) & topic
                }
            if foreign:
                off_topic = {
                    "conclusion": named[:3],
                    "grounds": list(dict.fromkeys(foreign.values()))[:5],
                }
                conclusion_verdicts = [
                    (key, SupportStatus.NO_DATA if key in foreign else status)
                    for key, status in conclusion_verdicts
                ]

        # Вывод, который не назвал ни одного основания, — не «не проверялся»:
        # вывод есть, и он ни на чём не держится. Молчание читалось как «до
        # вывода не дошли» (`report.readiness`), и прогон выглядел
        # незавершённым, хотя ответ на вопрос получен (`T-77`).
        declared = Workflow._declares_grounds(data)
        missing_evidence = declared and not conclusion_verdicts
        if missing_evidence:
            verdicts.append(
                ClaimVerdict(
                    claim="",
                    source_id="",
                    status=SupportStatus.NO_DATA,
                    detail=MISSING_EVIDENCE_DETAIL,
                    is_conclusion=True,
                    kind=MISSING_EVIDENCE,
                )
            )
            rejected += 1
            details.append(MISSING_EVIDENCE_DETAIL)

        Workflow._queue_oov(oov, context=oov_context, role=role_name, session=session_id)

        conclusion = Workflow._conclusion_verdict(
            tally_grounds(conclusion_verdicts),
            ground_buckets(conclusion_verdicts),
            off_topic,
            unsupported_numbers,
            len(user_input_keys),
            declared=declared,
        )
        if conclusion is not None and missing_evidence:
            conclusion["reason"] = MISSING_EVIDENCE_DETAIL
        return GroundingResult(
            citations=confirmed,
            rejected=rejected,
            details=details,
            conclusion=conclusion,
            needs_review=needs_review,
            needs_review_details=needs_review_details,
            user_input=user_input_titles,
            user_input_grounds=user_input_grounds,
            verdicts=verdicts,
            handoffs=handoffs,
        )

    @staticmethod
    def _verify_metrics(data: Any) -> tuple[list[str], int, list[str], list[ClaimVerdict]]:
        """Проверить расчётные метрики пересчётом по источнику.

        Факт подтверждается дословной цитатой. Метрику подтверждает
        только пересчёт: число нельзя «процитировать», его можно только
        посчитать заново. Метрика без источника или с неверным
        значением отбрасывается — как и выдуманная цитата.

        Args:
            data: Результат роли.

        Returns:
            Четвёрка ``(подтверждённые ссылки, число отброшенных, пояснения,
            вердикты по метрикам)``.
        """
        metrics = getattr(data, "metrics", None) or []
        if not metrics:
            return [], 0, [], []

        from pf.materials import load_materials
        from pf.sources import verify_metric

        materials = load_materials()
        confirmed: list[str] = []
        rejected = 0
        details: list[str] = []
        verdicts: list[ClaimVerdict] = []
        for metric in metrics:
            support = verify_metric(metric, materials)
            name = getattr(metric, "name", "метрика")
            verdicts.append(
                ClaimVerdict(
                    claim=str(name),
                    source_id=str(getattr(metric, "source_id", "") or ""),
                    status=support.status,
                    detail=support.detail,
                )
            )
            if support.is_usable:
                confirmed.append(support.citation)
            else:
                rejected += 1
                details.append(f"{name}: {support.detail}")
        return confirmed, rejected, details, verdicts

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
            absence = bundle.absence_reason(named)
            if absence:
                from pf.ground import Support, SupportStatus

                return Support(
                    quote=quote,
                    source_id=source_id,
                    status=SupportStatus.NO_DATA,
                    detail=absence,
                    kind=bundle.absence_kind(named),
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
        from pf.roles import ROLES, resolve_role

        default = state.phase.next
        data = result.data
        if not isinstance(data, NextStep):
            return default

        # Модель называет роль по-русски: «план», «Сбор источников».
        # Реестр — английские идентификаторы, поэтому имя приводится к
        # ним: иначе решение роутера отбрасывалось как «неизвестное».
        chosen = resolve_role(data.role)
        if chosen not in ROLES:
            note = f"маршрут «{data.role}» неизвестен — идём по порядку: {default.title_ru}"
            self._remember_routing(state, data.role, default, accepted=False, note=note)
            return default

        target = _PHASE_BY_ROLE.get(chosen)
        if target is None:
            note = f"маршрут «{data.role}» без фазы — идём по порядку: {default.title_ru}"
            self._remember_routing(state, data.role, default, accepted=False, note=note)
            return default

        order = list(Phase)
        if target is Phase.PLANNING and order.index(state.phase) > order.index(Phase.PLANNING):
            return self._route_to_planning(state, data, default)

        # Маршрут вперёд может перескочить обязательный сбор источников: в
        # сессии 31 роутер назвал «анализ» из фазы маршрутизации, анализ
        # пошёл без материалов и упал. Сбор выполнен, только если его шаг
        # записан в результатах; иначе маршрут исправляется на «сбор».
        skips_collecting = (
            order.index(state.phase) < order.index(Phase.COLLECTING)
            and order.index(target) > order.index(Phase.COLLECTING)
            and Phase.COLLECTING.value not in state.results
        )
        if skips_collecting:
            note = (
                f"маршрут «{data.role}» вперёд через невыполненный сбор — "
                f"исправлено, идём по порядку: {Phase.COLLECTING.title_ru}"
            )
            self._remember_routing(state, data.role, Phase.COLLECTING, accepted=False, note=note)
            return Phase.COLLECTING

        if order.index(target) <= order.index(state.phase):
            note = (
                f"маршрут «{data.role}» ведёт назад — исправлено, "
                f"идём по порядку: {default.title_ru}"
            )
            self._remember_routing(state, data.role, default, accepted=False, note=note)
            return default

        note = f"маршрут: {data.role} — {target.title_ru}. Причина: {data.reason or 'не указана'}"
        self._remember_routing(state, data.role, target, accepted=True, note=note)
        return target

    #: Сколько раз за сессию можно вернуться к плану. Один возврат — это
    #: уточнение по объявленным пробелам; второй — уже хождение по кругу.
    MAX_REPLANS = 1

    def _route_to_planning(
        self,
        state: SessionState,
        data: NextStep,
        default: Phase,
    ) -> Phase:
        """Решить, можно ли вернуться к плану.

        Возврат осмыслен, только если разбор объявил пробелы: тогда план
        уточняется по ним. Без пробелов это повтор уже сделанного, а
        второй возврат — цикл. Планировщику при возврате приходят сами
        пробелы и посчитанные числа (см. ``_progress_brief``), иначе
        уточнять было бы нечем.

        Args:
            state: Состояние сессии.
            data: Решение роутера.
            default: Фаза, которая выполняется при отказе от возврата.

        Returns:
            Фаза плана, если возврат разрешён, иначе обычный порядок.
        """
        analysis = state.results.get(Phase.ANALYZING.value, {})
        gaps = analysis.get("gaps") if isinstance(analysis, dict) else None
        replans = int(state.results.get("replans") or 0)
        if gaps and replans < self.MAX_REPLANS:
            state.results["replans"] = replans + 1
            note = (
                f"маршрут: возврат к плану — разбор объявил {len(gaps)} пробел(ов). "
                f"Причина: {data.reason or 'не указана'}"
            )
            self._remember_routing(state, data.role, Phase.PLANNING, accepted=True, note=note)
            return Phase.PLANNING
        if replans >= self.MAX_REPLANS:
            note = (
                f"маршрут «{data.role}»: повторный возврат к плану запрещён — "
                f"идём по порядку: {default.title_ru}"
            )
        else:
            note = (
                f"маршрут «{data.role}» ведёт назад, но пробелов не объявлено — "
                f"план повторять незачем, идём по порядку: {default.title_ru}"
            )
        self._remember_routing(state, data.role, default, accepted=False, note=note)
        return default

    @staticmethod
    def _remember_routing(
        state: SessionState,
        role: str,
        target: Phase,
        *,
        accepted: bool,
        note: str,
    ) -> None:
        """Запомнить решение роутера — для человека и для журнала.

        Журнал append-only, а решение принимается после записи текущего
        шага: дописать его в уже записанную строку нельзя. Поэтому
        решение кладётся в состояние, и его забирает ближайшая запись —
        та, ради которой решение и принято. Иначе требование
        «исправлять ошибочный выбор» выполнялось бы, но нигде не было
        видно.

        Args:
            state: Состояние сессии.
            role: Роль, которую предложил роутер.
            target: Фаза, которая будет выполнена.
            accepted: Решение принято или исправлено.
            note: Объяснение для человека.
        """
        state.results["routing_note"] = note
        state.results["routing_decision"] = {
            "role": role,
            "target": target.value,
            "accepted": accepted,
            "note": note,
        }

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

        data, usage = self._call_role(
            role_name,
            payload,
            context,
            extras=self._prompt_extras(state),
        )
        # Числа плана и обязательств — тоже утверждения. Раньше они проходили
        # по совпадению основы слова: «удержание на уровне 87 %» не
        # отличалось от «удержание вырастет», хотя 87 — абонентская база.
        # Считаются **до** проверки оснований: статус вывода зависит от
        # неподтверждённых величин, и второй проход дал бы два статуса
        # одного шага (`T-48b`).
        plan_numbers = PlanNumberCheck()
        if phase in (Phase.STRATEGY, Phase.KEEPER):
            plan_numbers = self._check_plan_numbers(data)
        grounding = self._verify_claims(
            data,
            bundle,
            state=state,
            role_name=role_name,
            session_id=state.session_id,
            unsupported_numbers=plan_numbers.unsupported,
        )
        if phase is Phase.ANALYZING:
            # Вердикт гипотезам ставится здесь, до сериализации шага: иначе
            # «гипотеза» останется утверждением без исхода, и отчёт не сможет
            # сказать, проверяли её или забыли.
            self._apply_hypothesis_verdicts(data, grounding)
        lexical = self._lexical_report(data)
        # Критерии успеха проверяются вместе с основаниями: это тоже
        # утверждение о том, что и как будет измерено.
        criteria_notes: list[str] = []
        if phase is Phase.STRATEGY:
            criteria_notes.extend(self._check_success_criteria(data))
            # Ограничения задачи проверяются кодом, а не доверием к промпту:
            # направление, которое их нарушает, — отказ с причиной в журнале
            # и отчёте (`T-06b`).
            constraint_check = self._check_constraints(data, state)
            if constraint_check:
                state.results["constraint_check"] = constraint_check
        criteria_notes.extend(note.text for note in plan_numbers.notes)
        template_notes: list[str] = []
        if phase is Phase.KEEPER:
            # Обязательство без базы или метода — не отказ прогона, а
            # названное замечание: чинится заполнением шаблона, а не
            # остановкой (`T-50`).
            template_notes = self._check_commitment_template(data)
        if bundle is None and phase in (Phase.ANALYZING, Phase.STRATEGY):
            # Роутер может повести вперёд, перескочив сбор источников.
            # Тогда проверять основания нечем, и шаг обязан сказать это
            # прямо: пустой список отказов читается как «всё в порядке».
            criteria_notes.append(
                "источники не собраны: проверять основания нечем — шаг выполнен без материалов"
            )
        # Отказы уходят в журнал вместе с машинным видом. Пока его там не было,
        # отчёт, замер и документ демонстрации называли класс по русскому
        # тексту, и новый отказ (`indicator-demotion`) попадал в «прочее».
        plan_texts = {note.text for note in plan_numbers.notes}
        rejection_items: list[Rejection] = [
            Rejection(
                kind=verdict.kind,
                status=verdict.status.value,
                detail=verdict.detail,
                claim=verdict.claim,
            )
            for verdict in grounding.verdicts
            if verdict.status is not SupportStatus.CONFIRMED
        ]
        rejection_items.extend(
            Rejection(kind=note.kind, status="no_data", detail=note.text)
            for note in plan_numbers.notes
        )
        rejection_items.extend(
            Rejection(kind=classify_detail(text), status="no_data", detail=text)
            for text in criteria_notes
            if text not in plan_texts
        )
        rejection_items.extend(
            Rejection(kind=COMMITMENT_TEMPLATE, status="no_data", detail=text)
            for text in template_notes
        )
        # Ранний пробел (`T-81`): разбор вернул факты, но ни одна ссылка не
        # подтвердилась и ничего не ждёт проверки — значит, релевантных
        # источников по вопросу нет. Пробел называется один раз, а поток
        # отказов сворачивается в одну строку с разбивкой по видам: иначе на
        # вопросе вне данных в журнале и документе стоит 10–22 отказа одного
        # и того же смысла.
        rejected_total = grounding.rejected + len(criteria_notes) + len(template_notes)
        rejected_detail_lines = [*grounding.details, *criteria_notes, *template_notes]
        early_gap_text: str | None = None
        if phase is Phase.ANALYZING and bundle is not None:
            from pf.hypotheses import early_gap

            early_gap_text = early_gap(
                facts=len(getattr(data, "facts", None) or []),
                confirmed=len(grounding.citations),
                needs_review=len(grounding.needs_review),
                user_input=len(grounding.user_input_grounds),
            )
        if early_gap_text:
            data.gaps = [*list(getattr(data, "gaps", None) or []), early_gap_text]
            if rejection_items:
                counts: dict[str, int] = {}
                for item in rejection_items:
                    label = label_of(item.kind) if item.kind else "прочее"
                    counts[label] = counts.get(label, 0) + 1
                breakdown = ", ".join(f"{label} — {count}" for label, count in counts.items())
                summary = f"{early_gap_text} — отклонено {len(rejection_items)} ({breakdown})"
                rejection_items = [
                    Rejection(kind=NO_RELEVANT_SOURCES, status="no_data", detail=summary)
                ]
                rejected_total = 1
                rejected_detail_lines = [summary]
        if grounding.conclusion is not None:
            # Статус оснований вывода — часть того же шага: он попадает
            # в журнал вместе с выводом, а не остаётся в состоянии сессии.
            # Передача вывода (`T-46`) едет тем же путём: получатель сверяет
            # поля утверждения и хеш источника, а не пересказывает прозу.
            if grounding.handoffs:
                grounding.conclusion["transmission"] = grounding.handoffs
            state.results["conclusion_check"] = grounding.conclusion

        result = StepResult(
            role=role_name,
            phase=phase,
            data=data,
            citations=grounding.citations,
            rejected=rejected_total,
            rejected_details=rejected_detail_lines,
            rejections=[item.as_dict() for item in rejection_items],
            needs_review=grounding.needs_review,
            needs_review_details=grounding.needs_review_details,
            user_input_grounds=grounding.user_input_grounds,
            lexical=lexical,
            cost_units=usage.cost_units,
            tokens=usage.total_tokens,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            duration_ms=usage.duration_ms,
            context_limit=usage.context_limit,
        )

        state.results[phase.value] = self._serialize(data)
        state.steps_done += 1
        state.cost_units += usage.cost_units

        # Субагент включается до записи шага: его отчёт — часть того же
        # шага. Он перепроверяет основания с другой стороны: не «найдена
        # ли цитата», а «относится ли источник к задаче». Если писать
        # его после записи, второе мнение останется в состоянии сессии
        # и в поставляемый отчёт не попадёт.
        if phase is Phase.ANALYZING:
            claims = getattr(data, "facts", None) or []
            # Противоречие проверяется до записи шага: факты, говорящие
            # разное об одном показателе, — это пробел, а не два основания.
            contradictions = self._find_contradictions(list(claims))
            if contradictions:
                data.gaps = [*list(getattr(data, "gaps", None) or []), *contradictions]
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
            # Добор — часть того же шага разбора: кандидаты попадают в
            # журнал, а не остаются в состоянии сессии. Ищутся только
            # неизвестные гипотезы и пробелы; противоречия сюда не идут.
            plan = self._plan_collection(state, data, bundle)
            if plan is not None:
                state.results["collection"] = plan.as_detail()

        self._record(state, result)
        self._apply_hooks(state, result)
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
        # Объекты страниц нужны проверке оснований: по ним видно, что
        # цитата из внешнего источника действительно есть на странице.
        # Без этого ссылка на веб отклонялась как «источник не найден» —
        # то есть как выдумка, хотя страница честно скачана.
        state.results["web_pages"] = list(pages)
        state.results["web_source_objects"] = list(sources)
        # Веб-контекст попадает в модель вместе с локальными абзацами:
        # иначе агент выходит в интернет, но не видит, что там нашёл.
        state.results["web_context"] = as_context(sources) + "\n\n" + webfetch.as_context(pages)
        return pages

    def _apply_hooks(self, state: SessionState, result: StepResult) -> None:
        """Применить хуки к результату шага.

        Три хука:

        - ``sources.collected`` — подтверждённых оснований мало, значит
          вывод пока нечем подкрепить: план расширяется, чтобы поиск шёл
          дальше;
        - ``facts.contradicted`` — в разборе есть противоречия: раздел
          «Противоречия» в отчёте называет их число и ось (`T-65`);
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

        if result.phase is Phase.ANALYZING:
            gaps = list(getattr(result.data, "gaps", None) or [])
            count = sum(1 for gap in gaps if "противоречие" in str(gap).lower())
            contradicted = fire("facts.contradicted", float(count))
            if contradicted.fired:
                state.results["contradiction_hook"] = contradicted.detail

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

    #: Сколько записей каждого вида уходит в сводку прогресса. Длинный
    #: вход вытесняет схему ответа: модель возвращает структуру входа
    #: вместо своей. Предел измерен на блоке РАСЧЁТЫ.
    MAX_PROGRESS_ITEMS = 8

    @classmethod
    def _progress_brief(cls, state: SessionState) -> dict[str, Any]:
        """Сводка того, что уже выяснено.

        Планировщик без неё не может уточнить план: он видит только
        постановку задачи и потому либо повторяет её, либо спрашивает
        заново то, на что уже есть ответ. Роутер без неё решает «куда
        дальше» вслепую.

        Подтверждённые основания берутся из журнала, а не из ответа
        модели: подтверждает код, и только он.

        Args:
            state: Состояние сессии.

        Returns:
            Сводка: подтверждённые ссылки, число отказов, пробелы,
            гипотезы и посчитанные кодом метрики.
        """
        records = Journal(journal_path(state.session_id)).read()
        confirmed: list[str] = []
        rejected = 0
        for record in records:
            for citation in record.citations:
                if citation not in confirmed:
                    confirmed.append(citation)
            rejected += len(record.rejected_details)

        analysis = state.results.get(Phase.ANALYZING.value, {})
        if not isinstance(analysis, dict):
            analysis = {}
        gaps = [str(item) for item in (analysis.get("gaps") or [])]
        hypotheses = [
            str(item.get("text", ""))
            for item in (analysis.get("hypotheses") or [])
            if isinstance(item, dict)
        ]
        metrics = [
            {
                "name": metric.name,
                "value": metric.value,
                "unit": metric.unit,
                "citation": metric.citation,
                "formula": metric.formula,
            }
            for metric in cls._computed_metric_records()
        ]
        limit = cls.MAX_PROGRESS_ITEMS
        return {
            "confirmed": confirmed[:limit],
            "rejected": rejected,
            "gaps": gaps[:limit],
            "hypotheses": hypotheses[:limit],
            "metrics": metrics[:limit],
        }

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
            payload: dict[str, Any] = {
                "task": frame,
                "progress": self._progress_brief(state),
            }
            previous = state.results.get(Phase.PLANNING.value, {})
            if previous:
                payload["previous_plan"] = previous
            return "planner", payload, ""

        if phase is Phase.ROUTING:
            frame = state.results.get(Phase.INTAKE.value, {})
            plan = state.results.get(Phase.PLANNING.value, {})
            return (
                "router",
                {
                    "task": frame,
                    "plan": plan,
                    "progress": self._progress_brief(state),
                },
                "",
            )

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
            context = self._materials(bundle, state, self.config.context_passages)
            return "collector", {"query": query}, context

        if phase is Phase.ANALYZING:
            collector = state.results.get(Phase.COLLECTING.value, {})
            context = self._append_reference(
                self._materials(bundle, state, self.config.context_passages)
            )
            return "analyst", {"collected": collector}, context

        if phase is Phase.OPTIONS:
            analysis = state.results.get(Phase.ANALYZING.value, {})
            frame = state.results.get(Phase.INTAKE.value, {})
            return "options", {"task": frame, "analysis": analysis}, self._reference_context()

        if phase is Phase.REDTEAM:
            options = state.results.get(Phase.OPTIONS.value, {})
            analysis = state.results.get(Phase.ANALYZING.value, {})
            return (
                "redteam",
                {"variants": options, "analysis": analysis},
                self._reference_context(),
            )

        if phase is Phase.STRATEGY:
            options = state.results.get(Phase.OPTIONS.value, {})
            redteam = state.results.get(Phase.REDTEAM.value, {})
            analysis = state.results.get(Phase.ANALYZING.value, {})
            frame = state.results.get(Phase.INTAKE.value, {})
            context = self._append_reference(
                self._materials(bundle, state, self.config.context_passages)
            )
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
            # Основание обязательства — ссылка на материал. Список доступных
            # источников нужен и здесь: без него keeper называет source_id,
            # которого не видел ни он, ни контекст стратегии (`T-77`).
            context = bundle.sources_header() if bundle is not None else ""
            return (
                "keeper",
                {"strategy": strategy, "redteam": redteam},
                self._append_reference(context),
            )

        raise ValueError(f"для фазы «{phase.value}» нет роли")

    #: Сколько абзацев источников уходит в контекст модели. Ограничение
    #: существенно: на длинном контексте локальная модель малого размера
    #: начинает терять формат ответа и возвращать «историю диалога» вместо
    #: JSON. Пять абзацев — это и качество, и управляемость.
    CONTEXT_PASSAGES = 5

    @classmethod
    def _bundle_context(cls, bundle: SourceBundle | None, limit: int | None = None) -> str:
        """Собрать контекст из источников.

        Список доступных источников идёт перед абзацами: модель должна
        знать не только то, что попало в окно, но и то, на что вообще можно
        сослаться. Ссылка на источник вне списка — пробел в сборе, а не
        основание, и проверка оснований это подтверждает.

        Args:
            bundle: Локальные источники.
            limit: Сколько абзацев взять. ``None`` — константа
                ``CONTEXT_PASSAGES``; рабочий процесс передаёт бюджет из
                настроек (``PF_CONTEXT_PASSAGES``), чтобы его можно было
                замерить, а не менять правкой кода.

        Returns:
            Текст материалов либо пустая строка.
        """
        if bundle is None:
            return ""
        used = cls.CONTEXT_PASSAGES if limit is None else limit
        return bundle.context(limit=used, include_sources=True)

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

    @classmethod
    def _reference_context(cls) -> str:
        """Блок «СПРАВОЧНИК»: знание о применимости, отдельное от оснований.

        Справочник не пополняет подтверждённые основания: он отвечает на
        вопрос «как это обычно бывает», а не «что было у нас». Поэтому блок
        помечен «не факт о кейсе» и идёт отдельно от материалов (`T-85`).

        Returns:
            Текст блока либо пустая строка, если справочника нет.
        """
        from pf import reference

        lines = reference.as_lines()
        if not lines:
            return ""
        return "\n".join([REFERENCE_RULE, *lines])

    @classmethod
    def _append_reference(cls, context: str) -> str:
        """Дописать справочник к контексту роли.

        Args:
            context: Контекст роли (материалы или пустая строка).

        Returns:
            Контекст со справочником либо без изменений.
        """
        block = cls._reference_context()
        if not block:
            return context
        return f"{context}\n\n{block}" if context else block

    @staticmethod
    def _computed_metric_records() -> list[Metric]:
        """Готовые расчёты по материалам проекта — как данные.

        Считать должен код, а не модель: числа из таблиц — это то, что
        можно пересчитать и проверить. Записи нужны и блоку РАСЧЁТЫ для
        контекста, и плану: планировщик получает числа, чтобы не
        спрашивать заново о том, что уже измерено.

        Returns:
            Список метрик с provenance либо пустой список.
        """
        from pf.analysis import (
            PERIOD_COLUMNS,
            compute_from_rows,
            group_series,
            label_column_for,
            summarize_groups,
        )
        from pf.materials import load_materials

        records: list[Metric] = []
        for material in load_materials():
            if not material.rows:
                continue
            columns = list(material.rows[0].keys())
            value_column = next(
                (name for name in columns if "знач" in name.lower() or "value" in name.lower()),
                None,
            )
            if value_column is None:
                continue
            # Подпись ряда — «что за величина», а не «когда». Колонка с
            # периодом подписью быть не может: тогда каждый период
            # образовал бы свой ряд из одной строки, и изменение всегда
            # было бы нулевым.
            label_column = label_column_for(columns, value_column=value_column)
            metrics = compute_from_rows(
                material.rows,
                value_column=value_column,
                source_id=material.source_id,
                name=material.title,
                label_column=label_column or "",
            )
            if not metrics:
                continue
            records.extend(metrics[:MAX_METRICS_IN_CONTEXT])
            # Изменение считается по рядам: разные операторы и разные
            # годы нельзя складывать в один ряд, иначе получится
            # бессмысленная величина.
            # Ряд — это одна и та же величина во времени: тот же
            # оператор И тот же показатель. Иначе в ряд попадает
            # абонентская база рядом с долей на рынке.
            # Период — ось времени внутри ряда, а не признак ряда:
            # «2017», «2018» — это точки одной величины. А вот
            # показатель — признак: база и доля на рынке разные.
            group_columns = tuple(
                name
                for name in columns
                if name not in {value_column, label_column}
                and name.strip().lower() not in PERIOD_COLUMNS
            )
            groups = group_series(
                material.rows,
                label_column=label_column or "",
                group_columns=group_columns,
            )
            summary = summarize_groups(
                groups,
                value_column=value_column,
                source_id=material.source_id,
                name=material.title,
                unit=metrics[0].unit,
            )
            records.extend(summary[:MAX_METRICS_IN_CONTEXT])
        return records

    @classmethod
    def _computed_metrics(cls, state: SessionState) -> str:
        """Блок РАСЧЁТЫ для контекста модели.

        Args:
            state: Состояние сессии. Расчёты берутся из материалов, но
                подпись единая для всех фаз — поэтому параметр сохранён.

        Returns:
            Текст блока с расчётами либо пустая строка.
        """
        blocks: list[str] = []
        source_id = ""
        lines: list[str] = []
        for metric in cls._computed_metric_records():
            if metric.source_id != source_id:
                if source_id:
                    blocks.append(_calculations_block(source_id, lines))
                source_id = metric.source_id
                lines = []
            lines.append(_metric_line(metric))
        if source_id:
            blocks.append(_calculations_block(source_id, lines))
        return "\n\n".join(blocks)

    @classmethod
    def _materials(
        cls,
        bundle: SourceBundle | None,
        state: SessionState,
        limit: int | None = None,
    ) -> str:
        """Собрать все материалы для роли: локальные и внешние.

        Разделены намеренно: подтверждённые основания приходят из
        локального среза, внешние материалы требуют проверки человеком.
        Если смешать их в один список, агент перестанет различать, где
        правда, а где гипотеза.

        Args:
            bundle: Локальные источники.
            state: Состояние сессии.
            limit: Бюджет абзацев на роль. ``None`` — константа
                ``CONTEXT_PASSAGES``.

        Returns:
            Текст материалов.
        """
        local = cls._bundle_context(bundle, limit)
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

    @classmethod
    def human_question_to_stop(cls, state: SessionState, result: StepResult) -> str:
        """Вопрос, из-за которого работа останавливается.

        Человек подключается там, где без его решения работа потеряла бы
        смысл. Но повтор уже заданного вопроса работой не двигает: в
        прогоне `demo-v11` роутер спросил одно и то же дважды — человек
        ответил, `pf resume` повторил маршрутизацию, модель повторила
        вопрос, и сессия встала навсегда: ни шага вперёд, ни нового
        вопроса, хотя ответ у агента уже был.

        Args:
            state: Состояние сессии.
            result: Результат шага.

        Returns:
            Текст вопроса, если работа останавливается, иначе пустая строка.
            Повтор снимается **только** тогда, когда ответ уже записан:
            вопрос, которого человек не видел, пропускать нельзя.
        """
        if not isinstance(result.data, NextStep) or not result.data.need_human:
            return ""
        question = (result.data.question or "Нужно ваше решение, чтобы продолжить.").strip()
        if question and cls._question_is_answered(state, question):
            state.results["routing_note"] = (
                "роутер повторил уже заданный вопрос: ответ записан в сессии, "
                "работа продолжена по порядку фаз"
            )
            return ""
        return question

    @classmethod
    def _question_is_answered(cls, state: SessionState, question: str) -> bool:
        """Вопрос уже задан и на него ответили.

        Args:
            state: Состояние сессии.
            question: Вопрос роутера.

        Returns:
            ``True``, если такой вопрос задавали и человек дал ответ.
        """
        if not state.results.get("human_answers"):
            return False
        asked = {
            cls._normalize_question(item) for item in state.results.get("human_questions") or []
        }
        return cls._normalize_question(question) in asked

    @staticmethod
    def _normalize_question(text: str) -> str:
        """Вопрос без различий в регистре и пробелах.

        Args:
            text: Текст вопроса.

        Returns:
            Приведённый текст для сравнения.
        """
        return " ".join(str(text).lower().split())

    @staticmethod
    def _human_context(state: SessionState) -> str:
        """Собрать ответы пользователя для роли.

        Ответ, который человек дал на вопрос агента, должен попасть в
        контекст следующего шага. Иначе агент спрашивает, человек
        отвечает — и ответ теряется: работа продолжается вслепую.

        Подпись называет и стадию, на которой ответ дан, и ключ, которым на
        него ссылаются. Прежняя формулировка «на него нельзя ссылаться»
        запрещала ссылку вместо того, чтобы научить правильной: слова
        человека — законная опора, у них просто другой канал подтверждения
        (`D-49`).

        Args:
            state: Состояние сессии.

        Returns:
            Текст ответов либо пустая строка.
        """
        records = human_inputs(state)
        if not records:
            return ""
        lines: list[str] = []
        for record in records:
            stage = phase_of(record.phase)
            if stage is None:
                key = "ввод пользователя"
                where = "стадия не записана"
            else:
                key = f"ввод пользователя: {stage.value}"
                where = f"стадия «{stage.title_ru}»"
            lines.append(
                f"Ввод пользователя ({where}) — ссылка «{key}», "
                f"канал основания — «ввод пользователя». Это не документ: "
                f"подтверждается только тем, что пользователь это сказал: {record.text}"
            )
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
        detail = self._serialize(result.data)
        decision = state.results.pop("routing_decision", None)
        if isinstance(decision, dict):
            detail["routing"] = decision
        if result.phase is Phase.ANALYZING:
            fact_check = state.results.get("fact_check")
            if isinstance(fact_check, dict):
                detail["fact_check"] = fact_check
            # Добор оснований — часть шага разбора. Забирается из состояния,
            # а не пересчитывается: журнал — след шага, а не второй прогон.
            collection = state.results.pop("collection", None)
            if isinstance(collection, dict):
                detail["collection"] = collection
        if result.phase in (Phase.STRATEGY, Phase.KEEPER):
            # Статус оснований вывода пишется вместе с выводом: иначе он
            # останется в состоянии сессии и в поставляемый отчёт не попадёт.
            conclusion = state.results.pop("conclusion_check", None)
            if isinstance(conclusion, dict):
                detail["conclusion_check"] = conclusion
            # Нарушения ограничений задачи — тоже часть шага: отчёт называет
            # их отдельной строкой, а не выводит из прозы (`T-06b`).
            constraint_check = state.results.pop("constraint_check", None)
            if isinstance(constraint_check, dict):
                detail["constraints"] = constraint_check
        if result.lexical:
            # Покрытие — часть шага: по нему видно, что именно расширять
            # в словаре, и виден эффект от пополнения.
            detail["lexical"] = result.lexical
        if result.context_limit:
            # Окно и признак обрезки — в detail, а не отдельным полем записи:
            # старые журналы остаются читаемыми, а вопрос «видела ли модель
            # весь контекст» получает ответ в том же месте, где и остальные
            # наблюдения шага.
            detail["context"] = {
                "limit": result.context_limit,
                "prompt_tokens": result.prompt_tokens,
                "truncated": result.prompt_truncated,
            }
        journal = Journal(journal_path(state.session_id))
        record = StepRecord(
            ordinal=state.steps_done,
            role=result.role,
            phase=result.phase.value,
            summary=self._summarize(result),
            detail=detail,
            citations=result.citations,
            rejected_details=result.rejected_details,
            rejections=result.rejections,
            needs_review_details=result.needs_review_details,
            user_input_grounds=result.user_input_grounds,
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
        # Расчёты агента передаются вместе с источниками: агрегат (среднее,
        # изменение) в основании вывода подтверждается пересчётом по ним, а
        # не поиском величины в тексте материала, где её нет и быть не может.
        return collect(query, metrics=tuple(Workflow._computed_metric_records()))

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
        # подключает человека при существенном выборе. Повтор уже
        # заданного вопроса работой не двигает — ответ у агента есть.
        question = workflow.human_question_to_stop(state, result)
        if question:
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
