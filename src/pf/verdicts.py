"""Виды отказа: подпись для человека и владелец исправления.

Раньше класс отказа называли три места по-своему: отчёт разбирал русский текст
причины (`report.rejection_kind`), замер — статус и подстроку
(`scripts/replay_verification.py`), а журнал хранил только текст. Машинный вид
при этом существовал (`Support.kind`, `ABSENCE_KINDS`, `PlanNumberNote.kind`),
но терялся при записи шага. Из-за этого новый отказ `indicator-demotion` попал в
«прочее», а «прочее» не отвечает на вопрос «что чинить».

Здесь одна таблица: вид → подпись и владелец. Разбор текста остался только для
записей, сделанных до появления вида: журнал append-only, и переписать старые
формулировки нельзя. Записанный вид всегда сильнее разбора текста.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Вид, которого нет в таблице. Называется прямо, а не «прочее»: так видно,
#: что таблицу надо дополнить, а не что причина неизвестна природе.
UNKNOWN_KIND = "unclassified"

#: Подпись неизвестной причины. Осталась прежней: по ней читают старые отчёты.
FALLBACK_LABEL = "прочее"

#: Кто чинит отказ. Один вид — один владелец: иначе строка «что чинить»
#: превращается в перечисление всего.
FIX_PROMPT = "промпт"
FIX_CHECK = "проверка"
FIX_SCHEMA = "схема"
FIX_CONTEXT = "контекст"
FIX_CALC = "расчёт"

#: Виды отказа, у которых есть машинное имя в коде.
INDICATOR_DEMOTION = "indicator-demotion"
CALC_LINE_DEMOTION = "calc-line-demotion"
UNIQUE_GROUNDS = "unique-grounds"

#: Виды, которыми проверка **подтверждает** основание. Они не отказы, и
#: подписи «что чинить» у них нет: чинить нечего.
CONFIRMING_KINDS: frozenset[str] = frozenset({"exact", "number_tolerance"})


@dataclass(frozen=True, slots=True)
class Kind:
    """Как вид отказа называется для человека и кто его чинит.

    Attributes:
        label: Подпись в отчёте — по ней считают класс.
        fix: Владелец исправления (промпт, схема, контекст, расчёт, проверка).
    """

    label: str
    fix: str


#: Таблица видов: единственное место, где вид получает подпись.
#:
#: Ключи совпадают с машинными именами в коде: `Support.kind` (`ground.py`),
#: `ABSENCE_KINDS` (`sources.py`), `PlanNumberNote.kind` (`workflow.py`),
#: части `CONCLUSION_RULE_PARTS` (`ground.py`). Новый вид без записи здесь
#: роняет `tests/test_verdict_kinds.py`.
KINDS: dict[str, Kind] = {
    # Проверка цитаты (`ground.Support.kind`).
    "too_short": Kind("цитата короче минимума", FIX_PROMPT),
    "paraphrase": Kind("пересказ вместо цитаты", FIX_CHECK),
    "fabrication": Kind("цитата не найдена", FIX_PROMPT),
    "number_mismatch": Kind("значение не сошлось с источником", FIX_CHECK),
    "language": Kind("ответ не на языке задачи", FIX_PROMPT),
    # Ссылка не годится как основание (`sources.ABSENCE_KINDS`).
    "source_not_named": Kind("фантомный источник", FIX_SCHEMA),
    "source_unknown": Kind("фантомный источник", FIX_PROMPT),
    "human_input": Kind("ссылка на ввод человека", FIX_SCHEMA),
    "outside_slice": Kind("источник нерелевантен", FIX_CONTEXT),
    # Канал «ввод пользователя»: опора, а не отказ. Ссылка проверяется тем,
    # что человек это сказал, на той ли стадии и теми ли словами (`D-49`).
    "user_input": Kind("опора на ввод пользователя", FIX_PROMPT),
    "human_input_unrecorded": Kind("ссылка на ввод, которого не было", FIX_PROMPT),
    "human_input_phase_mismatch": Kind("стадия ввода названа не та", FIX_PROMPT),
    "human_input_quote_mismatch": Kind("ввод пересказан, а не приведён", FIX_CHECK),
    # Числа плана и обязательств (`workflow.NUMBER_*`).
    "foreign_indicator": Kind("число из чужого показателя", FIX_CHECK),
    "no_calculation": Kind("число без своего расчёта", FIX_CALC),
    # Понижение основания до отклонённого.
    INDICATOR_DEMOTION: Kind("основание про другой показатель", FIX_PROMPT),
    CALC_LINE_DEMOTION: Kind("значение посчитано как цитата", FIX_PROMPT),
    UNIQUE_GROUNDS: Kind("повтор ссылки не основание", FIX_CHECK),
    # Отказы, которые знает только разбор текста: вид появился после записи.
    "term_missing": Kind("термин не найден в источнике", FIX_CHECK),
    "criterion_indicator": Kind("критерий не про показатель", FIX_PROMPT),
    "sources_not_collected": Kind("источники не собраны", FIX_CONTEXT),
    "metric_name_column": Kind("имя метрики не совпало с колонкой", FIX_PROMPT),
    "calc_identity": Kind("формула-тождество", FIX_CALC),
    "no_provenance": Kind("у расчёта нет provenance", FIX_SCHEMA),
    "quote_not_in_row": Kind("цитата не найдена в строке", FIX_PROMPT),
}


#: Разбор текста причины для записей без машинного вида. Пары
#: «обязательные подстроки → вид»; проверяются по порядку, поэтому частное
#: стоит раньше общего: «источник не назван … не относится к задаче» — это
#: фантомный источник, а не нерелевантный.
_DETAIL_MARKERS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("посчитано, а не процитировано",), CALC_LINE_DEMOTION),
    (("говорит о другом",), INDICATOR_DEMOTION),
    (("ввод человека",), "human_input"),
    (("не назван",), "source_not_named"),
    (("не найден среди материалов",), "source_not_named"),
    (("цитировать нечего",), "source_not_named"),
    (("вне среза задачи",), "outside_slice"),
    (("не относится к задаче",), "outside_slice"),
    (("термин", "не найден в источнике"), "term_missing"),
    (("цитата короче",), "too_short"),
    (("не совпало с колонкой",), "metric_name_column"),
    (("формула — тождество",), "calc_identity"),
    (("критерий не относится к показателю",), "criterion_indicator"),
    (("источники не собраны",), "sources_not_collected"),
    (("не имеет структуры",), "no_provenance"),
    (("не совпадает со строкой",), "number_mismatch"),
    (("не совпадает с источником",), "number_mismatch"),
    (("не совпадает с расчётом по показателю",), "number_mismatch"),
    (("пересчёт по входам даёт",), "number_mismatch"),
    (("взято из расчёта",), "foreign_indicator"),
    (("ничем не подтверждено",), "no_calculation"),
    (("не найдена в строке",), "quote_not_in_row"),
    (("цитата не найдена",), "fabrication"),
)


def classify_detail(detail: str) -> str:
    """Назвать вид отказа по тексту причины.

    Нужен только записям, сделанным до появления машинного вида: журнал
    append-only, и старые шаги хранят фразу, а не вид.

    Args:
        detail: Строка отклонения из журнала.

    Returns:
        Машинный вид из :data:`KINDS` либо :data:`UNKNOWN_KIND`.
    """
    text = detail.lower()
    for markers, kind in _DETAIL_MARKERS:
        if all(marker in text for marker in markers):
            return kind
    return UNKNOWN_KIND


def label_of(kind: str) -> str:
    """Подпись вида для отчёта.

    Args:
        kind: Машинный вид отказа.

    Returns:
        Подпись из таблицы; для неизвестного вида — :data:`FALLBACK_LABEL`.
    """
    found = KINDS.get(kind)
    return found.label if found is not None else FALLBACK_LABEL


def fix_of(kind: str) -> str:
    """Кто чинит отказ этого вида.

    Args:
        kind: Машинный вид отказа.

    Returns:
        Владелец исправления; для неизвестного вида — «неизвестно».
    """
    found = KINDS.get(kind)
    return found.fix if found is not None else "неизвестно"


@dataclass(frozen=True, slots=True)
class Rejection:
    """Один отказ шага: вид, статус и текст для человека.

    Attributes:
        kind: Машинный вид отказа (ключ :data:`KINDS`) либо пустая строка,
            если проверка вида не назвала.
        status: Статус основания (``no_data``, ``needs_review``).
        detail: Строка для человека — что именно не сошлось.
        claim: Утверждение, к которому относится отказ (пусто у чисел плана).
    """

    kind: str
    status: str
    detail: str
    claim: str = ""

    def as_dict(self) -> dict[str, str]:
        """Представить отказ словарём — так он попадает в журнал.

        Returns:
            Словарь с ключами ``kind``, ``status``, ``detail``, ``claim``.
        """
        return {
            "kind": self.kind,
            "status": self.status,
            "detail": self.detail,
            "claim": self.claim,
        }
