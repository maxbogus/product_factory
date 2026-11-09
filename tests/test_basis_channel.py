"""Канал основания: откуда взято утверждение и на какой стадии это сказано.

Ввод пользователя — не документ, но и не пустая ссылка. Это **опора**: её
можно назвать (канал + стадия + слова), а решение «верить или нет» принимает
человек. Раньше подпись «Ответ человека» попадала в `source_id`, проверка
отвечала «не источник — ввод человека, а не документ» и выбрасывала
основание: 13 записей на трёх сохранённых прогонах (`D-45`). Подпись
«на него нельзя ссылаться» лечила симптом — она не давала сослаться, вместо
того чтобы дать сослаться правильно (`D-49`).

Здесь проверяется четыре свойства канала:

1. канал и стадия живут в ссылке (`basis_kind`, `basis_phase`), а не
   угадываются по формулировке;
2. ответ пользователя записывается **со стадией**, на которой он дан;
3. ссылка на ввод проверяется: был ли такой ввод, на той ли стадии, те ли
   слова — иначе это ссылка на слова, которых человек не говорил;
4. принятая опора не считается подтверждённым фактом и не поднимает статус
   вывода: «верить или нет» — решение человека.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pf.ground import (
    CONCLUSION_RULE,
    SupportStatus,
    conclusion_status,
)
from pf.schemas import (
    Analysis,
    Claim,
    CommunicationStrategy,
    DecisionRecord,
    ProductStrategy,
    SourceRef,
    Strategy,
    VerificationPlan,
)
from pf.sources import SourceBundle, human_input_stage
from pf.state import (
    Phase,
    SessionState,
    human_inputs,
    phase_of,
    record_human_input,
)
from pf.workflow import Workflow

#: Что пользователь сказал на стадии разбора.
ANSWER = "Внутренней телеметрии нет — работаем только на открытых источниках"

#: Ссылка на этот ввод в том виде, в каком её называет контекст роли.
CITATION = f"ввод пользователя: {Phase.ANALYZING.value}"


def _bundle() -> SourceBundle:
    """Пустой набор источников: проверять документы в этих тестах нечего.

    Returns:
        Набор без материалов.
    """
    return SourceBundle(passages=[], articles=[], query="почему падает удержание")


def _state(answer: str = ANSWER, phase: Phase = Phase.ANALYZING) -> SessionState:
    """Сессия с одним записанным ответом пользователя.

    Args:
        answer: Что сказал пользователь.
        phase: Стадия, на которой он это сказал.

    Returns:
        Состояние сессии.
    """
    state = SessionState(session_id="basis-channel")
    state.phase = phase
    record_human_input(state, answer)
    return state


def _analysis(source: SourceRef) -> Analysis:
    """Разбор с одним фактом, опирающимся на заданную ссылку.

    Args:
        source: Основание факта.

    Returns:
        Разбор с одним фактом.
    """
    return Analysis(facts=[Claim(kind="факт", text="Внутренней телеметрии нет", sources=[source])])


def _strategy(source: SourceRef) -> Strategy:
    """Стратегия, у которой единственное основание — заданная ссылка.

    Args:
        source: Основание вывода.

    Returns:
        Заполненная стратегия.
    """
    return Strategy(
        product=ProductStrategy(
            audience="клиенты мобильного тарифа",
            chosen_direction="снизить цену на тариф",
            evidence=[source],
        ),
        communication=CommunicationStrategy(
            audience="клиенты мобильного тарифа",
            barrier="цена выше, чем у альтернатив",
            key_message="мы держим цену под контролем",
        ),
        verification=VerificationPlan(
            expected_change="удержание вырастет на 2 п.п.",
            compare_against="уровень прошлого квартала",
            observation_period="3 месяца",
            adverse_effects=["маржа снизится"],
        ),
    )


# ── 1. Канал и стадия в ссылке ──────────────────────────────────────────────


def test_basis_kind_is_declared_and_defaults_to_fact() -> None:
    """Канал основания назван в ссылке; по умолчанию — факт.

    Умолчание «факт» обязательно: в сохранённых ответах ролей поля нет, и
    ответ без него должен читаться как раньше (`roles.validate`).
    """
    assert SourceRef().basis_kind == "факт"
    assert SourceRef(basis_kind="ввод пользователя").basis_kind == "ввод пользователя"
    assert SourceRef(basis_kind="ввод пользователя", basis_phase="анализ").basis_phase == "анализ"
    with pytest.raises(ValidationError):
        SourceRef(basis_kind="слух")


def test_basis_kind_synonym_is_normalized() -> None:
    """Модель называет канал по-своему — схема принимает те же понятия."""
    assert SourceRef(basis_kind="пользовательский ввод").basis_kind == "ввод пользователя"
    assert SourceRef(basis_kind="Ответ человека").basis_kind == "ввод пользователя"
    assert SourceRef(basis_kind="допущение").basis_kind == "гипотеза"


def test_old_reference_without_channel_parses() -> None:
    """Сохранённый ответ роли без поля канала читается как факт."""
    claim = Claim.model_validate(
        {"kind": "факт", "text": "абонентская база 87 млн", "sources": [{"source_id": "a.csv"}]}
    )
    assert claim.sources[0].basis_kind == "факт"
    assert claim.sources[0].basis_phase == ""


# ── 2. Запись ответа пользователя со стадией ────────────────────────────────


def test_human_input_record_keeps_stage() -> None:
    """Ответ записывается вместе со стадией, на которой он дан.

    Стадия — не оформление: по ней ссылка проверяется, а отчёт говорит, где
    именно пользователь вмешался.
    """
    state = SessionState(session_id="stage")
    state.phase = Phase.ANALYZING
    state.pending_question = "Есть ли внутренние данные по оттоку?"
    record_human_input(state, ANSWER)

    [record] = human_inputs(state)
    assert record.text == ANSWER
    assert record.phase == Phase.ANALYZING.value
    assert record.question == "Есть ли внутренние данные по оттоку?"
    assert record.at


def test_old_plain_string_answers_are_read() -> None:
    """Старые состояния хранят голые строки — они читаются без стадии.

    Стадию не выдумываем: у записи до правки её просто нет.
    """
    state = SessionState(session_id="old")
    state.results["human_answers"] = ["цену снижать нельзя"]

    [record] = human_inputs(state)
    assert record.text == "цену снижать нельзя"
    assert record.phase == ""


def test_phase_of_resolves_names() -> None:
    """Стадию называют по-русски, по-английски и заголовком фазы."""
    assert phase_of("разбор") is Phase.ANALYZING
    assert phase_of("Анализ") is Phase.ANALYZING
    assert phase_of("analyzing") is Phase.ANALYZING
    assert phase_of("Разбор источников") is Phase.ANALYZING
    assert phase_of("фиксация решения") is Phase.KEEPER
    assert phase_of("") is None
    assert phase_of("двенадцатая луна") is None


def test_human_input_citation_names_stage() -> None:
    """Ссылка на ввод узнаётся по префиксу и называет стадию."""
    assert human_input_stage(CITATION) == Phase.ANALYZING.value
    assert human_input_stage("Ответ человека") == ""
    assert human_input_stage("project/data/numbers.csv#row0") is None
    assert human_input_stage("") is None


def test_human_input_citation_reaches_role_context() -> None:
    """Контекст роли называет и стадию, и ключ, которым на ввод ссылаются.

    Прежняя подпись «на него нельзя ссылаться» неверна: сослаться можно —
    нельзя выдавать ввод за документ.
    """
    workflow = Workflow()
    state = _state()
    _role, _payload, context = workflow._step_inputs(state, _bundle())

    assert Phase.ANALYZING.title_ru in context
    assert CITATION in context


# ── 3. Проверка ссылки на ввод ──────────────────────────────────────────────


def test_user_input_ground_is_accepted_not_rejected() -> None:
    """Ссылка на записанный ввод — опора, а не отказ.

    Слова пользователя подтверждаются тем, что он их сказал: дословность
    проверяется по записи, а не по документу.
    """
    state = _state()
    grounding = Workflow._verify_claims(
        _analysis(SourceRef(source_id=CITATION, quote=ANSWER, basis_kind="ввод пользователя")),
        _bundle(),
        state=state,
    )

    assert grounding.rejected == 0
    assert grounding.details == []
    assert [verdict.kind for verdict in grounding.verdicts] == ["user_input"]
    assert grounding.verdicts[0].status is SupportStatus.NEEDS_REVIEW
    assert grounding.user_input
    assert Phase.ANALYZING.title_ru in grounding.user_input[0]


def test_human_input_allows_reference_without_stage_when_input_is_single() -> None:
    """Один ввод в сессии — стадия неоднозначна, и это сказано.

    Сохранённые прогоны ссылались на «Ответ человека» без стадии; отказывать
    им только за это значило бы выбросить законную опору.
    """
    state = _state()
    grounding = Workflow._verify_claims(
        _analysis(SourceRef(source_id="Ответ человека", quote=ANSWER)),
        _bundle(),
        state=state,
    )

    assert grounding.rejected == 0
    assert grounding.verdicts[0].kind == "user_input"
    assert "единственный" in grounding.user_input[0]


def test_unrecorded_human_input_is_rejected() -> None:
    """Ссылка на ввод, которого не было, — отказ, а не опора."""
    state = SessionState(session_id="empty")
    grounding = Workflow._verify_claims(
        _analysis(SourceRef(source_id=CITATION, quote=ANSWER)),
        _bundle(),
        state=state,
    )

    assert grounding.rejected == 1
    assert grounding.user_input == []
    assert grounding.verdicts[0].status is SupportStatus.NO_DATA
    assert grounding.verdicts[0].kind == "human_input_unrecorded"


def test_wrong_stage_is_rejected() -> None:
    """Ввод был на другой стадии — стадия названа не та."""
    state = _state()
    grounding = Workflow._verify_claims(
        _analysis(
            SourceRef(
                source_id=f"ввод пользователя: {Phase.STRATEGY.value}",
                quote=ANSWER,
            )
        ),
        _bundle(),
        state=state,
    )

    assert grounding.rejected == 1
    assert grounding.verdicts[0].kind == "human_input_phase_mismatch"


def test_paraphrased_input_is_rejected() -> None:
    """Слова, которых пользователь не говорил, — не его ввод."""
    state = _state()
    grounding = Workflow._verify_claims(
        _analysis(
            SourceRef(
                source_id=CITATION,
                quote="Пользователь разрешил снизить цену на 15 процентов",
            )
        ),
        _bundle(),
        state=state,
    )

    assert grounding.rejected == 1
    assert grounding.verdicts[0].kind == "human_input_quote_mismatch"


# ── 4. Доверие — решение человека, а не проверки ────────────────────────────


def test_conclusion_status_user_input_only() -> None:
    """Вывод на одном вводе пользователя: опора есть, документа нет."""
    assert (
        conclusion_status(1, 0, 0, needs_review=1, user_input=1) == "опирается на ввод пользователя"
    )
    # Ввод пользователя и подтверждённый документ — это «частично».
    assert conclusion_status(2, 1, 0, needs_review=1, user_input=1) == "подкреплено частично"


def test_human_input_does_not_raise_status() -> None:
    """Ввод пользователя не даёт «подкреплено»: это решение человека.

    Проверка не имеет права объявить слова пользователя подтверждённым
    фактом — иначе доверие к ним выставил бы код.
    """
    state = _state()
    grounding = Workflow._verify_claims(
        _strategy(SourceRef(source_id=CITATION, quote=ANSWER, basis_kind="ввод пользователя")),
        _bundle(),
        state=state,
    )

    assert grounding.conclusion is not None
    assert grounding.conclusion["status"] == "опирается на ввод пользователя"
    assert grounding.conclusion["user_input"] == 1
    assert grounding.conclusion["confirmed"] == 0


def test_rule_name_names_user_input_channel() -> None:
    """Имя правила счёта включает канал ввода пользователя.

    Меняется счёт оснований — меняется имя: иначе отчёт не отличит запись
    прогона от текущей оценки (`D-44`).
    """
    assert "user-input-channel" in CONCLUSION_RULE


def test_decision_keeps_user_input_basis() -> None:
    """Обязательство опирается на ввод пользователя так же, как стратегия."""
    state = _state()
    decision = DecisionRecord(
        decision="снизить цену на тариф",
        commitments=[
            {
                "statement": "удержание вырастет на 2 п.п.",
                "owner": "продукт",
                "metric": "удержание",
                "review_condition": "через квартал",
                "basis": [SourceRef(source_id=CITATION, quote=ANSWER).model_dump()],
            }
        ],
    )

    grounding = Workflow._verify_claims(decision, _bundle(), state=state)

    assert grounding.rejected == 0
    assert grounding.user_input
