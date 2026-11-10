"""Проверки вердикта гипотез.

Бриф требует «провести настоящее исследование и проверить гипотезы». Гипотезы
формулировались и всё: читатель видел «гипотеза: отток из-за цены» и не мог
понять, проверяли её или забыли.

Три правила, ради которых всё написано:

1. **Вердикт ставит код, а не модель.** Модель формулирует гипотезу; сказать
   «подтверждено» — не её право.
2. **`unknown` — результат, а не провал.** Гипотеза без проверки не считается
   подтверждённой и не прячется.
3. **Опровержение требует противоположного направления.** Молчание о
   направлении — не противоречие.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pf.journal import Journal, StepRecord, journal_path


def test_verdict_defaults_to_unknown() -> None:
    """Гипотеза без проверки — `unknown`, а не «подтверждено»."""
    from pf.hypotheses import Verdict
    from pf.schemas import Claim

    claim = Claim(kind="гипотеза", text="Отток растёт из-за роста цен на тарифы")
    assert claim.verdict is Verdict.UNKNOWN
    assert claim.verdict_basis == ""


def test_verdict_from_model_is_ignored() -> None:
    """Модель не может объявить гипотезу подтверждённой.

    Маркер вердикта — это утверждение модели о своей работе, а не свойство
    оснований: в сохранённых прогонах такие утверждения уже расходились с
    фактами (см. `docs/decisions.md`, D-23).
    """
    from pf.hypotheses import Verdict
    from pf.schemas import Claim

    claim = Claim(
        kind="гипотеза",
        text="Отток растёт из-за роста цен на тарифы",
        verdict="supported",
        verdict_basis="модель так решила",
    )
    assert claim.verdict is Verdict.UNKNOWN
    assert claim.verdict_basis == ""


def test_verdict_supported_needs_grounded_basis() -> None:
    """`supported` — только когда предмет гипотезы подтверждён дословно."""
    from pf.hypotheses import Verdict, verdict_of

    hypothesis = "Отток абонентов растёт из-за цен на тарифы"
    supported = verdict_of(hypothesis, ["Отток абонентов растёт третий квартал подряд"])
    assert supported.verdict is Verdict.SUPPORTED
    assert "Отток абонентов растёт" in supported.basis

    unknown = verdict_of(hypothesis, ["Выручка оператора выросла на 8 %"])
    assert unknown.verdict is Verdict.UNKNOWN
    assert unknown.basis == ""


def test_verdict_refuted_on_opposite_polarity() -> None:
    """Обратное направление по тому же предмету — `refuted`."""
    from pf.hypotheses import Verdict, verdict_of

    result = verdict_of(
        "Удержание абонентов падает",
        ["Удержание абонентов выросло до 87 % за квартал"],
    )
    assert result.verdict is Verdict.REFUTED
    assert result.basis


def test_verdict_without_direction_is_not_refuted() -> None:
    """Гипотеза без направления не опровергается ростом показателя.

    «Отток из-за цен» не утверждает, куда отток идёт: рост или падение
    показателя её не опровергает, а подтверждает предмет.
    """
    from pf.hypotheses import Verdict, verdict_of

    result = verdict_of(
        "Отток абонентов из-за цен на тарифы",
        ["Отток абонентов снизился на 2 п.п. за квартал"],
    )
    assert result.verdict is Verdict.SUPPORTED


def test_verdict_ignores_unconfirmed_grounds() -> None:
    """Неподтверждённое основание вердикта не даёт.

    `_apply_hypothesis_verdicts` берёт только подтверждённые вердикты:
    гипотеза, опирающаяся на «требует проверки», остаётся `unknown`.
    """
    from pf.ground import SupportStatus
    from pf.workflow import ClaimVerdict, GroundingResult, Workflow

    grounding = GroundingResult(
        verdicts=[
            ClaimVerdict(
                claim="Отток абонентов растёт",
                source_id="project/data/x.csv",
                status=SupportStatus.NEEDS_REVIEW,
                detail="иностранный источник",
            )
        ]
    )
    assert Workflow._confirmed_grounds(grounding) == []


def test_collection_targets_unknown_hypotheses_and_gaps() -> None:
    """Добор идёт только туда, где ответа нет.

    Поддержанная и опровергнутая гипотеза уже получила вердикт — искать по
    ним значит повторять работу. Противоречие тоже не ищется: поиск не
    отвечает на вопрос, какой из двух фактов верен, поэтому оно
    возвращается отдельным списком с причиной, а не превращается в запрос.
    """
    from pf.hypotheses import collection_nodes
    from pf.schemas import Claim

    unknown = Claim(kind="гипотеза", text="Отток абонентов растёт")
    supported = Claim(kind="гипотеза", text="Удержание снижается")
    supported.verdict = "supported"
    refuted = Claim(kind="гипотеза", text="Цены на тарифы вырастут")
    refuted.verdict = "refuted"
    contradiction = (
        "противоречие: «отток растёт» и «отток падает» говорят разное об одном показателе"
    )

    nodes, not_collectable = collection_nodes(
        [unknown, supported, refuted],
        ["нет данных об оттоке за квартал", contradiction],
    )

    assert [node.text for node in nodes] == [
        "Отток абонентов растёт",
        "нет данных об оттоке за квартал",
    ]
    assert nodes[0].kind == "гипотеза"
    assert nodes[1].kind == "пробел"
    assert nodes[0].query
    assert not_collectable and "противоречие" in not_collectable[0]


def test_collection_ranks_candidates_by_closeness() -> None:
    """Кандидаты упорядочены по близости к узлу, а не по порядку корпуса.

    Это то место, где `D-37` разрешает ранжирование: повышать статус
    ранжирование не может, а выбрать, что показать человеку первым, —
    его работа. Оценка — лексическая (`score_passages`), без моделей.
    """
    from pf.corpus import Article, Corpus
    from pf.hypotheses import plan_collection
    from pf.schemas import Claim

    corpus = Corpus(
        [
            Article(
                source_id="test/Физика",
                title="Эрланг",
                section="test",
                content=(
                    "Формула Эрланга описывает вероятность потери вызова "
                    "в телефонной сети и к тарифам отношения не имеет."
                ),
            ),
            Article(
                source_id="test/Отток",
                title="Отток",
                section="test",
                content=(
                    "Отток абонентов растёт третий квартал подряд, и оператор "
                    "связывает это с ростом цен на тарифы."
                ),
            ),
        ]
    )
    hypothesis = Claim(kind="гипотеза", text="Отток абонентов растёт из-за цен на тарифы")

    plan = plan_collection([hypothesis], [], corpus=corpus, per_node=1)

    assert len(plan.candidates) == 1
    assert plan.candidates[0].source_id == "test/Отток"
    assert plan.candidates[0].relevance > 0
    assert plan.outcomes[0].outcome == "new"


def test_collection_skips_known_and_repeated_passages() -> None:
    """Уже собранный абзац и его копия под другим адресом — не прогресс.

    Добор, который приносит то же знание, выглядит как работа, но ничего не
    добавляет: в `demo-v5` шесть отказов подряд были одним и тем же
    источником. Копия узнаётся по тексту (`pf.policy.content_key`), а не по
    ссылке.
    """
    from pf.corpus import Article, Corpus
    from pf.hypotheses import plan_collection
    from pf.schemas import Claim
    from pf.sources import Passage

    text = (
        "Отток абонентов растёт третий квартал подряд, и оператор связывает "
        "это с ростом цен на тарифы."
    )
    corpus = Corpus(
        [
            Article(source_id="test/Отток", title="Отток", section="test", content=text),
            Article(source_id="test/Копия", title="Копия", section="test", content=text),
        ]
    )
    known = Passage(
        source_id="test/Отток",
        title="Отток",
        section="test",
        paragraph_index=0,
        text=text,
        relevance=1.0,
    )
    hypothesis = Claim(kind="гипотеза", text="Отток абонентов растёт")

    plan = plan_collection([hypothesis], [], corpus=corpus, known=[known])

    assert plan.candidates == ()
    assert plan.outcomes[0].outcome == "duplicate"


def test_collection_closes_node_after_two_empty_rounds() -> None:
    """Два пустых раунда закрывают узел — правило `pf.policy`, а не счёт в коде.

    Один пустой результат — случайность, два — закономерность. Закрытый узел
    больше не запрашивается, и это видно причиной, а не молчанием.
    """
    from pf.corpus import Article, Corpus
    from pf.hypotheses import plan_collection
    from pf.schemas import Claim

    corpus = Corpus(
        [
            Article(
                source_id="test/Физика",
                title="Эрланг",
                section="test",
                content=(
                    "Формула Эрланга описывает вероятность потери вызова "
                    "в телефонной сети и к тарифам отношения не имеет."
                ),
            )
        ]
    )
    hypothesis = Claim(kind="гипотеза", text="Отток абонентов растёт")

    first = plan_collection([hypothesis], [], corpus=corpus)
    second = plan_collection([hypothesis], [], corpus=corpus, progress=first.progress)

    assert first.outcomes[0].outcome == "empty"
    assert second.outcomes[0].outcome == "no_progress"
    assert second.candidates == ()


def test_collection_does_not_change_hypothesis_verdict() -> None:
    """Кандидат — не основание: вердикт по-прежнему ставит код из оснований.

    Добор приносит абзац, который может ответить, но не подтверждает его:
    дословность проверяет `pf.ground`, а вердикт выводится из подтверждённых
    утверждений. Иначе поиск сам себе выдавал бы подтверждение.
    """
    from pf.corpus import Article, Corpus
    from pf.hypotheses import Verdict, plan_collection
    from pf.schemas import Claim

    corpus = Corpus(
        [
            Article(
                source_id="test/Отток",
                title="Отток",
                section="test",
                content="Отток абонентов растёт третий квартал подряд, а цены на тарифы выросли.",
            )
        ]
    )
    hypothesis = Claim(kind="гипотеза", text="Отток абонентов растёт")

    plan = plan_collection([hypothesis], [], corpus=corpus)

    assert plan.candidates
    assert hypothesis.verdict is Verdict.UNKNOWN
    assert hypothesis.verdict_basis == ""


def test_verdict_reaches_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Отчёт называет: поддержано столько, опровергнуто столько, неизвестно."""
    from pf.report import build_session_report

    monkeypatch.setattr("pf.config.settings.journal_dir", tmp_path)
    session_id = "session-verdict"
    journal = Journal(journal_path(session_id))
    journal.append(
        StepRecord(
            ordinal=1,
            role="analyst",
            phase="analyzing",
            summary="фактов: 1, гипотез: 3, пробелов: 0",
            detail={
                "facts": [{"kind": "факт", "text": "Отток абонентов растёт", "sources": []}],
                "hypotheses": [
                    {
                        "kind": "гипотеза",
                        "text": "Отток абонентов растёт",
                        "sources": [],
                        "verdict": "supported",
                        "verdict_basis": "Отток абонентов растёт",
                    },
                    {
                        "kind": "гипотеза",
                        "text": "Удержание абонентов падает",
                        "sources": [],
                        "verdict": "refuted",
                        "verdict_basis": "Удержание абонентов выросло до 87 %",
                    },
                    {
                        "kind": "гипотеза",
                        "text": "Цены на тарифы вырастут",
                        "sources": [],
                        "verdict": "unknown",
                        "verdict_basis": "",
                    },
                ],
                "gaps": [],
            },
        )
    )

    report = build_session_report(session_id)
    assert report.hypotheses == {"supported": 1, "refuted": 1, "unknown": 1}
    markdown = report.as_markdown()
    assert "поддержано 1" in markdown
    assert "опровергнуто 1" in markdown
    assert "требует проверки 1" in markdown


def test_collection_reaches_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Отчёт называет добор и говорит, что кандидаты — не основания.

    Иначе «нашли кандидатов» читалось бы как «вывод подкреплён». Числа
    берутся из журнала шага, как и остальные: отчёт — чтение прогона.
    """
    from pf.report import build_session_report

    monkeypatch.setattr("pf.config.settings.journal_dir", tmp_path)
    session_id = "session-collection"
    journal = Journal(journal_path(session_id))
    journal.append(
        StepRecord(
            ordinal=1,
            role="analyst",
            phase="analyzing",
            summary="фактов: 0, гипотез: 0, пробелов: 0",
            detail={
                "facts": [],
                "hypotheses": [],
                "gaps": [],
                "collection": {
                    "nodes": [
                        {"kind": "гипотеза", "text": "Отток абонентов растёт", "query": "отток"}
                    ],
                    "candidates": [
                        {
                            "node": "Отток абонентов растёт",
                            "citation": "test/Отток#p0",
                            "title": "Отток",
                            "relevance": 1.5,
                        }
                    ],
                    "outcomes": [
                        {"node": "Отток абонентов растёт", "outcome": "new", "reason": ""}
                    ],
                    "not_collectable": ["противоречие: два факта говорят разное"],
                },
            },
        )
    )

    report = build_session_report(session_id)
    markdown = report.as_markdown()

    assert report.collection_nodes == 1
    assert report.collection_candidates == 1
    assert report.collection_outcomes == {"new": 1}
    assert report.collection_not_collectable
    assert "Добор оснований" in markdown
    assert "основаниями не считаются" in markdown
    assert "test/Отток#p0" in markdown


def test_early_gap_declared_when_nothing_confirmed() -> None:
    """Разбор без подтверждённых оснований объявляет «данных нет».

    На вопросе вне данных проверка отклоняет каждое утверждение по
    отдельности, и набегает 10–22 отказа одного и того же смысла:
    «подтвердить нечем». Пробел объявляется один раз — вместо потока
    причин читатель видит один факт: данных нет (`T-81`).
    """
    from pf.hypotheses import EARLY_GAP_TEXT, early_gap

    assert early_gap(facts=12, confirmed=0) == EARLY_GAP_TEXT
    assert early_gap(facts=12, confirmed=3) is None
    assert early_gap(facts=0, confirmed=0) is None


def test_early_gap_keeps_review_and_user_input() -> None:
    """Материал, ждущий проверки, — не «данных нет».

    «Требует проверки» и опора на ввод человека — это опоры, а не
    отсутствие данных. Если такие есть, пробел не объявляется: решает
    человек, а не проверка (`D-49`).
    """
    from pf.hypotheses import early_gap

    assert early_gap(facts=5, confirmed=0, needs_review=1) is None
    assert early_gap(facts=5, confirmed=0, user_input=1) is None


def test_early_gap_collapses_rejections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Шаг разбора сворачивает поток отказов в один пробел.

    Двенадцать утверждений с выдуманными ссылками отклоняются все.
    Раньше шаг писал двенадцать строк «подтвердить нечем»; теперь он
    называет один пробел и одну причину с разбивкой по видам (`T-81`).
    """
    import json

    from pf.config import settings
    from pf.llm import LLMResponse, LLMUsage
    from pf.sources import Passage, SourceBundle
    from pf.state import Phase, SessionState
    from pf.workflow import Workflow

    class _Client:
        """Модель, которая отвечает разбором с выдуманными ссылками."""

        def complete(self, system: str, user: str, *, model: str = "") -> LLMResponse:
            answer = {
                "facts": [
                    {
                        "kind": "факт",
                        "text": f"Утверждение {index}",
                        "sources": [
                            {
                                "source_id": "phantom/Нет",
                                "paragraph": 0,
                                "quote": "цитата, которой нет",
                            }
                        ],
                    }
                    for index in range(12)
                ],
                "hypotheses": [],
                "forecasts": [],
                "gaps": [],
            }
            return LLMResponse(
                content=json.dumps(answer, ensure_ascii=False),
                usage=LLMUsage(prompt_tokens=10, completion_tokens=10),
            )

    monkeypatch.setattr(settings, "state_dir", tmp_path / "state")
    monkeypatch.setattr(settings, "journal_dir", tmp_path / "journal")

    passage = Passage(
        source_id="telecom/Мегафон (компания)",
        title="Мегафон (компания)",
        section="telecom",
        paragraph_index=0,
        text="Ввела специальный тариф «Транзитный».",
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
    bundle = SourceBundle(passages=[passage], articles=[article], query="удержание")

    workflow = Workflow(client=_Client(), config=settings)  # type: ignore[arg-type]
    # Добор по пробелу здесь не проверяется, а он поднимает весь корпус
    # (десятки секунд). Для этого теста важна только свёртка отказов.
    monkeypatch.setattr(Workflow, "_plan_collection", lambda *args, **kwargs: None)
    state = SessionState(session_id="early-gap", phase=Phase.ANALYZING, question="вопрос")
    state.results["collecting"] = {"facts": []}

    result = workflow.run_step(state, bundle)

    from pf.hypotheses import EARLY_GAP_TEXT

    assert EARLY_GAP_TEXT in result.data.gaps
    assert result.rejected == 1
    assert len(result.rejected_details) == 1
    assert EARLY_GAP_TEXT in result.rejected_details[0]
