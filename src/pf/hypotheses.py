"""Вердикт гипотезы: поддержана, опровергнута или неизвестна.

Зачем модуль. Гипотезы формулировались и больше не использовались: в ответе
стояло «гипотеза: отток из-за цены», и читатель не мог понять, проверяли её
или забыли. Это худший вид утверждения — выглядит как работа, а проверить
нечего.

Три правила, ради которых всё написано:

1. **Вердикт ставит код.** Модель формулирует гипотезу; сказать «подтверждено»
   — не её право. Вердикт выводится из подтверждённых оснований, а
   подтверждает основания проверка дословности (`pf.ground`).
2. **`unknown` — результат, а не провал.** Гипотеза, о которой в данных ничего
   нет, остаётся неизвестной и называется в отчёте. Молчание модели о
   вердикте не превращается в подтверждение.
3. **Опровержение требует противоположного направления.** «Отток из-за цен»
   не утверждает, куда идёт отток: и рост, и падение показателя совместимы с
   гипотезой. Опровергает только утверждение о противоположном направлении
   того же предмета.

Правило направления и предмета одно на репозиторий — ``pf.lexical.polarity``
и ``pf.lexical.topic_words``: противоречия в фактах (`Workflow`) и вердикт
гипотез обязаны считать направление одинаково, иначе «падает» в одном месте
и «снижается» в другом перестают быть одним и тем же.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from pf.corpus import Article, tokenize
from pf.lexical import polarity, topic_words
from pf.policy import NodeProgress, classify_content, content_key

if TYPE_CHECKING:
    from pf.sources import Passage


class Verdict(StrEnum):
    """Вердикт гипотезы."""

    #: Есть подтверждённое основание о том же предмете.
    SUPPORTED = "supported"
    #: Есть подтверждённое основание о противоположном направлении.
    REFUTED = "refuted"
    #: Проверять нечем: подтверждённых оснований о предмете нет.
    UNKNOWN = "unknown"

    @property
    def title_ru(self) -> str:
        """Человекочитаемое название вердикта."""
        return {
            Verdict.SUPPORTED: "поддержано",
            Verdict.REFUTED: "опровергнуто",
            Verdict.UNKNOWN: "требует проверки",
        }[self]


@dataclass(frozen=True, slots=True)
class HypothesisVerdict:
    """Вердикт гипотезы и то, на чём он держится.

    Attributes:
        verdict: Поддержана, опровергнута или неизвестна.
        basis: Подтверждённое утверждение, из которого следует вердикт.
            Пусто у ``unknown`` — и это честнее, чем выдуманное основание.
    """

    verdict: Verdict = Verdict.UNKNOWN
    basis: str = ""


def verdict_of(hypothesis: str, grounds: Iterable[str]) -> HypothesisVerdict:
    """Вывести вердикт гипотезы из подтверждённых утверждений.

    Основания перебираются по порядку, и первое совпадение по предмету решает:
    набор подтверждённых утверждений — это данные, а не голосование, поэтому
    результат не зависит от повторных вызовов.

    Args:
        hypothesis: Текст гипотезы.
        grounds: Тексты подтверждённых утверждений (дословность уже проверена).

    Returns:
        Вердикт и основание, на котором он держится.
    """
    text = hypothesis.strip()
    if not text:
        return HypothesisVerdict()

    subjects = topic_words(text)
    direction = polarity(text)
    for ground in grounds:
        candidate = str(ground).strip()
        if not candidate or not (topic_words(candidate) & subjects):
            continue
        candidate_direction = polarity(candidate)
        if direction is None:
            # Гипотеза не называет направление: совпадение предмета — уже
            # поддержка, а опровергать нечего.
            return HypothesisVerdict(verdict=Verdict.SUPPORTED, basis=candidate)
        if candidate_direction is None:
            # Основание о том же предмете, но о направлении молчит. Это не
            # противоречие: сравнивать не с чем.
            continue
        if candidate_direction == direction:
            return HypothesisVerdict(verdict=Verdict.SUPPORTED, basis=candidate)
        return HypothesisVerdict(verdict=Verdict.REFUTED, basis=candidate)
    return HypothesisVerdict()


# ── Ранний пробел ────────────────────────────────────────────────────────────

#: Текст раннего пробела: разбор ничего не подтвердил — данных нет.
EARLY_GAP_TEXT = "данных нет: ни одно утверждение разбора не подтверждено источниками"


def early_gap(
    *,
    facts: int,
    confirmed: int,
    needs_review: int = 0,
    user_input: int = 0,
) -> str | None:
    """Объявить пробел «данных нет», если разбор ничего не подтвердил.

    Зачем. На вопросе вне данных проверка отклоняет каждое утверждение по
    отдельности, и набегает 10–22 отказа одного и того же смысла:
    «подтвердить нечем». Читатель видит поток причин, а не один факт —
    данных нет. Пробел объявляется один раз, а отказы сворачиваются в одну
    строку с разбивкой по видам (см. ``Workflow.run_step``).

    «Требует проверки» и опора на ввод человека — это **опоры**, а не
    отсутствие данных: доверие к ним определяет человек, а не проверка
    (`D-49`). Поэтому при них пробел не объявляется.

    Args:
        facts: Сколько фактов вернул разбор.
        confirmed: Сколько оснований подтверждено (ссылки, а не записи).
        needs_review: Сколько оснований ждёт проверки человеком.
        user_input: Сколько опор на ввод пользователя.

    Returns:
        Текст пробела либо ``None``, если разбор что-то подтвердил или
        подтверждать было нечего.
    """
    if facts <= 0:
        return None
    if confirmed > 0 or needs_review > 0 or user_input > 0:
        return None
    return EARLY_GAP_TEXT


# ── Добор оснований по неизвестному ──────────────────────────────────────────


class CandidateSearch(Protocol):
    """Поиск кандидатов: корпус или его подмена в тесте.

    Протокол, а не конкретный класс: добору нужен только ``search``, и
    тест не обязан строить TF-IDF по восьми тысячам статей.
    """

    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        prefer_domain: bool = True,
    ) -> list[tuple[Article, float]]:
        """Найти статьи под запрос.

        Args:
            query: Поисковый запрос.
            top_k: Сколько статей вернуть.
            prefer_domain: Применять ли доменную обработку.

        Returns:
            Пары «статья — оценка», по убыванию оценки.
        """
        ...


@dataclass(frozen=True, slots=True)
class CollectionNode:
    """Что осталось выяснить и каким запросом это ищется.

    Attributes:
        kind: «гипотеза» или «пробел» — откуда взялся узел.
        text: Исходная формулировка (её видит человек).
        query: Запрос для поиска; сейчас совпадает с текстом, но отделён,
            чтобы формулировку для человека и запрос можно было развести.
    """

    kind: str
    text: str
    query: str


@dataclass(frozen=True, slots=True)
class CollectionCandidate:
    """Абзац, который может ответить на узел.

    Кандидат — **не** основание: дословность проверяет `pf.ground`, а
    вердикт ставит код по подтверждённым утверждениям. Поэтому здесь нет
    поля статуса — его нечем заполнить.

    Attributes:
        node: Текст узла, ради которого найден абзац.
        source_id: Идентификатор статьи.
        title: Заголовок статьи.
        citation: Ссылка на абзац для человека.
        text: Текст абзаца.
        relevance: Лексическая близость к узлу (`score_passages`).
    """

    node: str
    source_id: str
    title: str
    citation: str
    text: str
    relevance: float


@dataclass(frozen=True, slots=True)
class CollectionOutcome:
    """Чем кончился добор по одному узлу.

    Attributes:
        node: Текст узла.
        outcome: ``new``, ``duplicate``, ``empty``, ``no_progress`` или
            причина закрытия узла из `pf.policy`.
        reason: Пояснение для человека; пусто, когда всё в порядке.
    """

    node: str
    outcome: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class CollectionPlan:
    """Что добрать по неизвестному: узлы, кандидаты и исходы.

    Attributes:
        nodes: Что осталось выяснить.
        candidates: Найденные абзацы, по убыванию близости к своему узлу.
        outcomes: Чем кончился добор по каждому узлу.
        not_collectable: Пробелы, которые поиском не закрываются, с причиной.
        progress: Состояние узлов для следующего раунда. Обычный словарь, а
            не объекты `pf.policy`: состояние сессии сохраняется в JSON, и
            объект пришлось бы разбирать обратно при каждом чтении.
    """

    nodes: tuple[CollectionNode, ...] = ()
    candidates: tuple[CollectionCandidate, ...] = ()
    outcomes: tuple[CollectionOutcome, ...] = ()
    not_collectable: tuple[str, ...] = ()
    progress: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_detail(self) -> dict[str, Any]:
        """Представление для журнала шага.

        Текст кандидата не дублируется: в журнале достаточно ссылки и
        заголовка, а полный абзац уже лежит в источнике.

        Returns:
            Словарь с узлами, кандидатами, исходами и неразбираемыми пробелами.
        """
        return {
            "nodes": [
                {"kind": node.kind, "text": node.text, "query": node.query} for node in self.nodes
            ],
            "candidates": [
                {
                    "node": candidate.node,
                    "citation": candidate.citation,
                    "title": candidate.title,
                    "relevance": round(candidate.relevance, 3),
                }
                for candidate in self.candidates
            ],
            "outcomes": [
                {"node": item.node, "outcome": item.outcome, "reason": item.reason}
                for item in self.outcomes
            ],
            "not_collectable": list(self.not_collectable),
        }


#: Сколько статей смотреть под один узел. Это ответ на вопрос «есть ли
#: ответ вообще», а не обход корпуса: глубина и уровни — забота `D-15`.
DEFAULT_TOP_K = 3

#: Сколько абзацев одного узла показывать человеку.
DEFAULT_PER_NODE = 2

#: Признак пробела, который поиском не закрывается.
_CONTRADICTION_MARK = "противоречие"


def collection_nodes(
    hypotheses: Iterable[Any],
    gaps: Iterable[str],
) -> tuple[tuple[CollectionNode, ...], tuple[str, ...]]:
    """Узлы добора: неизвестные гипотезы и закрываемые пробелы.

    Гипотеза с вердиктом уже получила ответ — искать по ней значит повторять
    работу. Противоречие не ищется никогда: поиск не отвечает на вопрос,
    какой из двух фактов верен, поэтому оно возвращается отдельным списком с
    причиной, а не превращается в запрос.

    Args:
        hypotheses: Гипотезы разбора (объекты с полями ``text``, ``verdict``).
        gaps: Пробелы, объявленные разбором.

    Returns:
        Пары «узлы добора» и «пробелы, которые поиском не закрываются».
    """
    nodes: list[CollectionNode] = []
    not_collectable: list[str] = []

    for hypothesis in hypotheses:
        text = str(getattr(hypothesis, "text", "")).strip()
        if not text or _verdict_name(hypothesis) != Verdict.UNKNOWN.value:
            continue
        nodes.append(CollectionNode(kind="гипотеза", text=text, query=text))

    for gap in gaps:
        text = str(gap).strip()
        if not text:
            continue
        if _CONTRADICTION_MARK in text.lower():
            not_collectable.append(text)
            continue
        nodes.append(CollectionNode(kind="пробел", text=text, query=text))

    return tuple(nodes), tuple(not_collectable)


def plan_collection(
    hypotheses: Iterable[Any],
    gaps: Iterable[str],
    *,
    corpus: CandidateSearch,
    known: Iterable[Passage] = (),
    progress: Mapping[str, Any] | None = None,
    top_k: int = DEFAULT_TOP_K,
    per_node: int = DEFAULT_PER_NODE,
) -> CollectionPlan:
    """Спланировать добор оснований по неизвестному.

    Добор детерминированный: тот же корпус и тот же узел дают тех же
    кандидатов. Модель здесь не вызывается — иначе `make test` ходил бы в
    Ollama, а прогон перестал бы повторяться.

    Кандидат — **не** основание: он не попадает ни в ``SupportStatus``, ни в
    вердикт гипотезы. Вывод остаётся за `pf.ground`, а близость решает
    только, что показать человеку первым (`D-37`).

    Остановка — правило `pf.policy`, а не счёт в этом коде: дубль и пустой
    результат не дают знания, а два пустых раунда подряд закрывают узел,
    чтобы он не запрашивался вечно.

    Args:
        hypotheses: Гипотезы разбора.
        gaps: Пробелы разбора.
        corpus: Поиск по корпусу (в тесте — своя подмена).
        known: Уже собранные абзацы: их не предлагаем повторно.
        progress: Состояние узлов из предыдущего раунда.
        top_k: Сколько статей смотреть под узел.
        per_node: Сколько новых абзацев оставить на узел.

    Returns:
        План добора: узлы, кандидаты, исходы по узлам и состояние для
        следующего раунда.
    """
    from pf.sources import score_passages

    nodes, not_collectable = collection_nodes(hypotheses, gaps)
    state = _progress_of(progress)
    if not nodes:
        return CollectionPlan(not_collectable=not_collectable, progress=_dump_progress(state))

    known_list = list(known)
    known_citations = {passage.citation for passage in known_list}
    known_texts = {content_key(passage.text) for passage in known_list}

    candidates: list[CollectionCandidate] = []
    outcomes: list[CollectionOutcome] = []

    for node in nodes:
        node_state = state.setdefault(node.text, NodeProgress())
        if node_state.is_closed:
            outcomes.append(
                CollectionOutcome(
                    node=node.text,
                    outcome=node_state.closed_reason,
                    reason=(
                        f"узел закрыт: {node_state.empty_hits} раунд(а) без нового "
                        "знания — повторять нечего"
                    ),
                )
            )
            continue

        terms = set(tokenize(node.query))
        found = 0
        duplicate = 0
        for article, _score in corpus.search(node.query, top_k=top_k):
            for passage in score_passages(article, terms, per_node):
                if passage.citation in known_citations:
                    duplicate += 1
                    continue
                key = content_key(passage.text)
                if key in known_texts:
                    duplicate += 1
                    continue
                known_citations.add(passage.citation)
                known_texts.add(key)
                found += 1
                candidates.append(
                    CollectionCandidate(
                        node=node.text,
                        source_id=passage.source_id,
                        title=passage.title,
                        citation=passage.citation,
                        text=passage.text,
                        relevance=passage.relevance,
                    )
                )
                if found >= per_node:
                    break
            if found >= per_node:
                break

        outcome = classify_content(new=found, duplicate=duplicate)
        node_state.record(outcome)
        # Раунд, который закрывает узел, называется причиной закрытия, а не
        # «пусто»: читателю важно знать, что дальше по этому узлу не пойдут.
        value = node_state.closed_reason or outcome.value
        reason = {
            "duplicate": "найденное уже есть в собранных источниках",
            "empty": "по запросу ничего не нашлось",
        }.get(outcome.value, "")
        if node_state.closed_reason:
            reason = (
                f"узел закрыт: {node_state.empty_hits} раунд(а) без нового знания — "
                "повторять нечего"
            )
        outcomes.append(CollectionOutcome(node=node.text, outcome=value, reason=reason))

    # Порядок — по близости: человек читает сверху, и первым должно стоять
    # то, что ближе к вопросу. Внутри узла порядок уже задан `score_passages`.
    candidates.sort(key=lambda item: -item.relevance)

    return CollectionPlan(
        nodes=nodes,
        candidates=tuple(candidates),
        outcomes=tuple(outcomes),
        not_collectable=not_collectable,
        progress=_dump_progress(state),
    )


def _verdict_name(hypothesis: Any) -> str:
    """Название вердикта гипотезы строкой.

    Args:
        hypothesis: Гипотеза разбора.

    Returns:
        ``supported`` / ``refuted`` / ``unknown``; пусто, если поля нет.
    """
    verdict = getattr(hypothesis, "verdict", None)
    return str(getattr(verdict, "value", verdict) or "")


def _progress_of(
    progress: Mapping[str, Any] | None,
) -> dict[str, NodeProgress]:
    """Восстановить состояние узлов из сохранённого словаря.

    Args:
        progress: Словарь из предыдущего раунда.

    Returns:
        Состояние узлов. Незнакомые записи пропускаются: журнал мог быть
        записан другой версией, и падать из-за этого незачем.
    """
    state: dict[str, NodeProgress] = {}
    for node, raw in (progress or {}).items():
        if not isinstance(raw, Mapping):
            continue
        state[str(node)] = NodeProgress(
            empty_hits=int(raw.get("empty_hits") or 0),
            closed_reason=str(raw.get("closed_reason") or ""),
        )
    return state


def _dump_progress(state: Mapping[str, NodeProgress]) -> dict[str, dict[str, Any]]:
    """Сохранить состояние узлов в виде, пригодном для JSON.

    Args:
        state: Состояние узлов.

    Returns:
        Словарь «узел → счётчики и причина закрытия».
    """
    return {
        node: {"empty_hits": item.empty_hits, "closed_reason": item.closed_reason}
        for node, item in state.items()
    }
