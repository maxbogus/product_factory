"""Замер семантики: решает ли близость то, чего не решает лексика (D-37).

Зачем модуль. Проверка оснований лексическая: леммы, словарь близости, точные
термины, числа, отрицание (`D-19`). Её ограничение названо там же — честная
цитата, поставленная не к месту, проходит. Вопрос «не поставить ли модель
эмбеддингов» решается числами, а не вкусом: поэтому здесь живёт замер с
гейтом из четырёх критериев (`docs/decisions.md`, D-37).

Что именно измеряется:

- **флаг «дословно, но не по месту»** — цитата найдена в источнике буквально,
  но к утверждению не относится. Мера — полнота на размеченном наборе
  (`eval/pairs/out_of_place.jsonl`) при условии, что честные подтверждения от
  флага не теряются: точность подтверждений не падает;
- **отбор кандидатов** — выигрыш nDCG@5. Сейчас не измерен: во всех
  сохранённых журналах один и тот же вопрос, и `n = 1` даёт шум, а не
  число. Модуль возвращает `None` там, где данных нет, а не ноль;
- **бюджет** — доля времени прогона (по журналу) и память на модель.

Модуль намеренно ничего не решает за агента: семантика по `D-37` может только
**понижать** статус основания, а подтверждение по-прежнему пишет
`grounding-gate`. Здесь считаются метрики и вердикт замера — и всё.

Порог флага калибруется на половине набора (train) и замораживается, метрики
считаются на второй половине (test): иначе порог подгонялся бы под те же
пары, на которых отчитывается.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pf.ground import find_quote
from pf.lexical import exact_terms_in, stems
from pf.sanitize import for_comparison

#: Корень репозитория: файл лежит в ``src/pf/``.
ROOT = Path(__file__).resolve().parent.parent.parent

#: Размеченный набор пар.
PAIRS_PATH = ROOT / "eval" / "pairs" / "out_of_place.jsonl"

#: Доля времени прогона, которую замер разрешает слою (D-37).
MAX_TIME_SHARE = 0.10

#: Предел памяти на модель с векторами, МБ (D-37).
MAX_RSS_MB = 1024

#: Порог полноты флага «не по месту» (D-37).
MIN_FLAG_RECALL = 0.7

#: Порог выигрыша отбора кандидатов (D-37).
MIN_NDCG_GAIN = 0.05

#: Слова, которые есть в каждом тексте и потому ничего не различают.
#: Отбрасываются из TF-IDF: иначе близость считалась бы по «и», «в», «на».
#: Сравнение идёт с основами слов — так же, как в остальной проверке.
STOP_WORDS = frozenset(
    {
        "около",
        "более",
        "менее",
        "этот",
        "какой",
        "котор",
        "также",
        "через",
        "после",
        "перед",
        "между",
        "соста",
    }
)


class Label(StrEnum):
    """Метка пары: цитата подтверждает утверждение или нет."""

    #: Цитата к утверждению относится.
    IN_PLACE = "in_place"
    #: Цитата дословная, но утверждения не подтверждает.
    OUT_OF_PLACE = "out_of_place"


class Kind(StrEnum):
    """Откуда взялась пара."""

    #: Из сохранённого журнала: утверждение и цитата жили вместе.
    REAL = "real"
    #: Собрана из своих же данных: цитата одного утверждения выдана за
    #: основание другого.
    SYNTHETIC = "synthetic"


@dataclass(frozen=True, slots=True)
class Pair:
    """Пара «утверждение — цитата» с меткой.

    Attributes:
        pair_id: Идентификатор пары, уникальный в наборе.
        claim: Утверждение, которое собралось опираться на цитату.
        quote: Цитата — обязана быть дословной.
        source_id: Источник цитаты.
        paragraph: Номер абзаца источника, в котором цитата лежит.
        label: Подтверждает ли цитата утверждение.
        kind: Происхождение пары.
        why: Почему метка такая — словами. Метка без причины не проверяема.
    """

    pair_id: str
    claim: str
    quote: str
    source_id: str
    paragraph: int
    label: Label
    kind: Kind
    why: str


def _require_text(raw: Mapping[str, Any], key: str, where: str) -> str:
    """Взять обязательное непустое текстовое поле.

    Args:
        raw: Разобранная запись.
        key: Имя поля.
        where: Где запись — для сообщения об ошибке.

    Returns:
        Значение поля.

    Raises:
        ValueError: Поля нет или оно пустое.
    """
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: поле «{key}» должно быть непустой строкой")
    return value.strip()


def _require_int(raw: Mapping[str, Any], key: str, where: str) -> int:
    """Взять обязательное целое поле.

    Args:
        raw: Разобранная запись.
        key: Имя поля.
        where: Где запись.

    Returns:
        Значение поля.

    Raises:
        ValueError: Поля нет или это не целое.
    """
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{where}: поле «{key}» должно быть целым числом")
    return value


def load_pairs(path: Path | None = None) -> list[Pair]:
    """Прочитать размеченный набор пар.

    Args:
        path: Путь к набору. По умолчанию — :data:`PAIRS_PATH`.

    Returns:
        Список пар в порядке файла.

    Raises:
        ValueError: Запись неполная или метка/вид неизвестны — набор данных
            не должен читаться «наполовину».
    """
    target = path or PAIRS_PATH
    pairs: list[Pair] = []
    for line_no, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw: Any = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{target.name}:{line_no}: запись должна быть объектом")
        where = f"{target.name}:{line_no}"
        pairs.append(
            Pair(
                pair_id=_require_text(raw, "id", where),
                claim=_require_text(raw, "claim", where),
                quote=_require_text(raw, "quote", where),
                source_id=_require_text(raw, "source_id", where),
                paragraph=_require_int(raw, "paragraph", where),
                label=Label(_require_text(raw, "label", where)),
                kind=Kind(_require_text(raw, "kind", where)),
                why=_require_text(raw, "why", where),
            )
        )
    return pairs


def pairs_sha256(path: Path | None = None) -> str:
    """Хеш набора пар: снимок замера обязан относиться к текущему набору.

    Args:
        path: Путь к набору. По умолчанию — :data:`PAIRS_PATH`.

    Returns:
        Хеш содержимого файла в шестнадцатеричном виде.
    """
    return hashlib.sha256((path or PAIRS_PATH).read_bytes()).hexdigest()


def split_of(index: int, pairs: Sequence[Pair]) -> str:
    """Половина набора, к которой относится пара.

    Делится по порядку внутри своего класса: первая честная пара — train,
    вторая — test, и так же для пар «не по месту». Иначе половины
    разойдутся по составу: при делении по чётности индекса одна половина
    может целиком состоять из честных пар, и тогда точность на test не
    проверяется вовсе — замер выглядит строгим, а не проверяет ничего.

    Args:
        index: Порядковый номер пары в файле (с нуля).
        pairs: Набор пар целиком — по нему считается номер внутри класса.

    Returns:
        ``train`` или ``test``.
    """
    label = pairs[index].label
    position = sum(1 for earlier in pairs[:index] if earlier.label is label)
    return "train" if position % 2 == 0 else "test"


def source_passage(source_id: str, paragraph: int) -> str | None:
    """Текст абзаца источника по идентификатору.

    Args:
        source_id: Идентификатор источника.
        paragraph: Номер абзаца.

    Returns:
        Текст абзаца или ``None``, если источник или абзац не найдены.
    """
    from pf.materials import load_materials

    tail = source_id.rsplit("/", 1)[-1]
    for material in load_materials():
        if material.source_id != source_id and material.source_id.rsplit("/", 1)[-1] != tail:
            continue
        if 0 <= paragraph < len(material.passages):
            return material.passages[paragraph].text
        return None
    return None


def quote_is_verbatim(pair: Pair) -> bool:
    """Лежит ли цитата пары дословно в указанном абзаце источника.

    Args:
        pair: Пара для проверки.

    Returns:
        Цитата найдена дословно.
    """
    text = source_passage(pair.source_id, pair.paragraph)
    if text is None:
        return False
    found, _ = find_quote(text, pair.quote)
    return found


def _tokens(text: str) -> list[str]:
    """Значимые единицы текста для TF-IDF.

    Правило берётся из `pf.lexical`, а не пишется заново: основы слов
    (`stems`) склеивают «абонентская» и «абонентов», точные термины
    (`exact_terms_in`) не дают потерять «МТС». Своя третья токенизация в этом
    репозитории уже приводила к тому, что честная пара считалась
    несовпадающей.

    Числа в мешок слов **не** входят: они сравниваются как величины
    (`pf.quantities`), а совпадение числа — не смысловое сходство. Именно
    добавленные в мешок числа давали ложное совпадение чужой строки
    (утечка `p-032`: «27,4 млн» у 2016 года и у накопленного итога 2019-го).

    Args:
        text: Исходный текст.

    Returns:
        Список единиц: основы слов и точные термины.
    """
    clean = for_comparison(text)
    tokens = [token for token in stems(clean) if token not in STOP_WORDS]
    tokens.extend(term.lower() for term in exact_terms_in(clean) if len(term) <= 4)
    return tokens


def _inverse_frequency(documents: Sequence[Sequence[str]]) -> dict[str, float]:
    """Обратная частота слов по набору текстов.

    Args:
        documents: Тексты, разложенные на леммы.

    Returns:
        Отображение «лемма → вес».
    """
    total = len(documents)
    seen: dict[str, int] = {}
    for tokens in documents:
        for token in set(tokens):
            seen[token] = seen.get(token, 0) + 1
    return {token: math.log((1 + total) / (1 + count)) + 1.0 for token, count in seen.items()}


def _vector(tokens: Sequence[str], idf: Mapping[str, float]) -> dict[str, float]:
    """Взвешенный вектор текста с единичной длиной.

    Args:
        tokens: Леммы текста.
        idf: Веса слов.

    Returns:
        Отображение «лемма → вес», нормированное по длине.
    """
    counts: dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
    weighted = {token: count * idf.get(token, 1.0) for token, count in counts.items()}
    norm = math.sqrt(sum(value * value for value in weighted.values()))
    if norm == 0.0:
        return {}
    return {token: value / norm for token, value in weighted.items()}


def _cosine(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Косинусная близость двух нормированных векторов.

    Args:
        left: Первый вектор.
        right: Второй вектор.

    Returns:
        Близость от 0 до 1.
    """
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(token, 0.0) for token, weight in left.items())


def tfidf_scores(pairs: Sequence[Pair]) -> dict[str, float]:
    """Близость утверждения и цитаты по TF-IDF — пол замера.

    Веса считаются по самому набору пар: набор фиксирован, а сравнение идёт
    внутри него. Это дешёвая и полностью детерминированная мера — тот самый
    «пол», с которым сравниваются эмбеддинги.

    Args:
        pairs: Набор пар.

    Returns:
        Отображение «идентификатор пары → близость».
    """
    claims = [_tokens(pair.claim) for pair in pairs]
    quotes = [_tokens(pair.quote) for pair in pairs]
    idf = _inverse_frequency([*claims, *quotes])
    claim_vectors = [_vector(tokens, idf) for tokens in claims]
    quote_vectors = [_vector(tokens, idf) for tokens in quotes]
    return {
        pair.pair_id: round(_cosine(claim_vectors[index], quote_vectors[index]), 6)
        for index, pair in enumerate(pairs)
    }


@dataclass(frozen=True, slots=True)
class FlagScore:
    """Исход флага «дословно, но не по месту» на половине набора.

    Attributes:
        split: ``train`` или ``test``.
        threshold: Замороженный порог близости.
        true_positives: Пары «не по месту», которые флаг поймал.
        false_positives: Честные пары, которые флаг понизил.
        false_negatives: Пары «не по месту», которые флаг пропустил.
        true_negatives: Честные пары, которые флаг не тронул.
    """

    split: str
    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int

    @property
    def positives(self) -> int:
        """Сколько в половине пар «не по месту»."""
        return self.true_positives + self.false_negatives

    @property
    def precision(self) -> float:
        """Точность флага: доля верных срабатываний."""
        fired = self.true_positives + self.false_positives
        return round(self.true_positives / fired, 4) if fired else 0.0

    @property
    def recall(self) -> float:
        """Полнота флага: доля найденных пар «не по месту»."""
        return round(self.true_positives / self.positives, 4) if self.positives else 0.0

    @property
    def confirmations_lost(self) -> int:
        """Сколько честных подтверждений потеряно — это и есть точность.

        Семантика по `D-37` умеет только понижать статус, поэтому каждое
        ложное срабатывание — потерянное подтверждение, а не «погрешность».
        """
        return self.false_positives

    def as_dict(self) -> dict[str, Any]:
        """Исход в том виде, в каком он попадает в снимок замера.

        Returns:
            Словарь с числами и посчитанными метриками.
        """
        return {
            "split": self.split,
            "threshold": self.threshold,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "positives": self.positives,
            "precision": self.precision,
            "recall": self.recall,
            "confirmations_lost": self.confirmations_lost,
        }


def calibrate_threshold(scores: Mapping[str, float], pairs: Sequence[Pair]) -> float:
    """Порог флага по половине train: ниже него — «не по месту».

    Правило простое и потому проверяемое: порог — самая низкая близость
    среди честных пар train. Тогда на train ни одно честное подтверждение не
    понижается, а порог не подбирается под те же пары, на которых считается
    полнота.

    Args:
        scores: Близости по парам.
        pairs: Набор пар.

    Returns:
        Порог близости.

    Raises:
        ValueError: В train нет честных пар — калибровать не на чем.
    """
    honest = [
        scores[pair.pair_id]
        for index, pair in enumerate(pairs)
        if split_of(index, pairs) == "train" and pair.label is Label.IN_PLACE
    ]
    if not honest:
        raise ValueError("в train нет честных пар: порог калибровать не на чем")
    # Приведение к float обязательно: у эмбеддингов близость приходит
    # числом numpy, а снимок замера — JSON, и float32 в него не пишется.
    return float(min(honest))


def flag_precision_recall(
    scores: Mapping[str, float],
    pairs: Sequence[Pair],
    threshold: float,
    *,
    split: str = "train",
) -> FlagScore:
    """Посчитать исход флага на половине набора.

    Args:
        scores: Близости по парам.
        pairs: Набор пар.
        threshold: Замороженный порог: флаг срабатывает при близости ниже.
        split: ``train`` или ``test``.

    Returns:
        Исход флага.
    """
    true_positives = false_positives = false_negatives = true_negatives = 0
    for index, pair in enumerate(pairs):
        if split_of(index, pairs) != split:
            continue
        flagged = scores[pair.pair_id] < threshold
        if pair.label is Label.OUT_OF_PLACE:
            if flagged:
                true_positives += 1
            else:
                false_negatives += 1
        elif flagged:
            false_positives += 1
        else:
            true_negatives += 1
    return FlagScore(
        split=split,
        threshold=round(threshold, 6),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_negatives=true_negatives,
    )


def ndcg_at_k(
    ranked: Sequence[str],
    relevant: Mapping[str, float],
    *,
    k: int = 5,
) -> float | None:
    """nDCG@k отобранных кандидатов.

    Args:
        ranked: Идентификаторы кандидатов в порядке отбора.
        relevant: Оценки релевантности: 0 — нерелевантен, больше — тем важнее.
        k: Сколько первых позиций учитывать.

    Returns:
        nDCG от 0 до 1 либо ``None``, если измерять нечего: пустой отбор,
        пустая разметка или все оценки нулевые. Ноль здесь соврал бы — он
        выглядит как «плохо», а означает «нет данных».
    """
    if not ranked or not relevant or k <= 0:
        return None
    gains = [relevant.get(key, 0.0) for key in ranked[:k]]
    ideal = sorted(relevant.values(), reverse=True)[:k]
    dcg = sum(gain / math.log2(position + 2) for position, gain in enumerate(gains))
    idcg = sum(gain / math.log2(position + 2) for position, gain in enumerate(ideal))
    if idcg == 0.0:
        return None
    return round(dcg / idcg, 4)


@dataclass(frozen=True, slots=True)
class Budget:
    """Бюджет слоя: доля времени прогона и память на модель.

    Attributes:
        time_share: Доля времени прогона, которую занял слой.
        rss_delta_mb: Прирост памяти, МБ.
        within_time: Уложился ли слой в :data:`MAX_TIME_SHARE`.
        within_memory: Уложился ли слой в :data:`MAX_RSS_MB`.
    """

    time_share: float
    rss_delta_mb: float
    within_time: bool
    within_memory: bool


def budget_check(
    *,
    baseline_sec: float,
    encode_sec: float,
    rss_delta_mb: float,
) -> Budget:
    """Проверить бюджет слоя против прогона агента.

    Время сравнивается не с пустотой, а с прогоном: 10 % от 311 секунд
    (`demo-v12`) — это 31 секунда, а не «шесть». Числа берутся из журнала
    (``duration_ms`` шагов), поэтому бюджет не выдуман.

    Args:
        baseline_sec: Длительность прогона агента, секунды.
        encode_sec: Время слоя, секунды.
        rss_delta_mb: Прирост памяти, МБ.

    Returns:
        Бюджет с вердиктами по времени и памяти.
    """
    share = encode_sec / baseline_sec if baseline_sec > 0 else float("inf")
    return Budget(
        time_share=round(share, 6),
        rss_delta_mb=round(rss_delta_mb, 1),
        within_time=share <= MAX_TIME_SHARE,
        within_memory=rss_delta_mb <= MAX_RSS_MB,
    )


def roc_auc(scores: Mapping[str, float], pairs: Sequence[Pair]) -> float:
    """Площадь под ROC: как часто честная пара выше подмены по оценке.

    0,5 — оценка не различает классы; 1,0 — все честные выше всех подмен.
    Считается рангами без зависимостей: на шестидесяти парах это 900
    сравнений, библиотека не нужна (`D-09`).

    Args:
        scores: Оценки по парам. Для флага «не по месту» — близость, для
            степени поддержки — она сама.
        pairs: Набор пар.

    Returns:
        Площадь от 0 до 1.
    """
    honest = [scores[pair.pair_id] for pair in pairs if pair.label is Label.IN_PLACE]
    alien = [scores[pair.pair_id] for pair in pairs if pair.label is Label.OUT_OF_PLACE]
    if not honest or not alien:
        return 0.5
    wins = sum(
        1.0 if left > right else 0.5 if left == right else 0.0 for left in honest for right in alien
    )
    return wins / (len(honest) * len(alien))


def brier(scores: Mapping[str, float], pairs: Sequence[Pair]) -> float:
    """Средняя квадратичная ошибка оценки как вероятности.

    Оценка трактуется как вероятность того, что пара честная. Базовая ставка —
    0,5 (половина на половину), её Brier 0,25: если оценка не лучше, называть
    её вероятностью нельзя, и в отчёте она называется степенью.

    Args:
        scores: Оценки по парам.
        pairs: Набор пар.

    Returns:
        Ошибка от 0 до 1.
    """
    total = 0.0
    for pair in pairs:
        outcome = 1.0 if pair.label is Label.IN_PLACE else 0.0
        total += (scores[pair.pair_id] - outcome) ** 2
    return total / len(pairs) if pairs else 0.0


def evaluate_method(scores: Mapping[str, float], pairs: Sequence[Pair]) -> dict[str, Any]:
    """Оценить один метод: порог на train, метрики на train и test.

    Args:
        scores: Близости по парам.
        pairs: Набор пар.

    Returns:
        Словарь записи метода для снимка замера.
    """
    threshold = calibrate_threshold(scores, pairs)
    train = flag_precision_recall(scores, pairs, threshold, split="train")
    test = flag_precision_recall(scores, pairs, threshold, split="test")
    return {
        "threshold": round(threshold, 6),
        "auc": round(roc_auc(scores, pairs), 6),
        "brier": round(brier(scores, pairs), 6),
        "train": train.as_dict(),
        "test": test.as_dict(),
    }


@dataclass(frozen=True, slots=True)
class Criterion:
    """Один критерий гейта.

    Attributes:
        key: Машинное имя критерия.
        title: Как критерий называется человеку.
        measured: Есть ли данные, чтобы его измерить.
        passed: Выполнен ли он (``False``, если не измерен).
        detail: Числа или причина, по которой измерить нельзя.
    """

    key: str
    title: str
    measured: bool
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class GateVerdict:
    """Вердикт гейта: четыре критерия и итог.

    Attributes:
        passed: Выполнены ли все **измеримые** критерии.
        criteria: Критерии в порядке проверки.
        summary: Итог словами, с числами.
    """

    passed: bool
    criteria: tuple[Criterion, ...]
    summary: str


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    """Привести значение снимка к объекту.

    Args:
        value: Значение из снимка.
        where: Где оно лежит — для сообщения об ошибке.

    Returns:
        Объект.

    Raises:
        ValueError: Значение не объект.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: ожидался объект")
    return value


def _number(raw: Mapping[str, Any], key: str, where: str) -> float:
    """Взять обязательное числовое поле.

    Args:
        raw: Объект снимка.
        key: Имя поля.
        where: Где объект.

    Returns:
        Число.

    Raises:
        ValueError: Поля нет или это не число.
    """
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: поле «{key}» должно быть числом")
    return float(value)


def _methods(measurement: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Все методы замера.

    Args:
        measurement: Снимок замера.

    Returns:
        Отображение «имя метода → запись».

    Raises:
        ValueError: Методов нет — сравнивать нечего.
    """
    raw = _mapping(measurement.get("methods"), "замер.methods")
    methods: dict[str, Mapping[str, Any]] = {}
    for name, value in raw.items():
        methods[str(name)] = _mapping(value, f"замер.methods.{name}")
    if not methods:
        raise ValueError("замер без методов: сравнивать нечего")
    return methods


def _ru(value: float, digits: int = 2) -> str:
    """Число в русской записи: с запятой вместо точки.

    Args:
        value: Число.
        digits: Сколько знаков после запятой.

    Returns:
        Строка для документа.
    """
    return f"{value:.{digits}f}".replace(".", ",")


def _join_numbers(values: Mapping[str, float], digits: int, tail: str) -> str:
    """Список «метод — число» через точку-разделитель.

    Args:
        values: Значения по методам.
        digits: Сколько знаков после запятой.
        tail: Что дописать после каждого значения.

    Returns:
        Строка вида ``tfidf 0,33tail · rubert 0,71tail``.
    """
    return " · ".join(f"{name} {_ru(value, digits)}{tail}" for name, value in values.items())


#: Начало блока вердикта в `docs/decisions.md`.
MARKER_BEGIN = "<!-- semantic-verdict:begin -->"

#: Конец блока вердикта в `docs/decisions.md`.
MARKER_END = "<!-- semantic-verdict:end -->"


def _budgets(methods: Mapping[str, Mapping[str, Any]], run_sec: float) -> dict[str, Budget]:
    """Бюджет по каждому методу замера.

    Args:
        methods: Методы замера.
        run_sec: Длительность прогона агента, секунды.

    Returns:
        Отображение «имя метода → бюджет».
    """
    budgets: dict[str, Budget] = {}
    for name, method in methods.items():
        budgets[name] = budget_check(
            baseline_sec=run_sec,
            encode_sec=_number(method, "encode_sec", f"замер.methods.{name}"),
            rss_delta_mb=_number(method, "rss_delta_mb", f"замер.methods.{name}"),
        )
    return budgets


def _test_numbers(method: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Метрики метода на тестовой половине набора.

    Args:
        method: Запись метода.
        name: Имя метода — для сообщения об ошибке.

    Returns:
        Объект с метриками ``test``.
    """
    return _mapping(method.get("test"), f"замер.methods.{name}.test")


def gate_verdict(measurement: Mapping[str, Any]) -> GateVerdict:
    """Вердикт гейта по снимку замера.

    Критерии — из `D-37`, в том же порядке: точность подтверждений, полнота
    флага, выигрыш отбора, бюджет. Критерий без данных не считается
    выполненным и не выдаётся за ноль: он помечается «не измерено».

    Args:
        measurement: Снимок замера.

    Returns:
        Вердикт с четырьмя критериями.

    Raises:
        ValueError: Снимок неполон — вердикт по нему был бы выдумкой.
    """
    methods = _methods(measurement)
    baseline = _mapping(measurement.get("baseline"), "замер.baseline")
    run_sec = _number(baseline, "run_sec", "замер.baseline")

    lost = {
        name: _number(_test_numbers(method, name), "confirmations_lost", f"замер.{name}.test")
        for name, method in methods.items()
    }
    recalls = {
        name: _number(_test_numbers(method, name), "recall", f"замер.{name}.test")
        for name, method in methods.items()
    }
    budgets = _budgets(methods, run_sec)

    confirmations = Criterion(
        key="confirmations",
        title="точность подтверждений не падает",
        measured=True,
        passed=all(value == 0 for value in lost.values()),
        detail="понижений честных подтверждений: " + _join_numbers(lost, 0, ""),
    )
    flag = Criterion(
        key="flag_recall",
        title="полнота флага «не по месту»",
        measured=True,
        passed=all(value >= MIN_FLAG_RECALL for value in recalls.values()),
        detail="полнота флага: " + _join_numbers(recalls, 2, f" (порог {_ru(MIN_FLAG_RECALL)})"),
    )

    ranking_raw = measurement.get("ranking")
    ranking_data = _mapping(ranking_raw, "замер.ranking") if ranking_raw is not None else {}
    queries = ranking_data.get("queries")
    gain = ranking_data.get("ndcg_gain")
    measured_ranking = (
        isinstance(queries, int)
        and not isinstance(queries, bool)
        and queries >= 2
        and isinstance(gain, (int, float))
        and not isinstance(gain, bool)
    )
    if measured_ranking and isinstance(gain, (int, float)):
        ranking = Criterion(
            key="ranking",
            title="выигрыш отбора кандидатов",
            measured=True,
            passed=float(gain) >= MIN_NDCG_GAIN,
            detail=f"выигрыш nDCG@5: {_ru(float(gain), 3)} (порог {_ru(MIN_NDCG_GAIN)})",
        )
    else:
        note = str(ranking_data.get("note") or "данных отбора нет")
        ranking = Criterion(
            key="ranking",
            title="выигрыш отбора кандидатов",
            measured=False,
            passed=False,
            detail=f"не измерено: {note}",
        )

    budget = Criterion(
        key="budget",
        title="время и память",
        measured=True,
        passed=all(item.within_time and item.within_memory for item in budgets.values()),
        detail=(
            "время: "
            + _join_numbers(
                {name: item.time_share * 100 for name, item in budgets.items()},
                2,
                f" % прогона (предел {_ru(MAX_TIME_SHARE * 100, 0)} %)",
            )
            + "; память: "
            + _join_numbers(
                {name: item.rss_delta_mb for name, item in budgets.items()},
                0,
                f" МБ (предел {MAX_RSS_MB} МБ)",
            )
        ),
    )

    criteria = (confirmations, flag, ranking, budget)
    measured_items = [item for item in criteria if item.measured]
    failed = [item.title for item in measured_items if not item.passed]
    passed = not failed
    summary = "выполнен" if passed else "не выполнен — " + "; ".join(failed)

    return GateVerdict(passed=passed, criteria=criteria, summary=summary)


def verdict_markdown(measurement: Mapping[str, Any]) -> str:
    """Блок вердикта для `docs/decisions.md`.

    Блок собирается из снимка замера, а не пишется руками: иначе числа в
    решении разойдутся с числами в замере, и никто не заметит.

    Args:
        measurement: Снимок замера.

    Returns:
        Markdown-блок между маркерами.
    """
    methods = _methods(measurement)
    baseline = _mapping(measurement.get("baseline"), "замер.baseline")
    run_sec = _number(baseline, "run_sec", "замер.baseline")
    budgets = _budgets(methods, run_sec)
    verdict = gate_verdict(measurement)

    lines = [
        MARKER_BEGIN,
        "| Метод | Полнота флага (тест) | Ложных понижений | Время слоя | Память |",
        "|---|---|---|---|---|",
    ]
    for name, method in methods.items():
        test = _test_numbers(method, name)
        budget = budgets[name]
        recall = _number(test, "recall", f"замер.{name}.test")
        lost = _number(test, "confirmations_lost", f"замер.{name}.test")
        seconds = _number(method, "encode_sec", f"замер.{name}")
        lines.append(
            f"| `{name}` | {_ru(recall)} | {lost:.0f} "
            f"| {_ru(seconds, 1)} с ({_ru(budget.time_share * 100)} % прогона "
            f"из {_ru(run_sec, 1)} с) | {_ru(budget.rss_delta_mb, 0)} МБ |"
        )
    lines.append("")
    lines.append(f"**Вердикт замера:** {verdict.summary}.")
    lines.append(MARKER_END)
    return "\n".join(lines)
