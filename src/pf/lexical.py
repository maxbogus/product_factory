"""Лексическая близость: леммы, точные термины, числа и распознавание слов.

Проверка оснований сравнивала текст буквально. Из-за этого:

- «абоненты» из источника не покрывали «клиентов» в утверждении;
- число ``77,3`` не совпадало с ``77,26``, хотя это округление;
- «удержание не падает» проходило проверку как «удержание падает»:
  слово «не» выброшено как стоп-слово;
- незнакомое слово молча снижало покрытие, и было непонятно, что чинить.

Модуль отвечает за лексическую часть сравнения и ничего не решает о
статусе основания: решение принимает ``pf.ground``. Здесь только факты о
словах — какие они, к какому понятию относятся, распознались ли они.

Три уровня распознавания слова (от дешёвого к дорогому):

1. **Смешанная раскладка.** ``удержаниe`` с латинской ``e`` — это
   ``удержание``; гомоглифы заменяются, слово находится в словаре.
2. **Морфология.** ``pymorphy3`` знает слово (``is_known``) и даёт лемму.
3. **Эвристика.** Транслитерация (``megafon`` ↔ ``мегафон``) и опечатка
   (расстояние Дамерау—Левенштейна ≤ 1 против словаря лексикона).

Если не сработало ничего, слово получает статус ``OOV`` и попадает в
очередь распознания (``pf.lexicon``), чтобы словарь можно было расширить
осознанно, а не «на глаз».
"""

from __future__ import annotations

import functools
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from pf import synonyms as synonyms_module
from pf.corpus import lemmatize, tokenize

#: Слово: буквы и цифры, без служебных символов.
_WORD = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
#: Число: запятая или точка как разделитель, пробелы в разрядах.
#: Границы обязательны: иначе «5G» даст число 5, а «2024-й» — 2024.
_NUMBER = re.compile(
    r"(?<![0-9a-zа-яё])-?\d+(?:[ \u00a0\u202f]\d{3})*(?:[.,]\d+)?(?![0-9a-zа-яё])",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(не|нет|без|никогда|not|no|never|without)\b",
    re.IGNORECASE,
)
_CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
_LATIN = re.compile(r"[a-z]", re.IGNORECASE)

#: Латинские буквы, которые выглядят как кириллические. Заменяются только
#: в слове, где уже есть кириллица: иначе «opex» превратился бы в «орех».
HOMOGLYPHS: dict[str, str] = {
    "a": "а",
    "b": "в",
    "c": "с",
    "e": "е",
    "h": "н",
    "k": "к",
    "m": "м",
    "o": "о",
    "p": "р",
    "t": "т",
    "x": "х",
    "y": "у",
}

#: Таблица транслитерации: русская буква → латинские варианты.
#: Нужна, чтобы «megafon» и «мегафон» дали один ключ: это имена и бренды,
#: для которых перевод бессмысленен, а написание плавает.
_TRANSLIT: dict[str, str] = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "i",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "c",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "i",
    "ь": "",
    "э": "e",
    "ю": "u",
    "я": "a",
}


class TokenStatus(StrEnum):
    """Что известно о слове.

    Различие принципиально: «знаю слово» и «догадался о слове» — разные
    основания. Догадка может участвовать в покрытии, но не даёт права
    считать точный термин подтверждённым.
    """

    #: Слово есть в словаре лексикона.
    KNOWN = "known"
    #: Слово распознано только морфологией: лемма известна словарю.
    RESOLVED_MORPH = "resolved_morph"
    #: Слово распознано эвристикой: гомоглифы, транслит, опечатка.
    RESOLVED_HEURISTIC = "resolved_heuristic"
    #: Слово не распознано — очередь распознания.
    OOV = "oov"

    @property
    def is_recognized(self) -> bool:
        """Считается ли слово распознанным для покрытия."""
        return self is not TokenStatus.OOV


@dataclass(frozen=True, slots=True)
class TokenResolution:
    """Результат распознавания одного слова.

    Attributes:
        token: Исходное слово в нижнем регистре.
        normalized: Слово после нормализации раскладки.
        lemma: Лемма, если она известна.
        status: Что известно о слове.
        reason: Почему получен этот статус — для журнала и очереди.
        candidate: Ближайшее слово словаря, если сработала эвристика.
        distance: Расстояние до кандидата.
    """

    token: str
    normalized: str
    lemma: str
    status: TokenStatus
    reason: str = ""
    candidate: str = ""
    distance: int | None = None


@dataclass(frozen=True, slots=True)
class Overlap:
    """Смысловое пересечение утверждения и абзаца источника.

    Attributes:
        coverage: Доля значимых слов утверждения, покрытых абзацем.
        missing_terms: Точные термины утверждения, которых нет в абзаце.
        oov_tokens: Нераспознанные слова утверждения — в очередь.
        numbers_ok: Все числа утверждения нашлись в абзаце с допуском.
        numbers_detail: Пояснение по числу, которое не сошлось.
        negation_ok: Отрицание не появилось там, где его не было.
        matched: Итог: утверждение покрывается абзацем.
    """

    coverage: float = 0.0
    missing_terms: list[str] = field(default_factory=list)
    oov_tokens: list[str] = field(default_factory=list)
    numbers_ok: bool = True
    numbers_detail: str = ""
    negation_ok: bool = True
    matched: bool = False


def normalize_homoglyphs(word: str) -> str:
    """Заменить латинские гомоглифы кириллицей в смешанном слове.

    Args:
        word: Слово как его написал человек или модель.

    Returns:
        Слово с кириллицей вместо похожих латинских букв. Чисто латинские
        и чисто кириллические слова не меняются: иначе «opex» стал бы
        «орех», а «mts» — «мтс» без всякого основания.
    """
    has_cyrillic = bool(_CYRILLIC.search(word))
    has_latin = bool(_LATIN.search(word))
    if not (has_cyrillic and has_latin):
        return word
    return "".join(HOMOGLYPHS.get(char.lower(), char) for char in word)


def translit_key(word: str) -> str:
    """Транслитерационный ключ слова — общий для двух алфавитов.

    Args:
        word: Слово в нижнем регистре.

    Returns:
        Латинский ключ: «мегафон» и «megafon» дают одно и то же.
    """
    lowered = unicodedata.normalize("NFC", word.strip().lower())
    key = "".join(_TRANSLIT.get(char, char) for char in lowered)
    # Варианты записи плавают: «kh»/«h», «c»/«k», «y»/«i», удвоения.
    key = key.replace("kh", "h").replace("c", "k").replace("y", "i")
    return re.sub(r"(.)\1+", r"\1", key)


def damerau_distance(left: str, right: str) -> int:
    """Расстояние Дамерау—Левенштейна (правки и перестановки).

    Используется только для коротких слов словаря: это дешёвая проверка
    «похоже на опечатку», а не поиск по всему корпусу.

    Args:
        left: Первое слово.
        right: Второе слово.

    Returns:
        Минимальное число правок.
    """
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous_previous: list[int] = []
    previous = list(range(len(right) + 1))
    for i in range(1, len(left) + 1):
        current = [i] + [0] * len(right)
        for j in range(1, len(right) + 1):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            if i > 1 and j > 1 and left[i - 1] == right[j - 2] and left[i - 2] == right[j - 1]:
                current[j] = min(current[j], previous_previous[j - 2] + 1)
        previous_previous, previous = previous, current
    return previous[len(right)]


@functools.lru_cache(maxsize=50_000)
def lemma(word: str) -> str:
    """Лемма слова с ограниченным кэшем.

    ``pf.corpus.lemmatize`` кэширует без предела: на большом корпусе с
    незнакомыми словами кэш растёт бесконечно. Здесь предел есть.

    Args:
        word: Слово в нижнем регистре.

    Returns:
        Нормальная форма слова.
    """
    normalized = normalize_homoglyphs(word.strip().lower())
    if not normalized:
        return ""
    return lemmatize(normalized)


def is_known_word(word: str) -> bool:
    """Знает ли морфологический словарь это слово.

    Кэш обязателен: проверка вызывает функцию на каждом значимом слове
    утверждения, а ``pymorphy3.parse`` — не бесплатный.

    Args:
        word: Слово в нижнем регистре.

    Returns:
        Слово есть в словаре pymorphy3 (``is_known``).
    """
    return _is_known_cached(normalize_homoglyphs(word.strip().lower()))


@functools.lru_cache(maxsize=50_000)
def _is_known_cached(word: str) -> bool:
    """Закэшированная проверка «слово есть в словаре».

    Args:
        word: Нормализованное слово в нижнем регистре.

    Returns:
        Слово есть в словаре pymorphy3.
    """
    from pf.corpus import get_morph

    if not word:
        return False
    parsed = get_morph().parse(word)
    return bool(parsed) and parsed[0].is_known


def language(text: str) -> str:
    """Язык текста по алфавиту букв.

    Args:
        text: Любой текст.

    Returns:
        ``ru``, ``en`` либо ``other``, если букв мало или алфавит смешан.
    """
    cyrillic = len(_CYRILLIC.findall(text))
    latin = len(_LATIN.findall(text))
    total = cyrillic + latin
    if total < 4:
        return "other"
    if cyrillic / total >= 0.6:
        return "ru"
    if latin / total >= 0.6:
        return "en"
    return "other"


def has_negation(text: str) -> bool:
    """Есть ли в тексте отрицание.

    Args:
        text: Проверяемый текст.

    Returns:
        Найдено ли отрицание.
    """
    return bool(_NEGATION.search(text))


# ── Направление утверждения ────────────────────────────────────────────────
# Правило одно на репозиторий: противоречия в фактах, вердикт гипотез и
# проверка полярности плана раньше держали по своей копии этих списков, и
# копии расходились. Здесь — единственное место.

#: Слова о падении показателя.
DECLINE_MARKERS: tuple[str, ...] = (
    "падает",
    "снижается",
    "уменьшается",
    "сокращается",
    "упало",
    "снизилось",
    "снизился",
)

#: Слова о неизменности показателя.
STABLE_MARKERS: tuple[str, ...] = (
    "не изменилось",
    "не изменился",
    "не меняется",
    "без изменений",
    "осталось прежним",
)

#: Слова о росте показателя.
GROWTH_MARKERS: tuple[str, ...] = (
    "растет",
    "выросло",
    "вырос",
    "увеличилось",
    "увеличился",
)

#: Слова, которые встречаются почти в любом утверждении: показатель они не
#: определяют и общим предметом факты не делают.
GENERIC_WORDS: frozenset[str] = frozenset(
    {
        "клиент",
        "клиентов",
        "клиенты",
        "квартал",
        "квартала",
        "последний",
        "данные",
        "данных",
        "может",
        "могут",
    }
)


def polarity(text: str) -> str | None:
    """Направление, о котором говорит утверждение.

    Args:
        text: Текст утверждения.

    Returns:
        ``decline``, ``stable``, ``growth`` либо ``None``, если направление
        не названо.
    """
    normalized = text.lower().replace("ё", "е")
    if any(marker in normalized for marker in STABLE_MARKERS):
        return "stable"
    if any(marker in normalized for marker in DECLINE_MARKERS):
        return "decline"
    if any(marker in normalized for marker in GROWTH_MARKERS):
        return "growth"
    return None


def topic_words(text: str) -> set[str]:
    """Слова, которые называют предмет утверждения.

    Args:
        text: Текст утверждения.

    Returns:
        Множество значимых слов без служебных.
    """
    normalized = text.lower().replace("ё", "е")
    return set(re.findall(r"[а-я]{5,}", normalized)) - GENERIC_WORDS


def words(text: str) -> list[str]:
    """Значимые слова текста в форме лемм.

    Args:
        text: Исходный текст.

    Returns:
        Список лемм: так «абоненты» и «абонентов» становятся одним словом.
    """
    return [lemmatize(word) for word in tokenize(text)]


def significant_words(text: str) -> set[str]:
    """Значимые слова текста: без скобок, цифр и коротких служебных.

    Скобки отбрасываются: «удержание клиентов (МТС, абонентская база)» —
    это метрика про удержание, а база в скобках лишь уточняет, откуда
    взято число. Если оставить скобки, подмена показателя не видна:
    метрика про удержание подтверждалась бы словом «база» из скобок.

    Args:
        text: Название метрики или текст строки.

    Returns:
        Множество слов длиной от четырёх букв.
    """
    without_notes = re.sub(r"\([^)]*\)", " ", text)
    return set(re.findall(r"[a-zа-яё]{4,}", without_notes.lower().replace("ё", "е")))


def stems(text: str) -> set[str]:
    """Основы значимых слов: показатель сравнивается без словоформ.

    «Абонентская» и «абонентов» — одно понятие, но леммы у них разные
    («абонентский» и «абонент»), поэтому сравнение по леммам давало бы
    ложный отказ. Шесть букв — та же мера, что в проверке критериев
    успеха: разные показатели («удержание» и «база») остаются разными.

    Скобки отбрасываются так же, как в :func:`significant_words`. Правило
    живёт здесь одно на весь репозиторий: своя копия регэкспа, написанная
    в другой функции, уже пропускала подмену показателя по слову из
    скобок.

    Args:
        text: Текст утверждения, название метрики или строка таблицы.

    Returns:
        Множество основ значимых слов.
    """
    return {word[:6] for word in significant_words(text) if len(word) >= 5}


#: Основы служебных слов показателя. «Изменение за период» — форма расчёта, а
#: не понятие: в прогоне `cloud-deepseek-2` расчёт по ключевой ставке делил с
#: базой сравнения ровно одно слово — «период» — и считался расчётом того же
#: показателя. Из-за этого верные числа `7,39 млн` и `84,62 %` отклонялись
#: словами «не совпадает с расчётом по показателю «Ключевая ставка…»».
#:
#: Основы записаны так, как их отдаёт :func:`stems` (он усекает до шести
#: букв), иначе список не сработает.
GENERIC_STEMS: frozenset[str] = frozenset(
    {"период", "измене", "значен", "уровен", "данные", "показа"}
)


def indicator_stems(text: str) -> set[str]:
    """Основы показателя: значимые слова без служебных.

    Служебные слова описывают форму записи («изменение за период», «уровень»)
    и не говорят, о каком показателе речь. Пока они участвовали в сравнении,
    любой расчёт с таким словом в названии подходил под любое утверждение, где
    оно встретилось.

    Правило одно на весь репозиторий: им пользуются сверка чисел плана
    (``Workflow._check_plan_numbers``) и сверка показателя основания с
    показателем вывода, поэтому список живёт здесь, а не в двух местах.

    Args:
        text: Текст утверждения, название метрики или строка таблицы.

    Returns:
        Множество основ значимых слов без служебных.
    """
    return stems(text) - GENERIC_STEMS


def numbers_in(text: str) -> list[float]:
    """Вытащить числа из текста.

    Поддержаны русская запятая, точка, пробелы в разрядах и знак минус.

    Args:
        text: Исходный текст.

    Returns:
        Список чисел в порядке появления.
    """
    found: list[float] = []
    for raw in _NUMBER.findall(text):
        cleaned = raw.replace("\u00a0", "").replace("\u202f", "").replace(" ", "")
        cleaned = cleaned.replace(",", ".")
        try:
            found.append(float(cleaned))
        except ValueError:
            continue
    return found


def number_tolerance(
    value: float,
    *,
    abs_tolerance: float = 0.1,
    rel_tolerance: float = 5e-4,
) -> float:
    """Допуск для сравнения одного числа.

    Абсолютный допуск нужен для округления («77,3» вместо «77,26»), но
    для малых величин он опасен: 0,1 — это половина от 0,2. Поэтому
    абсолютная часть включается только начиная с единицы, а для остальных
    работает относительная.

    Args:
        value: Проверяемое число.
        abs_tolerance: Абсолютный допуск для чисел от единицы.
        rel_tolerance: Относительный допуск.

    Returns:
        Допуск, в пределах которого расхождение считается округлением.
    """
    magnitude = abs(value)
    relative = magnitude * rel_tolerance
    if magnitude >= 1:
        return max(abs_tolerance, relative)
    return relative


def numbers_match(
    claim_numbers: Sequence[float],
    source_numbers: Sequence[float],
    *,
    abs_tolerance: float = 0.1,
    rel_tolerance: float = 5e-4,
) -> tuple[bool, str]:
    """Совпадают ли числа утверждения с числами источника.

    Args:
        claim_numbers: Числа утверждения.
        source_numbers: Числа источника.
        abs_tolerance: Абсолютный допуск.
        rel_tolerance: Относительный допуск.

    Returns:
        Пара ``(совпало, пояснение)``. Пояснение заполняется только при
        расхождении — это причина отказа для журнала.
    """
    for value in claim_numbers:
        tolerance = number_tolerance(
            value, abs_tolerance=abs_tolerance, rel_tolerance=rel_tolerance
        )
        if any(abs(value - candidate) <= tolerance for candidate in source_numbers):
            continue
        return False, f"число {value:g} не совпадает с источником (допуск {tolerance:g})"
    return True, ""


def is_exact_term(token: str, groups: Mapping[str, str] | None = None) -> bool:
    """Является ли слово точным термином.

    Точный термин нельзя заменять синонимом или «похожим» словом: подмена
    бренда или единицы измерения меняет смысл утверждения.

    Args:
        token: Слово в исходном написании.
        groups: Отображение синонимов; слово из группы точным не считается.

    Returns:
        Термин обязан совпасть с источником буквально (с точностью до
        словоформы).
    """
    stripped = token.strip()
    if not stripped:
        return False
    # Числа проверяются отдельно и с допуском.
    if any(char.isdigit() for char in stripped):
        return True
    if any(char in stripped for char in "%₽$€"):
        return True
    # Аббревиатура: «МТС», «ARPU».
    letters = [char for char in stripped if char.isalpha()]
    if len(letters) >= 2 and all(char.isupper() for char in letters):
        return True
    table = groups if groups is not None else {}
    if stripped.lower() in table:
        return False
    # Латиница — название или термин; перевод её не заменяет.
    return bool(_LATIN.search(stripped) and not _CYRILLIC.search(stripped))


def _strong_terms() -> frozenset[str]:
    """Леммы сильных доменных терминов.

    Returns:
        Множество лемм; импорт ленивый, чтобы не тянуть корпус при
        импорте модуля.
    """
    from pf.corpus import strong_domain_lemmas

    return strong_domain_lemmas()


def _lexicon_groups() -> dict[str, str]:
    """Полное отображение синонимов: встроенный набор и файл словаря.

    Импорт ленивый: ``pf.lexicon`` сам импортирует этот модуль, поэтому
    на уровне модуля циклический импорт недопустим.

    Returns:
        Словарь «лемма → каноническая лемма».
    """
    from pf import lexicon

    return lexicon.groups()


def recognize(
    token: str,
    groups: Mapping[str, str] | None = None,
    *,
    max_distance: int = 1,
    min_length: int = 6,
) -> TokenResolution:
    """Распознать одно слово всеми доступными способами.

    Порядок попыток фиксирован: от дешёвой и точной к дорогой и
    приблизительной. Эвристика не должна перебивать морфологию.

    Args:
        token: Слово в исходном написании.
        groups: Отображение синонимов (лемма → каноническая лемма).
        max_distance: Максимальное расстояние для опечатки.
        min_length: Минимальная длина слова для поиска опечатки: короткие
            слова слишком легко «похожи».

    Returns:
        Результат распознавания.
    """
    lowered = token.strip().lower()
    normalized = normalize_homoglyphs(lowered)
    table = groups if groups is not None else synonyms_module.build_map()

    if normalized != lowered and is_known_word(normalized):
        return TokenResolution(
            token=lowered,
            normalized=normalized,
            lemma=lemma(normalized),
            status=TokenStatus.RESOLVED_HEURISTIC,
            reason="нормализованы гомоглифы",
        )

    if is_known_word(normalized):
        base = lemma(normalized)
        if base in table:
            return TokenResolution(
                token=lowered,
                normalized=normalized,
                lemma=base,
                status=TokenStatus.KNOWN,
                reason="слово в словаре синонимов",
            )
        return TokenResolution(
            token=lowered,
            normalized=normalized,
            lemma=base,
            status=TokenStatus.RESOLVED_MORPH,
            reason="слово распознано морфологией",
        )

    # Латиница и кириллица одного названия: «megafon» ↔ «мегафон».
    key = translit_key(normalized)
    for candidate in table:
        if translit_key(candidate) == key:
            return TokenResolution(
                token=lowered,
                normalized=normalized,
                lemma=candidate,
                status=TokenStatus.KNOWN,
                reason="слово найдено транслитерацией",
                candidate=candidate,
                distance=0,
            )

    # Опечатка: ищем ближайшее слово словаря. Только для длинных слов и
    # только один кандидат — иначе «тариф» склеится с «график».
    if len(normalized) >= min_length:
        best: tuple[int, str] | None = None
        for candidate in table:
            if len(candidate) < min_length:
                continue
            distance = damerau_distance(normalized, candidate)
            if distance == 0:
                continue
            if distance <= max_distance and (best is None or distance < best[0]):
                best = (distance, candidate)
        if best is not None:
            distance, candidate = best
            return TokenResolution(
                token=lowered,
                normalized=normalized,
                lemma=candidate,
                status=TokenStatus.RESOLVED_HEURISTIC,
                reason=f"распознано как «{candidate}», расстояние {distance}",
                candidate=candidate,
                distance=distance,
            )

    return TokenResolution(
        token=lowered,
        normalized=normalized,
        lemma=normalized,
        status=TokenStatus.OOV,
        reason="слово не распознано ни словарём, ни морфологией",
    )


def _lemmas_of(text: str) -> set[str]:
    """Леммы значимых слов текста.

    Кэш ``lemma`` делает повторные слова почти бесплатными: это важно,
    потому что проверка проходит по всем абзацам статьи.

    Args:
        text: Исходный текст.

    Returns:
        Множество лемм.
    """
    return {lemma(word) for word in tokenize(text)}


def _canonical_set(lemmas: set[str], groups: Mapping[str, str]) -> set[str]:
    """Привести леммы к каноническим понятиям.

    Args:
        lemmas: Леммы слов.
        groups: Отображение синонимов.

    Returns:
        Множество канонических лемм.
    """
    return {groups.get(item, item) for item in lemmas}


def shares_words(claim: str, paragraph: str) -> bool:
    """Есть ли у утверждения и абзаца общие значимые слова.

    Дешёвый предфильтр перед морфологическим разбором: длинная статья —
    это тысячи абзацев, и разбирать каждый нет смысла.

    Args:
        claim: Текст утверждения.
        paragraph: Текст абзаца.

    Returns:
        Есть ли пересечение по словам.
    """
    return bool(set(tokenize(claim)) & set(tokenize(paragraph)))


def exact_terms_in(text: str, groups: Mapping[str, str] | None = None) -> list[str]:
    """Точные термины текста, которые обязаны найтись в источнике.

    Args:
        text: Исходный текст.
        groups: Отображение синонимов.

    Returns:
        Список терминов в том написании, в котором они пришли.
    """
    table = groups if groups is not None else _lexicon_groups()
    strong = _strong_terms()
    terms: list[str] = []
    for raw in _WORD.findall(text):
        # Числа проверяются отдельно и с допуском: в словоформах их нет,
        # и поиск по леммам всегда давал бы «термин 2024 не найден».
        if all(char.isdigit() for char in raw):
            continue
        if not is_exact_term(raw, table) and lemma(raw) not in strong:
            continue
        terms.append(raw)
    return terms


def _term_present(
    term: str,
    paragraph_lemmas: set[str],
    paragraph_canonical: set[str],
    groups: Mapping[str, str],
) -> bool:
    """Есть ли точный термин в абзаце.

    Синонимы здесь не помогают: точный термин обязан совпасть понятием
    буквально, иначе «МТС» подменится «МегаФоном».

    Args:
        term: Точный термин.
        paragraph_lemmas: Леммы слов абзаца (посчитаны заранее).
        paragraph_canonical: Канонические леммы абзаца.
        groups: Отображение синонимов — нужен, чтобы не считать термином
            слово, у которого есть синоним.

    Returns:
        Термин найден в абзаце.
    """
    target = lemma(term)
    if not target:
        return True
    if is_exact_term(term, groups):
        return target in paragraph_lemmas
    canonical = groups.get(target, target)
    return canonical in paragraph_canonical


def overlap(
    claim: str,
    paragraph: str,
    *,
    groups: Mapping[str, str] | None = None,
    abs_tolerance: float = 0.1,
    rel_tolerance: float = 5e-4,
    min_overlap: float = 0.8,
) -> Overlap:
    """Сравнить утверждение с абзацем источника по смыслу слов.

    Args:
        claim: Текст утверждения.
        paragraph: Текст абзаца-источника.
        groups: Отображение синонимов. По умолчанию — встроенный набор.
        abs_tolerance: Абсолютный допуск для чисел.
        rel_tolerance: Относительный допуск для чисел.
        min_overlap: Порог покрытия значимых слов.

    Returns:
        Пересечение: покрытие, недостающие термины, нераспознанные слова,
        результат сверки чисел и отрицания.
    """
    table = groups if groups is not None else _lexicon_groups()
    claim_lemmas = _lemmas_of(claim)
    claim_words = _canonical_set(claim_lemmas, table)
    paragraph_lemmas = _lemmas_of(paragraph)
    paragraph_words = _canonical_set(paragraph_lemmas, table)

    coverage = 0.0
    if claim_words:
        coverage = len(claim_words & paragraph_words) / len(claim_words)

    missing_terms = [
        term
        for term in exact_terms_in(claim, table)
        if not _term_present(term, paragraph_lemmas, paragraph_words, table)
    ]
    # Распознавание слов утверждения — источник очереди распознания.
    # Разбираем только утверждение: слова источника не нужно «понимать»,
    # их достаточно процитировать.
    oov_tokens = sorted(
        {
            resolution.token
            for resolution in (recognize(word, table) for word in tokenize(claim))
            if resolution.status is TokenStatus.OOV
        }
    )

    numbers_ok, numbers_detail = numbers_match(
        numbers_in(claim),
        numbers_in(paragraph),
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
    )
    # Проверяем только появление отрицания: «не» в длинном абзаце может
    # относиться к другой мысли, а вот добавленное моделью «не» меняет
    # смысл утверждения на противоположный.
    negation_ok = not (has_negation(claim) and not has_negation(paragraph))

    matched = coverage >= min_overlap and not missing_terms and numbers_ok and negation_ok
    return Overlap(
        coverage=coverage,
        missing_terms=missing_terms,
        oov_tokens=oov_tokens,
        numbers_ok=numbers_ok,
        numbers_detail=numbers_detail,
        negation_ok=negation_ok,
        matched=matched,
    )
