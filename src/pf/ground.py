"""Проверка оснований: цитата должна дословно существовать в источнике.

Главный риск агента, который работает с языковой моделью, — правдоподобная
выдумка. Здесь это закрывается проверкой, а не просьбой к модели «не выдумывай».

Правило простое и проверяемое:

1. цитата из ответа модели должна **дословно** встречаться в исходном тексте
   статьи (сравнение по нормализованному тексту);
2. если цитата не найдена — основание отбрасывается, и поле получает статус
   «нет данных» вместо правдоподобного значения.

Смысл: модель может придумать *связный* текст, но не может придумать
*цитату из конкретной статьи*, не совпадающую с её содержимым. Поэтому
проверка идёт по цитате, а не по смыслу ответа.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING

from pf.config import settings
from pf.entities import KIND_AMBIGUOUS, KIND_MISMATCH
from pf.entities import identity as entity_identity
from pf.lexical import (
    Overlap,
    language,
    mask_numbers,
    normalize_homoglyphs,
    numbers_in,
    numbers_match,
    shares_words,
)
from pf.lexical import (
    overlap as lexical_overlap,
)
from pf.verdicts import ENTITY_AMBIGUOUS, ENTITY_MISMATCH

if TYPE_CHECKING:
    from pf.canonical import SupportScore

#: Нормализация текста перед сравнением: приводим к NFC, схлопываем пробелы
#: и убираем мягкие переносы. Регистр НЕ меняем — регистр может быть частью
#: смысла термина, но для поиска цитаты он мешает, поэтому сравниваем
#: в нижнем регистре.
_WHITESPACE = re.compile(r"\s+")
_HYPHEN = re.compile(r"[-‐‑‒–—―]")
#: Слово целиком: гомоглифы заменяются внутри слова, а не в тексте —
#: иначе латинское «opex» в русской фразе стало бы «орех».
_WORD = re.compile(r"\S+")

#: Кавычки — оформление цитаты, а не её содержание: «МТС» и МТС — одно слово.
#: Запятые и точки не снимаются: «82,5» после чистки стало бы «82 5».
_QUOTES = str.maketrans("", "", "«»„“”‚‘’" + chr(34) + chr(39))

#: Доля покрытия, выше которой недословная цитата считается пересказом, а не
#: выдумкой. Замер на `demo-v12`: пересказ строки таблицы — 1.00, выдуманная
#: цитата — 0.00. Порог посередине, чтобы решение не зависело от округления.
PARAPHRASE_COVERAGE = 0.6

#: Виды отказа цитаты, после которых утверждение проверяется полями. Выдумки
#: здесь нет: иначе выдуманная цитата стала бы способом пройти проверку.
QUOTE_FALLBACK_KINDS = frozenset({"too_short", "paraphrase", "number_mismatch"})


class SupportStatus(StrEnum):
    """Статус подтверждения основания.

    Три состояния принципиально различаются: «нашли», «нашли, но сомневаемся»
    и «не нашли». Без этого «нет данных» невозможно отличить от нуля.
    """

    #: Цитата найдена в источнике дословно.
    CONFIRMED = "confirmed"
    #: Цитата близка к источнику, но не совпала точно — нужен просмотр.
    NEEDS_REVIEW = "needs_review"
    #: Цитата не найдена вовсе — основание отброшено.
    NO_DATA = "no_data"

    @property
    def title_ru(self) -> str:
        """Человекочитаемое название статуса."""
        return {
            SupportStatus.CONFIRMED: "подтверждено",
            SupportStatus.NEEDS_REVIEW: "требует проверки",
            SupportStatus.NO_DATA: "нет данных",
        }[self]


@dataclass(frozen=True, slots=True)
class Support:
    """Основание утверждения: цитата и её происхождение.

    Attributes:
        quote: Цитата из ответа модели.
        source_id: Идентификатор источника (``раздел/Заголовок``).
        paragraph_index: Номер абзаца в источнике, если найден.
        status: Статус подтверждения.
        detail: Пояснение для человека — что именно не сошлось.
        kind: Вид проверки или отказа — машинное имя, по которому отчёт
            группирует причины, не разбирая русский текст: ``exact``,
            ``number_tolerance``, ``too_short``, ``paraphrase``,
            ``number_mismatch``, ``fabrication``, ``language``.
        oov: Нераспознанные слова утверждения — их место в очереди
            распознания, а не в подозрениях на выдумку.
    """

    quote: str
    source_id: str
    status: SupportStatus
    paragraph_index: int | None = None
    detail: str = ""
    oov: tuple[str, ...] = ()
    kind: str = ""

    @property
    def is_usable(self) -> bool:
        """Можно ли опереться на это основание."""
        return self.status is SupportStatus.CONFIRMED

    @property
    def citation(self) -> str:
        """Человекочитаемая ссылка на источник.

        Якорь один: если в идентификаторе уже есть ``#rowN`` или
        ``#pN``, второй не добавляется. Иначе ссылка выходила бы
        «project/файл.csv#row3#p3» — такой якорь не открывается.
        """
        if self.paragraph_index is None:
            return self.source_id
        if "#" in self.source_id:
            return self.source_id
        return f"{self.source_id}#p{self.paragraph_index}"


def mark_external(support: Support, *, reason: str = "") -> Support:
    """Ограничить статус основания внешним источником.

    Внешний источник нельзя подтвердить: неизвестно, что за страницу
    отдала выдача. Дословность цитаты при этом проверяется, поэтому
    «нашли дословно» превращается в «требует проверки», а не в отказ.

    Args:
        support: Результат обычной проверки цитаты.
        reason: Пояснение для человека. По умолчанию — про внешний источник.
            Сырая выгрузка из хранилища знаний ограничена тем же правилом,
            но причина у неё своя.

    Returns:
        То же основание со статусом не выше ``needs_review``.
    """
    if support.status is not SupportStatus.CONFIRMED:
        return support
    return replace(
        support,
        status=SupportStatus.NEEDS_REVIEW,
        detail=reason or "внешний источник: цитата найдена дословно, требует проверки",
    )


#: Насколько вердикт «хуже» для основания. Худший вердикт побеждает: если
#: на одну строку источника сослались и подтверждённой цитатой, и выдумкой,
#: основание считается отклонённым — иначе повтор верной цитаты прятал бы
#: рядом стоящую выдумку.
_SEVERITY: dict[SupportStatus, int] = {
    SupportStatus.CONFIRMED: 0,
    SupportStatus.NEEDS_REVIEW: 1,
    SupportStatus.NO_DATA: 2,
}

#: Имена групп вердиктов — так они попадают в журнал шага.
GROUND_BUCKETS: tuple[str, ...] = ("confirmed", "rejected", "needs_review")


def citation_key(source_id: str) -> str:
    """Ключ основания: путь и первый якорь, без повторов.

    Ссылка приходит как ``приложение/файл.csv#row3#p3`` — якорь может
    продублироваться. Для счёта оснований это одна строка таблицы, а не
    две, поэтому ключ у них общий. Без общего ключа отчёт печатал
    «уникальных оснований 2» и «проверено 8» в одном документе.

    Args:
        source_id: Ссылка на источник из ответа модели.

    Returns:
        Нормализованный ключ основания.
    """
    path, _, fragment = source_id.partition("#")
    parts = [part for part in fragment.split("#") if part]
    return f"{path}#{parts[0]}" if parts else path


@dataclass(frozen=True, slots=True)
class GroundTally:
    """Счёт оснований по уникальным ключам.

    Attributes:
        checked: Сколько уникальных оснований проверялось.
        confirmed: Сколько из них подтверждено.
        rejected: Сколько отклонено.
        needs_review: Сколько ждёт проверки человеком.
    """

    checked: int = 0
    confirmed: int = 0
    rejected: int = 0
    needs_review: int = 0


def _worst_by_key(verdicts: Iterable[tuple[str, SupportStatus]]) -> dict[str, SupportStatus]:
    """Худший вердикт по каждому ключу основания.

    Args:
        verdicts: Пары «ключ основания — вердикт».

    Returns:
        Словарь «ключ — худший вердикт» в порядке появления ключей.
    """
    worst: dict[str, SupportStatus] = {}
    for key, status in verdicts:
        current = worst.get(key)
        if current is None or _SEVERITY[status] > _SEVERITY[current]:
            worst[key] = status
    return worst


def ground_buckets(verdicts: Iterable[tuple[str, SupportStatus]]) -> dict[str, list[str]]:
    """Ключи оснований, разложенные по вердиктам.

    Нужно, чтобы отчёт мог объединить шаги: у стратегии и фиксации решения
    одни и те же строки источника, и складывать их счётчики нельзя.

    Args:
        verdicts: Пары «ключ основания — вердикт».

    Returns:
        Словарь ``confirmed`` / ``rejected`` / ``needs_review`` со ключами.
    """
    names = {
        SupportStatus.CONFIRMED: "confirmed",
        SupportStatus.NO_DATA: "rejected",
        SupportStatus.NEEDS_REVIEW: "needs_review",
    }
    buckets: dict[str, list[str]] = {name: [] for name in GROUND_BUCKETS}
    for key, status in _worst_by_key(verdicts).items():
        buckets[names[status]].append(key)
    return buckets


def tally_grounds(verdicts: Iterable[tuple[str, SupportStatus]]) -> GroundTally:
    """Посчитать основания по уникальным ключам.

    Повторная ссылка на ту же строку источника — одно основание. Именно
    этого не хватало: в прогоне ``demo-v9`` стратегия и решение объявляли
    по четыре основания из двух строк, и документ называл их и «8 из 8»,
    и «уникальных 2» одновременно.

    Args:
        verdicts: Пары «ключ основания — вердикт».

    Returns:
        Счёт по уникальным ключам, худший вердикт побеждает.
    """
    buckets = ground_buckets(verdicts)
    return GroundTally(
        checked=sum(len(keys) for keys in buckets.values()),
        confirmed=len(buckets["confirmed"]),
        rejected=len(buckets["rejected"]),
        needs_review=len(buckets["needs_review"]),
    )


#: Из чего состоит правило счёта оснований вывода. Имя пишется в журнал
#: вместе со статусом: без него унаследованный статус не отличить от
#: текущего. Части, а не номер версии: когда правило теряет часть, разница
#: называется, а не угадывается.
#:
#: * ``unique-grounds`` — повторная ссылка на ту же строку источника не
#:   добавляет основания (`D-21`);
#: * ``calc-line-demotion`` — посчитанное значение в строке расчёта не
#:   считается цитатой (`D-36`);
#: * ``indicator-demotion`` — основание не про показатель вывода не считается
#:   подтверждённым (`D-41`, `D-46`);
#: * ``unsupported-number-cap`` — число плана без расчёта того же показателя
#:   не даёт «подкреплено» (`D-40`, следствие для вывода);
#: * ``user-input-channel`` — ввод пользователя не считается подтверждённым
#:   документом: «подкреплено» он не даёт, но и отказом не является (`D-49`).
CONCLUSION_RULE_PARTS: tuple[str, ...] = (
    "unique-grounds",
    "calc-line-demotion",
    "indicator-demotion",
    "unsupported-number-cap",
    "user-input-channel",
)

#: Имя правила счёта оснований вывода, которое действует сейчас.
CONCLUSION_RULE: str = "+".join(CONCLUSION_RULE_PARTS)


def conclusion_status(
    checked: int,
    confirmed: int,
    rejected: int,
    needs_review: int = 0,
    *,
    unsupported_numbers: int = 0,
    user_input: int = 0,
) -> str:
    """Статус оснований вывода — единственное место с этим правилом.

    Четыре состояния различаются принципиально: «подкреплено» — хотя бы одно
    основание подтверждено и ни одно не отклонено; «подкреплено частично» —
    часть отклонена, ждёт проверки или опирается на ввод пользователя;
    «опирается на ввод пользователя» — подтверждённых документов нет, зато
    есть слова человека: опора есть, но верить им или нет — его решение;
    «неподкреплено» — ни одного подтверждённого. Молчать нельзя: вывод без
    основания выглядит в документе так же, как вывод с основанием.

    Правило было записано дважды — в рабочем процессе и в отчёте, — и
    версии разошлись: отчёт не знал про «ждёт проверки» и называл такой
    вывод подкреплённым. Теперь правило одно.

    Подтверждённые основания могут быть не про тот показатель, о котором
    вывод: дословная цитата про абонентскую базу не подкрепляет изменение
    удержания. Такое основание сюда приходит **отклонённым**: понижение
    считается там, где собираются вердикты (`indicator-demotion`), а не
    отдельным признаком. Признак пришлось бы считать дважды — здесь и в
    отчёте, — и на выводе, где понижено одно основание из двух, два счёта
    разошлись бы (`D-46`).

    Число, названное в плане или обязательстве, тоже утверждение: если
    расчёта того же показателя нет, величина ни из чего не следует, и
    «подкреплено» она не даёт. Это потолок, а не отдельный статус:
    неподтверждённое число не делает вывод неподкреплённым, но и
    «подкреплено» рядом с ним стоять не может.

    Потолок умеет только понижать. «неподкреплено» и «ждёт проверки» он не
    трогает: иначе ограничение превратилось бы в подмену статуса.

    Args:
        checked: Сколько было проверено.
        confirmed: Сколько подтверждено.
        rejected: Сколько отклонено.
        needs_review: Сколько ждёт проверки человеком.
        unsupported_numbers: Сколько величин плана осталось без своего
            расчёта.
        user_input: Сколько опор — ввод пользователя. Они входят в
            ``needs_review``; отдельным числом называются, чтобы отличить
            «нечем подтвердить» от «это слова человека».

    Returns:
        Статус оснований вывода.
    """
    if checked <= 0:
        return "не проверялось"
    if confirmed <= 0 and rejected <= 0 and user_input > 0:
        # Опорой служат слова самого человека. Сказать «неподкреплено» —
        # значит объявить его слова пустотой; сказать «подкреплено» — значит
        # выставить за него доверие, которого он не давал.
        return "опирается на ввод пользователя"
    if confirmed <= 0 and needs_review > 0:
        return "ждёт проверки"
    if confirmed <= 0:
        return "неподкреплено"
    if rejected > 0 or needs_review > 0 or unsupported_numbers > 0:
        return "подкреплено частично"
    return "подкреплено"


@dataclass(frozen=True, slots=True)
class GroundEntry:
    """Одно основание в отчёте о качестве заземления.

    Attributes:
        citation: Ссылка на источник.
        status: Статус подтверждения.
        kind: Что это — факт или метрика.
    """

    citation: str
    status: str
    kind: str = "факт"


@dataclass(frozen=True, slots=True)
class GroundReport:
    """Отчёт о качестве оснований: какая доля подтверждена кодом.

    Самооценка «оснований достаточно» ничего не значит. Значима
    доля подтверждённых оснований — и отдельно по фактам и по
    метрикам: у них разный способ проверки.

    Метрики намеренно разделены: «факты подтверждены на 90 %» и
    «метрики воспроизводятся на 60 %» говорят разные вещи, а одно
    среднее число скрывает разницу.
    """

    entries: list[GroundEntry] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Сколько всего оснований."""
        return len(self.entries)

    @property
    def confirmed(self) -> int:
        """Сколько оснований подтверждено."""
        return sum(1 for entry in self.entries if entry.status == "confirmed")

    @property
    def grounding_rate(self) -> float:
        """Доля подтверждённых оснований от всех.

        Returns:
            Доля от 0 до 1; ноль для пустого отчёта.
        """
        if not self.entries:
            return 0.0
        return self.confirmed / self.total

    def _rate(self, kind: str) -> float:
        """Доля подтверждённых оснований одного вида.

        Args:
            kind: Вид основания.

        Returns:
            Доля от 0 до 1; ноль, если таких оснований нет.
        """
        selected = [entry for entry in self.entries if entry.kind == kind]
        if not selected:
            return 0.0
        confirmed = sum(1 for entry in selected if entry.status == "confirmed")
        return confirmed / len(selected)

    @property
    def fact_grounding_rate(self) -> float:
        """Доля подтверждённых фактов."""
        return self._rate("факт")

    @property
    def metric_grounding_rate(self) -> float:
        """Доля подтверждённых метрик."""
        return self._rate("метрика")

    @property
    def summary(self) -> str:
        """Краткая сводка для вывода.

        Returns:
            Текст сводки.
        """
        return (
            f"оснований: {self.total}, подтверждено: {self.confirmed} "
            f"({self.grounding_rate:.0%}); факты: {self.fact_grounding_rate:.0%}, "
            f"метрики: {self.metric_grounding_rate:.0%}"
        )


def normalize(text: str) -> str:
    """Нормализовать текст для сравнения.

    Args:
        text: Исходный текст.

    Returns:
        Нормализованный текст: нижний регистр, единые пробелы, без мягких
        переносов, тире, кавычек и различия «ё»/«е».
    """
    normalized = unicodedata.normalize("NFC", text).lower().translate(_QUOTES)
    normalized = normalized.replace("ё", "е")
    normalized = _HYPHEN.sub("-", normalized)
    normalized = _WHITESPACE.sub(" ", normalized)
    # Смешанная раскладка — то же слово: «абонентoв» с латинской «o» должно
    # найтись в источнике с «абонентов». Замена идёт по словам, поэтому
    # чистая латиница («opex») остаётся как есть.
    normalized = _WORD.sub(lambda match: normalize_homoglyphs(match.group(0)), normalized)
    return normalized.strip()


def find_quote(source_text: str, quote: str) -> tuple[bool, int | None]:
    """Найти цитату в исходном тексте.

    Args:
        source_text: Исходный текст источника.
        quote: Искомая цитата.

    Returns:
        Пара ``(найдена, номер_абзаца)``. Номер абзаца — ``None``, если
        цитата найдена, но не внутри абзаца (например, в заголовке).
    """
    normalized_quote = normalize(quote)
    if not normalized_quote:
        return False, None

    paragraphs = source_text.split("\n")
    for index, paragraph in enumerate(paragraphs):
        if normalized_quote in normalize(paragraph):
            return True, index

    if normalized_quote in normalize(source_text):
        return True, None
    return False, None


def split_source_id(source_id: str) -> tuple[str, int | None]:
    """Разделить идентификатор источника на путь и номер строки.

    Модель часто склеивает их в одну строку. Якоря бывают двух видов:

    - ``#p126`` — номер абзаца статьи;
    - ``#row3`` — номер строки таблицы (так ссылаются метрики).

    Разбор делается здесь, чтобы остальной код работал с чистыми
    полями. Оба вида якоря дают номер строки: у абзацев он тоже
    нужен — по нему ищется фрагмент для проверки цитаты.

    Args:
        source_id: Идентификатор как прислала модель.

    Returns:
        Пара ``(source_id, номер строки)``.
    """
    text = source_id.strip()
    for anchor in ("#p", "#row"):
        if anchor in text:
            path, _, suffix = text.rpartition(anchor)
            if path and suffix.isdigit():
                return path, int(suffix)
    return text, None


def _groups() -> dict[str, str]:
    """Отображение синонимов из словаря лексикона.

    Импорт ленивый: ``pf.lexicon`` читает файлы, а ``pf.ground`` должен
    импортироваться быстро и без побочных эффектов.

    Returns:
        Словарь «лемма → каноническая лемма».
    """
    from pf import lexicon

    return lexicon.groups()


def _tolerances(
    abs_tolerance: float | None,
    rel_tolerance: float | None,
) -> tuple[float, float]:
    """Допуски для сравнения чисел.

    Args:
        abs_tolerance: Абсолютный допуск либо ``None`` — взять из настроек.
        rel_tolerance: Относительный допуск либо ``None``.

    Returns:
        Пара ``(абсолютный, относительный)``.
    """
    absolute = settings.metric_abs_tolerance if abs_tolerance is None else abs_tolerance
    relative = settings.metric_rel_tolerance if rel_tolerance is None else rel_tolerance
    return absolute, relative


def token_overlap(claim: str, paragraph: str) -> float:
    """Доля значимых слов утверждения, покрытых абзацем.

    Сравнение идёт по леммам и с учётом словаря синонимов: «абоненты» в
    источнике покрывают «клиентов» в утверждении. Это не ослабляет
    проверку: точные термины и числа сверяются отдельно и без синонимов.

    Args:
        claim: Текст утверждения.
        paragraph: Текст абзаца-источника.

    Returns:
        Доля от 0.0 до 1.0.
    """
    result = lexical_overlap(claim, paragraph, groups=_groups())
    return result.coverage


def _overlap_reason(result: Overlap, min_overlap: float) -> str:
    """Назвать причину, по которой утверждение не покрыто абзацем.

    Глухое «покрытие 0 %» читается как выдумка, хотя причина может быть
    другой: подменён термин, разошлось число, появилось отрицание.

    Args:
        result: Результат сравнения.
        min_overlap: Порог покрытия.

    Returns:
        Текст причины для журнала.
    """
    if result.missing_terms:
        return f"термин «{result.missing_terms[0]}» не найден в источнике"
    if not result.numbers_ok:
        return result.numbers_detail
    if not result.negation_ok:
        return "в утверждении появилось отрицание, которого нет в источнике"
    return (
        f"слова утверждения не подтверждаются абзацем "
        f"(покрытие {result.coverage:.0%} при пороге {min_overlap:.0%})"
    )


def _field_reason(result: SupportScore) -> str:
    """Назвать поле, по которому утверждение не подтвердилось.

    «Не совпало» не говорит ничего: причина может быть в периоде, субъекте,
    единице или в том, что одна строка не подтверждает динамику. Решение
    принимает степень поддержки, а причина называется полем и числом.

    Args:
        result: Степень поддержки по абзацу.

    Returns:
        Текст причины для журнала и отчёта.
    """
    if result.reason:
        return f"{result.reason} ({result.explain()})"
    return f"степень поддержки ниже порога: {result.explain()}"


def _cap_by_readings(support: Support, source_text: str, claim_text: str) -> Support:
    """Ограничить подтверждение прочтениями аббревиатуры.

    Дословная цитата — сильное свидетельство, но не свидетельство того, что
    речь о той же сущности: цитата из строки с уточнением дословна и при
    этом не подтверждает утверждение про голое имя (`D-51`). Поэтому у
    подтверждённой цитаты статус проверяется ещё и прочтениями: чужая
    сущность — отказ, не названное прочтение — «требует проверки» со ссылкой.

    Args:
        support: Основание, подтверждённое цитатой.
        source_text: Текст источника.
        claim_text: Текст утверждения.

    Returns:
        Основание со статусом не выше, чем позволяют прочтения.
    """
    # Импорт ленивый: `pf.canonical` читает словарь и чистку.
    from pf.canonical import parse_fields

    identity = entity_identity(
        claim_text,
        source_text,
        claim_row=bool(parse_fields(claim_text)),
        source_row=bool(parse_fields(source_text)),
    )
    if identity.kind == KIND_MISMATCH:
        return replace(
            support,
            status=SupportStatus.NO_DATA,
            kind=ENTITY_MISMATCH,
            detail=f"цитата дословна, но {identity.note}",
        )
    if identity.kind == KIND_AMBIGUOUS:
        return replace(
            support,
            status=SupportStatus.NEEDS_REVIEW,
            kind=ENTITY_AMBIGUOUS,
            detail=f"прочтение не установлено: {identity.note}",
        )
    return support


def field_verdict(result: SupportScore, threshold: float) -> tuple[SupportStatus, str, str]:
    """Превратить степень поддержки в статус: три исхода, а не два.

    Неоднозначное прочтение аббревиатуры **понижает** подтверждение: основание
    остаётся в силе, но не подтверждает, пока человек не выберет прочтение
    (`D-51`). Отказ оно не отменяет: если степень поддержки низка из-за другого
    поля (единицы, периода, величины), это и есть причина, и подменять её на
    «требует проверки» значило бы обещать проверку там, где всё решено.

    Функция открыта для замера: `make replay` и набор пар считают статус тем же
    правилом, что и прогон.

    Args:
        result: Степень поддержки по абзацу.
        threshold: Порог, при котором степень становится подтверждением.

    Returns:
        Пару «статус, вид, пояснение».
    """
    ambiguous = result.entity_kind == KIND_AMBIGUOUS
    if result.supported(threshold):
        if ambiguous:
            return (
                SupportStatus.NEEDS_REVIEW,
                ENTITY_AMBIGUOUS,
                f"прочтение не установлено: {result.entity_note} ({result.explain()})",
            )
        suffix = f" · {result.entity_note}" if result.entity_note else ""
        return (
            SupportStatus.CONFIRMED,
            "",
            f"утверждение подтверждено: {result.explain()}{suffix}",
        )
    if ambiguous and result.support_without_subject >= threshold:
        # Остальные поля сходятся: спор идёт только о прочтении.
        return (
            SupportStatus.NEEDS_REVIEW,
            ENTITY_AMBIGUOUS,
            f"прочтение не установлено: {result.entity_note} ({result.explain()})",
        )
    kind = ENTITY_MISMATCH if result.entity_kind == KIND_MISMATCH else ""
    return (SupportStatus.NO_DATA, kind, _field_reason(result))


def verify_claim(
    source_text: str,
    claim_text: str,
    source_id: str,
    paragraph_index: int | None = None,
    quote: str = "",
    *,
    support_threshold: float | None = None,
    abs_tolerance: float | None = None,
    rel_tolerance: float | None = None,
) -> Support:
    """Подтвердить утверждение по источнику.

    Работают два пути:

    1. **Цитата.** Если модель прислала цитату, она проверяется дословно,
       а при расхождении — с допуском на округление чисел.
    2. **Поля.** Если цитаты нет, считается **степень поддержки** полями
       (`pf.canonical.score`): понятие показателя, субъект, регион, период,
       величины, термины, отрицание и опора утверждения. Порог, при котором
       степень становится подтверждением, — настройка
       (`claim_support_threshold`), а не константа: он подбирается на замере.

    Цитата — не способ обойти проверку и не повод выбросить верное
    утверждение. Если она не годится как доказательство (короче порога,
    пересказ, другое число), утверждение проверяется полями — тем же
    правилом, что и без цитаты, — а вид отказа остаётся в ``kind`` и
    называется в пояснении. Выдумка в этот список не входит: там поля и так
    не сходятся (`D-02`).

    Args:
        source_text: Полный текст источника.
        claim_text: Текст утверждения.
        source_id: Идентификатор источника.
        paragraph_index: Номер абзаца, если указан.
        quote: Дословная цитата, если прислана.
        support_threshold: Порог степени поддержки. По умолчанию — из настроек.
        abs_tolerance: Абсолютный допуск чисел. По умолчанию — из настроек.
        rel_tolerance: Относительный допуск чисел.

    Returns:
        Основание со статусом подтверждения.
    """
    if not quote.strip():
        return _verify_fields(
            source_text,
            claim_text,
            source_id,
            paragraph_index,
            support_threshold=support_threshold,
            abs_tolerance=abs_tolerance,
            rel_tolerance=rel_tolerance,
        )

    support = verify_quote(
        source_text,
        quote,
        source_id,
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
    )
    if support.status is SupportStatus.CONFIRMED:
        # Дословность не отвечает на вопрос «та ли сущность» (`D-51`).
        support = _cap_by_readings(support, source_text, claim_text)
    if support.status is not SupportStatus.NO_DATA or support.kind not in QUOTE_FALLBACK_KINDS:
        return support

    fields = _verify_fields(
        source_text,
        claim_text,
        source_id,
        paragraph_index,
        support_threshold=support_threshold,
        abs_tolerance=abs_tolerance,
        rel_tolerance=rel_tolerance,
    )
    return replace(
        fields,
        kind=fields.kind or support.kind,
        detail=(
            f"цитата не годится как доказательство ({support.detail}): "
            f"проверено полями — {fields.detail}"
        ),
    )


def _verify_fields(
    source_text: str,
    claim_text: str,
    source_id: str,
    paragraph_index: int | None = None,
    *,
    support_threshold: float | None = None,
    abs_tolerance: float | None = None,
    rel_tolerance: float | None = None,
) -> Support:
    """Проверить утверждение по полям: цитаты нет либо она не годится.

    Args:
        source_text: Полный текст источника.
        claim_text: Текст утверждения.
        source_id: Идентификатор источника.
        paragraph_index: Номер абзаца, если указан.
        support_threshold: Порог степени поддержки. По умолчанию — из настроек.
        abs_tolerance: Абсолютный допуск чисел. По умолчанию — из настроек.
        rel_tolerance: Относительный допуск чисел.

    Returns:
        Основание со статусом подтверждения.
    """
    # Импорт ленивый: `pf.canonical` читает словарь и чистку, а `pf.ground`
    # должен импортироваться быстро и без побочных эффектов.
    from pf.canonical import score as canonical_score

    threshold = settings.claim_support_threshold if support_threshold is None else support_threshold
    absolute, relative = _tolerances(abs_tolerance, rel_tolerance)
    groups = _groups()
    paragraphs = source_text.split("\n")

    def compare(paragraph: str) -> SupportScore:
        """Сравнить утверждение с одним абзацем по полям."""
        return canonical_score(
            claim_text,
            paragraph,
            groups=groups,
            abs_tolerance=absolute,
            rel_tolerance=relative,
        )

    if paragraph_index is not None and 0 <= paragraph_index < len(paragraphs):
        paragraph = paragraphs[paragraph_index].strip()
        result = compare(paragraph)
        status, kind, detail = field_verdict(result, threshold)
        quote = paragraph[:400] if status is not SupportStatus.NO_DATA else claim_text.strip()[:400]
        return Support(
            quote=quote,
            source_id=source_id,
            status=status,
            paragraph_index=paragraph_index,
            detail=detail,
            oov=tuple(result.oov_tokens),
            kind=kind,
        )

    # Абзац не указан — ищем, в каком абзаце утверждение подтверждается.
    claim_numbers = numbers_in(claim_text)
    best: SupportScore | None = None
    best_paragraph = ""
    for index, paragraph in enumerate(paragraphs):
        if len(paragraph.strip()) < 40:
            continue
        # Дешёвый предфильтр: у статьи тысячи абзацев, и морфологический
        # разбор каждого — самая дорогая часть проверки. Пропускаем абзац,
        # только если у него нет ни общих слов, ни чисел утверждения.
        paragraph_numbers_ok, _ = numbers_match(
            claim_numbers,
            numbers_in(paragraph),
            abs_tolerance=absolute,
            rel_tolerance=relative,
        )
        if not paragraph_numbers_ok and not shares_words(claim_text, paragraph):
            continue
        result = compare(paragraph)
        status, kind, detail = field_verdict(result, threshold)
        if best is None or result.support > best.support:
            best, best_paragraph = result, paragraph
        if status is SupportStatus.CONFIRMED:
            return Support(
                quote=paragraph.strip()[:400],
                source_id=source_id,
                status=status,
                paragraph_index=index,
                detail=detail,
                oov=tuple(result.oov_tokens),
                kind=kind,
            )

    if best is None:
        return Support(
            quote=claim_text.strip()[:400],
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail=(
                "в источнике нет абзаца, пригодного для сверки: "
                "ни общих слов, ни чисел с утверждением"
            ),
        )

    status, kind, detail = field_verdict(best, threshold)
    return Support(
        quote=best_paragraph.strip()[:400],
        source_id=source_id,
        status=status,
        detail=detail,
        oov=tuple(best.oov_tokens),
        kind=kind,
    )


def verify_quote(
    source_text: str,
    quote: str,
    source_id: str,
    *,
    min_length: int = 12,
    abs_tolerance: float | None = None,
    rel_tolerance: float | None = None,
) -> Support:
    """Проверить, подтверждается ли цитата исходным текстом.

    Три исхода, и все честные:

    1. **Дословно** — подтверждено.
    2. **С допуском на округление.** Если в цитате есть числа и все они
       нашлись в одном абзаце с допуском, а слова совпали по леммам,
       цитата отличается от источника только точностью записи числа.
    3. **Другой язык.** Дословный перевод не проверяется: сверяются
       только числа, а статус остаётся «требует проверки».

    Args:
        source_text: Исходный текст источника.
        quote: Цитата из ответа модели.
        source_id: Идентификатор источника.
        min_length: Минимальная длина цитаты. Короткие совпадения вроде
            «да» или «и» находятся почти в любом тексте и ничего не доказывают.
        abs_tolerance: Абсолютный допуск чисел. По умолчанию — из настроек.
        rel_tolerance: Относительный допуск чисел.

    Returns:
        Основание со статусом подтверждения.
    """
    stripped = quote.strip()
    if len(stripped) < min_length:
        return Support(
            quote=stripped,
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail=f"цитата короче {min_length} символов — подтвердить нечего",
            kind="too_short",
        )

    found, paragraph_index = find_quote(source_text, stripped)
    if found:
        return Support(
            quote=stripped,
            source_id=source_id,
            status=SupportStatus.CONFIRMED,
            paragraph_index=paragraph_index,
            detail="цитата найдена в источнике",
            kind="exact",
        )

    absolute, relative = _tolerances(abs_tolerance, rel_tolerance)

    # Другой язык источника: дословность недостижима. Сверяем числа —
    # единственное, что не зависит от перевода, — и не выдаём цитату за
    # подтверждённую.
    quote_language = language(stripped)
    source_language = language(source_text)
    if "other" not in {quote_language, source_language} and quote_language != source_language:
        numbers_ok, numbers_detail = numbers_match(
            numbers_in(stripped),
            numbers_in(source_text),
            abs_tolerance=absolute,
            rel_tolerance=relative,
        )
        if not numbers_ok:
            return Support(
                quote=stripped,
                source_id=source_id,
                status=SupportStatus.NO_DATA,
                detail=numbers_detail,
                kind="language",
            )
        return Support(
            quote=stripped,
            source_id=source_id,
            status=SupportStatus.NEEDS_REVIEW,
            detail=(
                "иностранный источник: дословный перевод не проверяется, "
                "сверены только числа — нужен взгляд человека"
            ),
            kind="language",
        )

    # Округление — это различие **только** в записи числа. Маска числа
    # отделяет округление от пересказа: пересказ строки таблицы сохраняет
    # слова, но теряет метки колонок, и совпадения масок не будет.
    quote_numbers = numbers_in(stripped)
    if quote_numbers:
        masked = mask_numbers(normalize(stripped))
        for index, paragraph in enumerate(source_text.split("\n")):
            # Нужны оба условия: числа совпали **с допуском** и различие в
            # тексте только в их записи. Маска без сверки чисел подтверждала
            # бы любое другое число: «99,99» и «77,26» после маски совпадают.
            numbers_ok, _ = numbers_match(
                quote_numbers,
                numbers_in(paragraph),
                abs_tolerance=absolute,
                rel_tolerance=relative,
            )
            if numbers_ok and masked in mask_numbers(normalize(paragraph)):
                return Support(
                    quote=stripped,
                    source_id=source_id,
                    status=SupportStatus.CONFIRMED,
                    paragraph_index=index,
                    detail="цитата совпала с источником с точностью до допуска чисел",
                    kind="number_tolerance",
                )

    groups = _groups()
    quote_numbers = numbers_in(stripped)
    best: Overlap | None = None
    for paragraph in source_text.split("\n"):
        # Абзац, где не нашлось ни одного числа цитаты и ни одного общего
        # слова, не может быть её источником — не тратим на него разбор.
        paragraph_numbers_ok, _ = numbers_match(
            quote_numbers,
            numbers_in(paragraph),
            abs_tolerance=absolute,
            rel_tolerance=relative,
        )
        if not paragraph_numbers_ok and not shares_words(stripped, paragraph):
            continue
        result = lexical_overlap(
            stripped,
            paragraph,
            groups=groups,
            abs_tolerance=absolute,
            rel_tolerance=relative,
            min_overlap=settings.claim_min_overlap,
        )
        if best is None or result.coverage > best.coverage:
            best = result

    if best is not None and best.coverage >= settings.claim_min_overlap and not best.numbers_ok:
        # Слова совпали, а число нет: это не пересказ и не выдумка, а другое
        # число. «Не найдена» читалось бы как выдумка.
        return Support(
            quote=stripped,
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail=_overlap_reason(best, settings.claim_min_overlap),
            oov=tuple(best.oov_tokens),
            kind="number_mismatch",
        )
    if best is not None and best.coverage >= PARAPHRASE_COVERAGE:
        # Пересказ — не выдумка: слова и числа источника на месте, но
        # цитаты как таковой нет. Это отдельный вид, и он проверяется полями.
        return Support(
            quote=stripped,
            source_id=source_id,
            status=SupportStatus.NO_DATA,
            detail=(
                f"цитата не дословна (покрытие {best.coverage:.0%}) — это пересказ, а не цитата"
            ),
            oov=tuple(best.oov_tokens),
            kind="paraphrase",
        )
    return Support(
        quote=stripped,
        source_id=source_id,
        status=SupportStatus.NO_DATA,
        detail="цитата не найдена в источнике дословно",
        kind="fabrication",
    )


def verify_quote_as_external(
    source_text: str,
    quote: str,
    source_id: str,
) -> Support:
    """Проверить цитату во внешнем источнике.

    Дословность проверяется как обычно, но подтвердить внешний источник
    нечем: никто не знает, что за страницу отдала выдача. Поэтому статус
    не выше «требует проверки».

    Args:
        source_text: Текст страницы.
        quote: Цитата из ответа модели.
        source_id: Идентификатор внешнего источника.

    Returns:
        Основание со статусом не выше ``needs_review``.
    """
    return mark_external(verify_quote(source_text, quote, source_id))
