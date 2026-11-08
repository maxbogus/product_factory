"""Перевод внешних материалов на русский — слой, а не украшение ответа.

Зачем модуль. Английский источник сегодня почти бесполезен: проверка
оснований для пары «английская цитата / английский источник» сверяет только
числа (``pf.ground.verify_quote``), а слова, термины и отрицание не проверяет
вовсе. Когда рядом с оригиналом лежит перевод, цитату можно искать дословно
в нём, а числа по-прежнему сверять с оригиналом.

Три правила, ради которых всё написано:

1. **Оригинал не меняется.** Перевод — отдельный текст; источник остаётся
   единственным местом, где ищется дословность.
2. **Числа, единицы и идентификаторы сохраняются.** «87,5 %» не имеет права
   стать «88 %»: перевод с потерянным числом непригоден и в кэш не попадает.
3. **Кэш обязателен.** Перевод недетерминирован, а прогон обязан
   воспроизводиться: ``make test`` и ``make replay`` в модель не ходят.

Недоступная модель — не ошибка прогона, а отсутствие перевода: возвращается
``Translation(ok=False)``, и вызывающий код остаётся на оригинале.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from pf.config import settings
from pf.lexical import language
from pf.llm import LLMClient, LLMError, build_client

#: Короче этого переводить нечего: получится «да» вместо абзаца.
MIN_TRANSLATABLE_CHARS = 40

#: Движки перевода: модель, отсутствие перевода, готовый кэш.
ENGINE_LLM = "llm"
ENGINE_NONE = "none"
ENGINE_CACHE = "cache"

#: Расширение файлов кэша.
CACHE_SUFFIX = ".yaml"

#: Требование языка, общее для всех ролей. Оно не отменяет дословности:
#: цитата остаётся на языке источника, переводится только объяснение.
LANGUAGE_BLOCK = (
    "Отвечай по-русски. Цитаты из источников приводи дословно, на языке "
    "источника: перевод цитаты основанием не является."
)

#: Инструкция переводчику. Заморожена в коде, а не в данных: перевод —
#: часть проверки, и его правила не должны меняться без теста.
SYSTEM_PROMPT = (
    "Ты переводишь научный и деловой текст на русский язык. "
    "Правила: 1) не меняй числа, единицы измерения, проценты и идентификаторы "
    "(#rowN, T-25, arXiv:1511.04023, econ.GN); 2) имена собственные, названия "
    "компаний и аббревиатуры оставляй как в оригинале; 3) ничего не добавляй и "
    "не объясняй; 4) если термин не переводится — оставь оригинал в скобках; "
    "5) верни только перевод, без вступления и комментариев."
)

#: Пробел между разрядами числа: «1 250» и «1<NBSP>250» — одно число.
_DIGIT_SPACE = re.compile(r"(?<=\d)[\s\u00a0\u202f](?=\d)")

#: Число: целое или дробное, с точкой или запятой.
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

#: Единицы, которые обязаны пережить перевод. Проценты и пункты — потому что
#: это единица измерения, а не слово.
_UNITS: tuple[str, ...] = ("%", "п.п.", "млн", "млрд", "руб", "§")

#: Идентификаторы: ссылки на строку, номера задач и решений, адреса препринтов,
#: коды категорий arXiv. Всё это модель не имеет права «перевести».
_ID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"#row\d+"),
    re.compile(r"\b[TДD]-\d{2}\b"),
    re.compile(r"arXiv:\s*[\w./-]+"),
    re.compile(r"\b[a-z]{2,6}\.[A-Z]{2,4}\b"),
)

#: Границы предложений — для нарезки длинного текста на куски.
_SENTENCE = re.compile(r"(?<=[.!?;:])\s+")

#: Границы абзацев.
_PARAGRAPH = re.compile(r"\n\s*\n")


@dataclass(frozen=True, slots=True)
class Translation:
    """Результат перевода.

    Attributes:
        text: Переведённый текст; пустой, если перевода нет.
        engine: Движок: ``llm``, ``cache`` либо ``none`` (текст уже русский).
        model: Модель, которая переводила.
        ok: Пригоден ли перевод к использованию.
        notes: Пояснения — что потеряно, что добавлено, почему отказано.
        usage_tokens: Расход на перевод.
    """

    text: str = ""
    engine: str = ENGINE_LLM
    model: str = ""
    ok: bool = True
    notes: tuple[str, ...] = ()
    usage_tokens: int = 0

    @property
    def is_usable(self) -> bool:
        """Можно ли опираться на перевод.

        Returns:
            Перевод есть и числа сходятся.
        """
        return self.ok and bool(self.text.strip())


@dataclass(frozen=True, slots=True)
class TranslationRecord:
    """Запись кэша переводов.

    Attributes:
        source_sha1: Ключ — хеш исходного текста.
        engine: Движок перевода.
        model: Модель.
        created_at: Когда перевод получен.
        text_ru: Перевод.
        numbers_ok: Сошлись ли числа на момент записи.
    """

    source_sha1: str
    engine: str
    model: str
    created_at: str
    text_ru: str
    numbers_ok: bool = True


def numbers_of(text: str) -> Counter[str]:
    """Собрать числа, единицы и идентификаторы текста.

    Мультимножество, а не множество: повтор числа тоже значим — если «87,5 %»
    встречается дважды, а в переводе один раз, часть утверждения потеряна.

    Args:
        text: Любой текст.

    Returns:
        Счётчик «число/единица/идентификатор → сколько раз».
    """
    cleaned = _DIGIT_SPACE.sub("", text)
    counter: Counter[str] = Counter()
    for raw in _NUMBER.findall(cleaned):
        # «87,5» и «87.5» — одно и то же число: разделитель зависит от языка.
        counter[raw.replace(",", ".")] += 1
    lowered = text.lower()
    for unit in _UNITS:
        found = lowered.count(unit)
        if found:
            counter[f"единица:{unit}"] += found
    for pattern in _ID_PATTERNS:
        for found in pattern.findall(text):
            counter[f"идентификатор:{found.strip()}"] += 1
    return counter


def preserved(original: str, translated: str) -> tuple[bool, tuple[str, ...]]:
    """Проверить, что перевод сохранил числа, единицы и идентификаторы.

    Потерянное и придуманное число одинаково опасны: первое делает цитату
    слабее оригинала, второе — выдумкой.

    Args:
        original: Исходный текст.
        translated: Перевод.

    Returns:
        Пары «сохранено ли» и «пояснения».
    """
    before = numbers_of(original)
    after = numbers_of(translated)
    notes: list[str] = []
    lost = sorted((before - after).elements())
    added = sorted((after - before).elements())
    if lost:
        notes.append("перевод потерял: " + ", ".join(lost))
    if added:
        notes.append("перевод добавил: " + ", ".join(added))
    return (not notes, tuple(notes))


class TranslationStore:
    """Кэш переводов: файл на исходный текст.

    Ключ — хеш **исходного** текста. Перевод ключом не бывает: иначе испорченный
    перевод закреплял бы сам себя.
    """

    def __init__(self, directory: Path | None = None) -> None:
        """Создать хранилище.

        Args:
            directory: Каталог кэша. По умолчанию — из настроек.
        """
        base = directory if directory is not None else settings.translations_dir
        self.directory = Path(base)

    @staticmethod
    def key(text: str) -> str:
        """Ключ кэша для исходного текста.

        Args:
            text: Исходный текст.

        Returns:
            Первые 16 символов sha1.
        """
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def path_for(self, text: str) -> Path:
        """Путь файла кэша для текста.

        Args:
            text: Исходный текст.

        Returns:
            Путь к YAML-файлу.
        """
        return self.directory / f"{self.key(text)}{CACHE_SUFFIX}"

    def get(self, text: str) -> Translation | None:
        """Прочитать перевод из кэша.

        Числа сверяются заново: кэш — файл, его могли отредактировать, а
        доверять переводу без сверки нельзя.

        Args:
            text: Исходный текст.

        Returns:
            Перевод либо ``None``, если записи нет или она непригодна.
        """
        path = self.path_for(text)
        if not path.is_file():
            return None
        try:
            data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (yaml.YAMLError, OSError):
            return None
        if not isinstance(data, dict):
            return None
        cached = str(data.get("text_ru") or "").strip()
        if not cached:
            return None
        ok, _ = preserved(text, cached)
        if not ok:
            return None
        return Translation(
            text=cached,
            engine=ENGINE_CACHE,
            model=str(data.get("model") or ""),
            ok=True,
            notes=("из кэша",),
        )

    def put(self, text: str, translation: Translation) -> Path:
        """Записать перевод в кэш.

        Args:
            text: Исходный текст.
            translation: Пригодный перевод.

        Returns:
            Путь к записанному файлу.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(text)
        record = TranslationRecord(
            source_sha1=self.key(text),
            engine=translation.engine,
            model=translation.model,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            text_ru=translation.text,
            numbers_ok=translation.ok,
        )
        payload = {
            "source_sha1": record.source_sha1,
            "engine": record.engine,
            "model": record.model,
            "created_at": record.created_at,
            "text_ru": record.text_ru,
            "numbers_ok": record.numbers_ok,
        }
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=True), encoding="utf-8"
        )
        return path


def split_for_translation(text: str, *, max_chars: int = 1200) -> tuple[str, ...]:
    """Нарезать текст на куски, которые модель не потеряет.

    Сначала по абзацам, затем по границам предложений. Предложение длиннее
    лимита остаётся целым: разрезать его по середине — значит потерять число
    или отрицание на стыке.

    Args:
        text: Исходный текст.
        max_chars: Предел длины куска.

    Returns:
        Куски в исходном порядке.
    """
    chunks: list[str] = []
    for block in _PARAGRAPH.split(text):
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            chunks.append(block)
            continue
        current = ""
        for sentence in _SENTENCE.split(block):
            if not current:
                current = sentence
            elif len(current) + 1 + len(sentence) <= max_chars:
                current = f"{current} {sentence}"
            else:
                chunks.append(current)
                current = sentence
        if current:
            chunks.append(current)
    return tuple(chunks)


def translate(
    text: str,
    *,
    client: LLMClient | None = None,
    max_chars: int = 1200,
    store: TranslationStore | None = None,
    refresh: bool = False,
) -> Translation:
    """Перевести текст на русский.

    Args:
        text: Исходный текст.
        client: Клиент модели. По умолчанию — из настроек.
        max_chars: Предел длины куска при нарезке.
        store: Кэш переводов. По умолчанию — каталог из настроек.
        refresh: Пересчитать перевод, не веря кэшу.

    Returns:
        Результат перевода; при отказе — с ``ok=False`` и пояснением.
    """
    stripped = text.strip()
    if len(stripped) < MIN_TRANSLATABLE_CHARS:
        return Translation(
            ok=False,
            engine=ENGINE_NONE,
            notes=(f"текст короче {MIN_TRANSLATABLE_CHARS} символов — переводить нечего",),
        )
    if language(stripped) == "ru":
        return Translation(text=stripped, engine=ENGINE_NONE, notes=("текст уже на русском",))

    cache = store if store is not None else TranslationStore()
    if not refresh:
        cached = cache.get(stripped)
        if cached is not None:
            return cached

    translator = client if client is not None else build_client()
    pieces: list[str] = []
    tokens = 0
    try:
        for part in split_for_translation(stripped, max_chars=max_chars):
            response = translator.complete(SYSTEM_PROMPT, part)
            pieces.append(str(response.content).strip())
            usage = getattr(response, "usage", None)
            tokens += int(getattr(usage, "total_tokens", 0) or 0)
    except LLMError as exc:
        return Translation(
            engine=ENGINE_LLM,
            model=settings.effective_model,
            ok=False,
            notes=(f"модель недоступна: {exc}",),
            usage_tokens=tokens,
        )

    joined = "\n\n".join(piece for piece in pieces if piece)
    ok, notes = preserved(stripped, joined)
    result = Translation(
        text=joined,
        engine=ENGINE_LLM,
        model=settings.effective_model,
        ok=ok,
        notes=notes,
        usage_tokens=tokens,
    )
    if ok:
        # Непригодный перевод не кэшируется: иначе ошибка закрепится.
        cache.put(stripped, result)
    return result
