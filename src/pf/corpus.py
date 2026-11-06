"""Локальный корпус Википедии: загрузка, индекс, поиск.

Корпус — снимок 7 966 статей из открытых разделов Википедии (см.
``data/wiki/SOURCE.md``). Он используется как **источник фактов**: агент
обязан ссылаться на конкретный абзац, а не пересказывать «по памяти».

Поиск сделан на TF-IDF: этого достаточно для 8 тысяч статей, и он не
требует ни эмбеддингов, ни тяжёлых моделей — индексация занимает секунды.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]

from pf.config import settings

#: Шаблон выделения слов. Только кириллица и латиница длиной от 4 знаков:
#: короткие служебные слова дают шум, а не сигнал.
_TOKEN_PATTERN = r"[а-яёa-z]{4,}"

#: Слова, которые встречаются везде и не различают статьи.
_STOP_WORDS = frozenset(
    [
        "и",
        "в",
        "на",
        "с",
        "по",
        "для",
        "что",
        "не",
        "как",
        "это",
        "к",
        "от",
        "из",
        "о",
        "же",
        "у",
        "или",
        "за",
        "то",
        "же",
        "мы",
        "вы",
        "он",
        "она",
        "они",
        "быть",
        "был",
        "была",
        "были",
        "будет",
        "также",
        "который",
        "которые",
        "при",
        "так",
        "уже",
        "или",
        "же",
        "свой",
        "более",
        "менее",
        "очень",
        "первый",
        "весь",
        "сам",
        "между",
        "через",
        "однако",
        "потому",
        "если",
        "когда",
        "можно",
        "нужно",
        "который",
        "этого",
        "этого",
        "эту",
        "такие",
        "такие",
        "там",
        "тут",
        "где",
        "который",
        "article",
        "about",
        "from",
        "this",
        "that",
        "with",
        "have",
        "been",
        "were",
        "will",
        "would",
        "could",
    ]
)


#: Доменные термины предметной области. Их вес при поиске выше общего
#: словаря: «оператор» в статье про спектр оператора — это математика,
#: а в статье про МТС — это телеком. Без различения поиск отдаёт шум.
#: Сильные доменные признаки: без них совпадение доменного слова
#: считается случайным (например, «оператор» в математике).
STRONG_DOMAIN_TERMS: frozenset[str] = frozenset(
    [
        "абонент",
        "абонентов",
        "абонентский",
        "мобильный",
        "мобильная",
        "мобильных",
        "сотовая",
        "сотовый",
        "сотово",
        "связь",
        "симкар",
        "симкарты",
        "телеком",
        "телекоммуникация",
        "тариф",
        "тарифы",
        "тарифный",
        "пакет",
        "пакеты",
        "трафик",
        "роуминг",
        "интернет",
        "провайдер",
    ]
)

DOMAIN_TERMS: frozenset[str] = frozenset(
    [
        "оператор",
        "операторов",
        "абонент",
        "абонентов",
        "абонентский",
        "удержание",
        "отток",
        "тариф",
        "тарифы",
        "тарифный",
        "подписка",
        "подписки",
        "контракт",
        "контрактный",
        "мобильный",
        "мобильная",
        "сотовая",
        "сотовый",
        "связь",
        "операторский",
        "симкар",
        "интернет",
        "пакет",
        "пакеты",
        "трафик",
        "роуминг",
        "корпоративный",
        "массовый",
        "телеком",
        "телекоммуникация",
        "операция",
        "операций",
        "стоимость",
        "тарификация",
    ]
)


@dataclass(frozen=True, slots=True)
class Article:
    """Статья корпуса.

    Attributes:
        source_id: Устойчивый идентификатор источника, вида ``economics/Цена``.
        title: Заголовок статьи.
        section: Раздел корпуса, из которого взята статья.
        content: Полный текст статьи.
    """

    source_id: str
    title: str
    section: str
    content: str

    @property
    def paragraphs(self) -> list[str]:
        """Абзацы статьи без пустых строк."""
        return [p.strip() for p in self.content.split("\n") if p.strip()]

    @property
    def domain_hit(self) -> float:
        """Сколько доменных терминов в заголовке статьи.

        Заголовок — сильнейший сигнал: статья «МТС (компания)» про телеком,
        даже если её абзацы набраны общими словами.

        Одного совпадения недостаточно: слово «оператор» есть и в статье
        «Спектр оператора» (математика). Поэтому термин засчитывается, только
        если в заголовке есть **сильный** доменный признак — «абонент»,
        «тариф», «мобильный», «связь», «телеком». Статью «Спектр оператора»
        это отсекает, а «МТС (компания)» оставляет: там есть «Мобильные
        ТелеСистемы».
        """
        words = set(tokenize(self.title))
        if words & STRONG_DOMAIN_TERMS:
            return float(len(words & DOMAIN_TERMS))
        # В заголовке признака нет — смотрим тело статьи, но с меньшим весом:
        # «Спектр оператора» не содержит слов «абонент» или «тариф»,
        # а статья про МТС содержит.
        body_words = set(tokenize(self.content[:4000]))
        strong_in_body = len(body_words & STRONG_DOMAIN_TERMS)
        if strong_in_body < 2:
            return 0.0
        return len(words & DOMAIN_TERMS) + strong_in_body / 4


class Corpus:
    """Корпус статей с полнотекстовым поиском по TF-IDF.

    Экземпляр загружается один раз и переиспользуется: индексация 8 тысяч
    статей занимает несколько секунд, поэтому держать её в памяти дешевле,
    чем пересчитывать на каждый запрос.
    """

    def __init__(self, articles: list[Article]) -> None:
        self._articles = articles
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: Any = None

    # ── Загрузка ────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, wiki_dir: Path | None = None) -> Corpus:
        """Загрузить корпус из каталога с JSON-файлами разделов.

        Args:
            wiki_dir: Каталог с файлами разделов. По умолчанию — из настроек.

        Returns:
            Готовый корпус.

        Raises:
            FileNotFoundError: Каталог корпуса не найден.
        """
        directory = wiki_dir or settings.wiki_dir
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Каталог корпуса не найден: {directory}. "
                "Скопируйте data/wiki/ из репозитория или задайте PF_WIKI_DIR."
            )

        articles: list[Article] = []
        seen_titles: set[str] = set()
        for path in sorted(directory.glob("*.json")):
            section = path.stem
            payload = json.loads(path.read_text(encoding="utf-8"))
            for raw in payload.get("articles", []):
                title = str(raw.get("title", "")).strip()
                content = str(raw.get("content", ""))
                if not title or not content:
                    continue
                # Статья может попасть в два раздела (пересечение тематик).
                # Дубли одинакового заголовка отбрасываем: иначе в выдаче
                # один и тот же текст занимает несколько мест.
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                articles.append(
                    Article(
                        source_id=f"{section}/{title}",
                        title=title,
                        section=section,
                        content=content,
                    )
                )
        return cls(articles)

    # ── Свойства ────────────────────────────────────────────────────────────

    @property
    def articles(self) -> list[Article]:
        """Все статьи корпуса."""
        return self._articles

    def __len__(self) -> int:
        return len(self._articles)

    def get(self, source_id: str) -> Article | None:
        """Найти статью по её идентификатору."""
        for article in self._articles:
            if article.source_id == source_id:
                return article
        return None

    # ── Индекс и поиск ──────────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        """Построить TF-IDF-индекс, если он ещё не построен."""
        if self._vectorizer is not None:
            return
        self._vectorizer = TfidfVectorizer(
            max_features=60_000,
            token_pattern=_TOKEN_PATTERN,
            lowercase=True,
        )
        self._matrix = self._vectorizer.fit_transform(a.content for a in self._articles)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        prefer_domain: bool = True,
    ) -> list[tuple[Article, float]]:
        """Найти статьи, релевантные запросу.

        Args:
            query: Поисковый запрос.
            top_k: Сколько статей вернуть. По умолчанию — из настроек.

        Returns:
            Список пар ``(статья, оценка)``, отсортированный по убыванию оценки.
            Оценка — косинусная близость TF-IDF.
        """
        self._ensure_index()
        limit = top_k or settings.source_top_k
        if not query.strip() or self._matrix is None:
            return []

        vector = self._vectorizer.transform([query])  # type: ignore[union-attr]
        scores = (self._matrix @ vector.T).toarray().ravel()

        # Доменный вес заголовка: статья о телекоме должна подниматься выше
        # статьи о математике, даже если обе содержат слово «оператор».
        # Буст щедрый (1.0 + 1.2 * hit), потому что статья без сильного
        # доменного признака в нашей предметной области — почти всегда шум.
        domain_boost = np.array(
            [float(self._articles[i].domain_hit) for i in range(len(self._articles))],
            dtype=float,
        )
        weights = np.where(domain_boost > 0, 1.0 + 1.2 * domain_boost, 1.0)
        scores = scores * weights

        ranked = sorted(
            ((self._articles[i], float(scores[i])) for i in range(len(self._articles))),
            key=lambda pair: pair[1],
            reverse=True,
        )
        # Статьи с нулевой оценкой не несут информации — отбрасываем.
        found = [(article, score) for article, score in ranked if score > 0.0]

        if prefer_domain and self._query_is_domain(query):
            # Сначала статьи с доменным признаком, потом остальные — но с
            # пониженным весом, чтобы они не вытесняли тематические.
            strong = [(a, s) for a, s in found if a.domain_hit > 0]
            weak = [(a, s * 0.35) for a, s in found if a.domain_hit == 0]
            found = strong + weak

        return found[:limit]

    @staticmethod
    def _query_is_domain(query: str) -> bool:
        """Похож ли запрос на нашу предметную область."""
        words = set(tokenize(query))
        return bool(words & STRONG_DOMAIN_TERMS)


@lru_cache(maxsize=1)
def get_corpus() -> Corpus:
    """Корпус процесса (загружается один раз)."""
    return Corpus.load()


def tokenize(text: str) -> list[str]:
    """Разбить текст на значимые слова.

    Args:
        text: Исходный текст.

    Returns:
        Список слов в нижнем регистре без стоп-слов.
    """
    words = re.findall(_TOKEN_PATTERN, text.lower())
    return [word for word in words if word not in _STOP_WORDS]
