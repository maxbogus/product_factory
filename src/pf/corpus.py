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
import pymorphy3  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]

from pf import domain
from pf.config import settings

#: Шаблон выделения слов. Только кириллица и латиница длиной от 4 знаков:
#: короткие служебные слова дают шум, а не сигнал.
_TOKEN_PATTERN = r"[а-яёa-z]{4,}"

#: Морфологический анализатор и кэш лемм. Анализатор тяжёлый (словари),
#: поэтому создаётся один раз; кэш нужен потому, что одни и те же слова
#: встречаются в тексте тысячи раз.
_MORPH: pymorphy3.MorphAnalyzer | None = None
_LEMMA_CACHE: dict[str, str] = {}

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


#: Доменные термины, сильные признаки и названия операторов — данные домена
#: (`knowledge/domain/entities.yaml`, `pf.domain`). Здесь остаются только
#: правила: как считать доменный счёт статьи и как отличить компанию от
#: омонима в заголовке («оператор» в математике — не наш домен).


#: Насколько статья про оператора поднимается над прочими доменными.
#: Величина подобрана так, чтобы оператор обгонял статью с лучшим
#: TF-IDF, но не вытеснял её из выдачи: сортировка по сумме.
OPERATOR_BONUS: float = 2.0


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

        Заголовок — сильнейший сигнал: статья «Компания А (компания)» про
        телеком, даже если её абзацы набраны общими словами.

        Одного совпадения недостаточно: слово «оператор» есть и в статье
        «Спектр оператора» (математика). Поэтому термин засчитывается, только
        если в заголовке есть **сильный** доменный признак — «абонент»,
        «тариф», «мобильный», «связь», «телеком». Статью «Спектр оператора»
        это отсекает, а «Компания А (компания)» оставляет: там есть «Мобильные
        ТелеСистемы».

        Наборы терминов приходят из домена (`pf.domain`), а правило счёта
        остаётся здесь.
        """
        words = set(tokenize(self.title))
        if words & domain.strong_terms():
            return float(len(words & domain.domain_terms()))
        # В заголовке признака нет — смотрим тело статьи, но с меньшим весом:
        # «Спектр оператора» не содержит слов «абонент» или «тариф»,
        # а статья про оператора содержит.
        body_words = set(tokenize(self.content[:4000]))
        strong_in_body = len(body_words & domain.strong_terms())
        if strong_in_body < 2:
            return 0.0
        return len(words & domain.domain_terms()) + strong_in_body / 4


class Corpus:
    """Корпус статей с полнотекстовым поиском по TF-IDF.

    Экземпляр загружается один раз и переиспользуется: индексация 8 тысяч
    статей занимает несколько секунд, поэтому держать её в памяти дешевле,
    чем пересчитывать на каждый запрос.
    """

    #: Во сколько раз больше кандидатов берётся для доменного запроса,
    #: прежде чем применяется доменная фильтрация. Нужный запас большой:
    #: после догрузки телеком-раздела статья про оператора стоит на
    #: 22-м месте общего топа (а «Мегафон» — на 95-м), тогда как короткие
    #: телеком-статьи вроде «Мобильный интернет» ближе по TF-IDF.
    DOMAIN_CANDIDATE_MULTIPLIER: int = 40

    #: Доля от лучшего счёта, ниже которой статья в доменный срез не идёт.
    #: Подобрано замером: статья с оценкой 0,15 при лучшей 2,13 — это шум
    #: («Эрланг» из математики), а не ответ на вопрос про удержание.
    DOMAIN_SCORE_CUTOFF: float = 0.5

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
        """Построить TF-IDF-индекс, если он ещё не построен.

        Индекс строится по **основам** слов. Это не украшение: без него
        запрос «почему падает удержание у мобильном тарифе» не совпадает со
        статьёй, где написано «тариф» и «мобильный», — TF-IDF даёт нулевое
        сходство, и нужная статья в срез не попадала вовсе.
        """
        if self._vectorizer is not None:
            return
        self._vectorizer = TfidfVectorizer(
            max_features=60_000,
            analyzer=analyze,
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

        Для доменного запроса работают два механизма сразу:

        1. **Пул кандидатов.** В поиск идут только статьи, содержащие доменные
           термины запроса. Без этого в срез попадала физика: TF-IDF по
           длинной статье плохо совпадает с коротким запросом, и нужная
           статья оказывалась 25-й из 25.
        2. **Бонус оператора.** Статья про оператора связи поднимается выше
           прочих: это ядро предметной области, и именно её ищет
           руководитель цифрового продукта.

        Для запроса вне предметной области работает обычный TF-IDF — агент
        должен отвечать и на чужую тему.

        Args:
            query: Поисковый запрос.
            top_k: Сколько статей вернуть. По умолчанию — из настроек.
            prefer_domain: Применять ли доменную обработку.

        Returns:
            Список пар ``(статья, оценка)``, отсортированный по убыванию оценки.
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

        if prefer_domain and is_domain_query(query):
            # Кандидатов берём с запасом: доменная фильтрация идёт после
            # TF-IDF, и нужная статья может стоять ниже топ-6.
            candidates = found[: limit * self.DOMAIN_CANDIDATE_MULTIPLIER]
            return self._prefer_domain(candidates, found, query)[:limit]

        return found[:limit]

    def _prefer_domain(
        self,
        candidates: list[tuple[Article, float]],
        all_found: list[tuple[Article, float]],
        query: str,
    ) -> list[tuple[Article, float]]:
        """Оставить в выдаче статьи по предметной области запроса.

        Шаги:

        1. из кандидатов оставляются те, что содержат **чистый** доменный
           термин запроса. Чистый — отобранный замером по корпусу: «абонент»
           встречается в 5 статьях из чужих разделов, а «связь» — в 1 402,
           и химическая «двойная связь» прошла бы фильтр наравне с
           телекомом;
        2. статья про оператора получает бонус и проходит в срез всегда:
           операторов в корпусе всего четыре, и это ядро предметной области;
        3. хвост отсекается по порогу доли от лучшего счёта: доменный срез
           должен быть коротким и точным, а не длинным и формальным.

        Args:
            candidates: Отобранные TF-IDF кандидаты.
            all_found: Вся найденная выдача — не используется, оставлен
                для совместимости вызова.
            query: Поисковый запрос.

        Returns:
            Переупорядоченная выдача.
        """
        query_lemmas = set(analyze(query))
        # Фильтр — по чистым терминам. «Связь» и «интернет» тоже доменные,
        # но они двусмысленны: химическая «двойная связь» и IT-«связь»
        # проходят по ним наравне с телекомом. Чистые термины отобраны
        # замером по корпусу (`clean: true` в словаре домена).
        wanted = query_lemmas & domain_lemmas()
        if not wanted:
            wanted = query_lemmas & strong_domain_lemmas()
        if not wanted:
            return candidates

        pool: list[tuple[Article, float]] = []
        for article, score in candidates:
            is_operator = is_operator_article(article.title)
            if is_operator:
                # Статья про оператора попадает в срез всегда: это ядро
                # предметной области, даже если конкретное слово запроса
                # в ней не встречается. «Мегафон» не содержит слова
                # «тариф», но вопрос — именно про него.
                pool.append((article, score + OPERATOR_BONUS))
            elif wanted & _article_lemmas(article):
                pool.append((article, score))

        if not pool:
            return candidates

        scored = pool
        scored.sort(key=lambda pair: pair[1], reverse=True)

        # Хвост отсекается: статья, чей счёт кратно ниже лучшего, в срез
        # не идёт. В доменном срезе нужен десяток сильных статей, а не
        # сотня формально подходящих.
        best = scored[0][1]
        cutoff = best * self.DOMAIN_SCORE_CUTOFF
        return [(article, score) for article, score in scored if score >= cutoff]


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


def get_morph() -> pymorphy3.MorphAnalyzer:
    """Морфологический анализатор — создаётся один раз на процесс.

    Returns:
        Анализатор pymorphy3.
    """
    global _MORPH
    if _MORPH is None:
        _MORPH = pymorphy3.MorphAnalyzer()
    return _MORPH


def lemmatize(word: str) -> str:
    """Привести слово к нормальной форме.

    Лемматизация взята из проверенного решения: ``pymorphy3`` с русским
    словарём. Эвристический срез суффиксов здесь не годится — он путает
    «связь» и «связью» (даёт «связ») и «удержание» с «удержания»
    (даёт «удержание» против «удержан»). На этих словах держится вся
    сопоставимость запроса и словаря.

    Args:
        word: Слово в нижнем регистре.

    Returns:
        Нормальная форма слова.
    """
    cached = _LEMMA_CACHE.get(word)
    if cached is not None:
        return cached
    parsed = get_morph().parse(word)
    normal = parsed[0].normal_form if parsed else word.lower()
    _LEMMA_CACHE[word] = normal
    return normal


def domain_lemmas() -> frozenset[str]:
    """Леммы чистых доменных терминов — по ним ищется предметная область.

    Кэша здесь нет намеренно: словарь домена кэшируется по версии файла в
    `pf.domain`, а второй кэш в модуле пережил бы подмену домена в тестах и
    отвечал бы леммами прошлого домена.

    Returns:
        Множество лемм чистых доменных терминов.
    """
    return frozenset(lemmatize(term) for term in domain.clean_terms())


def strong_domain_lemmas() -> frozenset[str]:
    """Леммы сильных доменных признаков.

    Сильный признак — тот, что не встречается в чужих областях: «оператор»
    есть и в статье про операторы в C++, а «абонент» — только в телекоме.

    Returns:
        Множество лемм сильных доменных терминов.
    """
    return frozenset(lemmatize(term) for term in domain.strong_terms())


def is_operator_article(title: str) -> bool:
    """Статья про компанию-оператора связи?

    Args:
        title: Заголовок статьи.

    Returns:
        Заголовок называет оператора, а не банк, смартфон или единицу
        измерения с тем же названием.
    """
    return bool(domain.company_pattern().search(title.strip()))


def analyze(text: str) -> list[str]:
    """Разложить текст на значимые слова в форме леммы.

    Используется как анализатор TF-IDF: и корпус, и запрос проходят через
    одну и ту же функцию, поэтому словоформы сопоставимы.

    Args:
        text: Исходный текст.

    Returns:
        Список лемм значимых слов.
    """
    return [lemmatize(word) for word in tokenize(text)]


def _article_lemmas(article: Article) -> frozenset[str]:
    """Основы значимых слов статьи — для доменной фильтрации.

    Args:
        article: Статья корпуса.

    Returns:
        Множество основ слов статьи.
    """
    return frozenset(analyze(article.title + " " + article.content[:8000]))


def is_domain_query(query: str) -> bool:
    """Похож ли запрос на нашу предметную область.

    Сравнение идёт по основам слов, а не по словоформам: запрос «почему падает
    удержание у мобильном тарифе» доменный, хотя слова «мобильном» и
    «тарифе» в словаре нет.

    Args:
        query: Поисковый запрос.

    Returns:
        Запрос относится к предметной области.
    """
    lemmas = set(analyze(query))
    return bool(lemmas & strong_domain_lemmas())
