#!/usr/bin/env python3
"""Догрузка телеком-статей из Википедии.

Скрипт dev-инструмент: он нужен, чтобы пополнить корпус, а не во время
работы агента. Агент работает на том, что лежит в ``data/wiki/``.

Основа — проверенный сборщик из ``study/diplomMagistrate``:
``scripts/run_harvest.py`` (класс ``WikipediaRuClient``): поиск по запросам,
загрузка полного текста статьи, retry с exponential backoff, кэш в формате
``{"articles": [...], "failed_titles": [...], "last_updated": ...}`` — формат
совпадает с уже лежащими в ``data/wiki/`` файлами, поэтому загрузчик
``pf.corpus.Corpus.load()`` подхватывает результат без правок.

Отличия от оригинала:
- свой User-Agent: библиотека ``wikipedia`` со своим заголовком получает
  от Википедии 403, а с заголовком, где указан контакт, — 200;
- поисковые запросы — по телеком-тематике, а не по десяти доменам;
- выходной раздел — ``telecom``.

Запуск:

    python scripts/fetch_telecom.py --limit 120
    python scripts/fetch_telecom.py --limit 40 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

#: Заголовок запроса. Википедия отвечает 403 на безликие запросы.
USER_AGENT = "product-factory-strategist/0.1 (contact@example.org)"

#: Куда складывать статьи.
WIKI_DIR = Path(__file__).resolve().parent.parent / "data" / "wiki"

#: Раздел корпуса, который пополняется.
SECTION = "telecom"

#: Поисковые запросы: по ним ищутся статьи телеком-тематики.
SEARCH_QUERIES: list[str] = [
    "мобильный оператор связи",
    "тарифы на мобильную связь",
    "отток клиентов",
    "удержание клиентов",
    "сотовая связь",
    "Мегафон",
    "МТС",
    "Билайн",
    "Ростелеком",
    "конкуренция операторов связи",
    "рынок мобильной связи",
    "абонентская плата",
    "роуминг",
    "мобильный интернет в России",
]

#: Задержка между запросами к API. Без неё Википедия начинает отвечать 429.
API_REQUEST_DELAY = 1.5

#: Сколько попыток на одну статью.
MAX_RETRIES = 3

#: Начальная пауза между попытками.
BASE_BACKOFF = 5.0

#: Верхняя граница паузы.
MAX_BACKOFF = 60.0

logger = logging.getLogger("fetch_telecom")


@dataclass(slots=True)
class Cache:
    """Кэш статей одного раздела.

    Attributes:
        articles: Загруженные статьи.
        failed_titles: Заголовки, которые не удалось загрузить.
        last_updated: Когда обновлялся кэш.
    """

    articles: list[dict[str, str]] = field(default_factory=list)
    failed_titles: list[str] = field(default_factory=list)
    last_updated: str = ""

    @classmethod
    def load(cls, path: Path) -> Cache:
        """Прочитать кэш с диска.

        Args:
            path: Путь к файлу кэша.

        Returns:
            Кэш (пустой, если файла нет).
        """
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("не удалось прочитать кэш %s: %s", path, exc)
            return cls()
        return cls(
            articles=list(data.get("articles", [])),
            failed_titles=list(data.get("failed_titles", [])),
            last_updated=str(data.get("last_updated", "")),
        )

    def save(self, path: Path) -> None:
        """Записать кэш на диск.

        Args:
            path: Путь к файлу кэша.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.last_updated = datetime.now(UTC).isoformat()
        payload = {
            "articles": self.articles,
            "failed_titles": self.failed_titles,
            "last_updated": self.last_updated,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_page(title: str) -> dict[str, str] | None:
    """Загрузить полный текст статьи с повторами.

    Args:
        title: Заголовок статьи.

    Returns:
        Статья с заголовком и текстом либо ``None``, если не вышло.
    """
    import wikipedia

    for attempt in range(MAX_RETRIES):
        try:
            page = wikipedia.page(title, auto_suggest=False)
            return {"title": page.title, "content": page.content}
        except wikipedia.exceptions.DisambiguationError as exc:
            # Статья-Disambiguation: берём первый вариант.
            try:
                page = wikipedia.page(exc.options[0], auto_suggest=False)
                return {"title": page.title, "content": page.content}
            except Exception:  # noqa: BLE001 — статья не нужна, идём дальше
                return None
        except wikipedia.exceptions.PageError:
            return None
        except Exception as exc:  # noqa: BLE001 — сеть может отдать что угодно
            backoff = min(BASE_BACKOFF * (2**attempt), MAX_BACKOFF)
            logger.debug("попытка %d для %r: %s", attempt + 1, title, exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(backoff)
    return None


def fetch(limit: int, dry_run: bool) -> int:
    """Собрать статьи и записать в раздел корпуса.

    Args:
        limit: Сколько статей нужно.
        dry_run: Только показать, что будет найдено, ничего не писать.

    Returns:
        Сколько статей записано.
    """
    import wikipedia

    wikipedia.set_lang("ru")
    wikipedia.set_user_agent(USER_AGENT)

    path = WIKI_DIR / f"{SECTION}.json"
    cache = Cache.load(path)
    seen = {article["title"] for article in cache.articles}
    logger.info("в кэше уже %d статей", len(cache.articles))

    per_query = max(3, limit // len(SEARCH_QUERIES) + 1)
    collected = 0

    for query in SEARCH_QUERIES:
        if collected >= limit:
            break
        try:
            titles = wikipedia.search(query, results=per_query * 2)
        except Exception as exc:  # noqa: BLE001 — сеть может отдать что угодно
            logger.warning("поиск %r не удался: %s", query, exc)
            time.sleep(BASE_BACKOFF)
            continue

        logger.info("запрос %r: найдено %d заголовков", query, len(titles))
        for title in titles:
            if collected >= limit:
                break
            if title in seen:
                continue
            article = load_page(title)
            if article is None:
                cache.failed_titles.append(title)
                continue
            seen.add(article["title"])
            cache.articles.append(article)
            collected += 1
            if collected % 20 == 0:
                logger.info("загружено %d/%d", collected, limit)
                if not dry_run:
                    cache.save(path)
        time.sleep(API_REQUEST_DELAY)

    if dry_run:
        logger.info("пробный запуск: найдено %d статей, файл не тронут", collected)
        for article in cache.articles[-collected:] if collected else []:
            logger.info("  %s (%d символов)", article["title"], len(article["content"]))
        return collected

    cache.save(path)
    logger.info(
        "готово: %d новых статей, всего %d, файл %s",
        collected,
        len(cache.articles),
        path,
    )
    return collected


def main(argv: list[str] | None = None) -> int:
    """Точка входа.

    Args:
        argv: Аргументы командной строки.

    Returns:
        Код возврата.
    """
    parser = argparse.ArgumentParser(description="Догрузить телеком-статьи из Википедии")
    parser.add_argument("--limit", type=int, default=120, help="сколько статей нужно")
    parser.add_argument("--dry-run", action="store_true", help="ничего не записывать")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        fetch(args.limit, args.dry_run)
    except KeyboardInterrupt:
        logger.warning("прервано")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
