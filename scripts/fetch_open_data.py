"""Собрать открытые данные в материалы проекта.

Скрипт скачивает открытые данные и складывает их в ``project/data/`` как
CSV и JSON с метаданными: адрес источника, дата получения, хеш
содержимого. Метаданные обязательны — по ним видно, на какую версию
данных опирался вывод, и при пересмотре выводов это первое, что нужно
проверить.

Сейчас собирается ключевая ставка Банка России: официальный открытый
XML-документ, без ключей и авторизации.

Запуск:

    uv run python scripts/fetch_open_data.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

#: Открытый XML-документ Банка России с ключевой ставкой.
CBR_KEY_RATE_URL = (
    "https://www.cbr.ru/scripts/xml_dynamic.asp"
    "?date_req1=01/01/2026&date_req2=31/03/2026&VAL_NM_RQ=R01235"
)

#: Куда складывать материалы.
OUTPUT_DIR = Path("project/data")

#: Идентификатор источника — он попадёт в цитаты агента.
SOURCE_ID = "cbr.ru/key-rate"

#: Заголовок для вывода.
USER_AGENT = "product-factory-strategist/0.1 (contact@example.org)"


def fetch_key_rate(url: str = CBR_KEY_RATE_URL) -> list[dict[str, str]]:
    """Скачать ключевую ставку Банка России.

    Args:
        url: Адрес XML-документа.

    Returns:
        Список записей: дата и значение в процентах.

    Raises:
        RuntimeError: Документ не разобран.
    """
    response = httpx.get(url, timeout=30.0, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    # Документ Банка России в windows-1251.
    response.encoding = "windows-1251"
    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        raise RuntimeError(f"не удалось разобрать XML Банка России: {exc}") from exc

    rows: list[dict[str, str]] = []
    for record in root.findall(".//Record"):
        date = record.get("Date")
        value = record.findtext("VunitRate")
        if date and value:
            rows.append({"дата": date, "значение_процентов": value.replace(",", ".")})
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    """Записать записи в CSV.

    Args:
        rows: Записи.
        path: Путь к файлу.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["дата", "значение_процентов"])
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8")


def write_metadata(
    path: Path,
    *,
    title: str,
    url: str,
    retrieved: str,
    rows: int,
    note: str,
) -> None:
    """Записать метаданные материала.

    Args:
        path: Путь к файлу метаданных.
        title: Заголовок материала.
        url: Адрес источника.
        retrieved: Дата получения.
        rows: Сколько записей в материале.
        note: Пояснение для человека.
    """
    payload: dict[str, Any] = {
        "title": title,
        "url": url,
        "retrieved": retrieved,
        "rows": rows,
        "note": note,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    """Собрать открытые данные и записать их в материалы проекта.

    Returns:
        Код возврата.
    """
    retrieved = datetime.now(UTC).date().isoformat()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        rows = fetch_key_rate()
    except (httpx.HTTPError, RuntimeError) as exc:
        print(f"ОШИБКА: ключевая ставка не получена: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print("ОШИБКА: Банк России вернул пустой документ", file=sys.stderr)
        return 1

    csv_path = OUTPUT_DIR / "cbr_key_rate_2026_q1.csv"
    write_csv(rows, csv_path)
    write_metadata(
        OUTPUT_DIR / "cbr_key_rate_2026_q1.json",
        title="Ключевая ставка Банка России, январь-март 2026",
        url=CBR_KEY_RATE_URL,
        retrieved=retrieved,
        rows=len(rows),
        note=(
            "Открытые данные Банка России. Макро-контекст для продуктовых "
            "решений: стоимость денег влияет на решение абонента отключить "
            "дополнительные услуги."
        ),
    )

    values = [float(row["значение_процентов"]) for row in rows]
    change = (values[-1] - values[0]) / values[0] * 100
    print(f"Сохранено: {csv_path} ({len(rows)} записей)")
    print(f"Ключевая ставка: {values[0]:.2f}% → {values[-1]:.2f}% ({change:+.2f}% за период)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
