"""Материалы проекта: свои документы, таблицы и данные.

Требование брифа: «работать как минимум с документами, таблицами и сайтами
из разрешённых источников» и «сохранять ссылку на источник каждого
существенного сведения». Материалы проекта — третий слой знаний: они
относятся к конкретной задаче и не должны попадать в инструкции ролей.

Каждый материал становится источником с тем же идентификатором
``project/<файл>``, что и статьи корпуса, и проходит **ту же** проверку
дословности: цитата из таблицы подтверждается так же, как цитата из
статьи. Поэтому материалы можно смешивать с корпусом в одном срезе
источников — агент не различает, откуда пришёл текст, и не может
выдумать ни то, ни другое.

У каждого материала хранятся метаданные: адрес источника, дата
получения и хеш содержимого. Они нужны для пересмотра выводов: по
хешу видно, что вывод опирается на ту же версию данных.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pf.config import settings

#: Минимальная длина абзаца, пригодного для цитирования. Такой же порог,
#: как у корпуса: из короткой строки нечего цитировать.
MIN_PASSAGE_CHARS = 12

#: Максимальная длина абзаца в контексте.
MAX_PASSAGE_CHARS = 1200

#: Расширения документов, которые читаются как текст.
TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".text"})

#: Расширения таблиц.
TABLE_SUFFIXES = frozenset({".csv", ".tsv"})

#: Расширения структурированных данных.
DATA_SUFFIXES = frozenset({".json"})

#: Расширения документов с текстом: разбираются библиотеками, если
#: они установлены. Отсутствие библиотеки не должно ломать загрузку —
#: материал просто пропускается, а агент видит меньше источников.
DOCUMENT_SUFFIXES = frozenset({".pdf", ".docx", ".doc", ".xlsx"})

#: Все поддерживаемые расширения.
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | TABLE_SUFFIXES | DATA_SUFFIXES | DOCUMENT_SUFFIXES

#: Файлы, которые описывают материалы, а не содержат данные. README в
#: каталоге проекта — это инструкция человеку: если считать её источником,
#: агент начнёт цитировать собственную документацию как факт.
SERVICE_FILE_NAMES = frozenset({"readme.md", "readme.txt", "readme"})

#: Суффикс файла с метаданными. Метаданные относятся к соседнему файлу
#: с тем же именем: ``данные.csv`` + ``данные.meta.json``. Отдельный
#: файл метаданных не был бы источником сам по себе — в нём нет данных,
#: которые можно процитировать.
METADATA_SUFFIX = ".meta.json"


class MaterialError(RuntimeError):
    """Ошибка чтения материала."""


@dataclass(frozen=True, slots=True)
class MaterialPassage:
    """Абзац материала, пригодный для цитирования.

    Attributes:
        index: Номер абзаца в материале (нумерация с нуля).
        text: Текст абзаца.
    """

    index: int
    text: str


@dataclass(frozen=True, slots=True)
class Material:
    """Материал проекта с метаданными.

    Attributes:
        source_id: Идентификатор источника вида ``project/<файл>``.
        title: Заголовок материала.
        kind: Вид материала: документ, таблица или данные.
        passages: Абзацы, пригодные для цитирования.
        url: Адрес источника, если материал оттуда получен.
        retrieved: Дата получения материала.
        sha256: Хеш содержимого — версия данных, на которые опирался вывод.
        path: Путь к файлу на диске.
    """

    class Kind(StrEnum):
        """Вид материала."""

        #: Текстовый документ.
        DOCUMENT = "document"
        #: Таблица.
        TABLE = "table"
        #: Структурированные данные.
        DATA = "data"

    source_id: str
    title: str
    kind: Kind
    passages: list[MaterialPassage] = field(default_factory=list)
    url: str = ""
    retrieved: str = ""
    sha256: str = ""
    path: str = ""
    rows: list[dict[str, str]] = field(default_factory=list)

    @property
    def content(self) -> str:
        """Весь текст материала — абзацы, соединённые переводом строки."""
        return "\n".join(passage.text for passage in self.passages)

    def table_rows(self) -> list[dict[str, str]]:
        """Строки таблицы как словари «колонка → значение».

        Нужны для расчётов: по строке можно посчитать метрику и
        потом проверить её пересчётом. Для нетабличных материалов
        возвращается пустой список.

        Returns:
            Список строк.
        """
        return list(self.rows)


def _sha256(data: bytes) -> str:
    """Хеш содержимого материала.

    Args:
        data: Байты файла.

    Returns:
        Хеш в шестнадцатеричном виде.
    """
    return hashlib.sha256(data).hexdigest()


def _split_paragraphs(text: str) -> list[str]:
    """Нарезать текст на абзацы.

    Args:
        text: Текст документа.

    Returns:
        Список непустых абзацев.
    """
    blocks: list[str] = []
    for raw in re.split(r"\n\s*\n|\r\n\r\n", text):
        cleaned = " ".join(raw.split())
        if cleaned:
            blocks.append(cleaned[:MAX_PASSAGE_CHARS])
    return blocks


def _csv_rows(text: str, delimiter: str) -> list[str]:
    """Превратить таблицу в строки-абзацы.

    Строка читается как ``значение: значение, значение: значение`` —
    так она остаётся дословным фрагментом, который можно процитировать
    и проверить. Заголовок включается в каждую строку: без него цитата
    «8,1 %» не говорит, что это за метрика.

    Args:
        text: Содержимое таблицы.
        delimiter: Разделитель полей.

    Returns:
        Список строк.
    """
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return []

    header, *body = rows
    header = [cell.strip() for cell in header]
    lines: list[str] = []
    for row in body:
        cells = [cell.strip() for cell in row]
        pairs = [
            f"{header[index]}: {value}"
            for index, value in enumerate(cells)
            if index < len(header) and header[index] and value
        ]
        if pairs:
            lines.append(", ".join(pairs)[:MAX_PASSAGE_CHARS])
    return lines


def _csv_dicts(text: str, delimiter: str) -> list[dict[str, str]]:
    """Прочитать таблицу в словари «колонка → значение».

    Строки нужны для расчётов: метрика ссылается на номер строки, и
    по ней значение можно пересчитать и проверить.

    Args:
        text: Содержимое таблицы.
        delimiter: Разделитель полей.

    Returns:
        Список словарей. Первая строка файла считается заголовком.
    """
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return []
    header = [cell.strip() for cell in rows[0]]
    result: list[dict[str, str]] = []
    for row in rows[1:]:
        record: dict[str, str] = {}
        for index, cell in enumerate(row):
            if index < len(header) and header[index]:
                record[header[index]] = cell.strip()
        if record:
            result.append(record)
    return result


def _json_records(payload: Any) -> tuple[str, str, str, list[str]]:
    """Разобрать структурированные данные.

    Поддерживается два вида: объект с полями ``title``/``url``/
    ``retrieved``/``text`` и список записей с теми же полями.

    Args:
        payload: Разобранный JSON.

    Returns:
        Кортеж ``(заголовок, адрес, дата получения, абзацы)``.

    Raises:
        MaterialError: Формат не распознан.
    """
    if isinstance(payload, dict):
        items = [payload]
    elif isinstance(payload, list):
        items = [item for item in payload if isinstance(item, dict)]
    else:
        raise MaterialError("ожидался объект или список объектов")

    title = ""
    url = ""
    retrieved = ""
    blocks: list[str] = []
    for item in items:
        title = title or str(item.get("title", ""))
        url = url or str(item.get("url", ""))
        retrieved = retrieved or str(item.get("retrieved", ""))
        text = item.get("text") or item.get("content")
        if isinstance(text, str) and text.strip():
            blocks.extend(_split_paragraphs(text))
            continue
        # Запись без готового текста: поля становятся «значение: значение»,
        # но вложенный список (например, records) разворачивается в свои
        # же записи — иначе все метрики склеиваются в один абзац.
        for key, value in item.items():
            if key in {"title", "url", "retrieved", "note"}:
                continue
            if isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
                nested_title, nested_url, nested_retrieved, nested_blocks = _json_records(value)
                title = title or nested_title
                url = url or nested_url
                retrieved = retrieved or nested_retrieved
                blocks.extend(nested_blocks)
                continue
            if value not in (None, ""):
                blocks.append(f"{key}: {value}"[:MAX_PASSAGE_CHARS])
    return title, url, retrieved, blocks


def _read_document(path: Path) -> str:
    """Прочитать текст из документа.

    Args:
        path: Путь к файлу.

    Returns:
        Текст документа.

    Raises:
        MaterialError: Формат не поддерживается или библиотека не установлена.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pdfplumber
        except ImportError as exc:
            raise MaterialError("для чтения PDF нужен pdfplumber: uv add pdfplumber") from exc
        pages: list[str] = []
        with pdfplumber.open(path) as document:
            for page in document.pages:
                pages.append(page.extract_text() or "")
        return "\n\n".join(pages)

    if suffix in {".docx", ".doc"}:
        try:
            from docx import Document
        except ImportError as exc:
            raise MaterialError("для чтения DOCX нужен python-docx: uv add python-docx") from exc
        document = Document(str(path))
        return "\n\n".join(paragraph.text for paragraph in document.paragraphs)

    if suffix == ".xlsx":
        try:
            from openpyxl import load_workbook
        except ImportError as exc:
            raise MaterialError("для чтения XLSX нужен openpyxl: uv add openpyxl") from exc
        workbook = load_workbook(path, read_only=True, data_only=True)
        blocks: list[str] = []
        for sheet in workbook.worksheets:
            blocks.append(f"Лист «{sheet.title}»")
            for row in sheet.iter_rows(values_only=True):
                cells = [str(value) for value in row if value is not None]
                if cells:
                    blocks.append(", ".join(cells))
        workbook.close()
        return "\n\n".join(blocks)

    raise MaterialError(f"формат {suffix} не поддерживается")


def _read_sidecar(path: Path) -> dict[str, str]:
    """Прочитать метаданные материала из соседнего файла.

    Метаданные лежат в файле ``<имя>.meta.json`` рядом с материалом и
    описывают его: адрес источника, дату получения, пояснение.

    Args:
        path: Путь к файлу материала.

    Returns:
        Словарь с полями ``title``, ``url``, ``retrieved``, ``note``.
    """
    sidecar = path.with_name(path.name + METADATA_SUFFIX)
    if not sidecar.is_file():
        return {}
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        "title": str(payload.get("title", "")),
        "url": str(payload.get("url", "")),
        "retrieved": str(payload.get("retrieved", "")),
        "note": str(payload.get("note", "")),
    }


def _material_from_file(path: Path, root: Path) -> Material | None:
    """Собрать материал из файла.

    Args:
        path: Путь к файлу.
        root: Корень каталога материалов — из него строится идентификатор.

    Returns:
        Материал либо ``None``, если формат не поддерживается.
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return None

    try:
        raw = path.read_bytes()
    except OSError:
        return None

    sidecar = _read_sidecar(path)
    title = sidecar.get("title") or path.stem
    url = sidecar.get("url", "")
    retrieved = sidecar.get("retrieved", "")
    kind = Material.Kind.DOCUMENT
    blocks: list[str] = []
    rows: list[dict[str, str]] = []

    try:
        if suffix in TEXT_SUFFIXES:
            blocks = _split_paragraphs(raw.decode("utf-8", errors="replace"))
        elif suffix in TABLE_SUFFIXES:
            kind = Material.Kind.TABLE
            delimiter = "\t" if suffix == ".tsv" else ","
            text = raw.decode("utf-8", errors="replace")
            blocks = _csv_rows(text, delimiter)
            rows = _csv_dicts(text, delimiter)
        elif suffix in DATA_SUFFIXES:
            kind = Material.Kind.DATA
            payload = json.loads(raw.decode("utf-8", errors="replace"))
            title, url, retrieved, blocks = _json_records(payload)
        else:
            blocks = _split_paragraphs(_read_document(path))
    except (MaterialError, json.JSONDecodeError, OSError, ValueError):
        # Материал, который не прочитался, не должен ронять загрузку:
        # агент работает с тем, что доступно, и называет пробелы.
        return None

    passages = [
        MaterialPassage(index=index, text=block)
        for index, block in enumerate(blocks)
        if len(block) >= MIN_PASSAGE_CHARS
    ]
    if not passages:
        return None

    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(path.name)
    source_id = f"project/{relative.as_posix()}"

    return Material(
        source_id=source_id,
        title=title,
        kind=kind,
        passages=passages,
        url=url,
        retrieved=retrieved,
        sha256=_sha256(raw),
        path=str(path),
        rows=rows,
    )


@dataclass(frozen=True, slots=True)
class ResolvedCitation:
    """Ссылка, разрешённая в адрес источника.

    Attributes:
        source_id: Идентификатор источника.
        url: Адрес, по которому можно открыть источник.
        paragraph: Номер абзаца, если ссылка была на абзац.
        row: Номер строки, если ссылка была на строку таблицы.
        title: Заголовок источника.
    """

    source_id: str
    url: str
    title: str = ""
    paragraph: int | None = None
    row: int | None = None

    @property
    def anchor(self) -> str:
        """Якорь внутри адреса — на абзац или строку.

        Returns:
            Якорь вида ``#p12`` или ``#row3`` либо пустая строка.
        """
        if self.paragraph is not None:
            return f"#p{self.paragraph}"
        if self.row is not None:
            return f"#row{self.row}"
        return ""


def _wikipedia_url(title: str) -> str:
    """Адрес статьи Википедии по заголовку.

    Args:
        title: Заголовок статьи.

    Returns:
        Адрес статьи.
    """
    from urllib.parse import quote

    return f"https://ru.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"


def resolve_citation(
    citation: str,
    *,
    directory: Path | None = None,
) -> ResolvedCitation | None:
    """Разрешить ссылку на источник в адрес, по которому его можно открыть.

    Идентификатор вроде ``project/файл#row3`` или ``telecom/Статья#p12``
    — это не ссылка: человек по нему ничего не откроет. Здесь
    идентификатор превращается в адрес: для материала проекта берётся
    адрес из его метаданных, для статьи корпуса — адрес статьи
    Википедии.

    Args:
        citation: Ссылка вида ``источник#pN`` или ``источник#rowN``.
        directory: Каталог материалов. По умолчанию — ``project/``.

    Returns:
        Разрешённая ссылка либо ``None``, если источник неизвестен.
    """
    if not citation:
        return None

    clean, _, fragment = citation.partition("#")
    paragraph: int | None = None
    row: int | None = None
    if fragment.startswith("p") and fragment[1:].isdigit():
        paragraph = int(fragment[1:])
    elif fragment.startswith("row") and fragment[3:].isdigit():
        row = int(fragment[3:])

    root = directory or settings.project_dir
    if clean.startswith("project/"):
        relative = clean.removeprefix("project/")
        path = root / relative
        sidecar = _read_sidecar(path)
        if not path.is_file():
            return None
        return ResolvedCitation(
            source_id=clean,
            url=sidecar.get("url", "") or path.as_uri(),
            title=sidecar.get("title", "") or path.stem,
            paragraph=paragraph,
            row=row,
        )

    # Статья корпуса: раздел/заголовок. Адрес — статья Википедии.
    section, _, title = clean.partition("/")
    if not title:
        return None
    return ResolvedCitation(
        source_id=clean,
        url=_wikipedia_url(title),
        title=title,
        paragraph=paragraph,
        row=row,
    )


def load_materials(directory: Path | None = None) -> list[Material]:
    """Загрузить материалы проекта.

    Обходятся все поддерживаемые форматы. Неподдерживаемые и нечитаемые
    файлы пропускаются: отсутствие материала — это пробел в данных, а не
    повод остановить работу.

    Args:
        directory: Каталог материалов. По умолчанию — ``project/``.

    Returns:
        Список материалов.
    """
    root = directory or settings.project_dir
    if not root.is_dir():
        return []

    materials: list[Material] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.name.lower() in SERVICE_FILE_NAMES:
            continue
        if path.name.endswith(METADATA_SUFFIX):
            continue
        material = _material_from_file(path, root)
        if material is not None:
            materials.append(material)
    return materials
