# Подключение MCP-серверов к агентам

Бриф: «Среда создания и функционирования — любая. GigaCode, Кодик, Codex,
Claude Code, Cursor, GitHub Copilot, Windsurf, OpenCode и другие».
Агент работает и без MCP — с локальным корпусом и своими материалами. Но
MCP добавляют инструменты: разбор страниц, документы, слепки интерфейсов,
поисковый спрос.

## Что уже настроено

| Что | Где | Статус |
|---|---|---|
| Конфигурация серверов | `mcp/servers.json` | 6 серверов |
| Конфиг для GigaCode | `.gigacode_vsc/gigacode.jsonc` | готов, копируется в проект |
| Конфиг для Cursor | `docs/mcp/cursor-mcp.json` | готов, копируется в `~/.cursor/mcp.json` |
| Переменные | `.env` | `PF_MCP_DIR`, ключи к внешним серверам |

## Серверы

| Сервер | Что даёт | Путь |
|---|---|---|
| `static-parser` | Разбор веб-страниц: абзацы, метаданные, `robots.txt` | `PF_MCP_DIR/static-parser-mcp` |
| `parser-storage` | Хранилище разобранных документов: полнотекст, дедупликация | `PF_MCP_DIR/parser-storage-mcp` |
| `document-transformer` | Извлечение текста из PDF, DOCX и изображений с OCR | `PF_MCP_DIR/document-transformer-mcp` |
| `ux-snapshot` | Слепки интерфейсов: очищенный HTML для анализа UX | `PF_MCP_DIR/ux-snapshot-mcp` |
| `vision` | Скриншоты и распознавание текста на экране | `PF_MCP_DIR/vision-mcp` |
| `wordstat` | Поисковый спрос Яндекс Wordstat: частотность и динамика | `npx yandex-wordstat-mcp` |

## Шаг 1. Путь к серверам

Серверы живут в соседнем репозитории `mcp-servers` — у них тяжёлые
зависимости, несовместимые с лёгким окружением агента. Путь задаётся
переменной, иначе конфигурация перестаёт переноситься:

```bash
cp .env.example .env
# в .env:
PF_MCP_DIR=../mcp-servers
```

Значение читается **из `.env`** — переменная окружения процесса имеет
приоритет.

## Шаг 2. Проверить, что серверы поднимаются

```bash
pf mcp                       # список серверов
pf mcp static-parser         # инструменты сервера
pf mcp document-transformer  # инструменты сервера
```

Если сервер не поднялся, команда покажет причину: нет интерпретатора,
не установлены зависимости или не найден каталог.

## Шаг 3. Подключить к среде разработки

### Cursor

```bash
mkdir -p ~/.cursor
cp docs/mcp/cursor-mcp.json ~/.cursor/mcp.json
```

В файле замените `${PF_MCP_DIR}` на абсолютный путь к репозиторию
серверов — Cursor не читает `.env` проекта. Ключи к внешним серверам
(`wordstat`) в `~/.cursor/mcp.json` **не** кладите: держите их в
`~/.cursor/mcp.json` с правами `600` или в менеджере секретов.

### GigaCode

```bash
cp .gigacode_vsc/gigacode.jsonc <ваш-проект>/.gigacode_vsc/gigacode.jsonc
```

Пути в конфиге относительные: `../mcp-servers`. Если проект лежит в другом
месте — поправьте вручную.

### Cline

```json
{
  "mcpServers": {
    "static-parser": {
      "command": "/абсолютный/путь/mcp-servers/static-parser-mcp/.venv/bin/python",
      "args": ["/абсолютный/путь/mcp-servers/static-parser-mcp/server.py"]
    }
  }
}
```

## Ключи к внешним серверам

### Yandex Wordstat

Сервер `wordstat` работает с **Wordstat v2** — это Yandex Cloud Search
API. Старый OAuth-токен `api.wordstat.yandex.net` больше не принимается.

Что нужно:

1. Сервисный аккаунт в [Yandex Cloud](https://console.yandex.cloud/) с ролью
   `search-api.webSearch.user`.
2. API-ключ со scope `yc.search-api.execute`.
3. Идентификатор папки (folder ID).

Ключ кладётся в `.env` агента и **не коммитится**:

```bash
# .env
YANDEX_SEARCH_API_KEY=...
YANDEX_FOLDER_ID=...
```

Агент передаёт их серверу сам: клиент MCP умеет отдавать переменные
окружения из конфигурации.

## Чего MCP не делает

- Не заменяет локальный корпус: цитаты из MCP-страниц всегда имеют
  статус «требует проверки», а не «подтверждено».
- Не даёт доступа к закрытым данным: серверы работают с открытыми
  источниками.
- Не требует ключей для базовой работы: `static-parser`,
  `document-transformer`, `ux-snapshot` и `vision` работают без ключей.
