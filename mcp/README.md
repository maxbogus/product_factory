# MCP — подключение инструментов

MCP (Model Context Protocol) — протокол подключения данных и инструментов
к агенту. Агент может работать и без него: сейчас используется локальный
корпус.

## Конфигурация

```json
{
  "mcpServers": {
    "static-parser": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "tools/static-parser"
    }
  }
}
```

## Серверы

| Сервер | Что даёт | Ключ нужен |
|---|---|---|
| `static-parser` | разбор сайтов из разрешённых доменов, `robots.txt` | нет |
| `document-transformer` | текст из PDF, DOCX и изображений с OCR | нет |
| `parser-storage` | база знаний, полнотекст, дедупликация | нет |
| `ux-snapshot` | слепки интерфейсов: очищенный HTML для анализа UX | нет |
| `vision` | скриншоты и распознавание текста на экране | нет |
| `wordstat` | поисковый спрос: частотность и динамика запросов | да (Yandex Cloud) |

Реализация — переиспользуемый код из соседнего репозитория `mcp-servers`,
конфигурация и подключение — здесь.

## Подключение

Полная инструкция для Cursor, GigaCode и Cline — в
[`docs/MCP_SETUP.md`](../docs/MCP_SETUP.md). Коротко:

```bash
# 1. путь к репозиторию серверов
cp .env.example .env     # в .env: PF_MCP_DIR=../mcp-servers

# 2. проверка, что серверы поднимаются
pf mcp static-parser
pf mcp document-transformer
```

## Что важно знать

- **Ключи не в репозитории.** `wordstat` получает `YANDEX_SEARCH_API_KEY`
  и `YANDEX_FOLDER_ID` из `.env`; клиент сам передаёт их серверу.
- **Wordstat v2 — это Yandex Cloud API.** Старый OAuth-токен
  `api.wordstat.yandex.net` больше не принимается.
- **Внешнее ≠ подтверждено.** Даже если MCP-сервер вернул точную цитату,
  её статус — «требует проверки»: подтверждает код агента, а не сервер.
- **`.xlsx` не читается `document-transformer`** — его форматы: PDF, DOCX,
  DOC и изображения. Таблицы в XLSX агент читает сам, через
  [`materials.py`](../src/pf/materials.py).
