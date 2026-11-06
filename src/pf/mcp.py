"""Клиент MCP: подключение к внешним инструментам по протоколу.

MCP (Model Context Protocol) — протокол подключения данных и инструментов
к агенту. Серверы лежат в соседнем репозитории ``mcp-servers`` и имеют
тяжёлые зависимости (requests, beautifulsoup4, selectolax), несовместимые
с лёгким окружением агента. Поэтому сервер запускается **в своём
окружении**: путь к интерпретатору задаётся настройкой.

Клиент работает по stdio и держит сессию MCP. Инструменты вызываются по
имени с аргументами; ответ — текст.

Зачем это агенту: веб-источники по умолчанию выключены (см. ``PF_ENABLE_WEB``),
и MCP даёт способ подключить их явно, не таская в рантайм агента чужие
зависимости.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pf.config import settings


@dataclass(frozen=True, slots=True)
class Tool:
    """Инструмент, объявленный MCP-сервером.

    Attributes:
        name: Имя инструмента.
        description: Что он делает.
        schema: Схема аргументов.
    """

    name: str
    description: str = ""
    schema: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ServerSpec:
    """Описание MCP-сервера.

    Attributes:
        name: Имя сервера в конфигурации.
        command: Команда запуска.
        args: Аргументы команды.
        cwd: Рабочий каталог.
    """

    name: str
    command: str
    args: tuple[str, ...] = ()
    cwd: str = ""


class MCPError(RuntimeError):
    """Ошибка работы с MCP-сервером."""


def load_config(path: Path | None = None) -> dict[str, ServerSpec]:
    """Прочитать конфигурацию серверов.

    В конфигурации можно использовать ``${ПЕРЕМЕННАЯ}`` — подставляется
    значение переменной окружения. Так путь к соседнему репозиторию
    задаётся один раз (``PF_MCP_DIR``), а не зашивается в файл: иначе
    конфигурация перестаёт переноситься на другую машину.

    Args:
        path: Путь к конфигурации. По умолчанию — ``mcp/servers.json``.

    Returns:
        Словарь «имя сервера → описание».
    """
    target = path or settings.mcp_config_path
    if not target.exists():
        return {}
    raw = target.read_text(encoding="utf-8")
    raw = _expand_env(raw)
    payload = json.loads(raw)
    servers: dict[str, ServerSpec] = {}
    for name, spec in payload.get("mcpServers", {}).items():
        servers[name] = ServerSpec(
            name=name,
            command=str(spec.get("command", "python")),
            args=tuple(str(a) for a in spec.get("args", ())),
            cwd=str(spec.get("cwd", "")),
        )
    return servers


def _expand_env(raw: str) -> str:
    """Подставить переменные окружения в конфигурацию.

    Неизвестная переменная оставляется как есть: так видно в ошибке,
    что переменная не задана, а не что путь молча стал пустым.

    Args:
        raw: Текст конфигурации.

    Returns:
        Текст с подставленными значениями.
    """

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return os.environ.get(key, match.group(0))

    return re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", replace, raw)


async def list_tools(spec: ServerSpec) -> list[Tool]:
    """Получить список инструментов сервера.

    Args:
        spec: Описание сервера.

    Returns:
        Список инструментов.

    Raises:
        MCPError: Сервер недоступен или ответ разобран неверно.
    """
    from mcp.client.stdio import stdio_client

    from mcp import ClientSession, StdioServerParameters

    params = StdioServerParameters(
        command=spec.command,
        args=list(spec.args),
        cwd=spec.cwd or None,
    )
    try:
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.list_tools()
    except Exception as exc:  # noqa: BLE001 — сервер может подняться как угодно
        raise MCPError(f"сервер {spec.name} недоступен: {exc}") from exc

    tools: list[Tool] = []
    for item in result.tools:
        tools.append(
            Tool(
                name=item.name,
                description=getattr(item, "description", "") or "",
                schema=getattr(item, "inputSchema", None),
            )
        )
    return tools


async def call_tool(spec: ServerSpec, tool: str, arguments: dict[str, Any]) -> str:
    """Вызвать инструмент сервера.

    Args:
        spec: Описание сервера.
        tool: Имя инструмента.
        arguments: Аргументы вызова.

    Returns:
        Текстовый ответ сервера.

    Raises:
        MCPError: Вызов не удался.
    """
    from mcp.client.stdio import stdio_client

    from mcp import ClientSession, StdioServerParameters

    params = StdioServerParameters(
        command=spec.command,
        args=list(spec.args),
        cwd=spec.cwd or None,
    )
    try:
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(tool, arguments)
    except Exception as exc:  # noqa: BLE001 — сервер может подняться как угодно
        raise MCPError(f"вызов {tool} не удался: {exc}") from exc

    parts: list[str] = []
    for item in result.content:
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def list_tools_sync(server: str, config_path: Path | None = None) -> list[Tool]:
    """Синхронная обёртка над ``list_tools``.

    Args:
        server: Имя сервера.
        config_path: Путь к конфигурации.

    Returns:
        Список инструментов.

    Raises:
        MCPError: Сервер не описан в конфигурации.
    """
    servers = load_config(config_path)
    spec = servers.get(server)
    if spec is None:
        raise MCPError(f"сервер {server!r} не описан в конфигурации")
    return asyncio.run(list_tools(spec))
