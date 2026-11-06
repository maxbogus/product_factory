"""Проверки клиента MCP: конфигурация и подстановка переменных.

MCP-серверы лежат в соседнем репозитории, и их путь задаётся переменной
``PF_MCP_DIR``. Пользователь кладёт её в ``.env`` — так написано в
``.env.example``. Значит, клиент обязан читать её **оттуда**, а не только
из окружения процесса: иначе инструкция «скопируйте .env.example в .env»
не работает, а MCP не запускается.

Второе: серверы с ключами (Wordstat, Google Trends) получают доступ
через переменные окружения. Конфигурация обязана уметь их передавать.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pf.config import Settings
from pf.mcp import ServerSpec, load_config


def test_env_file_value_is_used_for_mcp_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PF_MCP_DIR из .env должен доходить до конфигурации серверов."""
    env_file = tmp_path / ".env"
    env_file.write_text("PF_MCP_DIR=/opt/mcp-servers\n", encoding="utf-8")
    monkeypatch.delenv("PF_MCP_DIR", raising=False)

    config = Settings(_env_file=env_file)
    servers = load_config(settings=config)
    spec = servers["static-parser"]
    assert spec.command.startswith("/opt/mcp-servers"), (
        f"путь к серверам должен подставляться из .env, получено: {spec.command}"
    )


def test_unknown_variable_is_left_visible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Незаданная переменная остаётся видимой — так ошибка заметна."""
    monkeypatch.delenv("PF_MCP_DIR", raising=False)
    config = Settings(_env_file=tmp_path / "none.env")

    payload = {
        "mcpServers": {
            "x": {"command": "${PF_MCP_DIR}/server.py"},
        }
    }
    path = tmp_path / "servers.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    servers = load_config(path, settings=config)
    assert "${PF_MCP_DIR}" in servers["x"].command, (
        "незаданная переменная должна остаться в команде, а не стать пустой строкой"
    )


def test_server_env_is_read_from_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Секреты сервера должны читаться из конфигурации."""
    monkeypatch.delenv("YANDEX_SEARCH_API_KEY", raising=False)
    config = Settings(_env_file=tmp_path / "none.env")

    payload = {
        "mcpServers": {
            "wordstat": {
                "command": "npx",
                "args": ["-y", "yandex-wordstat-mcp"],
                "env": {"YANDEX_SEARCH_API_KEY": "${YANDEX_SEARCH_API_KEY}"},
            }
        }
    }
    path = tmp_path / "servers.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    servers = load_config(path, settings=config)
    assert servers["wordstat"].env == {"YANDEX_SEARCH_API_KEY": "${YANDEX_SEARCH_API_KEY}"}


def test_server_spec_carries_env() -> None:
    """Описание сервера умеет хранить переменные окружения."""
    spec = ServerSpec(name="x", command="npx", env={"TOKEN": "abc"})
    assert spec.env["TOKEN"] == "abc"


def test_all_servers_are_configured() -> None:
    """В конфигурации должны быть все заявленные серверы.

    Заявленные серверы — это то, что агент обещает уметь: разбор
    страниц, документы, слепки интерфейсов, поисковый спрос.
    """
    servers = load_config()
    expected = {
        "static-parser",
        "parser-storage",
        "document-transformer",
        "ux-snapshot",
        "vision",
        "wordstat",
    }
    assert expected <= set(servers), f"не хватает серверов: {expected - set(servers)}"


def test_cursor_config_matches_server_config() -> None:
    """Конфиг для Cursor должен описывать те же серверы, что и основной."""
    root = Path(__file__).resolve().parent.parent
    cursor = json.loads((root / "docs" / "mcp" / "cursor-mcp.json").read_text(encoding="utf-8"))
    servers = load_config()
    assert set(cursor["mcpServers"]) == set(servers), (
        "конфиг для Cursor разошёлся с основной конфигурацией"
    )


def test_gigacode_config_matches_server_config() -> None:
    """Конфиг для GigaCode должен описывать те же серверы, что и основной."""
    root = Path(__file__).resolve().parent.parent
    gigacode = json.loads((root / ".gigacode_vsc" / "gigacode.jsonc").read_text(encoding="utf-8"))
    servers = load_config()
    assert set(gigacode["mcpServers"]) == set(servers), (
        "конфиг для GigaCode разошёлся с основной конфигурацией"
    )


def test_wordstat_key_is_passed_to_server() -> None:
    """Ключ к Wordstat должен доходить до сервера.

    Без переменных окружения сервер запустится, но не получит доступ к
    API: ошибка будет выглядеть как «сервер не отвечает», а не как
    «нет ключа».
    """
    root = Path(__file__).resolve().parent.parent
    payload = json.loads((root / "mcp" / "servers.json").read_text(encoding="utf-8"))
    env = payload["mcpServers"]["wordstat"]["env"]
    assert env["YANDEX_SEARCH_API_KEY"] == "${YANDEX_SEARCH_API_KEY}"
    assert env["YANDEX_FOLDER_ID"] == "${YANDEX_FOLDER_ID}"
