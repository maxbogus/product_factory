"""Живые тесты с локальной моделью.

Эти тесты бьют по настоящей модели через Ollama: их нельзя заменить
подделкой, потому что проверяется именно способность модели работать
с агентом — отвечать строгим JSON по схеме и цитировать дословно.

Они помечены ``slow`` и не входят в обычный прогон: ``make test-slow``.

Если Ollama недоступен, тесты пропускаются, а не падают: отсутствие
модели — не дефект кода.
"""

from __future__ import annotations

import pytest

from pf.config import settings
from pf.ground import SupportStatus, verify_quote
from pf.llm import build_client, extract_json
from pf.schemas import Variants

pytestmark = pytest.mark.slow


def _model_available() -> bool:
    """Отвечает ли локальная модель.

    Returns:
        Модель доступна.
    """
    import httpx

    try:
        response = httpx.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
    except httpx.HTTPError:
        return False
    names = [item["name"] for item in response.json().get("models", [])]
    return settings.ollama_model in names or bool(names)


requires_model = pytest.mark.skipif(
    not _model_available(), reason="локальная модель недоступна — запустите Ollama"
)


@requires_model
def test_model_answers_valid_json() -> None:
    """Модель отвечает JSON, который агент может разобрать."""
    client = build_client()
    response = client.complete(
        "Ты отвечаешь ТОЛЬКО валидным JSON без пояснений.",
        'Верни JSON: {"facts": [{"text": "текст"}], "gaps": []}',
    )
    payload = extract_json(response.content)
    assert isinstance(payload, dict)
    assert "facts" in payload


@requires_model
def test_model_produces_three_variants() -> None:
    """Модель может собрать три варианта с возражениями.

    Это проверка того, с чем агент работает на фазе вариантов: без
    возражений вариант не проходит схему.
    """
    client = build_client()
    response = client.complete(
        "Ты отвечаешь ТОЛЬКО валидным JSON без пояснений.",
        """Верни JSON с тремя вариантами решения:
{"variants": [{"title": "А", "summary": "с", "objections": ["о"]},
{"title": "Б", "summary": "с", "objections": ["о"]},
{"title": "В", "summary": "с", "objections": ["о"]}]}""",
    )
    payload = extract_json(response.content)
    variants = Variants.model_validate(payload)
    assert len(variants.variants) >= 3
    assert all(item.objections for item in variants.variants)


@requires_model
def test_model_quotes_source_verbatim() -> None:
    """Модель умеет привести дословную цитату.

    Без этого механизм честности не работает: цитата — единственное
    основание, которое агент принимает.
    """
    source = (
        "По состоянию на июнь 2024 года компания обслужила 77,26 миллиона "
        "абонентов."
    )
    client = build_client()
    response = client.complete(
        "Ты цитируешь источник дословно.",
        f"Приведи цитату из источника: {source}",
    )
    support = verify_quote(source, response.content.strip().strip('"'), "test/s")
    assert support.status is SupportStatus.CONFIRMED, support.detail


@requires_model
def test_agent_survives_whole_workflow() -> None:
    """Агент проходит весь рабочий процесс на живой модели."""
    from pf.state import Phase
    from pf.workflow import run_full_workflow

    session = "slow-workflow"
    state = run_full_workflow(
        "Почему падает удержание у мобильного тарифа и что с этим делать?",
        session,
    )
    assert state.phase in {Phase.DONE, Phase.KEEPER}
    assert state.steps_done >= 5
