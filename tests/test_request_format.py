"""Проверки того, что модель получает схему ответа.

Отдельная роль проверяется в других файлах. Здесь — другое: **форма
запроса**. Модель возвращает структуру того, что видела во входных данных,
а не той схемы, которую ждёт код: на разборе с шестью фактами стратег
возвращал поля варианта (``title``, ``summary``, ``approach``) вместо полей
стратегии (``product``, ``communication``, ``verification``).

Причина не в промпте: схема в нём есть, но длинный вход вытесняет её из
внимания модели. Поэтому схему передаёт код — из той же схемы, по которой
ответ потом проверяется. Расхождение промпта и кода исключено по
построению.
"""

from __future__ import annotations

from pf.roles import schema_for
from pf.workflow import Workflow


def _user_for(role: str) -> str:
    """Собрать запрос роли.

    Args:
        role: Идентификатор роли.

    Returns:
        Текст запроса.
    """
    return Workflow._build_user({"question": "тест"}, "", role)


def test_request_contains_response_schema() -> None:
    """В запросе есть JSON-схема ожидаемого ответа."""
    user = _user_for("strategy")
    assert "СХЕМА ОТВЕТА" in user
    assert "product" in user
    assert "communication" in user
    assert "verification" in user


def test_schema_matches_code_schema() -> None:
    """Схема в запросе — та же, по которой проверяется ответ.

    Иначе модель получает одну форму, а код ждёт другую.
    """
    user = _user_for("strategy")
    schema = schema_for("strategy").model_json_schema()

    for field in ("product", "communication", "verification"):
        assert field in schema.get("properties", {})
        assert field in user


def test_schema_names_required_fields() -> None:
    """В запросе перечислены обязательные поля схемы."""
    user = _user_for("keeper")
    schema = schema_for("keeper").model_json_schema()

    for field in schema.get("required", []):
        assert field in user


def test_request_tells_model_not_to_echo_input() -> None:
    """В запросе сказано не возвращать входные ключи.

    Модель по умолчанию повторяет структуру того, что видела.
    """
    user = _user_for("strategy")
    assert "входные" in user.lower()


def test_nested_fields_are_expanded() -> None:
    """Поля вложенных блоков перечислены явно.

    В JSON-схеме вложенные модели лежат в ``$defs`` и подставляются
    ссылкой ``$ref``. Без развёртки подсказка показывала бы пустые блоки —
    и модель возвращала бы вместо стратегии структуру варианта.
    """
    user = _user_for("strategy")
    assert "key_message" in user
    assert "chosen_direction" in user
    assert "expected_change" in user
    assert "$ref" not in user
    assert "$defs" not in user


def test_required_fields_are_marked() -> None:
    """Обязательные поля отличаются от необязательных."""
    user = _user_for("strategy")
    # key_message обязателен, proof — нет.
    assert "key_message: str" in user
    assert "proof: str (необязательное)" in user


def test_input_data_is_not_json() -> None:
    """Входные данные поданы текстом, а не JSON.

    Это измерено: при JSON во входе модель отвечает его эхом — на
    материалах по удержанию возвращала пустые списки фактов (за три
    попытки: 1, 0, 0), а на текстовой подаче — факт каждый раз
    (1, 1, 1). JSON во входе задаёт форму ответа, и модель заполняет её
    структурой входа вместо извлечения содержимого.
    """
    user = _user_for("collector")
    data_block = user.split("ДАННЫЕ")[1].split("СХЕМА ОТВЕТА")[0]
    assert "{" not in data_block
    assert "}" not in data_block
    assert "[{" not in data_block


def test_input_data_keeps_its_content() -> None:
    """Текстовая подача не теряет содержимое входных данных."""
    user = Workflow._build_user(
        {"query": "удержание клиентов", "plan": {"questions": ["какие тарифы"]}},
        "",
        "collector",
    )
    data_block = user.split("ДАННЫЕ")[1].split("СХЕМА ОТВЕТА")[0]
    assert "удержание клиентов" in data_block
    assert "какие тарифы" in data_block


def test_data_block_is_marked_as_not_citable() -> None:
    """Блок входных данных помечен как непригодный для цитирования.

    Модель цитировала вопросы из плана исследования, а не абзацы
    источников: блок ДАННЫЕ шёл без пояснения, и модель путала, что
    можно процитировать.
    """
    user = _user_for("collector")
    assert "цитировать их нельзя" in user


def test_materials_block_says_where_to_take_source_ids() -> None:
    """Блок материалов говорит, откуда брать source_id.

    Без этого модель придумывала названия источников («Churn Rate»,
    «query») — и все её основания отклонялись.
    """
    user = Workflow._build_user({"query": "удержание"}, "[a/b#p1] текст", "collector")
    materials = user.split("МАТЕРИАЛЫ")[1].split("ДАННЫЕ")[0]
    assert "source_id" in materials
