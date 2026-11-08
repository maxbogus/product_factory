"""Язык ответа задаётся явно и одинаково для всех ролей.

Ответы агента и схема ответа — русские, но ни в одном из девяти промптов
ролей упоминания языка не было вовсе (`grep` по `prompts/roles/*.md` пуст).
Пока контекст русский, это незаметно; на английском материале (препринты,
отчёты) модель отвечала по-английски, и JSON приходил с русскими ключами и
английскими значениями — формально валидный, содержательно чужой.

Блок общий: он добавляется в запрос каждой роли через ``_prompt_extras``.
Поэтому проверяются оба утверждения — что ролей девять и что блок доходит
до запроса.
"""

from __future__ import annotations

from pf.roles import ROLES
from pf.state import Phase, SessionState
from pf.translate import LANGUAGE_BLOCK
from pf.workflow import Workflow


def test_all_roles_get_language_block() -> None:
    """Все девять ролей получают требование отвечать по-русски."""
    assert len(ROLES) == 9, "ролей девять — блок общий, но проверить надо все"
    state = SessionState(
        session_id="language", phase=Phase.PLANNING, question="почему падает удержание"
    )

    extras = Workflow._prompt_extras(state)

    assert any(LANGUAGE_BLOCK in block for block in extras), extras


def test_language_block_keeps_quotes_verbatim() -> None:
    """Перевод не отменяет дословности: цитата остаётся на языке источника."""
    assert "по-русски" in LANGUAGE_BLOCK
    assert "дословно" in LANGUAGE_BLOCK
