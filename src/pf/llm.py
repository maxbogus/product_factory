"""Клиент языковой модели.

По умолчанию агент работает с локальной моделью через Ollama: это даёт
демонстрацию без единого секрета и без затрат. Провайдер с ключом
(OpenAI-совместимый) поддержан как опция.

Модуль отвечает за три вещи, важные для надёжности:

1. **Таймаут и повтор.** Модель может зависнуть; агент не должен.
2. **Разбор JSON.** Модель часто оборачивает JSON в пояснение или в
   ```-блок. Разбор терпим к этим обёрткам, но не терпим к невалидному JSON.
3. **Стоимость.** Из ответа извлекается число токенов — оно попадает в
   журнал, чтобы руководитель видел, сколько стоил результат.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from pf.config import Settings, settings

#: Кодовый блок с JSON — модель часто оборачивает ответ в него.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMError(RuntimeError):
    """Ошибка обращения к модели."""


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """Расход одного вызова.

    Attributes:
        prompt_tokens: Токены в запросе.
        completion_tokens: Токены в ответе.
        duration_ms: Длительность вызова.
        model: Модель, которая отвечала.
        context_limit: Окно контекста, в котором считался промпт; ``0`` —
            окно неизвестно (облачный провайдер).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: int = 0
    model: str = ""
    context_limit: int = 0

    @property
    def total_tokens(self) -> int:
        """Всего токенов."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def prompt_truncated(self) -> bool:
        """Упёрся ли промпт в окно контекста.

        Ollama считает ``prompt_eval_count`` уже после обрезки: значение,
        равное окну, означает, что начало промпта отброшено. Это не
        «ровно столько токенов», а признак потери контекста — молча
        игнорировать его нельзя, иначе починка окна непроверяема.
        """
        return self.context_limit > 0 and self.prompt_tokens >= self.context_limit

    @property
    def cost_units(self) -> float:
        """Стоимость в условных единицах.

        Единица — миллионная доля токена. Реальная цена зависит от модели
        и провайдера; здесь считается объём, который человек может перевести
        в деньги по своему тарифу. Подробнее — в ``docs/decisions.md``.
        """
        return self.total_tokens / 1_000_000


@dataclass(slots=True)
class LLMResponse:
    """Ответ модели.

    Attributes:
        content: Текст ответа.
        usage: Расход вызова.
        attempts: Сколько попыток потребовалось.
    """

    content: str
    usage: LLMUsage
    attempts: int = 1


class LLMClient(Protocol):
    """Протокол клиента модели."""

    def complete(self, system: str, user: str) -> LLMResponse:
        """Выполнить запрос."""
        ...


def extract_json(raw: str) -> Any:
    """Извлечь JSON из ответа модели.

    Модель может вернуть чистый JSON, JSON в ```-блоке или JSON с
    пояснением вокруг. Все три варианта разбираются; невалидный JSON —
    исключение, чтобы шаг можно было повторить с подсказкой.

    Args:
        raw: Сырой текст ответа.

    Returns:
        Разобранный объект.

    Raises:
        LLMError: JSON не найден или не разбирается.
    """
    text = raw.strip()
    if not text:
        raise LLMError("модель вернула пустой ответ")

    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Ищем самый внешний объект или массив в тексте.
    for opening, closing in (("{", "}"), ("[", "]")):
        start = text.find(opening)
        end = text.rfind(closing)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    # Ответ мог быть обрезан по длине: закрывающий скобки нет, а начало —
    # валидный JSON. Это отдельный случай, его стоит назвать прямо.
    if raw.count("{") > raw.count("}") or raw.count("[") > raw.count("]"):
        raise LLMError(
            "ответ модели обрезан по длине — JSON не закрыт. "
            "Увеличьте PF_MAX_TOKENS или попросите модель отвечать кратче."
        )

    raise LLMError(f"не удалось разобрать JSON из ответа модели: {raw[:200]}")


@dataclass(slots=True)
class _Retry:
    """Состояние повторов."""

    attempts: int = 0
    max_attempts: int = 2
    errors: list[str] = field(default_factory=list)


class OllamaClient:
    """Клиент локальной модели Ollama."""

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or settings

    def complete(self, system: str, user: str) -> LLMResponse:
        """Выполнить запрос к локальной модели.

        Args:
            system: Системная инструкция.
            user: Текст задачи.

        Returns:
            Ответ модели с расходом.

        Raises:
            LLMError: Модель недоступна или вернула ошибку.
        """
        payload = {
            "model": self.config.effective_model,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.effective_max_tokens,
                # Окно задаётся явно. Без него Ollama берёт 4096 и обрезает
                # промпт: начало контекста отбрасывается, а модель всё равно
                # отвечает — потеря не видна ни в ответе, ни в журнале.
                "num_ctx": self.config.effective_num_ctx,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # Thinking-режим: reasoning-модели (qwen3, deepseek-r1) по умолчанию
        # пишут рассуждение и могут вернуть его вместо JSON. Агент разбирает
        # строгий JSON, поэтому режим выключается явно, а не «как получится».
        payload["think"] = self.config.ollama_think
        url = f"{self.config.effective_base_url.rstrip('/')}/api/chat"

        try:
            response = httpx.post(url, json=payload, timeout=self.config.timeout_seconds)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"модель {self.config.effective_model} не ответила за "
                f"{self.config.timeout_seconds} с"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"не удалось обратиться к модели: {exc}") from exc

        data = response.json()
        message = data.get("message", {}).get("content", "")
        usage = LLMUsage(
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
            duration_ms=int(data.get("total_duration", 0) / 1_000_000),
            model=self.config.effective_model,
            # Окно запоминается в расходе: по нему видно, упёрся ли промпт
            # в предел. Без этого «модель ответила» не отличить от
            # «модель видела весь контекст».
            context_limit=self.config.effective_num_ctx,
        )
        return LLMResponse(content=message, usage=usage)


class OpenAICompatibleClient:
    """Клиент любого OpenAI-совместимого API."""

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or settings

    def complete(self, system: str, user: str) -> LLMResponse:
        """Выполнить запрос к API с ключом.

        Args:
            system: Системная инструкция.
            user: Текст задачи.

        Returns:
            Ответ модели с расходом.

        Raises:
            LLMError: Ключ не задан или API вернул ошибку.
        """
        if not self.config.llm_api_key:
            raise LLMError(
                "не задан PF_LLM_API_KEY. Скопируйте .env.example в .env "
                "и заполните ключ, либо используйте локальную модель "
                "(PF_LLM_PROVIDER=ollama)."
            )

        url = f"{self.config.effective_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.config.effective_model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.effective_max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Authorization": f"Bearer {self.config.llm_api_key}"}

        try:
            response = httpx.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise LLMError("модель не ответила вовремя") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"не удалось обратиться к модели: {exc}") from exc

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise LLMError("модель вернула пустой ответ")
        content = choices[0].get("message", {}).get("content", "")
        usage_data = data.get("usage", {})
        usage = LLMUsage(
            prompt_tokens=int(usage_data.get("prompt_tokens", 0)),
            completion_tokens=int(usage_data.get("completion_tokens", 0)),
            duration_ms=0,
            model=self.config.effective_model,
        )
        return LLMResponse(content=content, usage=usage)


class ResilientClient:
    """Клиент с повторами при сбоях.

    Повтор делается только на временных ошибках (таймаут, 5xx). Ошибка
    формата ответа повторяется с подсказкой формата — это помогает модели
    исправить JSON, а не просто повторяет тот же запрос.
    """

    def __init__(self, inner: LLMClient, max_attempts: int = 2) -> None:
        self.inner = inner
        self.max_attempts = max_attempts

    def complete(self, system: str, user: str) -> LLMResponse:
        """Выполнить запрос с повторами.

        Args:
            system: Системная инструкция.
            user: Текст задачи.

        Returns:
            Ответ модели с расходом.

        Raises:
            LLMError: Все попытки исчерпаны.
        """
        last_error: Exception | None = None
        attempts = 0
        current_user = user

        for attempt in range(1, self.max_attempts + 1):
            attempts = attempt
            try:
                response = self.inner.complete(system, current_user)
                if attempt > 1:
                    response.attempts = attempts
                return response
            except LLMError as exc:
                last_error = exc
                if attempt >= self.max_attempts:
                    break
                # Пауза между попытками: даём модели время, но не долго.
                time.sleep(min(2.0 * attempt, 5.0))
                current_user = (
                    f"{user}\n\n"
                    "ПРЕДЫДУЩИЙ ОТВЕТ БЫЛ НЕПРИЕМЛЕМ: "
                    f"{exc}. Верни ТОЛЬКО валидный JSON без пояснений."
                )

        raise LLMError(f"не удалось получить пригодный ответ за {attempts} попыток: {last_error}")


def build_client(config: Settings | None = None) -> LLMClient:
    """Собрать клиента по настройкам.

    Args:
        config: Настройки. По умолчанию — из окружения.

    Returns:
        Готовый клиент с повторами.
    """
    config = config or settings
    inner: LLMClient
    if config.llm_provider == "openai":
        inner = OpenAICompatibleClient(config)
    else:
        inner = OllamaClient(config)
    return ResilientClient(inner)
