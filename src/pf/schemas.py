"""Схемы вывода агента.

Схемы — это контракт между моделью и кодом. Модель обязана вернуть JSON,
который проходит валидацию; если не проходит, шаг считается неудачным и
повторяется с подсказкой. Поэтому в схемах есть не только «что должно быть»,
но и минимальные ограничения, без которых результат бессмыслен:

- у варианта должно быть возражение (иначе это не проверка стратегии);
- у факта должна быть ссылка на источник (иначе это пересказ);
- в стратегии обязателен план проверки (иначе решения нельзя принять).

Проверка цитат по источнику выполняется отдельно, в ``ground.py``: схема
проверяет форму, гейт — происхождение.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def as_text_list(value: object) -> list[str]:
    """Привести значение к списку строк.

    Модель часто отвечает списком объектов там, где ожидается список
    строк (``"alternatives": [{"title": "...", "summary": "..."}]``).
    Объект превращается в его заголовок либо в текст — так информация
    не теряется.

    Args:
        value: Значение от модели.

    Returns:
        Список строк (пустой, если пришёл пустой список).
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        # Некоторые модели отвечают объектом с парой «эффект → реакция».
        # Для списка строк важна суть, поэтому берём обе части через точку
        # с запятой — иначе теряется половина смысла.
        parts: list[str] = []
        for key in ("title", "name", "text", "summary", "statement", "effect"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                parts.append(candidate.strip())
                break
        for key in ("response", "action", "mitigation", "description", "detail"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                parts.append(candidate.strip())
                break
        if parts:
            return ["; ".join(parts)]
        return [as_text(value)]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    result.append(item.strip())
            elif isinstance(item, dict):
                for key in ("title", "name", "text", "summary", "statement"):
                    candidate = item.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        result.append(candidate.strip())
                        break
                else:
                    result.append(as_text(item))
            else:
                text = as_text(item)
                if text:
                    result.append(text)
        return result
    text = as_text(value)
    return [text] if text else []


def _as_mapping(value: object) -> dict[str, object]:
    """Привести значение к словарю полей.

    Работает и с обычным словарём, и с уже созданным объектом схемы:
    во втором случае берутся его поля. Пустой результат означает «блока
    нет» — тогда проверка строгости отвергнет ответ и шаг будет повторён.

    Args:
        value: Значение от модели или готовый объект схемы.

    Returns:
        Словарь полей (пустой, если блока нет).
    """
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, BaseModel):
        return dict(value.model_dump())
    return {}


def as_text(value: object) -> str:
    """Привести значение к строке.

    Модель часто отвечает списком там, где схема ждёт строку
    (``"channels": ["внутренние сети", "по продукту"]``). Список — это
    нормальный ответ, терять его нельзя, поэтому приводим к тексту через
    точку с запятой.

    Args:
        value: Значение от модели.

    Returns:
        Строковое представление.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value)


# ── Речевые роли ────────────────────────────────────────────────────────────

FactKind = Literal["факт", "гипотеза", "прогноз"]

#: Бренды операторов связи. В ключевом сообщении они недопустимы:
#: коммуникация продукта обращается к аудитории своего продукта, а не
#: конкурента. В контексте рынка (`barrier`, анализ) они допустимы.
COMPETITOR_BRANDS: tuple[str, ...] = ("мтс", "билайн", "теле2", "tele2", "ростелеком")


class SourceRef(BaseModel):
    """Ссылка на источник утверждения.

    Attributes:
        source_id: Идентификатор источника (``раздел/Заголовок``).
        paragraph: Номер абзаца, если известен.
        quote: Дословная цитата из источника.
    """

    source_id: str = Field(default="")
    paragraph: int | None = None
    quote: str = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        """Пустая ссылка не является основанием — но и не ошибкой схемы.

        Модель иногда оставляет источник пустым, если не смогла привязать
        утверждение. Такой источник просто не будет считаться подтверждённым
        при проверке — это честнее, чем ронять весь шаг.
        """
        if isinstance(data, dict):
            data = dict(data)
            data["source_id"] = as_text(data.get("source_id"))
            data["quote"] = as_text(data.get("quote"))
            paragraph = data.get("paragraph")
            if isinstance(paragraph, str):
                data["paragraph"] = int(paragraph) if paragraph.strip().isdigit() else None
        return data

    @property
    def citation(self) -> str:
        """Человекочитаемая ссылка."""
        if self.paragraph is None:
            return self.source_id
        return f"{self.source_id}#p{self.paragraph}"


class MetricRef(BaseModel):
    """Расчётная метрика со ссылкой на источник данных.

    Отличие от факта: факт подтверждается дословной цитатой, метрика —
    числом, посчитанным по строке. Поэтому у метрики другой источник
    (строка таблицы) и обязательно формула.

    Attributes:
        name: Название метрики.
        value: Значение.
        unit: Единица измерения.
        source_id: Идентификатор источника данных.
        row: Номер строки в источнике (нумерация с нуля).
        formula: Как считалось.
        inputs: Входные значения.
        note: Пояснение для человека.
    """

    name: str = Field(min_length=1)
    value: float
    unit: str = Field(default="")
    source_id: str = Field(default="")
    row: int | None = Field(default=None)
    formula: str = Field(default="")
    inputs: dict[str, Any] = Field(default_factory=dict)
    note: str = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        """Привести поля к ожидаемому виду: модель отвечает строками."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data["name"] = as_text(data.get("name"))
        data["unit"] = as_text(data.get("unit"))
        data["source_id"] = as_text(data.get("source_id"))
        data["formula"] = as_text(data.get("formula"))
        row = data.get("row")
        if isinstance(row, str):
            data["row"] = int(row) if row.strip().isdigit() else None
        return data

    @property
    def citation(self) -> str:
        """Ссылка на источник значения."""
        if not self.source_id:
            return ""
        if self.row is None:
            return self.source_id
        return f"{self.source_id}#row{self.row}"

    @property
    def is_reproducible(self) -> bool:
        """Можно ли пересчитать метрику по её же данным.

        Метрика без источника — это утверждение, а не расчёт.
        """
        return bool(self.source_id)


class Claim(BaseModel):
    """Утверждение с его основанием.

    Attributes:
        kind: Тип утверждения — факт, гипотеза или прогноз.
        text: Само утверждение.
        sources: Основания. У факта должен быть хотя бы один.
    """

    kind: FactKind
    text: str = Field(min_length=1)
    sources: list[SourceRef] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _repair_kind(cls, data: object) -> object:
        """Починить перепутанный тип утверждения.

        Модель иногда отвечает наоборот: кладёт текст утверждения в поле
        ``kind``, а настоящий тип не указывает. Разбираем характерные
        случаи:

        - ``kind`` содержит текст (длинная строка) — значит, тип потерян,
          а текст лежит в поле типа; тип восстанавливается по наличию
          источников: с источником — факт, без — гипотеза;
        - ``kind`` — неизвестное слово — приводим к гипотезе: безопаснее
          пометить как гипотезу, чем выдать за факт.
        """
        if not isinstance(data, dict):
            return data

        data = dict(data)
        kind = data.get("kind")
        text = data.get("text")

        if isinstance(kind, str) and len(kind) > 40:
            # Тип потерян, а в нём лежит текст утверждения.
            if not text or not isinstance(text, str) or len(text) < 3:
                data["text"] = kind
            data["kind"] = "факт" if data.get("sources") else "гипотеза"
            return data

        if isinstance(kind, str):
            normalized = kind.strip().lower()
            aliases = {
                "fact": "факт",
                "hypothesis": "гипотеза",
                "forecast": "прогноз",
                "prediction": "прогноз",
                "гипотеза": "гипотеза",
                "факт": "факт",
                "прогноз": "прогноз",
            }
            if normalized in aliases:
                data["kind"] = aliases[normalized]
            elif normalized not in {"факт", "гипотеза", "прогноз"}:
                data["kind"] = "гипотеза"

        if not data.get("kind"):
            data["kind"] = "факт" if data.get("sources") else "гипотеза"

        return data

    @model_validator(mode="after")
    def _fact_needs_source(self) -> Claim:
        """Факт без ссылки на источник — не факт, а пересказ."""
        if self.kind == "факт" and not self.sources:
            raise ValueError("факт должен ссылаться хотя бы на один источник")
        return self


class TaskFrame(BaseModel):
    """Постановка задачи: что именно нужно понять и для кого.

    Attributes:
        question: Вопрос руководителя продукта.
        goal: Цель, которую нужно достичь.
        audience: Получатель результата.
        constraints: Ограничения, которые нельзя нарушить.
        available_data: Известные доступные данные.
        open_questions: Что осталось уточнить у человека.
    """

    question: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    audience: str = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)
    available_data: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    """План исследования.

    Attributes:
        questions: Вопросы, ответы на которые могут изменить решение.
        priority_order: Порядок работы над вопросами.
        gaps_expected: Какие пробелы в данных ожидаются заранее.
    """

    questions: list[str] = Field(min_length=1)
    priority_order: list[int] = Field(default_factory=list)
    gaps_expected: list[str] = Field(default_factory=list)


class Analysis(BaseModel):
    """Разбор источников: факты, гипотезы, прогнозы и пробелы.

    Attributes:
        facts: Проверяемые утверждения с ссылками на источники.
        hypotheses: Гипотезы, которые ещё не подтверждены.
        forecasts: Прогнозы.
        gaps: Чего агент не знает и почему.
    """

    facts: list[Claim] = Field(default_factory=list)
    hypotheses: list[Claim] = Field(default_factory=list)
    forecasts: list[Claim] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    metrics: list[MetricRef] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _unwrap(cls, data: object) -> object:
        """Снять лишнюю обёртку, если модель её добавила.

        Модель иногда отвечает объектом вида
        ``{"system": {...}}`` или ``{"analysis": {...}}`` — это тот же ответ,
        просто в лишнем слое. Обёртка снимается, если внутри неё лежит
        единственный словарь с ожидаемыми ключами.
        """
        if isinstance(data, dict):
            expected = {"facts", "hypotheses", "forecasts", "gaps"}
            if not (expected & set(data.keys())):
                for key in ("system", "analysis", "result", "response", "output", "answer"):
                    inner = data.get(key)
                    if isinstance(inner, dict) and (expected & set(inner.keys())):
                        return inner
        return data

    @model_validator(mode="before")
    @classmethod
    def _normalize_shape(cls, data: object) -> object:
        """Привести ответ к ожидаемой форме.

        Модель отвечает на три вопроса по-разному:

        - объектом с ключами ``facts``/``hypotheses``/``gaps`` — обычный случай;
        - **массивом** утверждений без обёртки — все элементы считаются
          фактами, если у них есть основание, иначе гипотезами;
        - объектом с другими именами ключей (``known``/``unknown``,
          ``assumptions``/``unknowns``) — приводим к нашим.

        Args:
            data: Ответ модели.

        Returns:
            Данные в форме словаря с нашими ключами.
        """
        if isinstance(data, list):
            facts: list[object] = []
            hypotheses: list[object] = []
            gaps: list[object] = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                entry = dict(item)
                kind = entry.get("kind")
                if kind in {"гипотеза", "hypothesis"} or (
                    not entry.get("sources") and kind != "прогноз"
                ):
                    entry.setdefault("kind", "гипотеза")
                    hypotheses.append(entry)
                else:
                    entry.setdefault("kind", "факт")
                    facts.append(entry)
            return {"facts": facts, "hypotheses": hypotheses, "gaps": gaps}

        if isinstance(data, dict):
            data = dict(data)
            aliases = {
                "known": "facts",
                "verified": "facts",
                "assumptions": "hypotheses",
                "hypotheses_list": "hypotheses",
                "unknowns": "gaps",
                "missing": "gaps",
                "limitations": "gaps",
                "predictions": "forecasts",
            }
            for alias, target in aliases.items():
                if alias in data and target not in data:
                    data[target] = data.pop(alias)
            data["gaps"] = as_text_list(data.get("gaps"))
            return data

        return data

    @model_validator(mode="after")
    def _has_content(self) -> Analysis:
        """Пустой разбор бесполезен — значит, шаг не удался."""
        if not (self.facts or self.hypotheses or self.forecasts or self.gaps or self.metrics):
            raise ValueError("разбор не содержит ни фактов, ни гипотез, ни метрик, ни пробелов")
        return self


class Variant(BaseModel):
    """Вариант решения с возражениями.

    Attributes:
        title: Название варианта.
        summary: Суть в одном предложении.
        approach: Что предлагается сделать.
        objections: Сильные возражения против варианта.
        distinct_axis: По какой оси вариант отличается от остальных.
    """

    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    approach: str = Field(default="")
    objections: list[str] = Field(default_factory=list)
    distinct_axis: str = Field(default="")

    @field_validator("objections")
    @classmethod
    def _needs_objections(cls, value: list[str]) -> list[str]:
        """Вариант без возражения — это не проверенный вариант."""
        if not value:
            raise ValueError("у варианта должно быть хотя бы одно сильное возражение")
        return value


class Variants(BaseModel):
    """Набор вариантов решения.

    Attributes:
        variants: Варианты — не меньше трёх.
        keep_current_marker: Название варианта «оставить как есть».
        close_direction_marker: Название варианта «закрыть направление».
    """

    variants: list[Variant] = Field(min_length=3)
    keep_current_marker: str = Field(default="")
    close_direction_marker: str = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        """Привести разные формы ответа к одной.

        Модель может вернуть: список вариантов, одиночный вариант,
        либо объект с ключом ``variants``/``options``/``alternatives``.
        """
        if isinstance(data, list):
            # Ответ — голый список вариантов. Оборачиваем: иначе модель
            # получает ошибку валидации, вместо которой можно было бы
            # просто принять ответ.
            return {"variants": data}
        if isinstance(data, dict):
            data = dict(data)
            raw = data.get("variants")
            if raw is None:
                for alias in ("options", "alternatives", "variants_list"):
                    if data.get(alias) is not None:
                        raw = data.get(alias)
                        break
            if isinstance(raw, dict):
                raw = [raw]
            data["variants"] = raw if isinstance(raw, list) else []
            data["keep_current_marker"] = as_text(data.get("keep_current_marker"))
            data["close_direction_marker"] = as_text(data.get("close_direction_marker"))
            return data
        return data

    @model_validator(mode="after")
    def _diverse(self) -> Variants:
        """Варианты должны различаться по названию — иначе это один вариант."""
        titles = [v.title.strip().lower() for v in self.variants]
        if len(set(titles)) != len(titles):
            raise ValueError("названия вариантов должны различаться")
        return self


class RedTeamReport(BaseModel):
    """Результат проверки вариантов критиком.

    Attributes:
        findings: Что слабого найдено в каждом варианте.
        strongest_objection: Самое сильное возражение по совокупности.
        unresolved: Что осталось неразрешённым и требует решения человека.
    """

    findings: list[str] = Field(default_factory=list)
    strongest_objection: str = Field(default="")
    unresolved: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        """Привести разные формы ответа к одной.

        Модель отвечает либо объектом с ключами, либо просто списком
        находок. Список трактуется как ``findings``.
        """
        if isinstance(data, list):
            return {"findings": as_text_list(data)}
        if isinstance(data, dict):
            data = dict(data)
            data["findings"] = as_text_list(data.get("findings") or data.get("issues"))
            data["unresolved"] = as_text_list(data.get("unresolved") or data.get("open_questions"))
            data["strongest_objection"] = as_text(data.get("strongest_objection"))
        return data


class ProductStrategy(BaseModel):
    """Продуктовая стратегия.

    Attributes:
        audience: Аудитория и её задача.
        alternatives: Рассмотренные альтернативы.
        chosen_direction: Выбранное направление.
        deliberate_omissions: Осознанные отказы.
        value_to_cost: Связь пользы с затратами и доходами.
        evidence: Основания выбора.
    """

    audience: str = Field(min_length=1)
    alternatives: list[str] = Field(default_factory=list)
    chosen_direction: str = Field(min_length=1)
    deliberate_omissions: list[str] = Field(default_factory=list)
    value_to_cost: str = Field(default="")
    evidence: list[SourceRef] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_text(cls, data: object) -> object:
        """Строки, списки и синонимы полей привести к ожидаемому виду.

        Модель называет поля по-своему: «направление» вместо
        ``chosen_direction``, «аудитория» вместо ``audience``. Без
        синонимов стратегия отбрасывалась целиком, хотя по сути была
        заполнена — и агент терял работу на последнем шаге.
        """
        if isinstance(data, dict):
            data = dict(data)
            for alias, target in (
                ("аудитория", "audience"),
                ("audience_task", "audience"),
                ("направление", "chosen_direction"),
                ("name", "chosen_direction"),
                ("chosen", "chosen_direction"),
                ("direction", "chosen_direction"),
                ("выбранное_направление", "chosen_direction"),
                ("осознанные_отказы", "deliberate_omissions"),
                ("отказы", "deliberate_omissions"),
                ("польза_затраты", "value_to_cost"),
                ("value_cost", "value_to_cost"),
            ):
                if alias in data and not data.get(target):
                    data[target] = data[alias]
            data["value_to_cost"] = as_text(data.get("value_to_cost"))
            data["alternatives"] = as_text_list(data.get("alternatives"))
            data["deliberate_omissions"] = as_text_list(data.get("deliberate_omissions"))
            data["audience"] = as_text(data.get("audience"))
            data["chosen_direction"] = as_text(data.get("chosen_direction"))
            # Обязательные поля нельзя оставлять пустыми: стратегия без
            # аудитории и без направления — это не стратегия. Если
            # модель их не заполнила, берём то, что есть в блоке, и
            # честно помечаем пробел — выдумывать ничего нельзя.
            if not data["audience"]:
                for source in ("целевая_аудитория", "target_audience", "аудитория_и_задача"):
                    candidate = as_text(data.get(source))
                    if candidate:
                        data["audience"] = candidate
                        break
            if not data["chosen_direction"]:
                for source in ("направление", "выбранное_направление", "direction"):
                    candidate = as_text(data.get(source))
                    if candidate:
                        data["chosen_direction"] = candidate
                        break
        return data


class CommunicationStrategy(BaseModel):
    """Коммуникационная стратегия.

    Attributes:
        audience: Аудитория и её текущая ситуация.
        barrier: Барьер понимания или выбора.
        key_message: Ключевое сообщение.
        proof: Чем сообщение подтверждается.
        channels: Роль каналов.
        promise_grounding: На каких фактах основаны обещания.
    """

    audience: str = Field(min_length=1)
    barrier: str = Field(min_length=1)
    key_message: str = Field(min_length=1)
    proof: str = Field(default="")
    channels: str = Field(default="")
    promise_grounding: list[SourceRef] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_text(cls, data: object) -> object:
        """Строковые поля привести к строке: модель отвечает списком."""
        if isinstance(data, dict):
            data = dict(data)
            data["proof"] = as_text(data.get("proof"))
            data["channels"] = as_text(data.get("channels"))
            data["barrier"] = as_text(data.get("barrier"))
        return data

    @model_validator(mode="after")
    def _message_owns_our_brand(self) -> CommunicationStrategy:
        """Ключевое сообщение должно говорить от нашего имени.

        Коммуникация обращается к аудитории нашего продукта. Если в
        сообщении назван конкурент, это не наша коммуникация: в
        демонстрации так вышло «МТС предлагает прозрачные тарифы» для
        задачи Фабрики Мегафона. Ловим кодом, а не просьбой модели.

        В `barrier` конкурент упоминать можно: это контекст рынка,
        а не обращение к аудитории.

        Raises:
            ValueError: В сообщении назван чужой бренд.
        """
        named = [brand for brand in COMPETITOR_BRANDS if brand in self.key_message.lower()]
        if named:
            raise ValueError(
                "ключевое сообщение называет чужой бренд "
                f"({', '.join(named)}): коммуникация должна идти от нашего имени"
            )
        return self


class VerificationPlan(BaseModel):
    """План проверки решения.

    Attributes:
        expected_change: Ожидаемое изменение.
        compare_against: С чем сравниваем.
        observation_period: Срок наблюдения.
        adverse_effects: Нежелательные последствия, которых ждём.
        success_criteria: Признаки, по которым признаём успех.
    """

    expected_change: str = Field(min_length=1)
    compare_against: str = Field(min_length=1)
    observation_period: str = Field(min_length=1)
    adverse_effects: list[str] = Field(min_length=1)
    success_criteria: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_lists(cls, data: object) -> object:
        """Списки привести к спискам строк: модель отвечает объектами."""
        if isinstance(data, dict):
            data = dict(data)
            data["adverse_effects"] = as_text_list(data.get("adverse_effects"))
            data["success_criteria"] = as_text_list(data.get("success_criteria"))
        return data

    @model_validator(mode="after")
    def _plan_is_complete(self) -> VerificationPlan:
        """План проверки обязан содержать нежелательные последствия.

        Требование брифа: «план проверки — ожидаемое изменение, с чем
        сравниваем, срок наблюдения и нежелательные последствия». Пустой
        список последствий — это не «последствий нет», а отсутствующая
        часть плана: стратегия без «а если» — не стратегия.

        Raises:
            ValueError: Нежелательные последствия не указаны.
        """
        if not self.adverse_effects:
            raise ValueError(
                "в плане проверки должны быть нежелательные последствия: "
                "что может пойти не так и что делаем, если это произойдёт"
            )
        return self


class Strategy(BaseModel):
    """Итоговая стратегия.

    Attributes:
        product: Продуктовая стратегия.
        communication: Коммуникационная стратегия.
        verification: План проверки.
        consistency_note: Как связаны продуктовая и коммуникационная части.
    """

    product: ProductStrategy
    communication: CommunicationStrategy
    verification: VerificationPlan
    consistency_note: str = Field(default="")

    @model_validator(mode="before")
    @classmethod
    def _unwrap(cls, data: object) -> object:
        """Снять лишнюю обёртку: {"strategy": {...}} → {...}."""
        if isinstance(data, dict):
            expected = {"product", "communication", "verification"}
            if not (expected & set(data.keys())):
                for key in ("strategy", "system", "result", "response", "output", "answer"):
                    inner = data.get(key)
                    if isinstance(inner, dict) and (expected & set(inner.keys())):
                        return inner
        return data

    @model_validator(mode="before")
    @classmethod
    def _normalize_text(cls, data: object) -> object:
        """Свести поля к ожидаемому виду.

        Коммуникационная часть и план проверки — обязательные, но модель
        иногда называет их иначе («communication_plan», «comms») или
        складывает в один блок. Если блока нет вовсе, подставляется
        минимальный: так стратегия остаётся пригодной, а недостаток
        виден в выводе — поле остаётся пустым и его видно.
        """
        if not isinstance(data, dict):
            return data

        data = dict(data)
        data["consistency_note"] = as_text(data.get("consistency_note"))

        communication = _as_mapping(data.get("communication"))
        for alias in ("comms", "communication_plan", "comm_strategy"):
            if not communication:
                communication = _as_mapping(data.get(alias))
        if communication:
            communication["proof"] = as_text(communication.get("proof"))
            communication["channels"] = as_text(communication.get("channels"))
            communication["barrier"] = as_text(communication.get("barrier"))
            data["communication"] = communication

        verification = _as_mapping(data.get("verification"))
        for alias in (
            "verification_plan",
            "plan",
            "testing_plan",
            "plan_verification",
            "verification_of_results",
            "check_plan",
        ):
            if not verification:
                verification = _as_mapping(data.get(alias))
        if verification:
            data["verification"] = verification

        product = _as_mapping(data.get("product"))
        if product:
            product["value_to_cost"] = as_text(product.get("value_to_cost"))
            product["alternatives"] = as_text_list(product.get("alternatives"))
            product["deliberate_omissions"] = as_text_list(product.get("deliberate_omissions"))
            data["product"] = product

        return data


class Commitment(BaseModel):
    """Обязательство, принятое вместе с решением.

    Attributes:
        statement: Что именно решили.
        owner: Кто отвечает.
        metric: Показатель.
        review_condition: Условие пересмотра.
        review_by: Когда пересмотреть.
        basis: Основания решения.
    """

    statement: str = Field(min_length=1)
    owner: str = Field(default="")
    metric: str = Field(default="")
    review_condition: str = Field(default="")
    review_by: str = Field(default="")
    basis: list[SourceRef] = Field(default_factory=list)


class DecisionRecord(BaseModel):
    """Запись о принятом решении.

    Attributes:
        decision: Суть решения.
        commitments: Обязательства.
        superseded_claims: Выводы, которые решение заменяет.
    """

    decision: str = Field(min_length=1)
    commitments: list[Commitment] = Field(default_factory=list)
    superseded_claims: list[str] = Field(default_factory=list)


class NextStep(BaseModel):
    """Решение роутера: что делать дальше.

    Attributes:
        role: Роль, которая должна выполнить следующий шаг.
        reason: Почему именно она.
        need_human: Нужно ли подключать человека.
        question: Вопрос человеку, если нужен.
    """

    role: str = Field(min_length=1)
    reason: str = Field(default="")
    need_human: bool = False
    question: str = Field(default="")
