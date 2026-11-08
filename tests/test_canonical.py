"""Проверки канонического сравнения: поля вместо одного порога покрытия.

Зачем отдельный слой. Решение «основание подтверждено» принималось одним
порогом по покрытию (0,8). Из этого следовали оба измеренных провала: честный
пересказ «почти сорок миллионов» против строки «значение: 38.9» отклонялся
(покрытие 0,17), а подмена периода «за 2016 год» против «накопленно с 2013
года, сентябрь 2019» подтверждалась (покрытие 1,00).

**Ответ — число.** Каждое поле даёт долю от 0 до 1, итог — произведение долей
в степенях весов, и порог превращения в статус — настройка
(`claim_support_threshold`), а не константа. Поэтому здесь проверяется не
только «подтверждено/нет», но и **различимость**: округление и подмена дают
разные числа.

Все тесты названы `test_fields_*`: критерий приёмки задачи
(`uv run pytest tests/test_canonical.py -k fields -q`) обязан выбирать все
тесты задачи, иначе красный прогон покажет «1 selected» вместо «N failed».
"""

from __future__ import annotations

from pf.canonical import canonicalize, column_semantics, parse_fields, score
from pf.config import settings
from pf.ground import SupportStatus, find_quote, verify_claim, verify_quote_as_external

#: Строка МТС: аббревиатура короче четырёх букв — её нет в мешке слов.
ROW_MTS = (
    "оператор: МТС, показатель: абонентская база в России, значение: 82, "
    "период: конец 3-го квартала 2024, "
    "источник: https://ru.wikipedia.org/wiki/МТС_(компания)"
)

#: Строка Tele2 без единицы: миллионы подразумеваются.
ROW_TELE2 = "оператор: Tele2, показатель: абонентская база, значение: 38.9, период: конец июля"

#: Строка накопленного итога: 27,4 млн — сумма к сентябрю 2019, а не за год.
ROW_CUMULATIVE = (
    "оператор: Россия (ЦНИИС), показатель: заявки на перенос номера накопленно "
    "с 2013 года, млн, значение: 27.4, период: сентябрь 2019"
)

#: Строка доли: единица названа шапкой колонки, а не числом.
ROW_SHARE = (
    "оператор: МТС, показатель: доля на рынке спутникового ТВ, значение: 6.2, "
    "период: 3-й квартал 2024"
)


def test_fields_score_is_float_and_graded() -> None:
    """Округление и подмена дают разные числа, а не два одинаковых «нет»."""
    rounded = score("Tele2 обслуживала 39 млн абонентов", ROW_TELE2)
    substituted = score("Tele2 обслуживала 7,39 млн абонентов", ROW_TELE2)

    assert 0.0 < rounded.support <= 1.0
    assert 0.0 <= substituted.support < 1.0
    assert rounded.support > substituted.support + 0.3
    assert rounded.support > settings.claim_support_threshold
    assert substituted.support < settings.claim_support_threshold


def test_fields_score_is_monotone_per_field() -> None:
    """Ухудшение любого решающего поля строго снижает степень поддержки."""
    base_claim = "Абонентская база МТС в России составляет 82 млн человек"
    baseline = score(base_claim, ROW_MTS).support

    worse_subject = score(base_claim.replace("МТС", "ВымпелКом"), ROW_MTS).support
    worse_value = score(base_claim.replace("82", "99"), ROW_MTS).support
    worse_period = score(base_claim + " за 2016 год", ROW_MTS).support

    assert worse_subject < baseline
    assert worse_value < baseline
    assert worse_period < baseline


def test_fields_abbreviation_term_is_found() -> None:
    """«МТС» находится в строке, которая начинается с «оператор: МТС».

    Аббревиатура короче четырёх букв не попадает в мешок слов
    (`corpus._TOKEN_PATTERN`), поэтому раньше проверка терминов не могла
    найти «МТС» никогда — измерено: 12 из 24 ложных отказов на наборе пар.
    """
    result = score("Абонентская база МТС в России составляет 82 млн человек", ROW_MTS)

    assert result.fields["термин"].share == 1.0
    assert result.support >= settings.claim_support_threshold


def test_fields_word_form_is_one_concept() -> None:
    """«Абонентскую» и «абонентов» связывает основа, а не лемма."""
    source = "Численность абонентов виртуальных операторов составляла 6 миллионов человек."
    result = score("Виртуальные операторы наращивали абонентскую базу", source)

    assert result.fields["показатель"].share >= 0.5


def test_fields_paraphrase_with_number_words_is_confirmed() -> None:
    """«Почти сорок миллионов» подтверждается значением 38.9."""
    result = score("Tele2 обслуживала почти сорок миллионов абонентов", ROW_TELE2)

    assert result.support >= settings.claim_support_threshold
    assert result.reason == ""


def test_fields_scale_comes_from_column_header() -> None:
    """Шкала «млн» берётся из шапки колонки, а не от соседнего слова."""
    declared = parse_fields(ROW_CUMULATIVE)
    scale, _unit = column_semantics(declared)

    assert declared["показатель"].endswith("млн")
    assert scale == 1_000_000


def test_fields_year_is_period_not_quantity() -> None:
    """Год — это период: он не сравнивается со значениями источника."""
    parsed = canonicalize("Заявки на перенос номера в 2016 году — 27,4 млн")
    values = [item.value for item in parsed.quantities]
    result = score("Заявки на перенос номера в 2016 году — 27,4 млн", ROW_CUMULATIVE)

    assert 2016.0 not in values
    assert values == [27.4]
    assert result.fields["величина"].share == 1.0
    assert result.fields["период"].share < 0.5


def test_fields_period_substitution_lowers_score_and_names_the_field() -> None:
    """Подмена периода отклоняется, и причина называет период."""
    result = score("Заявки на перенос номера в 2016 году — 27,4 млн", ROW_CUMULATIVE)

    assert result.support < settings.claim_support_threshold
    assert result.reason.startswith("период")
    assert "2016" in result.reason


def test_fields_operator_substitution_is_rejected() -> None:
    """Строка другого оператора при том же числе не подтверждает утверждение."""
    result = score("ВымпелКом обслуживал 38,9 млн абонентов", ROW_TELE2)

    assert result.support < settings.claim_support_threshold
    assert result.reason.startswith("субъект")


def test_fields_region_substitution_is_rejected() -> None:
    """Страна утверждения против другой строки — подмена региона."""
    result = score("Абонентская база МТС в России — 5,7 млн", ROW_TELE2)

    assert result.support < settings.claim_support_threshold


def test_fields_unit_substitution_is_rejected() -> None:
    """Процент, выданный за миллионы абонентов, — другая величина."""
    result = score("Доля МТС на рынке спутникового ТВ — 6,2 млн абонентов", ROW_SHARE)

    assert result.fields["величина"].share < 0.5
    assert result.support < settings.claim_support_threshold


def test_fields_single_row_cannot_support_trend() -> None:
    """Одна строка не подтверждает динамику: это другой вид основания."""
    result = score("Удержание абонентов Tele2 выросло", ROW_TELE2)

    assert result.fields["опора"].share < 0.5
    assert result.reason.startswith("опора")


def test_fields_threshold_is_a_setting() -> None:
    """Порог — настройка, а не константа в коде."""
    assert hasattr(settings, "claim_support_threshold")
    assert 0.0 < settings.claim_support_threshold < 1.0


def test_fields_status_follows_threshold() -> None:
    """Одно и то же сравнение даёт разные статусы при разных порогах.

    Утверждение с округлением (39 млн против 38.9) — не единица поддержки, и
    это то, что нужно: порог ниже степени подтверждает, порог выше — нет.
    """
    claim = "Tele2 обслуживала 39 млн абонентов"

    confirmed = verify_claim(ROW_TELE2, claim, "test/строка", paragraph_index=0)
    rejected = verify_claim(
        ROW_TELE2, claim, "test/строка", paragraph_index=0, support_threshold=0.95
    )

    assert confirmed.status is SupportStatus.CONFIRMED
    assert rejected.status is SupportStatus.NO_DATA


def test_fields_reason_names_the_field_in_support_detail() -> None:
    """Причина отказа в основании называет поле и число, а не «не совпало»."""
    support = verify_claim(
        ROW_TELE2,
        "ВымпелКом обслуживал 38,9 млн абонентов",
        "test/строка",
        paragraph_index=0,
    )

    assert support.status is SupportStatus.NO_DATA
    assert "субъект" in support.detail
    assert "поддержка" in support.detail


def test_fields_verbatim_still_uses_raw_text() -> None:
    """Дословность по-прежнему проверяется по сырому тексту (`D-38`).

    Текст сравнения и текст цитаты — разные тексты: цитата без адреса
    источника дословной не становится, хотя после чистки совпадёт.
    """
    without_source = ROW_MTS.replace(", источник: https://ru.wikipedia.org/wiki/МТС_(компания)", "")

    found_raw, _ = find_quote(ROW_MTS, ROW_MTS)
    found_stripped, _ = find_quote(without_source, ROW_MTS)

    assert found_raw is True
    assert found_stripped is False


def test_fields_external_keeps_status() -> None:
    """Внешний источник не поднимается выше «требует проверки» (`D-29`)."""
    support = verify_quote_as_external(ROW_MTS, "абонентская база в России", "web/МТС")

    assert support.status is SupportStatus.NEEDS_REVIEW


def test_fields_threshold_is_frozen_by_measurement() -> None:
    """Порог из настройки лежит между кластерами набора — и это проверяется.

    Порог не «поставлен на глаз»: он заморожен по замеру (`make
    tune-threshold`) и обязан отделять подмены от честных подтверждений.
    Поменяли веса полей или правило — полоса поехала, тест красный, порог
    надо перезамерить. Иначе подобранное число тихо переживёт свой замер.
    """
    from pf.semantic_bench import load_pairs, source_passage

    threshold = settings.claim_support_threshold
    supports: dict[str, float] = {}
    for pair in load_pairs():
        passage = source_passage(pair.source_id, pair.paragraph)
        assert passage is not None, pair.pair_id
        supports[pair.pair_id] = score(pair.claim, passage).support

    pairs = load_pairs()
    top_substitution = max(
        supports[pair.pair_id] for pair in pairs if pair.label.value == "out_of_place"
    )
    weakest_honest = min(
        supports[pair.pair_id]
        for pair in pairs
        if pair.label.value == "in_place" and supports[pair.pair_id] >= threshold
    )

    assert top_substitution < threshold
    assert threshold <= weakest_honest


def test_fields_pair_set_confirms_paraphrase_and_catches_substitution() -> None:
    """Набор из 60 пар: пересказ подтверждается, подмена периода — нет.

    Числа заморожены по замеру (`D-39`): до слоя полей подтверждалось 6
    честных пар из 30 и утекала 1 подмена из 30. Тест держит полноту и утечку
    одновременно: порогом в одиночку одно достигается вырождением другого.
    """
    from pf.semantic_bench import load_pairs, source_passage

    threshold = settings.claim_support_threshold
    confirmed = {label: 0 for label in ("in_place", "out_of_place")}
    totals = {label: 0 for label in ("in_place", "out_of_place")}
    for pair in load_pairs():
        passage = source_passage(pair.source_id, pair.paragraph)
        assert passage is not None, pair.pair_id
        totals[pair.label.value] += 1
        if score(pair.claim, passage).support >= threshold:
            confirmed[pair.label.value] += 1

    assert confirmed["in_place"] >= 27, confirmed
    assert confirmed["out_of_place"] <= 2, confirmed
    assert totals == {"in_place": 30, "out_of_place": 30}
