"""Проверки eval-раннера.

Набор сценариев в ``eval/cases/scenarios.jsonl`` — это записанные правила
поведения агента. Пока их никто не исполнял, часть правил не покрывалась
тестами, и расхождение между описанием и поведением оставалось
незамеченным.

Раннер закрывает эту дыру: каждый кейс исполняется, и результат
сверяется с ожиданием.
"""

from __future__ import annotations

from pf.evalrun import Case, Expectation, load_cases, run_all


def test_all_cases_have_handler() -> None:
    """У каждого кейса есть обработчик.

    Иначе кейс молча не проверяется — хуже, чем его нет.
    """
    results = run_all()
    assert results
    missing = [r.case_id for r in results if "нет обработчика" in r.detail]
    assert not missing


def test_all_cases_pass() -> None:
    """Все кейсы проходят: поведение соответствует описанию."""
    results = run_all()
    failed = [f"{r.case_id}: {r.detail}" for r in results if not r.passed]
    assert not failed, "; ".join(failed)


def test_every_case_is_checked() -> None:
    """Каждый кейс из набора исполнен."""
    cases = load_cases()
    results = run_all()
    assert len(results) == len(cases)
    assert {c.id for c in cases} == {r.case_id for r in results}


def test_expectations_are_known() -> None:
    """У кейсов только известные ожидания."""
    for case in load_cases():
        assert case.expectation in set(Expectation)


def test_case_parsing_keeps_optional_fields() -> None:
    """Отсутствующие поля — пустые, а не ошибка."""
    case = Case.from_json(
        {
            "id": "X-01",
            "expected": "PASS",
            "title": "кейс",
            "must_contain": ["да"],
        }
    )
    assert case.must_contain == ("да",)
    assert case.must_not_contain == ()
