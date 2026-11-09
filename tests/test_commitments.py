"""Проверки обязательства по шаблону: база, метод, контроль, откат.

Обязательство без базы нельзя ни проверить, ни исполнить: «удержание
вырастет на 5 %» не говорит, от чего. Без метода неясно, как считать. Без
контрольной группы не с чем сравнивать. Условие отката — не то же, что
условие пересмотра: «пересмотреть» и «откатить» — разные действия.

Здесь проверяется, что шаблон есть в схеме, что старый журнал читается без
миграции, что отчёт называет заполненность и что число в базовой линии
попадает в сверку чисел плана.
"""

from __future__ import annotations

from pf.report import ConclusionRecord, conclusion_markdown
from pf.schemas import Commitment, DecisionRecord
from pf.workflow import PLAN_NUMBER_FACT, Workflow


def test_commitment_template_fields_default_empty() -> None:
    """Новые поля шаблона есть в схеме и по умолчанию пусты.

    Пустые по умолчанию — чтобы прогон без базы остался возможным, но
    названным, а не падал на валидации.
    """
    commitment = Commitment(statement="Удержание вырастет на 2 п.п.")

    assert commitment.baseline == ""
    assert commitment.measurement == ""
    assert commitment.control_group == ""
    assert commitment.rollback == ""


def test_commitment_template_tolerates_old_journal() -> None:
    """Старый журнал без новых полей читается без миграции."""
    commitment = Commitment.model_validate(
        {"statement": "Удержание вырастет", "owner": "команда тарифа"}
    )

    assert commitment.baseline == ""
    assert commitment.measurement == ""


def test_commitment_template_rendered_in_report() -> None:
    """Отчёт показывает базу, метод, контрольную группу и откат; без базы — числом."""
    lines = conclusion_markdown(
        ConclusionRecord(
            decision="Внедрить гибкие тарифные планы",
            commitments=[
                {
                    "statement": "Удержание вырастет на 2 п.п.",
                    "owner": "команда тарифа",
                    "metric": "удержание",
                    "baseline": "уровень прошлого квартала",
                    "measurement": "доля продлений",
                    "control_group": "сегмент A",
                    "rollback": "откатить, если прироста нет",
                    "review_condition": "если прироста нет за 3 месяца",
                },
                {
                    "statement": "Уведомления до списания",
                    "owner": "маркетинг",
                    "metric": "доля уведомлений",
                },
            ],
        )
    )
    text = "\n".join(lines)

    assert "База" in text
    assert "Метод" in text
    assert "Контрольная группа" in text
    assert "Откат" in text
    assert "Без базы: 1 из 2" in text


def test_commitment_template_flags_missing_baseline() -> None:
    """Обязательство без базы или метода называется замечанием, а не молчит."""
    data = DecisionRecord(
        decision="решение",
        commitments=[
            Commitment(statement="Удержание вырастет на 2 п.п."),
            Commitment(
                statement="Удержание вырастет на 3 п.п.",
                baseline="уровень прошлого квартала",
                measurement="доля продлений",
            ),
        ],
    )

    notes = Workflow._check_commitment_template(data)

    assert len(notes) == 2, notes
    assert any("база" in note.lower() for note in notes)
    assert any("метод" in note.lower() for note in notes)


def test_commitment_template_baseline_numbers_checked() -> None:
    """Число в базовой линии — утверждение о мире и попадает в сверку чисел."""
    data = DecisionRecord(
        decision="решение",
        commitments=[
            Commitment(
                statement="Удержание вырастет на 2 п.п.",
                baseline="удержание 87 % на конец квартала",
            )
        ],
    )

    pairs = Workflow._plan_number_texts(data)
    baselines = [pair for pair in pairs if "база" in pair[0].lower()]

    assert baselines, pairs
    assert baselines[0][2] == PLAN_NUMBER_FACT
    assert "87" in baselines[0][1]
