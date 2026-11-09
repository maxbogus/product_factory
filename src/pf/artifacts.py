"""Артефакты прогона из одного пути: документ, отчёт и перепроверка.

Документ демонстрации (`demo/RESULT.md`) и отчёт по сессии собираются из
**одного** журнала и одним кодом. Раньше сборка жила в скрипте, а отчёт — в
`pf.report`; документ печатал статусы, посчитанные кодом в момент прогона, и
читатель не мог понять, что перед ним: запись прогона или проверка
сегодняшним кодом (`T-63`).

Поэтому здесь три вещи:

* `provenance` — откуда документ: журнал, его хеш, шаги, правило счёта. Всё
  берётся из файлов, ничего не пишется «сейчас», иначе документ нельзя было бы
  сравнить с пересборкой;
* `build_document` — сам документ (перенесён из `scripts/build_demo_doc.py`);
* `recheck` — что даёт **текущий** код на сохранённых ответах ролей и какие
  основания переехали. Отдельно от документа: commit и время меняются, а
  поставляемый документ обязан быть воспроизводимым.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pf.config import settings
from pf.ground import CONCLUSION_RULE
from pf.journal import Journal, StepRecord, journal_path
from pf.replay import replay_session
from pf.report import SessionReport, build_session_report, conclusion_markdown
from pf.verdicts import label_of

ROOT = Path(__file__).resolve().parent.parent

#: Причины отказа, которые дискредитируют сам источник. Остальные отказы
#: говорят о форме конкретной цитаты, а не о том, что источник чужой.
SOURCE_REJECTIONS: tuple[str, ...] = (
    "не относится к задаче",
    "не найден среди материалов",
)

#: Названия исходов проверки для человека — одни на отчёт и перепроверку.
STATUS_TITLES: dict[str, str] = {
    "confirmed": "подтверждено",
    "needs_review": "требует проверки",
    "no_data": "нет данных",
}


def _citation_key(citation: str) -> str:
    """Ключ основания: путь и первый якорь.

    Якорь может продублироваться (``путь#row0#p0``) — для счёта и сравнения
    это одно и то же основание.

    Args:
        citation: Ссылка на источник.

    Returns:
        Ссылка без повторного якоря.
    """
    path, _, fragment = citation.partition("#")
    parts = [part for part in fragment.split("#") if part]
    return f"{path}#{parts[0]}" if parts else path


def _rejected_keys(records: list[StepRecord]) -> set[str]:
    """Ключи ссылок, которые проверка отвергла **как источник**.

    Отказ по форме цитаты («значение посчитано, а не процитировано»,
    «цитата не найдена в строке») говорит о том, как оформлено одно
    утверждение, а не о том, что источник чужой: та же строка таблицы может
    подтверждать другой вывод. Дискредитирует источник только отказ по
    существу — он не относится к задаче, или такого файла нет.

    Args:
        records: Записи журнала.

    Returns:
        Множество ключей дискредитированных источников.
    """
    rejected: set[str] = set()
    for record in records:
        for line in record.rejected_details:
            if not any(marker in line for marker in SOURCE_REJECTIONS):
                continue
            rejected.add(_citation_key(line.split(":", 1)[0].strip()))
    return rejected


def _confirmed_grounds(records: list[StepRecord]) -> list[tuple[str, str]]:
    """Подтверждённые основания в порядке появления.

    Основание подтверждено, если код нашёл его в источнике и ни один шаг не
    отверг ту же ссылку. Отклонение сильнее: ссылку, которую хотя бы раз не
    подтвердили, выдавать за подтверждённую нельзя.

    Args:
        records: Записи журнала.

    Returns:
        Пары «где подтверждено — ссылка».
    """
    rejected = _rejected_keys(records)
    seen: set[str] = set()
    grounds: list[tuple[str, str]] = []
    for record in records:
        for citation in record.citations:
            key = _citation_key(citation)
            if key in seen or key in rejected:
                continue
            seen.add(key)
            grounds.append((f"{record.role}, {record.phase}", citation))
    return grounds


def _commit() -> str:
    """Короткий хеш текущей версии кода.

    Returns:
        Хеш либо пустая строка, если git недоступен.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


@dataclass(frozen=True, slots=True)
class Provenance:
    """Откуда взят артефакт: журнал, его хеш, шаги и правило счёта.

    Всё выводится из файлов журнала. Ни времени сборки, ни commit здесь нет
    намеренно: поставляемый документ обязан совпадать с пересборкой на любом
    HEAD, иначе тест «документ = журнал» не может пройти никогда.

    Attributes:
        session_id: Идентификатор сессии.
        journal: Путь к журналу.
        digest: Первые 12 символов sha256 журнала.
        steps: Сколько шагов записано.
        last_step: Время последнего шага.
        rule: Правило счёта оснований, записанное прогоном; пусто — не
            записано.
        rule_is_current: Совпадает ли записанное правило с текущим.
    """

    session_id: str
    journal: str
    digest: str
    steps: int
    last_step: str
    rule: str
    rule_is_current: bool

    def lines(self) -> list[str]:
        """Строки провенанса для шапки документа.

        Returns:
            Три строки: журнал и хеш, правило, команда перепроверки.
        """
        if not self.rule:
            rule_line = (
                f"Правило счёта оснований: не записано — сверять не с чем (`{CONCLUSION_RULE}`)."
            )
        else:
            status = "совпадает с текущим" if self.rule_is_current else "не совпадает с текущим"
            rule_line = f"Правило счёта оснований: `{self.rule}` — {status} (`{CONCLUSION_RULE}`)."
        return [
            f"Собрано из журнала `{self.journal}` (sha256 `{self.digest}`, "
            f"шагов {self.steps}, последний шаг {self.last_step or 'не записан'}).",
            rule_line,
            f"Проверить запись текущим кодом: `pf report --session {self.session_id} --recheck`.",
        ]


def provenance(session_id: str) -> Provenance:
    """Собрать провенанс сессии из файлов.

    Args:
        session_id: Идентификатор сессии.

    Returns:
        Провенанс: путь и хеш журнала, число шагов, последний шаг, правило.
    """
    path = journal_path(session_id)
    raw = path.read_bytes() if path.is_file() else b""
    records = Journal(path).read()
    report = build_session_report(session_id)
    rule = report.conclusion.rule if report.conclusion is not None else ""
    return Provenance(
        session_id=session_id,
        journal=str(path),
        digest=hashlib.sha256(raw).hexdigest()[:12],
        steps=len(records),
        last_step=records[-1].created_at if records else "",
        rule=rule,
        rule_is_current=bool(rule) and rule == CONCLUSION_RULE,
    )


@dataclass(frozen=True, slots=True)
class Recheck:
    """Перепроверка записи прогона текущим кодом.

    Attributes:
        session_id: Идентификатор сессии.
        available: Есть ли сохранённые ответы ролей. Без них перепроверять
            нечем, и это говорится прямо, а не показывается нулём.
        commit: Версия кода, которой считали.
        recorded_confirmed: Сколько подтверждено в записи журнала.
        recorded_rejected: Сколько отклонено в записи.
        recorded_status: Статус оснований вывода в записи.
        current_confirmed: Сколько подтверждено сейчас.
        current_rejected: Сколько отклонено сейчас.
        current_user_input: Сколько опор на ввод пользователя сейчас.
        current_status: Статус оснований вывода сейчас.
        moved: Основания, которые переехали между исходами.
    """

    session_id: str
    available: bool
    commit: str = ""
    recorded_confirmed: int = 0
    recorded_rejected: int = 0
    recorded_status: str = ""
    current_confirmed: int = 0
    current_rejected: int = 0
    current_user_input: int = 0
    current_status: str = ""
    moved: list[str] = field(default_factory=list)


def recheck(session_id: str) -> Recheck:
    """Сравнить запись журнала с тем, что даёт текущий код.

    Запись журнала не переписывается: это след прогона. Перепроверка —
    отдельный результат на сохранённых ответах ролей (`pf.replay`), поэтому
    видно не только «сколько», но и **какие** основания переехали.

    Args:
        session_id: Идентификатор сессии.

    Returns:
        Перепроверка; ``available=False``, если сохранённых ответов нет.
    """
    report = build_session_report(session_id)
    recorded_status = report.conclusion.status if report.conclusion is not None else ""
    payload = replay_session(session_id)
    if payload is None:
        return Recheck(
            session_id=session_id,
            available=False,
            commit=_commit(),
            recorded_confirmed=report.confirmed_citations,
            recorded_rejected=report.rejected_citations,
            recorded_status=recorded_status,
        )

    kinds: dict[str, int] = payload["kinds"]
    current_confirmed = int(kinds.get("подтверждено") or 0)
    current_user_input = int(kinds.get("опора на ввод пользователя") or 0)
    moved = _moved_grounds(Journal(journal_path(session_id)).read(), payload["verdicts"])

    statuses = [
        str(counts.get("conclusion_status") or "")
        for counts in payload["roles"].values()
        if counts.get("conclusion_status")
    ]
    return Recheck(
        session_id=session_id,
        available=True,
        commit=_commit(),
        recorded_confirmed=report.confirmed_citations,
        recorded_rejected=report.rejected_citations,
        recorded_status=recorded_status,
        current_confirmed=current_confirmed,
        current_rejected=int(kinds.get("нет данных") or 0),
        current_user_input=current_user_input,
        current_status=statuses[-1] if statuses else "",
        moved=moved,
    )


def _moved_grounds(
    records: list[StepRecord],
    verdicts: list[dict[str, Any]],
) -> list[str]:
    """Основания, которые были отклонены в записи, а сейчас приняты.

    Два ключа, потому что журналы разные. Новый журнал хранит отказы с
    машинным видом и утверждением (`StepRecord.rejections`) — по нему строка
    называет утверждение. Старый хранит только текст (`rejected_details`), и
    тогда ключом служит ссылка: утверждения в записи нет, выдумывать его
    нельзя.

    Args:
        records: Записи журнала.
        verdicts: Вердикты текущей проверки.

    Returns:
        Строки «было — стало», без повторов.
    """
    by_claim: dict[str, str] = {}
    by_key: dict[str, str] = {}
    for verdict in verdicts:
        status = str(verdict.get("status") or "")
        if status == "no_data":
            continue
        title = STATUS_TITLES.get(status, status)
        claim = str(verdict.get("claim") or "").strip()
        if claim:
            by_claim.setdefault(claim, title)
        key = _citation_key(str(verdict.get("source_id") or ""))
        if key:
            by_key.setdefault(key, title)

    moved: list[str] = []
    for record in records:
        for item in record.rejections or []:
            if str(item.get("status") or "") != "no_data":
                continue
            claim = str(item.get("claim") or "").strip()
            kind = str(item.get("kind") or "")
            claim_title = by_claim.get(claim) if claim else None
            if not claim_title:
                continue
            line = f"«{claim[:90]}»: было «{label_of(kind)}», сейчас «{claim_title}»"
            if line not in moved:
                moved.append(line)
        for detail in record.rejected_details:
            key = _citation_key(detail.split(":", 1)[0].strip())
            key_title = by_key.get(key) if key else None
            if not key_title:
                continue
            line = f"`{key}`: было отклонено, сейчас «{key_title}»"
            if line not in moved:
                moved.append(line)
    return moved


def recheck_markdown(result: Recheck) -> list[str]:
    """Оформить перепроверку для человека.

    Args:
        result: Результат перепроверки.

    Returns:
        Строки раздела в markdown.
    """
    lines: list[str] = ["## Перепроверка текущим кодом", ""]
    commit = f"`{result.commit}`" if result.commit else "неизвестна"
    lines.append(f"Версия кода: {commit}.")
    lines.append("")
    if not result.available:
        lines.append(
            "**Перепроверить нечем:** сохранённых ответов ролей нет "
            f"(`state/session-{result.session_id}.json`). Запись журнала при этом "
            "остаётся: ниже — только она, и это не «расхождений нет»."
        )
        lines.append("")
        return lines

    lines.append("| Исход | В записи журнала | Сейчас |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Подтверждено | {result.recorded_confirmed} | {result.current_confirmed} |")
    lines.append(f"| Отклонено | {result.recorded_rejected} | {result.current_rejected} |")
    lines.append(f"| Опор на ввод пользователя | — | {result.current_user_input} |")
    lines.append("")
    recorded = result.recorded_status or "не записан"
    current = result.current_status or "не посчитан"
    if recorded != current:
        lines.append(
            f"**Статус вывода изменился:** в журнале «{recorded}», сейчас «{current}». "
            "Документ демонстрации показывает запись прогона — пересоберите его, "
            "если нужен документ по текущему правилу."
        )
        lines.append("")
    if result.moved:
        lines.append("**Основания, которые переехали:**")
        lines.append("")
        for item in result.moved:
            lines.append(f"- {item}")
        lines.append("")
    else:
        lines.append("Основания не переезжали: вердикты совпадают с записью.")
        lines.append("")
    return lines


def _render(report: SessionReport, records: list[StepRecord], origin: Provenance) -> str:
    """Собрать текст документа.

    Args:
        report: Отчёт по сессии.
        records: Записи журнала.
        origin: Откуда документ — журнал, хеш, правило.

    Returns:
        Текст документа в markdown.
    """
    lines: list[str] = []
    lines.append("# Результат демонстрационной задачи")
    lines.append("")
    if report.question:
        lines.append(f"Задача: **«{report.question}»**")
    else:
        lines.append("Задача: не записана в состоянии сессии")
    lines.append("")
    lines.append(f"Сессия: {report.session_id}")
    lines.append("")
    lines.extend(origin.lines())
    lines.append("")
    lines.append("Документ собран из журнала шагов: основания и расчёты взяты")
    lines.append("оттуда, а статусы проставил код. Поэтому он не может разойтись")
    lines.append("с тем, что агент действительно сделал. Это запись прогона:")
    lines.append("если код проверки изменился, статусы здесь — прежние.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 1. Что измерено")
    lines.append("")
    lines.append("| Показатель | Значение |")
    lines.append("|---|---:|")
    lines.append(f"| Шагов выполнено | {report.steps} |")
    lines.append(f"| Подтверждённых записей | {report.confirmed_citations} |")
    lines.append(f"| Подтверждённых оснований (уникальных) | {report.unique_citations} |")
    lines.append(f"| Использовано источников | {report.unique_sources} |")
    lines.append(f"| Оснований отклонено | {report.rejected_citations} |")
    lines.append(f"| Доля подтверждённых | {report.grounding_rate:.0%} |")
    lines.append(f"| Токенов | {sum(report.tokens.values())} |")
    lines.append(f"| Расход | {report.total_cost_units:.6f} условных единиц |")
    lines.append("")
    report_file = settings.reports_dir / f"session-{report.session_id}.md"
    if report_file.is_file():
        lines.append(f"Полный отчёт с разбивкой расхода по ролям — `{report_file}`.")
        lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 2. Подтверждённые основания")
    lines.append("")
    lines.append("Основание подтверждает код: цитата должна найтись в источнике.")
    lines.append("Ссылка, которую проверка отвергла как источник — не по теме или")
    lines.append("такого файла нет, — сюда не попадает. Отказ по форме цитаты")
    lines.append("(«значение посчитано, а не процитировано») источник не дискредитирует:")
    lines.append("та же строка может подтверждать другой вывод.")
    lines.append("")
    grounds = _confirmed_grounds(records)
    if grounds:
        lines.append("| Подтверждено на шаге | Источник |")
        lines.append("|---|---|")
        for where, citation in grounds:
            lines.append(f"| {where} | `{citation}` |")
    else:
        lines.append("Подтверждённых оснований нет: проверка не нашла ни одной")
        lines.append("цитаты, дословно совпавшей с источником.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 3. Расчёты по данным")
    lines.append("")
    lines.append("Числа посчитаны кодом по строкам таблицы, а не взяты из текста")
    lines.append("модели. Колонка «Статус» показывает, что из этого прошло проверку")
    lines.append("пересчётом.")
    lines.append("")
    if report.metrics:
        lines.append("| Метрика | Значение | Строка источника | Как считалось | Статус |")
        lines.append("|---|---:|---|---|---|")
        for metric in report.metrics:
            citation = f"`{metric.citation}`" if metric.citation else "—"
            lines.append(
                f"| {metric.name} | {metric.formatted} | {citation} | "
                f"{metric.formula_note} | {metric.status} |"
            )
    else:
        lines.append("В этом прогоне расчётов нет: агент не вернул ни одной метрики")
        lines.append("с источником и строкой.")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## 4. Контроль качества")
    lines.append("")
    lines.append("| Что проверено | Значение |")
    lines.append("|---|---:|")
    lines.append(
        f"| Расчётов с формулой | {report.metrics_with_formula} из {len(report.metrics)} |"
    )
    for kind, count in sorted(report.rejection_kinds.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| Отклонено: {kind} | {count} |")
    if report.fact_check:
        lines.append(
            f"| Повторная проверка (fact-check) | {report.fact_check.get('summary', '—')} |"
        )
    lines.append("")
    if report.routing_notes:
        lines.append("**Решения роутера.**")
        lines.append("")
        for note in report.routing_notes:
            lines.append(f"- {note}")
        lines.append("")
    if report.constraints:
        # Направление против запрета из постановки задачи — отказ с причиной,
        # а не отказ основания: чинится выбранная линия (`T-06b`).
        lines.append("**Ограничения задачи нарушены.**")
        lines.append("")
        for item in report.constraints:
            lines.append(f"- {item}")
        lines.append("")
    lines.append("---")
    lines.append("")

    if report.rejected_details:
        lines.append("## 5. Отклонённые основания")
        lines.append("")
        lines.append("Что проверка не приняла и почему. Отклонение — это результат,")
        lines.append("а не сбой: лучше видимый отказ, чем правдоподобное число.")
        lines.append("")
        for detail in report.rejected_details:
            lines.append(f"- {detail}")
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.extend(_conclusion_section(report))
    lines.append("---")
    lines.append("")

    lines.append("## 7. Чего этот замер не показывает")
    lines.append("")
    for item in report.limitations:
        lines.append(f"1. {item}")
    lines.append("")
    return "\n".join(lines)


def _conclusion_section(report: SessionReport) -> list[str]:
    """Раздел «Вывод» для документа демонстрации.

    Текст берётся из того же рендера, что и отчёт: документ и отчёт обязаны
    одинаково говорить, что решили и на чём это держится.

    Args:
        report: Отчёт по сессии.

    Returns:
        Строки раздела в markdown.
    """
    section = conclusion_markdown(report.conclusion)
    section[0] = "## 6. Вывод"
    return section


def build_document(session_id: str) -> str:
    """Собрать документ демонстрации по журналу сессии.

    Один путь сборки на все артефакты: этим же кодом документ собирает
    `make demo-doc`, и этим же — проверка «документ совпадает с журналом».

    Args:
        session_id: Идентификатор сессии.

    Returns:
        Текст документа.
    """
    report = build_session_report(session_id)
    records = Journal(journal_path(session_id)).read()
    return _render(report, records, provenance(session_id))
