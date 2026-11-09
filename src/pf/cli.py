"""Командный интерфейс агента.

Команды:

- ``pf run "вопрос"`` — поставить задачу и провести работу до стратегии;
- ``pf resume`` — продолжить сохранённую сессию;
- ``pf answer "текст"`` — ответить на вопрос, который задал агент;
- ``pf stop`` — остановить текущую сессию;
- ``pf log`` — показать журнал шагов;
- ``pf cost`` — показать расход;
- ``pf sources "запрос"`` — показать, какие источники агент найдёт;
- ``pf sessions`` — список сохранённых сессий;
- ``pf review`` — пересмотр выводов: какие из них затронуты новыми данными;
- ``pf eval`` — выполнить eval-сценарии и проверить правила поведения;
- ``pf web "запрос"`` — найти внешние источники (требуют проверки);
- ``pf translate <файл>`` — перевести английский материал на русский.

Интерфейс командный, а не графический, сознательно: руководителю продукта
нужно видеть продвижение работы и останавливать её, а не изучать устройство
агента (требование «удобство» и «прозрачность»).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import TYPE_CHECKING

from pf import __version__
from pf.config import settings
from pf.journal import Journal, journal_path
from pf.state import (
    Phase,
    SessionState,
    list_sessions,
    load,
    record_human_input,
    save,
)

if TYPE_CHECKING:
    from pf.materials import MaterialScan

#: Символы для вывода прогресса.
CHECK = "✓"
ARROW = "→"


def _out(message: str = "") -> None:
    """Вывести строку."""
    print(message)


def _rule(title: str = "") -> None:
    """Вывести разделитель с заголовком."""
    width = 72
    if title:
        _out(f"── {title} " + "─" * max(0, width - len(title) - 4))
    else:
        _out("─" * width)


def _session_id(explicit: str | None = None) -> str:
    """Определить идентификатор сессии.

    Args:
        explicit: Явно заданный идентификатор.

    Returns:
        Идентификатор сессии.
    """
    if explicit:
        return explicit
    sessions = list_sessions()
    return sessions[0] if sessions else ""


# ── run ──────────────────────────────────────────────────────────────────────


def _session_taken(session_id: str) -> bool:
    """Занят ли идентификатор сессии.

    Args:
        session_id: Проверяемый идентификатор.

    Returns:
        Есть ли состояние или журнал с таким идентификатором.
    """
    return load(session_id) is not None or journal_path(session_id).exists()


def _new_session_id() -> str:
    """Свободный идентификатор сессии.

    Идентификатор собирался как ``s{int(time.time())}`` — с разрешением в
    секунду, поэтому два запуска подряд попадали в один журнал и писали шаги
    с одинаковыми номерами: дубли `ordinal` были замечены глазами и не
    ловились кодом (T-32). Занятый идентификатор теперь пропускается.

    Returns:
        Идентификатор, которого нет ни в состоянии, ни в журнале.
    """
    base = f"s{int(time.time())}"
    if not _session_taken(base):
        return base
    for index in range(2, 100):
        candidate = f"{base}-{index}"
        if not _session_taken(candidate):
            return candidate
    return f"{base}-{time.time_ns()}"


def cmd_run(args: argparse.Namespace) -> int:
    """Поставить задачу и провести работу до стратегии."""
    from pf.workflow import run_full_workflow

    session_id = args.session or _new_session_id()
    _rule("Работа агента")
    _out(f"Сессия: {session_id}")
    _out(f"Вопрос: {args.question}")
    _out()

    def show(state: SessionState, result: object) -> None:
        citations = getattr(result, "citations", [])
        rejected = getattr(result, "rejected", 0)
        detail = _describe(getattr(result, "data", None))
        line = f"{CHECK} {state.phase.title_ru:<24} {detail}"
        if citations:
            line += f"  (оснований: {len(citations)})"
        if rejected:
            line += f"  (отклонено оснований: {rejected})"
        _out(line)

    try:
        state = run_full_workflow(args.question, session_id, on_step=show)
    except Exception as exc:  # noqa: BLE001 — CLI должен показать ошибку
        _out()
        _out(f"ОШИБКА: {exc}")
        _out(f"Прогресс сохранён, продолжить: pf resume --session {session_id}")
        return 1

    _out()
    _rule("Итог")
    _print_strategy(state)
    _rule("Расход")
    _print_cost_summary(session_id)
    _out()
    _out(f"Продолжить или посмотреть журнал: pf log --session {session_id}")
    return 0


def _describe(data: object) -> str:
    """Кратко описать результат шага."""
    from pf.schemas import (
        Analysis,
        DecisionRecord,
        NextStep,
        RedTeamReport,
        ResearchPlan,
        Strategy,
        TaskFrame,
        Variants,
    )

    if isinstance(data, TaskFrame):
        return data.question[:70]
    if isinstance(data, ResearchPlan):
        return f"{len(data.questions)} вопросов, пробелов: {len(data.gaps_expected)}"
    if isinstance(data, Analysis):
        return (
            f"фактов {len(data.facts)}, гипотез {len(data.hypotheses)}, "
            f"прогнозов {len(data.forecasts)}, пробелов {len(data.gaps)}"
        )
    if isinstance(data, Variants):
        return f"{len(data.variants)} вариантов"
    if isinstance(data, RedTeamReport):
        return f"{len(data.findings)} находок"
    if isinstance(data, Strategy):
        return "продуктовая + коммуникационная + план проверки"
    if isinstance(data, DecisionRecord):
        return f"{len(data.commitments)} обязательств"
    if isinstance(data, NextStep):
        return f"далее: {data.role}"
    return ""


def _print_cost_summary(session_id: str) -> None:
    """Показать итог расхода по сессии.

    Args:
        session_id: Идентификатор сессии.
    """
    journal = Journal(journal_path(session_id))
    records = journal.read()
    if not records:
        _out("  шагов не было")
        return
    _out(f"  Шагов:     {len(records)}")
    _out(f"  Токенов:   {journal.total_tokens}")
    _out(f"  Стоимость: {journal.total_cost_units:.6f} условных единиц")
    for role, cost in journal.cost_by_role().items():
        _out(f"    {role:<12} {cost:.6f}")


def _print_strategy(state: SessionState) -> None:
    """Показать итоговую стратегию."""
    strategy = state.results.get(Phase.STRATEGY.value)
    if not strategy:
        _out("Стратегия не получена — работа не дошла до фазы стратегии.")
        return

    product = strategy.get("product", {})
    communication = strategy.get("communication", {})
    verification = strategy.get("verification", {})

    _out("ПРОДУКТОВАЯ СТРАТЕГИЯ")
    _out(f"  Аудитория и задача: {product.get('audience', '—')}")
    _out(f"  Направление:        {product.get('chosen_direction', '—')}")
    if product.get("deliberate_omissions"):
        _out("  Осознанные отказы:")
        for item in product["deliberate_omissions"]:
            _out(f"    - {item}")
    if product.get("value_to_cost"):
        _out(f"  Польза / затраты:   {product['value_to_cost']}")

    _out()
    _out("КОММУНИКАЦИОННАЯ СТРАТЕГИЯ")
    _out(f"  Барьер:            {communication.get('barrier', '—')}")
    _out(f"  Ключевое сообщение: {communication.get('key_message', '—')}")
    if communication.get("proof"):
        _out(f"  Подтверждение:     {communication['proof']}")
    if communication.get("channels"):
        _out(f"  Роль каналов:      {communication['channels']}")

    _out()
    _out("ПЛАН ПРОВЕРКИ")
    _out(f"  Ожидаемое изменение: {verification.get('expected_change', '—')}")
    _out(f"  С чем сравниваем:    {verification.get('compare_against', '—')}")
    _out(f"  Срок наблюдения:     {verification.get('observation_period', '—')}")
    if verification.get("adverse_effects"):
        _out("  Нежелательные последствия:")
        for item in verification["adverse_effects"]:
            _out(f"    - {item}")


# ── resume / answer / stop ──────────────────────────────────────────────────


def cmd_resume(args: argparse.Namespace) -> int:
    """Продолжить сохранённую сессию."""
    from pf.workflow import Workflow

    session_id = _session_id(args.session)
    if not session_id:
        _out('Нет сохранённых сессий. Начните: pf run "ваш вопрос"')
        return 1

    state = load(session_id)
    if state is None:
        _out(f"Сессия {session_id} не найдена.")
        return 1

    _rule("Продолжение")
    _out(f"Сессия: {session_id}")
    _out(f"Фаза:   {state.phase.title_ru}")
    _out(f"Вопрос: {state.question}")
    if state.pending_question:
        _out(f"Ждём ответа: {state.pending_question}")
    _out()

    workflow = Workflow()
    # Источники собираются для тех фаз, которым они нужны. Без этого
    # продолжение со сбора источников дало бы сборщику пустой блок
    # МАТЕРИАЛЫ, и он начал бы сочинять цитаты.
    bundle = workflow.bundle_for(state)

    try:
        result = workflow.run_step(state, bundle)
    except Exception as exc:  # noqa: BLE001
        _out(f"ОШИБКА: {exc}")
        return 1

    _out(f"{CHECK} {result.phase.title_ru}: {_describe(result.data)}")

    # Роутер может снова попросить человека: вопрос показывается, работа
    # не продолжается вслепую. Повтор уже заданного вопроса не
    # останавливает: человек на него ответил, и второй раз спрашивать нечего.
    question = workflow.human_question_to_stop(state, result)
    if question:
        state.pending_question = question
        state.results.setdefault("human_questions", []).append(question)
        save(state)
        _out()
        _out(f"Ждём ответа: {question}")
        _out('Ответить: pf answer "текст ответа"')
        return 0

    if state.phase is Phase.KEEPER:
        state.phase = Phase.DONE
    elif state.phase is Phase.ROUTING:
        state.phase = workflow.route_next(state, result)
    else:
        state.phase = state.phase.next
    save(state)
    _out(f"Следующая фаза: {state.phase.title_ru}")
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    """Ответить на вопрос, который задал агент."""
    session_id = _session_id(args.session)
    if not session_id:
        _out("Нет сохранённых сессий.")
        return 1

    state = load(session_id)
    if state is None:
        _out(f"Сессия {session_id} не найдена.")
        return 1

    state.pending_question = ""
    # Стадия записывается здесь: ответ дан там, где работа стояла, и это
    # единственное место, где она ещё известна. Без стадии ссылка на ввод
    # не проверяется — «где-то в сессии человек что-то сказал» (`D-49`).
    record_human_input(state, args.text)
    save(state)
    _out(f"{CHECK} ответ сохранён в сессию {session_id} (стадия «{state.phase.title_ru}»)")
    _out(f"Продолжить: pf resume --session {session_id}")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Остановить текущую сессию."""
    session_id = _session_id(args.session)
    if not session_id:
        _out("Нет сохранённых сессий.")
        return 1

    state = load(session_id)
    if state is None:
        _out(f"Сессия {session_id} не найдена.")
        return 1

    state.pending_question = ""
    state.results["stopped_at"] = state.phase.value
    save(state)
    _out(f"{CHECK} сессия {session_id} остановлена на фазе «{state.phase.title_ru}»")
    _out(f"Продолжить: pf resume --session {session_id}")
    return 0


# ── materials ───────────────────────────────────────────────────────────────


def cmd_materials(args: argparse.Namespace) -> int:
    """Показать материалы: свои документы, таблицы и данные, и сырые выгрузки.

    У каждого материала — идентификатор для цитирования, адрес источника
    и хеш содержимого. По хешу видно, на какую версию данных опирался
    вывод: при пересмотре это первое, что нужно проверить.

    Пропущенные файлы называются отдельно и с причиной: молчаливый пропуск
    выглядит как отсутствие документа, а это разные вещи.
    """
    from pf.materials import scan_materials

    scan = scan_materials()
    materials = scan.materials
    if not materials:
        _out("Материалов нет.")
        _out(
            f"Положите файлы в {settings.project_dir}/ — .md, .txt, .csv, .json, .pdf, .docx, .xlsx; "
            f"выгрузки читаются из {settings.knowledge_dir}/raw/<дата>/"
        )
        _report_skipped(scan)
        return 0

    from_knowledge = [material for material in materials if material.needs_review]
    _rule(f"Материалы: {len(materials)} (своих: {len(materials) - len(from_knowledge)})")
    for material in materials:
        _out(f"{ARROW} {material.source_id}")
        origin = "выгрузка — требует проверки" if material.needs_review else "материал проекта"
        _out(f"    вид: {material.kind.value}   абзацев: {len(material.passages)}   {origin}")
        if material.url:
            _out(f"    источник: {material.url}")
        if material.retrieved:
            _out(f"    получено: {material.retrieved}")
        _out(f"    sha256: {material.sha256[:16]}…")
        if material.passages:
            _out(f"    {material.passages[0].text[:150]}")
        _out()
    _report_skipped(scan)
    return 0


def _report_skipped(scan: MaterialScan) -> None:
    """Назвать пропущенные файлы и причины пропуска.

    Args:
        scan: Итог обхода материалов (``pf.materials.MaterialScan``).
    """
    if not scan.skipped:
        return
    _rule(f"Пропущено: {len(scan.skipped)}")
    for item in scan.skipped:
        _out(f"✗ {item.path}")
        _out(f"    причина: {item.reason}")
    _out()


# ── report ──────────────────────────────────────────────────────────────────


def cmd_report(args: argparse.Namespace) -> int:
    """Отчёт по прогону: числа, расход и границы замера.

    Отчёт строится по журналу шагов, поэтому цифры воспроизводимы.
    По умолчанию печатается markdown; с `--json` — словарь для
    машинного чтения, с `--save` — файл в каталоге отчётов.
    """
    import json as json_module
    from dataclasses import asdict

    from pf.artifacts import recheck, recheck_markdown
    from pf.report import build_session_report, report_as_dict

    session_id = _session_id(args.session)
    if not session_id:
        _out("Нет сохранённых сессий.")
        return 1

    report = build_session_report(session_id)

    if args.json:
        payload = report_as_dict(report)
        if args.recheck:
            # Перепроверка — отдельный результат: журнал не переписывается.
            payload["recheck"] = asdict(recheck(session_id))
        _out(json_module.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    text = report.as_markdown()
    if args.recheck:
        text = "\n".join([text.rstrip(), "", "---", "", *recheck_markdown(recheck(session_id))])
    if args.save:
        target = settings.reports_dir / f"session-{session_id}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        _out(f"Отчёт сохранён: {target}")
        return 0

    _out(text)
    return 0


# ── knowledge ───────────────────────────────────────────────────────────────


def cmd_knowledge(args: argparse.Namespace) -> int:
    """Показать хранилище знаний: сырые выгрузки и их provenance.

    Для каждой выгрузки — адрес источника, дата получения и хеш. По
    хешу видно, изменились ли данные с момента, когда на них опирался
    вывод: это первое, что проверяется при пересмотре выводов.
    """
    from pf.knowledge_store import list_snapshots, raw_dir, read_manifest

    snapshots = list_snapshots()
    if not snapshots:
        _out("Хранилище знаний пусто.")
        _out("Собрать открытые данные: uv run python scripts/fetch_open_data.py")
        return 0

    _rule(f"Хранилище знаний: {len(snapshots)} выгрузок")
    for date in snapshots:
        directory = raw_dir(date)
        entries = read_manifest(directory)
        _out(f"{ARROW} {date}  файлов: {len(entries)}")
        for entry in entries:
            _out(f"    {entry.get('name', '?')}")
            if entry.get("url"):
                _out(f"        источник: {entry['url']}")
            if entry.get("retrieved"):
                _out(f"        получено: {entry['retrieved']}")
            digest = str(entry.get("sha256", ""))
            if digest:
                _out(f"        sha256: {digest[:16]}…")
            if entry.get("note"):
                _out(f"        {entry['note'][:110]}")
        _out()
    return 0


# ── log / cost ──────────────────────────────────────────────────────────────


def cmd_log(args: argparse.Namespace) -> int:
    """Показать журнал шагов."""
    session_id = _session_id(args.session)
    if not session_id:
        _out("Нет сохранённых сессий.")
        return 1

    journal = Journal(journal_path(session_id))
    records = journal.read()
    if not records:
        _out(f"Журнал сессии {session_id} пуст.")
        return 0

    _rule(f"Журнал сессии {session_id}")
    for record in records:
        _out(f"{record.ordinal:>2}. {record.phase:<12} {record.role:<12} {record.summary}")
        if record.citations:
            _out(f"    основания: {', '.join(record.citations[:4])}")
            if len(record.citations) > 4:
                _out(f"    … ещё {len(record.citations) - 4}")
        if record.rejected_details:
            _out(f"    отклонено оснований: {len(record.rejected_details)}")
            for detail in record.rejected_details[:3]:
                _out(f"      · {detail}")
            if len(record.rejected_details) > 3:
                _out(f"      … ещё {len(record.rejected_details) - 3}")
        _out(
            f"    токены: {record.total_tokens}  "
            f"стоимость: {record.cost_units:.6f} ед.  "
            f"время: {record.created_at}"
        )
    return 0


def cmd_cost(args: argparse.Namespace) -> int:
    """Показать расход."""
    session_id = _session_id(args.session)
    if not session_id:
        _out("Нет сохранённых сессий.")
        return 1

    journal = Journal(journal_path(session_id))
    records = journal.read()
    if not records:
        _out(f"Журнал сессии {session_id} пуст.")
        return 0

    _rule(f"Расход сессии {session_id}")
    _out(f"  Шагов:        {len(records)}")
    _out(f"  Токенов:      {journal.total_tokens}")
    _out(f"  Стоимость:    {journal.total_cost_units:.6f} условных единиц")
    _out()
    _out("  По ролям:")
    for role, cost in journal.cost_by_role().items():
        _out(f"    {role:<12} {cost:.6f}")
    _out()
    _out("  Условная единица = 1/1 000 000 токена. Перевод в рубли —")
    _out("  по тарифу вашей модели (см. docs/decisions.md).")
    return 0


# ── sources / sessions ──────────────────────────────────────────────────────


def cmd_sources(args: argparse.Namespace) -> int:
    """Показать, какие источники агент найдёт по запросу."""
    from pf.sources import collect

    bundle = collect(args.query)
    if not bundle.passages:
        _out("Ничего не найдено. Попробуйте другой запрос или другой корпус.")
        return 1

    _rule(f"Источники по запросу: {args.query}")
    _out(f"Статей: {len(bundle.articles)}, абзацев: {len(bundle.passages)}")
    _out()
    for passage in bundle.passages[: args.limit]:
        _out(f"{ARROW} {passage.citation}  (релевантность {passage.relevance:.2f})")
        _out(f"    {passage.text[:150]}")
        _out()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Выполнить eval-сценарии и показать результаты."""
    from pf.evalrun import format_results, run_all

    results = run_all()
    _rule("Eval-сценарии")
    _out(format_results(results))
    failed = [r for r in results if not r.passed]
    return 1 if failed else 0


def cmd_web(args: argparse.Namespace) -> int:
    """Найти внешние источники по запросу и разобрать найденные страницы.

    Внешние источники никогда не становятся подтверждёнными основаниями:
    их статус — «требует проверки».
    """
    from pf.web import collect_web, search_web
    from pf.webfetch import fetch_pages

    if args.arxiv:
        # Отдельная ветка, а не канал в `search_web`: общий фильтр считает
        # основы русских слов и молча выбросил бы английские заголовки
        # (`D-26`). arXiv не ищет по-русски, поэтому запрос — на английском.
        from pf.arxiv import search_arxiv

        hits = search_arxiv(args.query, limit=args.limit)
        if not hits:
            _out("arXiv не ответил или ничего не нашёл — канал считается недоступным.")
            return 1
        _rule(f"Препринты arXiv: {args.query}")
        for hit in hits:
            _out(f"{ARROW} {hit.title}")
            _out(f"    {hit.url}")
            _out("    препринт: не рецензирован")
            if hit.snippet:
                _out(f"    {hit.snippet[:180]}")
            _out()
        _out("Статус любого внешнего источника — «требует проверки».")
        return 0

    if args.parse:
        hits = search_web(args.query, limit=args.limit)
        if not hits:
            _out("Ничего не найдено в интернете.")
            return 1
        _rule(f"Найдено в интернете: {args.query}")
        _out(f"Результатов: {len(hits)}. Разбираю страницы…")
        _out()
        pages = fetch_pages(hits, limit=args.limit)
        for hit in hits:
            mark = CHECK if any(page.url == hit.url for page in pages) else "-"
            _out(f"{mark} {hit.title}")
            _out(f"    {hit.url}")
            _out(f"    канал: {hit.provider}")
            if hit.snippet:
                _out(f"    {hit.snippet[:150]}")
            _out()
        if not pages:
            _out("Страницы не разобраны: пустой текст или запрет robots.txt.")
            return 1
        _rule("Разобранные страницы")
        for page in pages:
            _out(f"{ARROW} {page.title}")
            _out(f"    {page.url}")
            _out(f"    абзацев: {len(page.paragraphs)}  статус: {page.status}")
            if page.paragraphs:
                _out(f"    {page.paragraphs[0][:180]}")
            _out()
        _out("Статус любого внешнего источника — «требует проверки».")
        return 0

    sources = collect_web(args.query, limit=args.limit)
    if not sources:
        _out("Ничего не найдено в вебе.")
        return 1

    _rule(f"Внешние источники по запросу: {args.query}")
    _out(f"Найдено: {len(sources)}. Статус каждого — требует проверки.")
    _out()
    for source in sources:
        _out(f"{ARROW} {source.title}")
        _out(f"    {source.url}")
        _out(f"    {source.text[:150]}")
        _out()
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    """Показать инструменты MCP-сервера или вызвать инструмент.

    Серверы лежат в соседнем репозитории и запускаются в своём
    окружении: у них тяжёлые зависимости, несовместимые с лёгким
    окружением агента.
    """
    import asyncio

    from pf.mcp import ServerSpec, call_tool, list_tools, load_config

    servers = load_config()
    if not servers:
        _out("Серверы MCP не описаны в mcp/servers.json.")
        return 1

    if args.server not in servers:
        _out(f"Сервер «{args.server}» не описан. Доступны: {', '.join(servers)}")
        return 1

    spec: ServerSpec = servers[args.server]

    if args.tool:
        arguments: dict[str, object] = {}
        if args.args:
            try:
                arguments = json.loads(args.args)
            except json.JSONDecodeError as exc:
                _out(f"Аргументы должны быть JSON: {exc}")
                return 1
        try:
            result = asyncio.run(call_tool(spec, args.tool, arguments))
        except Exception as exc:  # noqa: BLE001 — CLI показывает ошибку
            _out(f"ОШИБКА: {exc}")
            return 1
        _rule(f"{args.server} → {args.tool}")
        _out(result[:4000])
        return 0

    try:
        tools = asyncio.run(list_tools(spec))
    except Exception as exc:  # noqa: BLE001 — CLI показывает ошибку
        _out(f"ОШИБКА: {exc}")
        return 1

    _rule(f"Инструменты сервера: {args.server}")
    for tool in tools:
        _out(f"{ARROW} {tool.name}")
        if tool.description:
            _out(f"    {tool.description[:110]}")
    _out()
    _out(f'Вызов: pf mcp {args.server} <инструмент> \'{{"аргумент": "значение"}}\'')
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Пересмотреть выводы сессии по изменившимся данным.

    Аргументы::

        args.session — идентификатор сессии;
        args.json — вывести результат машинно.

    Команда ничего не переписывает: журнал читается, пересмотр — отдельный
    результат. Так же, как у остальных команд, код возврата 0 означает
    «пересмотр выполнен», а не «всё в порядке»: изменившиеся данные —
    это новость, а не сбой.

    Returns:
        Код возврата.
    """
    from pf.review import review_markdown, review_session

    session_id = _session_id(args.session)
    if not session_id:
        _out("Нет сохранённых сессий: пересматривать нечего.")
        return 1

    result = review_session(session_id)
    if args.json:
        payload = {
            "session": result.session_id,
            "steps": result.steps_total,
            "checked": [
                {
                    "source_id": source.source_id,
                    "status": str(source.status),
                    "recorded": source.recorded,
                    "current": source.current,
                    "url": source.url,
                    "retrieved": source.retrieved,
                }
                for source in result.checked
            ],
            "affected_steps": [step.ordinal for step in result.affected_steps],
            "affected_commitments": [
                commitment.statement for commitment in result.affected_commitments
            ],
            "outside": result.outside,
        }
        _out(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    _rule("Пересмотр выводов")
    _out(review_markdown(result))
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    """Показать список сохранённых сессий."""
    sessions = list_sessions()
    if not sessions:
        _out("Нет сохранённых сессий.")
        return 0

    _rule("Сессии")
    for session_id in sessions:
        state = load(session_id)
        if state is None:
            continue
        _out(
            f"  {session_id}  фаза: {state.phase.title_ru:<24} "
            f"шагов: {state.steps_done}  стоимость: {state.cost_units:.6f}"
        )
        _out(f"      {state.question[:70]}")
    return 0


def cmd_lexicon(args: argparse.Namespace) -> int:
    """Лексикон: словарь близости и очередь распознания слов.

    Args:
        args: Аргументы командной строки.

    Returns:
        Код возврата.
    """
    from pf import lexicon

    if args.action == "stats":
        data = lexicon.stats()
        _out(
            f"Записей в словаре: {data['entries']} "
            f"(подтверждено {data['approved']}, предложено {data['proposed']})"
        )
        _out(f"Очередь распознания: {data['oov_total']} слов, ждут {data['oov_pending']}")
        for item in data["top"]:
            _out(f"  {item['token']}: {item['count']} раз ({item['status']})")
        return 0

    if args.action == "oov":
        records = lexicon.pending_oov(min_count=args.min_count)
        if not records:
            _out("Очередь пуста: незнакомых слов нет.")
            return 0
        _out(f"Ждут распознания: {len(records)}")
        for record in records[: args.limit]:
            _out(f"  {record.token}: {record.count} раз; {record.reason}")
            for context in record.contexts[:1]:
                _out(f"      контекст: {context[:120]}")
        _out("")
        _out("Разобрать: pf lexicon enrich --dry-run")
        return 0

    if args.action == "enrich":
        result = lexicon.enrich(
            limit=args.limit,
            min_count=args.min_count,
            dry_run=not args.apply,
            auto_approve=args.approve,
        )
        for proposal in result.proposals:
            target = f" → {proposal.canonical}" if proposal.canonical else ""
            _out(f"  {proposal.token}: {proposal.action}{target}")
        _out(result.summary)
        if not args.apply:
            _out("Это черновик: ничего не записано. Повторите с --apply.")
        return 0

    if args.token:
        changed = lexicon.approve(args.token, args.canonical or "")
        _out("Запись подтверждена." if changed else "Такой записи нет: сначала pf lexicon enrich.")
        return 0 if changed else 1

    _out("Укажите слово: pf lexicon approve <слово>")
    return 1


def cmd_translate(args: argparse.Namespace) -> int:
    """Перевести английский текст на русский и показать, что сохранено.

    Перевод — не основание: он существует, чтобы английский источник можно
    было проверить словами, а не только числами. Поэтому команда печатает и
    перевод, и замечания о потерянных числах.

    Args:
        args: Аргументы командной строки.

    Returns:
        Код возврата: 1, если перевода нет.
    """
    from pathlib import Path

    from pf.translate import translate

    source = Path(args.file).read_text(encoding="utf-8") if args.file else args.text or ""
    if not source.strip():
        _out("Нечего переводить: укажите файл или --text.")
        return 1

    result = translate(source, refresh=args.refresh)
    if args.json:
        _out(
            json.dumps(
                {
                    "engine": result.engine,
                    "model": result.model,
                    "ok": result.ok,
                    "notes": list(result.notes),
                    "usage_tokens": result.usage_tokens,
                    "text": result.text,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if result.is_usable else 1

    if not result.is_usable:
        _out("Перевода нет.")
        for note in result.notes:
            _out(f"  {note}")
        return 1

    _rule(f"Перевод ({result.engine}, {result.model or 'модель неизвестна'})")
    _out(result.text)
    if result.notes:
        _out("")
        for note in result.notes:
            _out(f"  {note}")
    _out("")
    _out("Перевод машинный и основанием не является: цитируется оригинал.")
    return 0


# ── main ────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Собрать разбор аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="pf",
        description="Стратег Фабрики продуктов — ИИ-агент для руководителя цифрового продукта.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="поставить задачу и провести работу до стратегии")
    run.add_argument("question", help="вопрос руководителя продукта")
    run.add_argument("--session", help="идентификатор сессии")
    run.set_defaults(func=cmd_run)

    resume = subparsers.add_parser("resume", help="продолжить сохранённую сессию")
    resume.add_argument("--session", help="идентификатор сессии")
    resume.set_defaults(func=cmd_resume)

    answer = subparsers.add_parser("answer", help="ответить на вопрос агента")
    answer.add_argument("text", help="ответ")
    answer.add_argument("--session", help="идентификатор сессии")
    answer.set_defaults(func=cmd_answer)

    stop = subparsers.add_parser("stop", help="остановить сессию")
    stop.add_argument("--session", help="идентификатор сессии")
    stop.set_defaults(func=cmd_stop)

    log = subparsers.add_parser("log", help="показать журнал шагов")
    log.add_argument("--session", help="идентификатор сессии")
    log.set_defaults(func=cmd_log)

    cost = subparsers.add_parser("cost", help="показать расход")
    cost.add_argument("--session", help="идентификатор сессии")
    cost.set_defaults(func=cmd_cost)

    sources = subparsers.add_parser("sources", help="показать найденные источники")
    sources.add_argument("query", help="поисковый запрос")
    sources.add_argument("--limit", type=int, default=8, help="сколько абзацев показать")
    sources.set_defaults(func=cmd_sources)

    materials = subparsers.add_parser(
        "materials", help="показать материалы проекта (документы и таблицы)"
    )
    materials.set_defaults(func=cmd_materials)

    report = subparsers.add_parser(
        "report", help="отчёт по прогону: основания, расход, границы замера"
    )
    report.add_argument("--session", help="идентификатор сессии")
    report.add_argument("--json", action="store_true", help="вывести в JSON")
    report.add_argument("--save", action="store_true", help="сохранить в reports/")
    report.add_argument(
        "--recheck",
        action="store_true",
        help="перепроверить запись текущим кодом на сохранённых ответах ролей",
    )
    report.set_defaults(func=cmd_report)

    knowledge = subparsers.add_parser(
        "knowledge", help="показать хранилище знаний (выгрузки и их источники)"
    )
    knowledge.set_defaults(func=cmd_knowledge)

    sessions = subparsers.add_parser("sessions", help="список сохранённых сессий")
    sessions.set_defaults(func=cmd_sessions)

    review_parser = subparsers.add_parser(
        "review", help="пересмотреть выводы по изменившимся данным"
    )
    review_parser.add_argument("--session", help="идентификатор сессии")
    review_parser.add_argument("--json", action="store_true", help="вывести результат машинно")
    review_parser.set_defaults(func=cmd_review)

    lexicon_parser = subparsers.add_parser(
        "lexicon", help="словарь близости и очередь распознания слов"
    )
    lexicon_parser.add_argument(
        "action",
        choices=["stats", "oov", "enrich", "approve"],
        help="stats — покрытие, oov — очередь, enrich — разобрать, approve — подтвердить",
    )
    lexicon_parser.add_argument("token", nargs="?", help="слово для подтверждения")
    lexicon_parser.add_argument("--canonical", help="каноническая лемма при подтверждении")
    lexicon_parser.add_argument("--limit", type=int, default=20, help="сколько слов показать")
    lexicon_parser.add_argument("--min-count", type=int, default=1, help="минимальная частота")
    lexicon_parser.add_argument(
        "--apply",
        action="store_true",
        help="записать предложения в словарь (по умолчанию — черновик)",
    )
    lexicon_parser.add_argument(
        "--approve",
        action="store_true",
        help="сразу подтвердить записи (минуя отдельный шаг approve)",
    )
    lexicon_parser.set_defaults(func=cmd_lexicon)

    eval_parser = subparsers.add_parser("eval", help="выполнить eval-сценарии и проверить правила")
    eval_parser.set_defaults(func=cmd_eval)

    web_parser = subparsers.add_parser("web", help="найти внешние источники (требуют проверки)")
    web_parser.add_argument("query", help="поисковый запрос")
    web_parser.add_argument("--limit", type=int, default=3, help="сколько источников")
    web_parser.add_argument(
        "--parse",
        action="store_true",
        help="искать в интернете и разбирать найденные страницы",
    )
    web_parser.add_argument(
        "--arxiv",
        action="store_true",
        help="искать препринты в arXiv (запрос на английском, не рецензированы)",
    )
    web_parser.set_defaults(func=cmd_web)

    translate_parser = subparsers.add_parser(
        "translate", help="перевести английский материал на русский"
    )
    translate_parser.add_argument("file", nargs="?", help="файл с текстом")
    translate_parser.add_argument("--text", default="", help="текст вместо файла")
    translate_parser.add_argument(
        "--refresh", action="store_true", help="пересчитать перевод, не веря кэшу"
    )
    translate_parser.add_argument("--json", action="store_true", help="вывести JSON")
    translate_parser.set_defaults(func=cmd_translate)

    mcp_parser = subparsers.add_parser("mcp", help="инструменты MCP-сервера")
    mcp_parser.add_argument("server", help="имя сервера из mcp/servers.json")
    mcp_parser.add_argument("tool", nargs="?", help="инструмент для вызова")
    mcp_parser.add_argument("args", nargs="?", help="аргументы вызова в виде JSON")
    mcp_parser.set_defaults(func=cmd_mcp)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Точка входа.

    Args:
        argv: Аргументы командной строки.

    Returns:
        Код возврата.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        parser.print_help()
        return 0

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        _out()
        _out("Прервано пользователем. Прогресс сохранён — pf resume продолжит.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
