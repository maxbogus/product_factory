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
- ``pf eval`` — выполнить eval-сценарии и проверить правила поведения;
- ``pf web "запрос"`` — найти внешние источники (требуют проверки).

Интерфейс командный, а не графический, сознательно: руководителю продукта
нужно видеть продвижение работы и останавливать её, а не изучать устройство
агента (требование «удобство» и «прозрачность»).
"""

from __future__ import annotations

import argparse
import json
import sys

from pf import __version__
from pf.journal import Journal, journal_path
from pf.state import Phase, SessionState, list_sessions, load, save

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


def cmd_run(args: argparse.Namespace) -> int:
    """Поставить задачу и провести работу до стратегии."""
    from pf.workflow import run_full_workflow

    session_id = args.session or f"s{int(__import__('time').time())}"
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
    bundle = (
        workflow.collect_sources(state)
        if state.phase
        in {
            Phase.ANALYZING,
            Phase.STRATEGY,
        }
        else None
    )

    try:
        result = workflow.run_step(state, bundle)
    except Exception as exc:  # noqa: BLE001
        _out(f"ОШИБКА: {exc}")
        return 1

    _out(f"{CHECK} {result.phase.title_ru}: {_describe(result.data)}")
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
    state.results.setdefault("human_answers", []).append(args.text)
    save(state)
    _out(f"{CHECK} ответ сохранён в сессию {session_id}")
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
    _out(f"Вызов: pf mcp {args.server} <инструмент> '{{\"аргумент\": \"значение\"}}'")
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

    sessions = subparsers.add_parser("sessions", help="список сохранённых сессий")
    sessions.set_defaults(func=cmd_sessions)

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
    web_parser.set_defaults(func=cmd_web)

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
