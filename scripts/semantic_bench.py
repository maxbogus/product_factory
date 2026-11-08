#!/usr/bin/env python3
"""Замер семантики: набор пар, TF-IDF против эмбеддингов, вердикт гейта.

Зачем скрипт. `D-37` говорит: прежде чем ставить модель эмбеддингов в
зависимости, надо измерить, решает ли она ту задачу, ради которой её берут.
Задача — флаг «цитата дословная, но не по месту»: честная цитата, поставленная
не к тому утверждению, сейчас проходит проверку оснований.

Что делает скрипт:

1. читает размеченный набор `eval/pairs/out_of_place.jsonl`;
2. считает близость «утверждение — цитата» каждым из методов: ``tfidf`` (пол,
   без зависимостей), ``rubert`` (`DeepPavlov/ruBERT-base`, среднее по
   токенам), ``sbert`` (`ai-forever/sbert_large_nlu_ru`);
3. калибрует порог флага на половине train, считает полноту и точность на
   половине test, меряет время и память;
4. пишет снимок замера и печатает вердикт по критериям `D-37`.

Модели не входят в `pyproject.toml` намеренно (`D-09`, `D-37`): замер идёт
одноразовым окружением, чтобы правило «один `.venv`» не нарушалось:

    uv run --with torch --with transformers --with sentence-transformers \\
        python scripts/semantic_bench.py --method tfidf --method rubert --method sbert

Без моделей работает `--method tfidf`: он и есть пол сравнения.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from pf.journal import Journal, journal_path
from pf.sanitize import for_comparison
from pf.semantic_bench import (
    Pair,
    evaluate_method,
    gate_verdict,
    load_pairs,
    pairs_sha256,
    tfidf_scores,
    verdict_markdown,
)

#: Корень репозитория: скрипт лежит в ``scripts/``.
REPO = Path(__file__).resolve().parent.parent

#: Куда пишутся снимки замера.
DEFAULT_OUT = REPO / "reports" / "semantic"

#: Сессия, по журналу которой берётся бюджет времени.
DEFAULT_SESSION = "demo-v12"

#: «Быстрая» модель из `D-37`. Идентификатор на HF — `rubert-base-cased`;
#: имя «ruBERT-base» из статьи модели на хабе не резолвится (401), и первый
#: прогон замера упал именно на этом (`docs/sessions/2026-11-08-session-12.md`).
RUBERT_MODEL = "DeepPavlov/rubert-base-cased"

#: «Качественная» модель из `D-37`.
SBERT_MODEL = "ai-forever/sbert_large_nlu_ru"

#: Методы, которые умеет скрипт.
METHODS = ("tfidf", "rubert", "sbert")


def _peak_rss_mb() -> float:
    """Пиковая память процесса, МБ.

    Returns:
        Пиковый RSS в мегабайтах.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _questions_of(session: str) -> list[str]:
    """Различные вопросы задачи из журнала сессии.

    Вопросов меньше двух — значит данные отбора кандидатов измерить не на
    чем: nDCG на одном вопросе даёт шум, а не число.

    Args:
        session: Идентификатор сессии.

    Returns:
        Список различных вопросов в порядке появления.
    """
    seen: list[str] = []
    for record in Journal(journal_path(session)).read():
        question = record.detail.get("question")
        if isinstance(question, str) and question.strip() and question not in seen:
            seen.append(question)
    return seen


def _baseline(session: str) -> dict[str, Any]:
    """Бюджет времени: длительность прогона агента по журналу.

    Args:
        session: Идентификатор сессии.

    Returns:
        Словарь с шагами и секундами.

    Raises:
        SystemExit: Журнала нет — бюджет брать неоткуда.
    """
    records = Journal(journal_path(session)).read()
    if not records:
        raise SystemExit(f"нет журнала сессии «{session}»: бюджет времени брать неоткуда")
    total_ms = sum(record.duration_ms for record in records)
    return {
        "session": session,
        "steps": len(records),
        "run_sec": round(total_ms / 1000.0, 3),
    }


def _cosine_scores(
    pairs: Sequence[Pair], encode: Callable[[list[str]], list[list[float]]]
) -> dict[str, float]:
    """Близость пар по готовому кодировщику.

    Args:
        pairs: Набор пар.
        encode: Функция «список текстов → список нормированных векторов».

    Returns:
        Отображение «идентификатор пары → близость».
    """
    # Чистка одна на все методы (`T-42`): иначе лексика сравнивает текст без
    # url и имён колонок, а эмбеддинги — с ними, и замеры несравнимы.
    texts = [for_comparison(pair.claim) for pair in pairs] + [
        for_comparison(pair.quote) for pair in pairs
    ]
    vectors = encode(texts)
    size = len(pairs)
    scores: dict[str, float] = {}
    for index, pair in enumerate(pairs):
        left = vectors[index]
        right = vectors[size + index]
        scores[pair.pair_id] = float(round(sum(a * b for a, b in zip(left, right, strict=False)), 6))
    return scores


def _rubert_scores(pairs: Sequence[Pair]) -> dict[str, float]:
    """Близость через `DeepPavlov/ruBERT-base` со средним по токенам.

    У этой модели нет настроек sentence-transformers, поэтому векторы
    считаются вручную: среднее по токенам с учётом маски и нормировка. Это
    ровно та модель, которую `D-37` называет «быстрой».

    Args:
        pairs: Набор пар.

    Returns:
        Отображение «идентификатор пары → близость».
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - одноразовое окружение
        raise SystemExit(
            "нет transformers/torch: запустите через "
            "uv run --with torch --with transformers python scripts/semantic_bench.py"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(RUBERT_MODEL)
    model = AutoModel.from_pretrained(RUBERT_MODEL)
    model.eval()

    def encode(texts: list[str]) -> list[list[float]]:
        result: list[list[float]] = []
        with torch.no_grad():
            for start in range(0, len(texts), 8):
                batch = tokenizer(
                    texts[start : start + 8],
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                hidden = model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
                normalised = torch.nn.functional.normalize(pooled, dim=1)
                result.extend(normalised.tolist())
        return result

    return _cosine_scores(pairs, encode)


def _sbert_scores(pairs: Sequence[Pair]) -> dict[str, float]:
    """Близость через `ai-forever/sbert_large_nlu_ru`.

    Args:
        pairs: Набор пар.

    Returns:
        Отображение «идентификатор пары → близость».
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - одноразовое окружение
        raise SystemExit(
            "нет sentence-transformers: запустите через "
            "uv run --with sentence-transformers python scripts/semantic_bench.py"
        ) from exc

    model = SentenceTransformer(SBERT_MODEL, device="cpu")

    def encode(texts: list[str]) -> list[list[float]]:
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=8)
        return [list(vector) for vector in vectors]

    return _cosine_scores(pairs, encode)


#: Кодировщики по имени метода.
SCORERS: dict[str, tuple[str, Callable[[Sequence[Pair]], dict[str, float]]]] = {
    "tfidf": ("леммы + TF-IDF (в `.venv`)", tfidf_scores),
    "rubert": (RUBERT_MODEL, _rubert_scores),
    "sbert": (SBERT_MODEL, _sbert_scores),
}


def _measure(pairs: Sequence[Pair], name: str) -> dict[str, Any]:
    """Замерить один метод: метрики, время, память.

    Args:
        pairs: Набор пар.
        name: Имя метода.

    Returns:
        Запись метода для снимка замера.
    """
    engine, scorer = SCORERS[name]
    rss_before = _peak_rss_mb()
    started = time.perf_counter()
    scores = scorer(pairs)
    elapsed = time.perf_counter() - started
    rss_delta = _peak_rss_mb() - rss_before

    record = evaluate_method(scores, pairs)
    record.update(
        {
            "method": name,
            "engine": engine,
            # Пик мог не вырасти: тогда слой меньше разрешения замера. Ноль
            # здесь означал бы «память не нужна», поэтому пишется минимум.
            "rss_delta_mb": round(max(rss_delta, 0.1), 1),
            "encode_sec": round(elapsed, 3),
        }
    )
    return record


def _ranking(session: str) -> dict[str, Any]:
    """Данные отбора кандидатов: измерим ли выигрыш nDCG.

    Args:
        session: Идентификатор сессии.

    Returns:
        Словарь: сколько различных вопросов и почему выигрыш не измерен.
    """
    questions = _questions_of(session)
    if len(questions) < 2:
        return {
            "queries": len(questions),
            "ndcg_gain": None,
            "note": (
                f"в журналах {len(questions)} различных вопросов: nDCG@5 на одном "
                "вопросе — шум, а не число (D-37)"
            ),
        }
    return {"queries": len(questions), "ndcg_gain": None, "note": "отбор кандидатов не замерялся"}


def main(argv: list[str] | None = None) -> int:
    """Собрать снимок замера и напечатать вердикт.

    Args:
        argv: Аргументы командной строки.

    Returns:
        Код возврата: 0 — гейт выполнен, 1 — нет.
    """
    parser = argparse.ArgumentParser(description="Замер семантики: флаг «дословно, но не по месту»")
    parser.add_argument("--method", action="append", choices=METHODS, default=[])
    parser.add_argument("--session", default=DEFAULT_SESSION, help="сессия для бюджета времени")
    parser.add_argument("--json", default="", help="куда записать снимок замера")
    parser.add_argument(
        "--markdown", action="store_true", help="напечатать блок вердикта для decisions.md"
    )
    args = parser.parse_args(argv)

    names = args.method or ["tfidf"]
    pairs = load_pairs()
    baseline = _baseline(args.session)

    methods: dict[str, Any] = {}
    for name in names:
        methods[name] = _measure(pairs, name)
        print(
            f"{name}: полнота {methods[name]['test']['recall']} "
            f"· ложных понижений {methods[name]['test']['confirmations_lost']} "
            f"· {methods[name]['encode_sec']} с · {methods[name]['rss_delta_mb']} МБ",
            file=sys.stderr,
        )

    measurement: dict[str, Any] = {
        "measured_at": date.today().isoformat(),
        "pairs_sha256": pairs_sha256(),
        "pairs_count": len(pairs),
        "baseline": baseline,
        "ranking": _ranking(args.session),
        "methods": methods,
    }
    verdict = gate_verdict(measurement)
    measurement["verdict"] = {"passed": verdict.passed, "summary": verdict.summary}

    # Имя снимка включает методы: иначе `make semantic-bench` (только TF-IDF)
    # молча перезаписал бы полный замер тремя методами.
    default_name = f"{measurement['measured_at']}-{'-'.join(names)}.json"
    target = Path(args.json) if args.json else DEFAULT_OUT / default_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(measurement, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nВердикт: {verdict.summary}")
    for item in verdict.criteria:
        state = (
            "не измерено" if not item.measured else ("выполнен" if item.passed else "НЕ выполнен")
        )
        print(f"  [{state}] {item.title}: {item.detail}")
    print(f"\nСнимок: {target}")

    if args.markdown:
        print()
        print(verdict_markdown(measurement))
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
