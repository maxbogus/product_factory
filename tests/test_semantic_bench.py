"""Проверки замера семантики: набор пар, гейт решения, вердикт (D-37).

Зачем отдельный файл. Семантика в проверку оснований не добавлена, и это
решение держится не на мнении, а на числах (`docs/decisions.md`, D-37). Числа
появляются из замера, а замер — это код: набор размеченных пар, метрики и
гейт с четырьмя критериями. Если такой код сломается молча, следующий
вердикт будет «нет» независимо от того, что показали модели, — поэтому
проверяется он сам.

Две группы проверок:

- ``-k gate`` (T-36) — набор пар, метрики, гейт. Ни модели, ни сети:
  TF-IDF считается по леммам, эмбеддинги подключаются только в замере
  (``scripts/semantic_bench.py``) и в тестах не нужны;
- ``-k verdict`` (T-37) — снимок замера: он соответствует набору пар, в нём
  есть числа по каждому методу, и вердикт в ``docs/decisions.md`` совпадает
  со снимком, а не написан руками.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

#: Размеченный набор пар: цитата дословная, но к утверждению не подходит.
PAIRS = ROOT / "eval" / "pairs" / "out_of_place.jsonl"

#: Снимок замера: им живут тесты, чтобы `make test` не грузил модель.
MEASUREMENT = ROOT / "tests" / "fixtures" / "semantic" / "measurement.json"

#: Методы, перечисленные в D-37: пол, «быстрый» и «качественный».
REQUIRED_METHODS = ("rubert", "sbert", "tfidf")


def _bench() -> ModuleType:
    """Загрузить модуль замера.

    Импорт внутри функции, а не на уровне модуля: до правки модуля нет, и
    красный прогон должен показать «failed», а не «collection error» —
    иначе не видно, сколько проверок держат правило.

    Returns:
        Модуль `pf.semantic_bench`.
    """
    return importlib.import_module("pf.semantic_bench")


def _measurement() -> dict[str, Any]:
    """Прочитать снимок замера.

    Returns:
        Словарь замера в том виде, в каком его пишет скрипт.
    """
    return json.loads(MEASUREMENT.read_text(encoding="utf-8"))


def _sample_measurement(recall: float = 0.75, lost: int = 0) -> dict[str, Any]:
    """Снимок замера для проверки логики гейта.

    Логика гейта проверяется на контролируемом входе, а не на настоящем
    снимке: иначе проверка «вердикт требует всех критериев» зависела бы от
    того, что намерили модели, и не могла бы покраснеть.

    Args:
        recall: Полнота флага на тестовой половине.
        lost: Сколько честных подтверждений понижено.

    Returns:
        Снимок замера с одним методом.
    """
    half = {
        "split": "test",
        "threshold": 0.2,
        "true_positives": 2,
        "false_positives": lost,
        "false_negatives": 1,
        "true_negatives": 2,
        "positives": 3,
        "precision": 1.0 if lost == 0 else 0.67,
        "recall": recall,
        "confirmations_lost": lost,
    }
    return {
        "measured_at": "2026-11-08",
        "pairs_sha256": "0" * 64,
        "pairs_count": 60,
        "baseline": {"session": "demo-v12", "steps": 9, "run_sec": 311.464},
        "ranking": {"queries": 1, "ndcg_gain": None, "note": "один вопрос на все журналы"},
        "methods": {
            "tfidf": {
                "method": "tfidf",
                "engine": "леммы + TF-IDF",
                "encode_sec": 2.0,
                "rss_delta_mb": 20.0,
                "threshold": 0.2,
                "train": dict(half, split="train"),
                "test": half,
            }
        },
        "verdict": {"passed": True, "summary": "выполнен"},
    }


# ── T-36: набор пар ─────────────────────────────────────────────────────────


def test_gate_pairs_are_verbatim_in_source() -> None:
    """Каждая цитата набора дословно лежит в указанном абзаце источника.

    Метка «дословно, но не по месту» имеет смысл только если слово
    «дословно» проверяемо. Иначе набор тихо сгниёт: правишь цитату — и
    меришь уже не тот класс ошибки.
    """
    bench = _bench()
    pairs = bench.load_pairs(PAIRS)

    missing = [pair.pair_id for pair in pairs if not bench.quote_is_verbatim(pair)]

    assert missing == [], f"цитаты не найдены дословно: {missing}"


def test_gate_pairs_have_both_labels_and_reasons() -> None:
    """В наборе есть оба класса, у каждой пары — обоснование метки."""
    bench = _bench()
    pairs = bench.load_pairs(PAIRS)

    assert len(pairs) >= 60, f"пар мало для полноты 0,7: {len(pairs)}"
    assert {pair.label for pair in pairs} == {bench.Label.IN_PLACE, bench.Label.OUT_OF_PLACE}
    assert {pair.kind for pair in pairs} == {bench.Kind.REAL, bench.Kind.SYNTHETIC}
    assert all(pair.why.strip() for pair in pairs), "метка без обоснования — самооценка"
    assert len({pair.pair_id for pair in pairs}) == len(pairs)
    out_of_place = [pair for pair in pairs if pair.label is bench.Label.OUT_OF_PLACE]
    assert len(out_of_place) >= 30, f"«не по месту» мало: {len(out_of_place)}"


# ── T-36: калибровка и метрики ──────────────────────────────────────────────


def test_gate_threshold_flags_nothing_honest_on_train() -> None:
    """Порог калибруется на train так, что честная пара не понижается."""
    bench = _bench()
    pairs = bench.load_pairs(PAIRS)
    scores = bench.tfidf_scores(pairs)

    threshold = bench.calibrate_threshold(scores, pairs)
    train = bench.flag_precision_recall(scores, pairs, threshold, split="train")

    assert train.false_positives == 0
    assert train.confirmations_lost == 0
    assert threshold > 0.0


def test_gate_recall_counts_only_out_of_place() -> None:
    """Полнота считается по парам «не по месту» тестовой половины."""
    bench = _bench()
    pairs = bench.load_pairs(PAIRS)
    scores = bench.tfidf_scores(pairs)
    threshold = bench.calibrate_threshold(scores, pairs)

    test = bench.flag_precision_recall(scores, pairs, threshold, split="test")
    expected_positives = sum(
        1
        for index, pair in enumerate(pairs)
        if pair.label is bench.Label.OUT_OF_PLACE and bench.split_of(index, pairs) == "test"
    )

    assert test.positives == expected_positives
    assert 0.0 <= test.recall <= 1.0
    # Точность считается по фактическим срабатываниям, а не назначается.
    if test.false_positives == 0:
        assert test.precision == 1.0 or test.positives == 0


def test_gate_missing_ranking_is_named_not_zero() -> None:
    """Без данных отбора nDCG — «не измерено», а не ноль."""
    bench = _bench()

    assert bench.ndcg_at_k([], {}, k=5) is None
    assert bench.ndcg_at_k(["a"], {"a": 1.0}, k=5) == 1.0

    measurement = _sample_measurement()
    measurement["ranking"] = {"queries": 0, "ndcg_gain": None, "note": "один вопрос на все журналы"}
    verdict = bench.gate_verdict(measurement)
    ranking = next(item for item in verdict.criteria if item.key == "ranking")

    assert ranking.measured is False
    assert "не измерено" in ranking.detail


def test_gate_budget_counts_share_of_run() -> None:
    """Бюджет слоя — доля времени прогона из журнала и предел памяти."""
    bench = _bench()

    fits = bench.budget_check(baseline_sec=311.4, encode_sec=2.0, rss_delta_mb=200)
    assert fits.time_share < 0.10
    assert fits.within_time is True
    assert fits.within_memory is True

    overrun = bench.budget_check(baseline_sec=311.4, encode_sec=31.5, rss_delta_mb=1100)
    assert overrun.within_time is False
    assert overrun.within_memory is False


def test_gate_needs_every_measured_criterion() -> None:
    """Вердикт «прошёл» требует всех измеримых критериев одновременно."""
    bench = _bench()

    measurement = _sample_measurement()
    verdict = bench.gate_verdict(measurement)
    measured = [item for item in verdict.criteria if item.measured]
    assert verdict.passed == all(item.passed for item in measured)

    broken = json.loads(json.dumps(measurement))
    broken["methods"]["tfidf"]["test"]["recall"] = 0.2
    broken_verdict = bench.gate_verdict(broken)

    assert broken_verdict.passed is False
    failed = [item for item in broken_verdict.criteria if item.measured and not item.passed]
    assert [item.key for item in failed] == ["flag_recall"]
    assert "полнота" in failed[0].detail


# ── T-37: снимок замера и вердикт в документах ──────────────────────────────


def test_verdict_measurement_matches_pairs_file() -> None:
    """Снимок замера относится к текущему набору пар, а не к прошлому."""
    bench = _bench()
    measurement = _measurement()

    assert measurement["pairs_sha256"] == bench.pairs_sha256(PAIRS)
    assert measurement["pairs_count"] == len(bench.load_pairs(PAIRS))


def test_verdict_has_methods_with_numbers() -> None:
    """По каждому методу D-37 в замере есть числа, а не обещания."""
    measurement = _measurement()

    assert tuple(sorted(measurement["methods"])) == REQUIRED_METHODS
    for name, method in measurement["methods"].items():
        assert method["encode_sec"] > 0, name
        assert method["rss_delta_mb"] > 0, name
        assert "train" in method and "test" in method, name
        assert method["engine"], name
    assert measurement["baseline"]["run_sec"] > 0
    assert measurement["verdict"]["passed"] in (True, False)


def test_verdict_block_in_decision_matches_measurement() -> None:
    """Блок вердикта в `decisions.md` собран из замера, а не написан руками."""
    bench = _bench()
    measurement = _measurement()
    decisions = (ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")

    block = bench.verdict_markdown(measurement)

    assert block.strip() in decisions, "вердикт в decisions.md разошёлся со снимком замера"
