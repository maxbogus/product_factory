"""Подбор порога степени поддержки на многократных разбиениях набора пар.

Зачем скрипт, а не подобранное число в коде. Степень поддержки
(`pf.canonical.score`) — число от 0 до 1, и порог, при котором оно
становится подтверждением, — настройка (`claim_support_threshold`). Порог
надо откуда-то взять: одно разбиение 30/30 — это шум, и порог, подобранный под
него, развалится на других абзацах. Поэтому порог берётся на повторных
разбиениях (`--folds` × `--repeats`), а в настройку попадает медиана с
отступом вниз.

Правило отбора — то же, что в замере семантики (`calibrate_threshold`): порог
не должен понижать ни одну честную пару **train**, а отступ
(`--margin`) оставляет запас на то, что продовые абзацы отличаются от набора.
Порог не подбирается максимумом F1 на тех же парах, где считается полнота.

Запуск:
    uv run python scripts/tune_threshold.py --json
    make tune-threshold
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections.abc import Sequence
from typing import Any

from pf.canonical import score
from pf.semantic_bench import Pair, brier, load_pairs, pairs_sha256, roc_auc, source_passage


def support_of(pairs: Sequence[Pair]) -> dict[str, float]:
    """Степень поддержки для каждой пары набора.

    Args:
        pairs: Пары набора.

    Returns:
        Отображение «идентификатор пары → степень поддержки».
    """
    scores: dict[str, float] = {}
    for pair in pairs:
        passage = source_passage(pair.source_id, pair.paragraph)
        if passage is None:
            continue
        scores[pair.pair_id] = score(pair.claim, passage).support
    return scores


def folds_of(pairs: Sequence[Pair], folds: int, seed: int) -> list[dict[str, set[str]]]:
    """Разбить набор на фолды внутри каждого класса.

    Деление внутри класса, а не по чётности номера: иначе одна половина
    оказалась бы целиком из честных пар, и точность на отложенной не
    проверялась бы вовсе — на этом замер уже ошибался (`D-37`).

    Args:
        pairs: Пары набора.
        folds: Число фолдов.
        seed: Зерно перемешивания — прогон должен повторяться.

    Returns:
        Список отображений «train/test → идентификаторы пар».
    """
    honest = [pair.pair_id for pair in pairs if pair.label.value == "in_place"]
    alien = [pair.pair_id for pair in pairs if pair.label.value == "out_of_place"]
    generator = random.Random(seed)
    assignment: dict[str, int] = {}
    for group in (honest, alien):
        order = list(group)
        generator.shuffle(order)
        for position, pair_id in enumerate(order):
            assignment[pair_id] = position % folds
    return [
        {
            "test": {pair_id for pair_id, index in assignment.items() if index == fold},
            "train": {pair_id for pair_id, index in assignment.items() if index != fold},
        }
        for fold in range(folds)
    ]



def _quantile(values: list[float], share: float) -> float | None:
    """Квантиль выборки без интерполяции.

    Нужен, чтобы один выброс не задавал порог: на наборе есть честная пара,
    которую поля отклоняют по названной причине (регион назван в
    утверждении, а в абзаце его нет, `p-037`), и правило «минимум по честным»
    из замера семантики утащило бы порог к 0,006 — тогда утекают 72 подмены
    из 300 слотов. Это измерено, а не предположено.

    Args:
        values: Значения.
        share: Доля от 0 до 1.

    Returns:
        Значение на позиции доли либо ``None`` для пустой выборки.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = round(share * (len(ordered) - 1))
    return ordered[min(max(index, 0), len(ordered) - 1)]


def evaluate(
    scores: dict[str, float],
    pairs: Sequence[Pair],
    *,
    folds: int,
    repeats: int,
    seed: int,
    margin: float,
) -> dict[str, Any]:
    """Посчитать порог по фолдам и метрики на отложенных фолдах.

    Args:
        scores: Степень поддержки по парам.
        pairs: Пары набора.
        folds: Число фолдов.
        repeats: Число повторных разбиений.
        seed: Начальное зерно.
        margin: Отступ вниз от минимальной честной степени train.

    Returns:
        Словарь с порогами по фолдам и метриками.
    """
    labels = {pair.pair_id: pair.label.value for pair in pairs}
    thresholds: list[float] = []
    recalls: list[float] = []
    leaks_total = 0
    alien_total = 0
    for repeat in range(repeats):
        for fold in folds_of(pairs, folds, seed + repeat):
            honest_train = [
                scores[pair_id]
                for pair_id in fold["train"]
                if labels[pair_id] == "in_place"
            ]
            alien_train = [
                scores[pair_id]
                for pair_id in fold["train"]
                if labels[pair_id] == "out_of_place"
            ]
            honest_edge = _quantile(honest_train, 0.1)
            alien_edge = _quantile(alien_train, 0.9)
            if honest_edge is None:
                continue
            if alien_edge is None:
                candidate = honest_edge
            else:
                # Середина между кластерами, а не край одного из них: порог,
                # прижатый к честным, пропускает подмены, прижатый к подменам —
                # рубит пересказы.
                candidate = 0.5 * (honest_edge + alien_edge)
                candidate = max(candidate, max(alien_train) * 1.05)
            candidate = min(max(candidate - margin, 0.01), 0.9)
            thresholds.append(candidate)
            honest_test = [
                pair_id for pair_id in fold["test"] if labels[pair_id] == "in_place"
            ]
            alien_test = [
                pair_id for pair_id in fold["test"] if labels[pair_id] == "out_of_place"
            ]
            if honest_test:
                recalls.append(
                    sum(1 for pair_id in honest_test if scores[pair_id] >= candidate)
                    / len(honest_test)
                )
            leaks_total += sum(1 for pair_id in alien_test if scores[pair_id] >= candidate)
            alien_total += len(alien_test)
    return {
        "folds": folds,
        "repeats": repeats,
        "margin": margin,
        "threshold_median": round(statistics.median(thresholds), 4) if thresholds else 0.0,
        "threshold_min": round(min(thresholds), 4) if thresholds else 0.0,
        "threshold_max": round(max(thresholds), 4) if thresholds else 0.0,
        "recall_mean": round(statistics.fmean(recalls), 4) if recalls else 0.0,
        "recall_min": round(min(recalls), 4) if recalls else 0.0,
        "leaks": leaks_total,
        "substitution_slots": alien_total,
    }


def parse_args() -> argparse.Namespace:
    """Разобрать аргументы командной строки.

    Returns:
        Аргументы.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5, help="фолдов на разбиение")
    parser.add_argument("--repeats", type=int, default=10, help="повторных разбиений")
    parser.add_argument("--seed", type=int, default=13, help="начальное зерно")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="сдвиг порога вниз после середины между кластерами",
    )
    parser.add_argument("--json", action="store_true", help="вывести JSON")
    return parser.parse_args()


def main() -> int:
    """Точка входа.

    Returns:
        Код возврата.
    """
    args = parse_args()
    pairs = load_pairs()
    scores = support_of(pairs)
    report = evaluate(
        scores,
        pairs,
        folds=args.folds,
        repeats=args.repeats,
        seed=args.seed,
        margin=args.margin,
    )
    # Замораживается верх полосы, округлённый вверх: подмена периода или
    # оператора — худшая ошибка из двух, и полоса показывает, насколько порог
    # зависит от состава train. Округление вверх оставляет запас на это.
    frozen = math.ceil(report["threshold_max"] * 100) / 100 if report["threshold_max"] else 0.0
    report["threshold_frozen"] = round(frozen, 4)
    # Степень названа вероятностью только если она лучше базовой ставки:
    # Brier 0,25 — это «всегда 0,5», то есть отсутствие информации.
    report["auc"] = round(roc_auc(scores, pairs), 4)
    report["brier"] = round(brier(scores, pairs), 4)
    report["pairs"] = len(pairs)
    report["pairs_sha256"] = pairs_sha256()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    print(f"Пар в наборе: {report['pairs']} (sha256 {report['pairs_sha256'][:12]})")
    print(
        f"Порог по фолдам: медиана {report['threshold_median']}, "
        f"от {report['threshold_min']} до {report['threshold_max']}"
    )
    print(
        f"Подтверждено честных на отложенных фолдах: {report['recall_mean']:.2f} "
        f"(худший фолд {report['recall_min']:.2f})"
    )
    print(
        f"Утечек: {report['leaks']} из {report['substitution_slots']} слотов "
        "«не по месту»"
    )
    print(f"Заморозить в PF_CLAIM_SUPPORT_THRESHOLD: {report['threshold_frozen']}")
    print(
        f"Различение классов: AUC {report['auc']}, Brier {report['brier']} "
        "(базовая ставка 0,25)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

