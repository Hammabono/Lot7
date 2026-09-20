"""過去の当せん番号から次回の候補を組み立てる予想エンジン。

重要: 抽せんは独立試行であり、ここで計算されるどのスコアにも将来の当せん確率を
動かす力はない。各モデルは「過去データの特徴を反映した選び方」を与えるだけで、
1 口あたりの理論当せん確率は常に 1/10,295,472 のままである。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .data import Draw
from .features import Stats, collect, count_consecutive_pairs
from .rules import ALL_NUMBERS, MAX_NUMBER, PICK, validate_selection

Weights = dict[int, float]


# --------------------------------------------------------------------------
# スコアリングモデル
# --------------------------------------------------------------------------


def _softmax(scores: dict[int, float], temperature: float) -> Weights:
    """スコアを正規化して確率重みに変換する。"""
    if temperature <= 0:
        raise ValueError("temperature は正の値が必要です")
    values = list(scores.values())
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    sd = math.sqrt(variance) or 1.0
    exponentials = {n: math.exp((v - mean) / sd / temperature) for n, v in scores.items()}
    total = sum(exponentials.values())
    return {n: v / total for n, v in exponentials.items()}


def uniform_scores(stats: Stats, draws: Sequence[Draw]) -> dict[int, float]:
    """基準線: すべての数字を等価に扱う。"""
    return {n: 0.0 for n in ALL_NUMBERS}


def hot_scores(stats: Stats, draws: Sequence[Draw], half_life: float = 60.0) -> dict[int, float]:
    """よく出ている数字を高く評価する（新しい抽せんほど重みを大きく）。"""
    scores = {n: 0.0 for n in ALL_NUMBERS}
    total = len(draws)
    for index, draw in enumerate(draws):
        age = total - 1 - index
        weight = 0.5 ** (age / half_life)
        for n in draw.numbers:
            scores[n] += weight
    return scores


def cold_scores(stats: Stats, draws: Sequence[Draw], half_life: float = 60.0) -> dict[int, float]:
    """出現の少ない数字を高く評価する（hot の符号反転）。"""
    return {n: -v for n, v in hot_scores(stats, draws, half_life).items()}


def overdue_scores(stats: Stats, draws: Sequence[Draw]) -> dict[int, float]:
    """平均間隔に対してご無沙汰な数字を高く評価する。"""
    return {n: stats.overdue_ratio(n) for n in ALL_NUMBERS}


def markov_scores(
    stats: Stats, draws: Sequence[Draw], depth: int = 1, decay: float = 0.6
) -> dict[int, float]:
    """直近の抽せん数字との共起しやすさで評価する。"""
    scores = {n: 0.0 for n in ALL_NUMBERS}
    if not draws:
        return scores
    for back, draw in enumerate(reversed(draws[-depth:])):
        weight = decay**back
        for recent in draw.numbers:
            counts = stats.co_occurrence.get(recent) or {}
            total = sum(counts.values()) or 1
            for n, count in counts.items():
                scores[n] += weight * count / total
    return scores


MODELS: dict[str, Callable[[Stats, Sequence[Draw]], dict[int, float]]] = {
    "uniform": uniform_scores,
    "hot": hot_scores,
    "cold": cold_scores,
    "overdue": overdue_scores,
    "markov": markov_scores,
}

#: ensemble モデルの既定の配合。
ENSEMBLE_MIX: dict[str, float] = {
    "hot": 0.35,
    "overdue": 0.35,
    "markov": 0.30,
}


def score_numbers(
    draws: Sequence[Draw], model: str = "ensemble", stats: Stats | None = None
) -> dict[int, float]:
    """モデル名からスコア（z 標準化済み）を計算する。"""
    stats = stats or collect(draws)
    if model == "ensemble":
        blended = {n: 0.0 for n in ALL_NUMBERS}
        for name, share in ENSEMBLE_MIX.items():
            part = _standardize(MODELS[name](stats, draws))
            for n, value in part.items():
                blended[n] += share * value
        return blended
    if model not in MODELS:
        raise ValueError(f"未知のモデル: {model}（利用可能: {sorted(MODELS) + ['ensemble']}）")
    return _standardize(MODELS[model](stats, draws))


def _standardize(scores: dict[int, float]) -> dict[int, float]:
    values = list(scores.values())
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    sd = math.sqrt(variance) or 1.0
    return {n: (v - mean) / sd for n, v in scores.items()}


def weights_from_scores(scores: dict[int, float], temperature: float = 1.0) -> Weights:
    return _softmax(scores, temperature)


# --------------------------------------------------------------------------
# 出目の形に関する制約
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Constraints:
    """過去の当せん番号の「形」に候補を合わせるための制約。"""

    sum_range: tuple[int, int] = (1, PICK * MAX_NUMBER)
    odd_range: tuple[int, int] = (0, PICK)
    low_range: tuple[int, int] = (0, PICK)
    max_consecutive_pairs: int = PICK - 1

    @classmethod
    def from_history(cls, stats: Stats, quantile: float = 0.05) -> "Constraints":
        """過去分布の中央 ``1 - 2*quantile`` の範囲を制約にする。"""
        if not stats.sums:
            return cls()
        return cls(
            sum_range=(
                int(_quantile(stats.sums, quantile)),
                int(_quantile(stats.sums, 1 - quantile)),
            ),
            odd_range=(
                int(_quantile(stats.odd_counts, quantile)),
                int(_quantile(stats.odd_counts, 1 - quantile)),
            ),
            low_range=(
                int(_quantile(stats.low_counts, quantile)),
                int(_quantile(stats.low_counts, 1 - quantile)),
            ),
            max_consecutive_pairs=int(_quantile(stats.consecutive_counts, 1 - quantile)),
        )

    def accepts(self, ticket: Sequence[int]) -> bool:
        total = sum(ticket)
        if not (self.sum_range[0] <= total <= self.sum_range[1]):
            return False
        odd = sum(1 for n in ticket if n % 2 == 1)
        if not (self.odd_range[0] <= odd <= self.odd_range[1]):
            return False
        low = sum(1 for n in ticket if n <= MAX_NUMBER // 2)
        if not (self.low_range[0] <= low <= self.low_range[1]):
            return False
        if count_consecutive_pairs(ticket) > self.max_consecutive_pairs:
            return False
        return True

    def describe(self) -> str:
        return (
            f"合計 {self.sum_range[0]}〜{self.sum_range[1]} / "
            f"奇数 {self.odd_range[0]}〜{self.odd_range[1]}個 / "
            f"前半(1-{MAX_NUMBER // 2}) {self.low_range[0]}〜{self.low_range[1]}個 / "
            f"連番ペア {self.max_consecutive_pairs}組以下"
        )


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("空の系列です")
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[int(position)])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


# --------------------------------------------------------------------------
# 候補生成
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    """1 口分の予想。"""

    numbers: tuple[int, ...]
    score: float

    def __str__(self) -> str:
        return " ".join(f"{n:2d}" for n in self.numbers) + f"   (score {self.score:+.3f})"


def _sample_without_replacement(weights: Weights, k: int, rng: random.Random) -> tuple[int, ...]:
    """Gumbel-top-k トリックによる重み付き非復元抽出。"""
    keys = []
    for n, w in weights.items():
        if w <= 0:
            continue
        # -log(-log(U)) を加えた上位 k 個が重み付き非復元抽出と一致する
        gumbel = -math.log(-math.log(rng.random()))
        keys.append((math.log(w) + gumbel, n))
    keys.sort(reverse=True)
    return tuple(sorted(n for _, n in keys[:k]))


def _select_diverse(
    pool: dict[tuple[int, ...], float], count: int, diversity: float
) -> list[tuple[tuple[int, ...], float]]:
    """スコアと口同士の重なりの少なさを両立させて貪欲に選ぶ。

    ``diversity`` が 0 なら純粋なスコア順。大きくすると既に選んだ口と
    数字が重なる候補にペナルティがかかり、買い目が散らばる。
    """
    ranked = sorted(pool.items(), key=lambda item: (-item[1], item[0]))
    if diversity <= 0:
        return ranked[:count]

    selected: list[tuple[tuple[int, ...], float]] = []
    used: list[set[int]] = []
    remaining = list(ranked)
    while remaining and len(selected) < count:
        best_index = 0
        best_value = -math.inf
        for index, (ticket, score) in enumerate(remaining):
            ticket_set = set(ticket)
            overlap = max((len(ticket_set & other) for other in used), default=0)
            value = score - diversity * overlap
            if value > best_value:
                best_value, best_index = value, index
        ticket, score = remaining.pop(best_index)
        selected.append((ticket, score))
        used.append(set(ticket))
    return selected


def generate(
    draws: Sequence[Draw],
    count: int = 5,
    model: str = "ensemble",
    temperature: float = 1.0,
    constraints: Constraints | None = None,
    seed: int | None = None,
    max_attempts: int = 20000,
    stats: Stats | None = None,
    diversity: float = 0.15,
    include: Iterable[int] = (),
    exclude: Iterable[int] = (),
) -> list[Prediction]:
    """候補を ``count`` 口生成する。

    重み付き抽出で候補を作り、制約に合うものだけを残し、
    スコアと買い目の散らばりを考慮して ``count`` 口を選ぶ。

    ``include`` の数字は全口に必ず含め、``exclude`` の数字は候補から外す。
    """
    stats = stats or collect(draws)
    scores = score_numbers(draws, model=model, stats=stats)
    if constraints is None:
        constraints = Constraints.from_history(stats)

    forced = frozenset(include)
    banned = frozenset(exclude)
    if forced & banned:
        raise ValueError(f"include と exclude が重複しています: {sorted(forced & banned)}")
    for n in forced | banned:
        if n not in ALL_NUMBERS:
            raise ValueError(f"数字は 1〜{MAX_NUMBER} の範囲です: {n}")
    if len(forced) > PICK:
        raise ValueError(f"include は {PICK} 個以下にしてください")
    remaining = PICK - len(forced)
    pool_numbers = [n for n in ALL_NUMBERS if n not in forced and n not in banned]
    if len(pool_numbers) < remaining:
        raise ValueError("除外が多すぎて 7 個を選べません")

    all_weights = weights_from_scores(scores, temperature=temperature)
    pool_total = sum(all_weights[n] for n in pool_numbers) or 1.0
    weights = {n: all_weights[n] / pool_total for n in pool_numbers}
    rng = random.Random(seed)

    def draw_ticket() -> tuple[int, ...]:
        sampled = _sample_without_replacement(weights, remaining, rng) if remaining else ()
        return tuple(sorted(forced | set(sampled)))

    def value(ticket: tuple[int, ...]) -> float:
        return sum(scores[n] for n in ticket) / PICK

    found: dict[tuple[int, ...], float] = {}
    for _ in range(max_attempts):
        if len(found) >= count * 40:
            break
        ticket = draw_ticket()
        if len(ticket) != PICK or ticket in found:
            continue
        if not constraints.accepts(ticket):
            continue
        found[ticket] = value(ticket)

    if not found:
        # 制約が厳しすぎて 1 口も作れない場合は制約を外して生成する
        for _ in range(max_attempts):
            if len(found) >= count:
                break
            ticket = draw_ticket()
            if len(ticket) == PICK:
                found[ticket] = value(ticket)

    selected = _select_diverse(found, count, diversity)
    return [Prediction(numbers=t, score=s) for t, s in selected]


def top_numbers(draws: Sequence[Draw], model: str = "ensemble", k: int = PICK) -> list[int]:
    """スコア上位 ``k`` 個の数字（決定論的な選び方）。"""
    scores = score_numbers(draws, model=model)
    ordered = sorted(ALL_NUMBERS, key=lambda n: (-scores[n], n))
    return sorted(ordered[:k])


def manual_ticket(numbers: Iterable[int]) -> tuple[int, ...]:
    return validate_selection(numbers)
