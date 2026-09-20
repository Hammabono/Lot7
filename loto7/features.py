"""過去の当せん番号から統計量を取り出す。"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from .data import Draw
from .rules import ALL_NUMBERS, MAX_NUMBER, PICK


@dataclass
class Stats:
    """抽せん履歴から計算した各種統計量。"""

    draw_count: int
    frequency: dict[int, int]
    """本数字としての出現回数。"""

    bonus_frequency: dict[int, int]
    last_seen: dict[int, int | None]
    """直近で出てから何回前か（前回の抽せんで出たら 0）。出たことがなければ ``None``。"""

    gaps: dict[int, list[int]] = field(default_factory=dict)
    """出現間隔の履歴。"""

    co_occurrence: dict[int, Counter] = field(default_factory=dict)
    """同時に出た数字のカウント。"""

    sums: list[int] = field(default_factory=list)
    odd_counts: list[int] = field(default_factory=list)
    low_counts: list[int] = field(default_factory=list)
    """1〜18 に入った個数。"""

    consecutive_counts: list[int] = field(default_factory=list)
    """連番ペアの個数。"""

    def mean_gap(self, number: int) -> float:
        """平均出現間隔。未出現なら履歴長を返す。"""
        observed = self.gaps.get(number) or []
        if not observed:
            return float(self.draw_count)
        return sum(observed) / len(observed)

    def overdue_ratio(self, number: int) -> float:
        """``直近未出現回数 / 平均出現間隔``。1 を超えるほど「ご無沙汰」。"""
        last = self.last_seen.get(number)
        if last is None:
            return float(self.draw_count)
        mean = self.mean_gap(number)
        return (last + 1) / mean if mean > 0 else 0.0


def summarize(number: int, draws: Sequence[Draw]) -> str:
    stats = collect(draws)
    return (
        f"{number:2d}: 出現 {stats.frequency[number]}回 / "
        f"直近 {stats.last_seen[number]}回前 / "
        f"平均間隔 {stats.mean_gap(number):.1f}回"
    )


def count_consecutive_pairs(numbers: Sequence[int]) -> int:
    ordered = sorted(numbers)
    return sum(1 for a, b in zip(ordered, ordered[1:]) if b - a == 1)


def collect(draws: Sequence[Draw]) -> Stats:
    """抽せん履歴をまとめて統計量に変換する。"""
    frequency = {n: 0 for n in ALL_NUMBERS}
    bonus_frequency = {n: 0 for n in ALL_NUMBERS}
    last_index: dict[int, int] = {}
    gaps: dict[int, list[int]] = defaultdict(list)
    co_occurrence: dict[int, Counter] = {n: Counter() for n in ALL_NUMBERS}
    sums: list[int] = []
    odd_counts: list[int] = []
    low_counts: list[int] = []
    consecutive_counts: list[int] = []

    for index, draw in enumerate(draws):
        for n in draw.numbers:
            frequency[n] += 1
            if n in last_index:
                gaps[n].append(index - last_index[n])
            last_index[n] = index
        for n in draw.bonus:
            bonus_frequency[n] += 1
        for i, a in enumerate(draw.numbers):
            for b in draw.numbers[i + 1 :]:
                co_occurrence[a][b] += 1
                co_occurrence[b][a] += 1
        sums.append(sum(draw.numbers))
        odd_counts.append(sum(1 for n in draw.numbers if n % 2 == 1))
        low_counts.append(sum(1 for n in draw.numbers if n <= MAX_NUMBER // 2))
        consecutive_counts.append(count_consecutive_pairs(draw.numbers))

    total = len(draws)
    last_seen = {
        n: (total - 1 - last_index[n]) if n in last_index else None for n in ALL_NUMBERS
    }

    return Stats(
        draw_count=total,
        frequency=frequency,
        bonus_frequency=bonus_frequency,
        last_seen=last_seen,
        gaps=dict(gaps),
        co_occurrence=co_occurrence,
        sums=sums,
        odd_counts=odd_counts,
        low_counts=low_counts,
        consecutive_counts=consecutive_counts,
    )


def chi_square_uniformity(stats: Stats) -> tuple[float, int]:
    """出現回数が一様分布から外れているかのカイ二乗統計量と自由度。

    帰無仮説「どの数字も等確率」が正しければ、統計量は自由度 36 のカイ二乗分布に従う。
    """
    observed_total = sum(stats.frequency.values())
    expected = observed_total / MAX_NUMBER
    if expected == 0:
        return 0.0, MAX_NUMBER - 1
    statistic = sum(
        (count - expected) ** 2 / expected for count in stats.frequency.values()
    )
    return statistic, MAX_NUMBER - 1


def chi_square_p_value(statistic: float, dof: int) -> float:
    """カイ二乗分布の上側確率（自由度が偶数のときの閉形式 + 一般の連分数展開）。"""
    if statistic <= 0:
        return 1.0
    return _upper_incomplete_gamma_regularized(dof / 2.0, statistic / 2.0)


def _upper_incomplete_gamma_regularized(s: float, x: float) -> float:
    """Q(s, x) = Γ(s, x) / Γ(s) を級数展開 / 連分数展開で計算する。"""
    if x < s + 1.0:
        # 下側不完全ガンマ関数の級数展開
        term = 1.0 / s
        total = term
        n = s
        for _ in range(1000):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return 1.0 - total * math.exp(-x + s * math.log(x) - math.lgamma(s))

    # Lentz のアルゴリズムによる連分数展開
    tiny = 1e-300
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h * math.exp(-x + s * math.log(x) - math.lgamma(s))


def expected_matches_per_ticket() -> float:
    """完全にランダムに 7 個選んだときの本数字一致数の期待値。"""
    return PICK * PICK / MAX_NUMBER
