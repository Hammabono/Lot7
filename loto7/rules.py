"""ロト7 のゲームルールと当せん等級の定義。

ロト7:
    * 1〜37 の 37 個の数字から 7 個を選ぶ
    * 本数字 7 個 + ボーナス数字 2 個が抽せんされる
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Iterable, Sequence

MIN_NUMBER = 1
MAX_NUMBER = 37
PICK = 7
BONUS_COUNT = 2

ALL_NUMBERS: tuple[int, ...] = tuple(range(MIN_NUMBER, MAX_NUMBER + 1))

#: 全組合せ数 C(37, 7)
TOTAL_COMBINATIONS = comb(MAX_NUMBER, PICK)


@dataclass(frozen=True)
class Tier:
    """当せん等級。"""

    rank: int
    label: str
    description: str


TIERS: tuple[Tier, ...] = (
    Tier(1, "1等", "本数字 7 個すべて一致"),
    Tier(2, "2等", "本数字 6 個 + ボーナス数字 1 個以上一致"),
    Tier(3, "3等", "本数字 6 個一致"),
    Tier(4, "4等", "本数字 5 個一致"),
    Tier(5, "5等", "本数字 4 個一致"),
    Tier(6, "6等", "本数字 3 個 + ボーナス数字 1 個以上一致"),
)

TIER_BY_RANK = {tier.rank: tier for tier in TIERS}


def validate_selection(numbers: Iterable[int]) -> tuple[int, ...]:
    """7 個の数字が規則を満たすか検証し、昇順タプルで返す。"""
    picked = tuple(sorted(numbers))
    if len(picked) != PICK:
        raise ValueError(f"選択数字は {PICK} 個である必要があります: {picked}")
    if len(set(picked)) != PICK:
        raise ValueError(f"数字が重複しています: {picked}")
    for n in picked:
        if not (MIN_NUMBER <= n <= MAX_NUMBER):
            raise ValueError(f"数字は {MIN_NUMBER}〜{MAX_NUMBER} の範囲です: {n}")
    return picked


def classify(
    ticket: Sequence[int],
    main_numbers: Sequence[int],
    bonus_numbers: Sequence[int] = (),
) -> int | None:
    """購入口 ``ticket`` の当せん等級を返す。はずれなら ``None``。"""
    ticket_set = set(ticket)
    matched = len(ticket_set & set(main_numbers))
    bonus_matched = len(ticket_set & set(bonus_numbers))

    if matched == 7:
        return 1
    if matched == 6:
        return 2 if bonus_matched >= 1 else 3
    if matched == 5:
        return 4
    if matched == 4:
        return 5
    if matched == 3 and bonus_matched >= 1:
        return 6
    return None


def tier_probabilities() -> dict[int, float]:
    """各等級の理論当せん確率（1 口あたり）。"""
    total = TOTAL_COMBINATIONS
    # 本数字 k 個・ボーナス b 個一致となる口数を数え上げる。
    # 母集団: 本数字 7 個 / ボーナス 2 個 / それ以外 28 個
    others = MAX_NUMBER - PICK - BONUS_COUNT

    def ways(main_hit: int, bonus_hit: int) -> int:
        rest = PICK - main_hit - bonus_hit
        if rest < 0 or rest > others:
            return 0
        return comb(PICK, main_hit) * comb(BONUS_COUNT, bonus_hit) * comb(others, rest)

    counts = {rank: 0 for rank in TIER_BY_RANK}
    for main_hit in range(PICK + 1):
        for bonus_hit in range(BONUS_COUNT + 1):
            n = ways(main_hit, bonus_hit)
            if not n:
                continue
            rank = classify(
                # ダミーの一致状況を等級判定ロジックに合わせて評価する
                ticket=list(range(1, main_hit + 1))
                + list(range(100, 100 + bonus_hit))
                + list(range(200, 200 + PICK - main_hit - bonus_hit)),
                main_numbers=list(range(1, PICK + 1)),
                bonus_numbers=list(range(100, 100 + BONUS_COUNT)),
            )
            if rank is not None:
                counts[rank] += n
    return {rank: counts[rank] / total for rank in sorted(counts)}
