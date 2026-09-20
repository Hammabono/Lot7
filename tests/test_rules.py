import pytest

from loto7.rules import (
    TOTAL_COMBINATIONS,
    classify,
    tier_probabilities,
    validate_selection,
)

MAIN = (1, 2, 3, 4, 5, 6, 7)
BONUS = (8, 9)


def test_total_combinations():
    assert TOTAL_COMBINATIONS == 10_295_472


@pytest.mark.parametrize(
    "ticket, expected",
    [
        ((1, 2, 3, 4, 5, 6, 7), 1),           # 7個一致
        ((1, 2, 3, 4, 5, 6, 8), 2),           # 6個 + ボーナス
        ((1, 2, 3, 4, 5, 6, 10), 3),          # 6個のみ
        ((1, 2, 3, 4, 5, 10, 11), 4),         # 5個
        ((1, 2, 3, 4, 10, 11, 12), 5),        # 4個
        ((1, 2, 3, 8, 10, 11, 12), 6),        # 3個 + ボーナス
        ((1, 2, 3, 10, 11, 12, 13), None),    # 3個のみ = はずれ
        ((1, 2, 10, 11, 12, 13, 14), None),   # 2個 = はずれ
        ((8, 9, 10, 11, 12, 13, 14), None),   # ボーナスのみ = はずれ
    ],
)
def test_classify(ticket, expected):
    assert classify(ticket, MAIN, BONUS) == expected


def test_five_matches_ignores_bonus():
    """4等(5個一致)はボーナスの有無で等級が変わらない。"""
    assert classify((1, 2, 3, 4, 5, 8, 9), MAIN, BONUS) == 4
    assert classify((1, 2, 3, 4, 5, 20, 21), MAIN, BONUS) == 4


def test_tier_probabilities_match_official_odds():
    """公式に公表されている当せん確率と一致すること。"""
    probabilities = tier_probabilities()
    expected = {1: 10_295_472, 2: 735_391, 3: 52_528, 4: 1_127, 5: 72, 6: 42}
    for rank, denominator in expected.items():
        assert round(1 / probabilities[rank]) == denominator


def test_validate_selection():
    assert validate_selection([7, 1, 5, 3, 9, 11, 13]) == (1, 3, 5, 7, 9, 11, 13)
    with pytest.raises(ValueError, match="7 個"):
        validate_selection([1, 2, 3])
    with pytest.raises(ValueError, match="重複"):
        validate_selection([1, 1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError, match="範囲"):
        validate_selection([1, 2, 3, 4, 5, 6, 38])
