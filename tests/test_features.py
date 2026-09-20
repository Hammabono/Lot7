from loto7.data import Draw
from loto7.features import (
    chi_square_p_value,
    chi_square_uniformity,
    collect,
    count_consecutive_pairs,
    expected_matches_per_ticket,
)


def make(round_no: int, numbers, bonus=(36, 37)) -> Draw:
    return Draw(round=round_no, date=None, numbers=tuple(numbers), bonus=tuple(bonus))


def test_count_consecutive_pairs():
    assert count_consecutive_pairs([1, 2, 3, 10, 20, 30, 31]) == 3
    assert count_consecutive_pairs([1, 3, 5, 7, 9, 11, 13]) == 0


def test_collect_frequency_and_last_seen():
    draws = [
        make(1, (1, 2, 3, 4, 5, 6, 7)),
        make(2, (1, 2, 3, 4, 5, 6, 8)),
        make(3, (1, 9, 10, 11, 12, 13, 14)),
    ]
    stats = collect(draws)
    assert stats.draw_count == 3
    assert stats.frequency[1] == 3
    assert stats.frequency[2] == 2
    assert stats.frequency[35] == 0
    # 1 は最終回に出ているので 0 回前、2 は 1 回前、35 は未出現
    assert stats.last_seen[1] == 0
    assert stats.last_seen[2] == 1
    assert stats.last_seen[35] is None
    assert stats.bonus_frequency[36] == 3


def test_collect_gaps_and_co_occurrence():
    draws = [
        make(1, (1, 2, 3, 4, 5, 6, 7)),
        make(2, (8, 9, 10, 11, 12, 13, 14)),
        make(3, (1, 2, 15, 16, 17, 18, 19)),
    ]
    stats = collect(draws)
    assert stats.gaps[1] == [2]          # 第1回 → 第3回 で間隔 2
    assert stats.mean_gap(1) == 2.0
    assert stats.co_occurrence[1][2] == 2  # 1 と 2 は 2 回同時に出ている
    assert stats.co_occurrence[1][8] == 0


def test_collect_shape_statistics():
    stats = collect([make(1, (1, 2, 3, 4, 5, 6, 7))])
    assert stats.sums == [28]
    assert stats.odd_counts == [4]       # 1,3,5,7
    assert stats.low_counts == [7]       # すべて 18 以下
    assert stats.consecutive_counts == [6]


def test_overdue_ratio_grows_with_absence():
    """毎回出ている数字より、間隔が空いた数字の overdue 比が大きくなる。"""
    # 1 は最初の 3 回連続で出た後 10 回不在（平均間隔 1 に対して 10 回空き）
    # 8 は全 13 回に出ている（平均間隔 1、直近も出ている）
    draws = [make(i, (1, 8, 20, 21, 22, 23, 24)) for i in range(1, 4)]
    draws += [make(i, (8, 9, 10, 11, 12, 13, 14)) for i in range(4, 14)]
    stats = collect(draws)
    assert stats.mean_gap(1) == 1.0
    assert stats.last_seen[1] == 10
    assert stats.overdue_ratio(1) == 11.0
    assert stats.overdue_ratio(8) == 1.0


def test_overdue_ratio_for_never_drawn_number():
    """一度も出ていない数字は履歴長そのものを返す。"""
    draws = [make(i, (1, 2, 3, 4, 5, 6, 7)) for i in range(1, 6)]
    stats = collect(draws)
    assert stats.last_seen[30] is None
    assert stats.overdue_ratio(30) == 5.0


def test_chi_square_on_random_history_is_not_significant(random_draws):
    """乱数由来の履歴では一様性は棄却されないはず。"""
    statistic, dof = chi_square_uniformity(collect(random_draws))
    assert dof == 36
    assert chi_square_p_value(statistic, dof) > 0.01


def test_chi_square_detects_a_rigged_history():
    """常に同じ数字が出る履歴なら一様性は強く棄却される。"""
    draws = [make(i, (1, 2, 3, 4, 5, 6, 7)) for i in range(1, 101)]
    statistic, dof = chi_square_uniformity(collect(draws))
    assert chi_square_p_value(statistic, dof) < 1e-10


def test_chi_square_p_value_known_values():
    # χ²=2, 自由度2 の上側確率は exp(-1)
    assert abs(chi_square_p_value(2.0, 2) - 0.36787944) < 1e-6
    # 自由度36 で χ²≈50.998 は p≈0.05
    assert abs(chi_square_p_value(50.998, 36) - 0.05) < 1e-3
    assert chi_square_p_value(0.0, 36) == 1.0


def test_expected_matches():
    assert abs(expected_matches_per_ticket() - 49 / 37) < 1e-12
