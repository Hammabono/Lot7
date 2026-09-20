import pytest

from loto7.backtest import TICKET_PRICE, format_comparison, walk_forward
from loto7.features import expected_matches_per_ticket
from loto7.rules import PICK


def test_walk_forward_basic_accounting(random_draws):
    result = walk_forward(random_draws, model="ensemble", tickets_per_draw=3, warmup=50)
    assert result.evaluated_draws == len(random_draws) - 50
    assert result.total_tickets == result.evaluated_draws * 3
    assert sum(result.match_histogram.values()) == result.total_tickets
    assert result.spend == result.total_tickets * TICKET_PRICE
    assert all(0 <= m <= PICK for m in result.match_histogram)


def test_walk_forward_requires_enough_history(random_draws):
    with pytest.raises(ValueError, match="warmup"):
        walk_forward(random_draws, warmup=len(random_draws))
    with pytest.raises(ValueError, match="warmup は 1 以上"):
        walk_forward(random_draws, warmup=0)


@pytest.mark.parametrize("model", ["uniform", "hot", "cold", "overdue", "markov", "ensemble"])
def test_no_model_beats_chance_on_random_history(model, random_draws):
    """乱数履歴に対して、どのモデルもランダムと有意差を出さないこと。

    これが本ツールの主張の中核。|z| が 3 を超えるようなら
    バックテストかモデルの実装を疑うべきである。
    """
    result = walk_forward(random_draws, model=model, tickets_per_draw=10, warmup=50)
    assert abs(result.z_score) < 3.0
    assert abs(result.mean_matches - expected_matches_per_ticket()) < 0.25


def test_walk_forward_uses_only_past_data(monkeypatch, random_draws):
    """予想の生成に未来の抽せんが渡っていないことを確認する（先読みバイアス検出）。"""
    seen_lengths: list[int] = []
    import loto7.backtest as backtest_module

    original = backtest_module.generate

    def spy(history, *args, **kwargs):
        seen_lengths.append(len(history))
        return original(history, *args, **kwargs)

    monkeypatch.setattr(backtest_module, "generate", spy)
    walk_forward(random_draws, tickets_per_draw=1, warmup=50)
    # 各ステップの学習データ長は 50, 51, 52, ... と増えていく
    assert seen_lengths == list(range(50, len(random_draws)))


def test_window_limits_training_history(monkeypatch, random_draws):
    seen_lengths: list[int] = []
    import loto7.backtest as backtest_module

    original = backtest_module.generate

    def spy(history, *args, **kwargs):
        seen_lengths.append(len(history))
        return original(history, *args, **kwargs)

    monkeypatch.setattr(backtest_module, "generate", spy)
    walk_forward(random_draws, tickets_per_draw=1, warmup=60, window=30)
    assert set(seen_lengths) == {30}


def test_walk_forward_is_reproducible(random_draws):
    a = walk_forward(random_draws, tickets_per_draw=3, warmup=150, seed=5)
    b = walk_forward(random_draws, tickets_per_draw=3, warmup=150, seed=5)
    assert a.match_histogram == b.match_histogram


def test_format_comparison_lists_every_model(random_draws):
    results = [
        walk_forward(random_draws, model=m, tickets_per_draw=2, warmup=150)
        for m in ("uniform", "ensemble")
    ]
    text = format_comparison(results)
    assert "uniform" in text and "ensemble" in text and "理論値" in text
