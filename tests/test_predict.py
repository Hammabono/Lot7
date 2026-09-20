import pytest

from loto7.data import Draw
from loto7.features import collect
from loto7.predict import (
    Constraints,
    ENSEMBLE_MIX,
    MODELS,
    generate,
    score_numbers,
    top_numbers,
    weights_from_scores,
)
from loto7.rules import ALL_NUMBERS, PICK


def make(round_no: int, numbers, bonus=(36, 37)) -> Draw:
    return Draw(round=round_no, date=None, numbers=tuple(numbers), bonus=tuple(bonus))


ALL_MODELS = [*sorted(MODELS), "ensemble"]


@pytest.mark.parametrize("model", ALL_MODELS)
def test_scores_cover_every_number(model, random_draws):
    scores = score_numbers(random_draws, model=model)
    assert set(scores) == set(ALL_NUMBERS)
    assert all(isinstance(v, float) for v in scores.values())


@pytest.mark.parametrize("model", ALL_MODELS)
def test_generate_produces_valid_tickets(model, random_draws):
    predictions = generate(random_draws, count=5, model=model, seed=7)
    assert len(predictions) == 5
    assert len({p.numbers for p in predictions}) == 5
    for prediction in predictions:
        assert len(prediction.numbers) == PICK
        assert len(set(prediction.numbers)) == PICK
        assert list(prediction.numbers) == sorted(prediction.numbers)
        assert all(n in ALL_NUMBERS for n in prediction.numbers)


def test_generate_is_reproducible(random_draws):
    a = generate(random_draws, count=5, seed=99)
    b = generate(random_draws, count=5, seed=99)
    assert [p.numbers for p in a] == [p.numbers for p in b]


def test_generate_respects_constraints(random_draws):
    constraints = Constraints(sum_range=(100, 120), odd_range=(3, 3), low_range=(0, 7))
    for prediction in generate(random_draws, count=10, constraints=constraints, seed=5):
        assert 100 <= sum(prediction.numbers) <= 120
        assert sum(1 for n in prediction.numbers if n % 2 == 1) == 3


def test_uniform_model_gives_flat_weights(random_draws):
    weights = weights_from_scores(score_numbers(random_draws, model="uniform"))
    assert all(abs(w - 1 / len(ALL_NUMBERS)) < 1e-12 for w in weights.values())


def test_hot_model_prefers_frequent_numbers():
    """1〜7 だけが出続けた履歴では hot がその 7 個を選ぶ。"""
    draws = [make(i, (1, 2, 3, 4, 5, 6, 7)) for i in range(1, 51)]
    assert top_numbers(draws, model="hot") == [1, 2, 3, 4, 5, 6, 7]


def test_cold_model_avoids_frequent_numbers():
    draws = [make(i, (1, 2, 3, 4, 5, 6, 7)) for i in range(1, 51)]
    assert not set(top_numbers(draws, model="cold")) & {1, 2, 3, 4, 5, 6, 7}


def test_overdue_model_prefers_absent_numbers():
    draws = [make(i, (1, 2, 3, 4, 5, 6, 7)) for i in range(1, 51)]
    # 8 以降は一度も出ていないので overdue スコアが最大になる
    assert not set(top_numbers(draws, model="overdue")) & {1, 2, 3, 4, 5, 6, 7}


def test_markov_model_follows_co_occurrence():
    """常に 1 とセットで出る数字が直近抽せん後に高評価される。"""
    draws = [make(i, (1, 2, 3, 4, 5, 6, 7)) for i in range(1, 41)]
    draws += [make(41, (1, 2, 3, 4, 5, 6, 7))]
    scores = score_numbers(draws, model="markov")
    assert scores[2] > scores[30]


def test_ensemble_mix_is_normalized():
    assert abs(sum(ENSEMBLE_MIX.values()) - 1.0) < 1e-12
    assert set(ENSEMBLE_MIX) <= set(MODELS)


def test_unknown_model_raises(random_draws):
    with pytest.raises(ValueError, match="未知のモデル"):
        score_numbers(random_draws, model="crystal_ball")


def test_constraints_from_history_covers_observed_draws(random_draws):
    """履歴から作った制約は、その履歴の大半の回を通す。"""
    stats = collect(random_draws)
    constraints = Constraints.from_history(stats, quantile=0.05)
    accepted = sum(1 for d in random_draws if constraints.accepts(d.numbers))
    assert accepted / len(random_draws) > 0.7


def test_diversity_reduces_overlap(random_draws):
    def max_overlap(predictions):
        sets = [set(p.numbers) for p in predictions]
        return max(
            len(a & b) for i, a in enumerate(sets) for b in sets[i + 1 :]
        )

    greedy = generate(random_draws, count=5, seed=3, diversity=0.0)
    spread = generate(random_draws, count=5, seed=3, diversity=1.0)
    assert max_overlap(spread) <= max_overlap(greedy)


def test_temperature_increases_spread(random_draws):
    """温度を上げるほど候補に登場する数字の種類が増える。"""
    cool = {n for p in generate(random_draws, count=20, seed=11, temperature=0.3) for n in p.numbers}
    warm = {n for p in generate(random_draws, count=20, seed=11, temperature=5.0) for n in p.numbers}
    assert len(warm) > len(cool)


def test_generate_forces_included_numbers(random_draws):
    predictions = generate(random_draws, count=5, seed=4, include=(7, 13, 21))
    assert len(predictions) == 5
    for prediction in predictions:
        assert {7, 13, 21} <= set(prediction.numbers)


def test_generate_never_uses_excluded_numbers(random_draws):
    banned = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    predictions = generate(random_draws, count=10, seed=4, exclude=banned)
    assert predictions
    for prediction in predictions:
        assert not banned & set(prediction.numbers)


def test_generate_with_seven_included_numbers(random_draws):
    fixed = (1, 5, 9, 14, 22, 30, 33)
    predictions = generate(random_draws, count=3, seed=4, include=fixed)
    assert [p.numbers for p in predictions] == [fixed]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"include": (5,), "exclude": (5,)}, "重複"),
        ({"include": (40,)}, "範囲"),
        ({"include": tuple(range(1, 10))}, "7 個以下"),
        ({"exclude": tuple(range(1, 35))}, "除外が多すぎ"),
    ],
)
def test_generate_rejects_impossible_filters(random_draws, kwargs, message):
    with pytest.raises(ValueError, match=message):
        generate(random_draws, count=3, **kwargs)
