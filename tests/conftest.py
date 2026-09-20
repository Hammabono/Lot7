import datetime as dt
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loto7.data import Draw
from loto7.rules import ALL_NUMBERS, BONUS_COUNT, PICK


@pytest.fixture
def random_draws() -> list[Draw]:
    """一様乱数で作った 200 回分の履歴。"""
    rng = random.Random(1234)
    draws = []
    for index in range(200):
        picked = rng.sample(ALL_NUMBERS, PICK + BONUS_COUNT)
        draws.append(
            Draw(
                round=index + 1,
                date=dt.date(2013, 4, 5) + dt.timedelta(weeks=index),
                numbers=tuple(sorted(picked[:PICK])),
                bonus=tuple(sorted(picked[PICK:])),
            )
        )
    return draws
