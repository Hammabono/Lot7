#!/usr/bin/env python3
"""動作確認用のシミュレーションデータを生成する。

ここで作られるのは**実際の当せん番号ではない**。一様乱数で作った架空の履歴であり、
プログラムがデータなしでも動くようにするためだけのもの。
実データは公式サイトから取得して `loto7 import` で差し替えること。
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loto7.data import Draw, save_csv
from loto7.rules import ALL_NUMBERS, BONUS_COUNT, PICK

#: ロト7 第1回の抽せん日（実際の日付）。
FIRST_DRAW_DATE = dt.date(2013, 4, 5)

NOTE = """\
!!! シミュレーションデータ / SIMULATED DATA !!!
これは一様乱数で生成した架空の抽せん結果であり、実際のロト7の当せん番号ではない。
動作確認・バックテストの基準線としてのみ使用すること。
実データはみずほ銀行の公式サイトから取得し、`python -m loto7 import <csv>` で置き換える。\
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=600, help="生成する回数")
    parser.add_argument("--seed", type=int, default=20130405, help="乱数シード")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "sample_simulated.csv",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    draws = []
    for index in range(args.rounds):
        picked = rng.sample(ALL_NUMBERS, PICK + BONUS_COUNT)
        draws.append(
            Draw(
                round=index + 1,
                date=FIRST_DRAW_DATE + dt.timedelta(weeks=index),
                numbers=tuple(sorted(picked[:PICK])),
                bonus=tuple(sorted(picked[PICK:])),
            )
        )

    save_csv(draws, args.out, header_note=NOTE)
    print(f"{len(draws)} 回分のシミュレーションデータを書き出しました: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
