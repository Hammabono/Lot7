"""予想モデルを過去データで検証する（ウォークフォワード検証）。

未来のデータを使わずに、各回の時点で「その時点までの履歴」だけからモデルを作り、
実際の当せん番号と突き合わせる。これがモデルに意味があるかを判断する唯一の方法である。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .data import Draw
from .features import collect, expected_matches_per_ticket
from .predict import Constraints, generate
from .rules import MAX_NUMBER, PICK, TIER_BY_RANK, classify
from .textutil import display_width, pad

#: 当せん金の目安（円）。ロト7 はパリミュチュエル方式のため回ごとに変動する概算値。
TYPICAL_PRIZES: dict[int, int] = {
    1: 600_000_000,
    2: 7_300_000,
    3: 730_000,
    4: 12_000,
    5: 1_400,
    6: 1_000,
}

TICKET_PRICE = 300


@dataclass
class BacktestResult:
    model: str
    tickets_per_draw: int
    evaluated_draws: int
    total_tickets: int
    match_histogram: Counter = field(default_factory=Counter)
    tier_histogram: Counter = field(default_factory=Counter)

    @property
    def total_matches(self) -> int:
        return sum(matches * n for matches, n in self.match_histogram.items())

    @property
    def mean_matches(self) -> float:
        return self.total_matches / self.total_tickets if self.total_tickets else 0.0

    @property
    def expected_mean_matches(self) -> float:
        return expected_matches_per_ticket()

    @property
    def z_score(self) -> float:
        """「ランダムと同じ」という帰無仮説に対する z 値。

        1 口あたりの一致数は超幾何分布 Hypergeometric(37, 7, 7) に従う。
        口同士は独立ではないので厳密な検定ではないが、桁感をつかむ目安になる。
        """
        n = self.total_tickets
        if n == 0:
            return 0.0
        mean = self.expected_mean_matches
        # 超幾何分布の分散: n*K/N*(1-K/N)*(N-n)/(N-1)
        big_n, k, draws = MAX_NUMBER, PICK, PICK
        variance = draws * (k / big_n) * (1 - k / big_n) * (big_n - draws) / (big_n - 1)
        se = math.sqrt(variance / n)
        return (self.mean_matches - mean) / se if se else 0.0

    @property
    def payout(self) -> int:
        return sum(TYPICAL_PRIZES[rank] * n for rank, n in self.tier_histogram.items())

    @property
    def spend(self) -> int:
        return self.total_tickets * TICKET_PRICE

    def format_report(self) -> str:
        lines = [
            f"モデル           : {self.model}",
            f"検証回数         : {self.evaluated_draws} 回",
            f"購入口数         : {self.total_tickets} 口 ({self.tickets_per_draw} 口/回)",
            "",
            "本数字の一致数の分布:",
        ]
        for matches in range(PICK + 1):
            count = self.match_histogram.get(matches, 0)
            share = count / self.total_tickets * 100 if self.total_tickets else 0.0
            bar = "#" * int(share / 2)
            lines.append(f"  {matches} 個一致: {count:6d} 口 ({share:5.2f}%) {bar}")
        lines += [
            "",
            f"平均一致数       : {self.mean_matches:.4f} 個",
            f"ランダムの期待値 : {self.expected_mean_matches:.4f} 個",
            f"z 値             : {self.z_score:+.2f}"
            "  （|z| < 2 なら偶然の範囲内 = ランダムと区別できない）",
            "",
            "当せん等級:",
        ]
        if self.tier_histogram:
            for rank in sorted(self.tier_histogram):
                tier = TIER_BY_RANK[rank]
                lines.append(
                    f"  {tier.label}: {self.tier_histogram[rank]:5d} 回 ({tier.description})"
                )
        else:
            lines.append("  なし")
        ratio = self.payout / self.spend * 100 if self.spend else 0.0
        lines += [
            "",
            f"購入額（概算）   : {self.spend:,} 円",
            f"当せん金（概算） : {self.payout:,} 円  … 回収率 {ratio:.1f}%",
        ]
        return "\n".join(lines)


def walk_forward(
    draws: Sequence[Draw],
    model: str = "ensemble",
    tickets_per_draw: int = 5,
    warmup: int = 100,
    window: int | None = None,
    temperature: float = 1.0,
    seed: int = 0,
    use_constraints: bool = True,
) -> BacktestResult:
    """各回について「それ以前のデータだけ」で予想し、実際の結果と突き合わせる。"""
    if warmup < 1:
        raise ValueError("warmup は 1 以上が必要です")
    if len(draws) <= warmup:
        raise ValueError(
            f"検証には warmup ({warmup}) より多い抽せん回数が必要です（現在 {len(draws)} 回）"
        )

    result = BacktestResult(
        model=model,
        tickets_per_draw=tickets_per_draw,
        evaluated_draws=0,
        total_tickets=0,
    )

    for index in range(warmup, len(draws)):
        history = draws[:index]
        if window:
            history = history[-window:]
        stats = collect(history)
        constraints = Constraints.from_history(stats) if use_constraints else Constraints()
        predictions = generate(
            history,
            count=tickets_per_draw,
            model=model,
            temperature=temperature,
            constraints=constraints,
            seed=seed + index,
            stats=stats,
        )
        actual = draws[index]
        for prediction in predictions:
            matched = len(set(prediction.numbers) & set(actual.numbers))
            result.match_histogram[matched] += 1
            rank = classify(prediction.numbers, actual.numbers, actual.bonus)
            if rank is not None:
                result.tier_histogram[rank] += 1
            result.total_tickets += 1
        result.evaluated_draws += 1

    return result


def compare_models(
    draws: Sequence[Draw],
    models: Sequence[str] = ("uniform", "hot", "cold", "overdue", "markov", "ensemble"),
    **kwargs,
) -> list[BacktestResult]:
    return [walk_forward(draws, model=model, **kwargs) for model in models]


def format_comparison(results: Sequence[BacktestResult]) -> str:
    header = (
        pad("モデル", 12)
        + pad("平均一致数", 12, ">")
        + pad("z 値", 9, ">")
        + pad("4個以上", 10, ">")
        + pad("回収率", 10, ">")
    )
    lines = [header, "-" * display_width(header)]
    for r in results:
        four_plus = sum(n for m, n in r.match_histogram.items() if m >= 4)
        ratio = r.payout / r.spend * 100 if r.spend else 0.0
        lines.append(
            pad(r.model, 12)
            + pad(f"{r.mean_matches:.4f}", 12, ">")
            + pad(f"{r.z_score:+.2f}", 9, ">")
            + pad(str(four_plus), 10, ">")
            + pad(f"{ratio:.1f}%", 10, ">")
        )
    lines.append("-" * display_width(header))
    lines.append(
        pad("理論値", 12)
        + pad(f"{expected_matches_per_ticket():.4f}", 12, ">")
        + pad("+0.00", 9, ">")
    )
    return "\n".join(lines)
