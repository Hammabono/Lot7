"""コマンドラインインターフェース。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .backtest import TICKET_PRICE, TYPICAL_PRIZES, compare_models, format_comparison, walk_forward
from .data import Draw, load_csv, save_csv
from .features import chi_square_p_value, chi_square_uniformity, collect
from .predict import Constraints, MODELS, generate, score_numbers
from .rules import (
    ALL_NUMBERS,
    MAX_NUMBER,
    PICK,
    TIERS,
    TOTAL_COMBINATIONS,
    classify,
    tier_probabilities,
    validate_selection,
)
from .textutil import display_width, pad

DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "sample_simulated.csv"

DISCLAIMER = """\
────────────────────────────────────────────────────────────
 ロト7 の抽せんは毎回独立した試行です。過去の当せん番号には次回を
 予測する情報は含まれておらず、どの数字を選んでも 1 口の当せん確率は
 1/10,295,472（1等）で変わりません。
 本ツールは過去データの統計的な性質を可視化し、その傾向に沿った
 買い目を組み立てるためのものです。的中を保証するものではありません。
 検証結果は `loto7 backtest` / `loto7 compare` で確認してください。
────────────────────────────────────────────────────────────"""


def _load(path: Path) -> list[Draw]:
    if not path.exists():
        raise SystemExit(
            f"データファイルが見つかりません: {path}\n"
            "`python tools/make_sample.py` でサンプルを生成するか、"
            "`python -m loto7 import <csv>` で実データを取り込んでください。"
        )
    draws = load_csv(path)
    if not draws:
        raise SystemExit(f"{path} に抽せん結果がありません。")
    return draws


def _parse_numbers(text: str | None) -> set[int]:
    if not text:
        return set()
    values = {int(part) for part in text.replace(",", " ").split()}
    for value in values:
        if value not in ALL_NUMBERS:
            raise SystemExit(f"数字は 1〜{MAX_NUMBER} の範囲です: {value}")
    return values


# --------------------------------------------------------------------------
# サブコマンド
# --------------------------------------------------------------------------


def cmd_predict(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    include = _parse_numbers(args.include)
    exclude = _parse_numbers(args.exclude)

    stats = collect(draws)
    constraints = (
        Constraints.from_history(stats, quantile=args.quantile)
        if not args.no_constraints
        else Constraints()
    )

    try:
        predictions = generate(
            draws,
            count=args.count,
            model=args.model,
            temperature=args.temperature,
            constraints=constraints,
            seed=args.seed,
            stats=stats,
            diversity=args.diversity,
            include=include,
            exclude=exclude,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(DISCLAIMER)
    print()
    print(f"データ  : {args.data}（第{draws[0].round}回〜第{draws[-1].round}回 / {len(draws)}回分）")
    print(f"モデル  : {args.model}  温度: {args.temperature}")
    print(f"制約    : {constraints.describe() if not args.no_constraints else 'なし'}")
    print(f"直近    : {draws[-1]}")
    print()
    if not predictions:
        print("条件に合う候補が見つかりませんでした。--no-constraints や条件の緩和を試してください。")
        return 1
    print(f"次回（第{draws[-1].round + 1}回）の買い目候補:")
    for i, prediction in enumerate(predictions, start=1):
        numbers = " ".join(f"{n:2d}" for n in prediction.numbers)
        total = sum(prediction.numbers)
        odd = sum(1 for n in prediction.numbers if n % 2 == 1)
        print(
            f"  候補{i}: {numbers}   [合計 {total:3d} / 奇数 {odd} / スコア {prediction.score:+.3f}]"
        )
    print()
    print(f"購入額の目安: {len(predictions)} 口 × {TICKET_PRICE} 円 = {len(predictions) * TICKET_PRICE:,} 円")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    stats = collect(draws)
    scores = score_numbers(draws, model=args.model, stats=stats)

    print(f"データ: {args.data}（第{draws[0].round}回〜第{draws[-1].round}回 / {len(draws)}回分）")
    print()
    expected = len(draws) * PICK / MAX_NUMBER
    header = (
        pad("数字", 5, ">")
        + pad("出現", 7, ">")
        + pad("期待", 8, ">")
        + pad("差", 8, ">")
        + pad("直近", 9, ">")
        + pad("平均間隔", 11, ">")
        + pad("スコア", 10, ">")
    )
    print(header)
    print("-" * display_width(header))
    order = sorted(ALL_NUMBERS, key=lambda n: (-scores[n], n)) if args.sort == "score" else ALL_NUMBERS
    for n in order:
        last = stats.last_seen[n]
        last_text = f"{last}回前" if last is not None else "未出現"
        print(
            pad(str(n), 5, ">")
            + pad(str(stats.frequency[n]), 7, ">")
            + pad(f"{expected:.1f}", 8, ">")
            + pad(f"{stats.frequency[n] - expected:+.1f}", 8, ">")
            + pad(last_text, 9, ">")
            + pad(f"{stats.mean_gap(n):.1f}", 11, ">")
            + pad(f"{scores[n]:+.2f}", 10, ">")
        )
    print()

    statistic, dof = chi_square_uniformity(stats)
    p_value = chi_square_p_value(statistic, dof)
    print("出現回数の一様性検定（帰無仮説: すべての数字が等確率）")
    print(f"  カイ二乗統計量 = {statistic:.2f} (自由度 {dof})")
    print(f"  p 値           = {p_value:.4f}")
    if p_value >= 0.05:
        print("  → 有意水準 5% で帰無仮説は棄却されない。")
        print("     出現回数のばらつきは偶然の範囲内で、「よく出る数字」は統計的に確認できない。")
    else:
        print("  → 有意水準 5% で帰無仮説が棄却された。")
        print("     ただし多重比較や標本数の影響もあるため、偏りの実在は慎重に判断すること。")
    print()

    def dist(name: str, values: list[int]) -> None:
        if not values:
            return
        mean = sum(values) / len(values)
        print(f"  {pad(name, 16)}: 平均 {mean:6.2f} / 最小 {min(values):3d} / 最大 {max(values):3d}")

    print("出目の形:")
    dist("本数字の合計", stats.sums)
    dist("奇数の個数", stats.odd_counts)
    dist(f"1-{MAX_NUMBER // 2} の個数", stats.low_counts)
    dist("連番ペア数", stats.consecutive_counts)
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    result = walk_forward(
        draws,
        model=args.model,
        tickets_per_draw=args.tickets,
        warmup=args.warmup,
        window=args.window,
        temperature=args.temperature,
        seed=args.seed,
        use_constraints=not args.no_constraints,
    )
    print(f"データ: {args.data}（{len(draws)}回分 / 学習開始 {args.warmup}回目以降を検証）")
    print()
    print(result.format_report())
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    models = args.models or (*sorted(MODELS), "ensemble")
    results = compare_models(
        draws,
        models=models,
        tickets_per_draw=args.tickets,
        warmup=args.warmup,
        window=args.window,
        temperature=args.temperature,
        seed=args.seed,
        use_constraints=not args.no_constraints,
    )
    print(f"データ: {args.data}（{len(draws)}回分）")
    print(f"各モデル {args.tickets} 口/回 × {results[0].evaluated_draws} 回 を検証")
    print()
    print(format_comparison(results))
    print()
    print("どのモデルも理論値の近傍に収まり、|z| が 2 を超えなければ")
    print("「ランダムに選ぶのと区別がつかない」と解釈するのが妥当です。")
    return 0


def cmd_odds(args: argparse.Namespace) -> int:
    probabilities = tier_probabilities()
    print(f"全組合せ数: C({MAX_NUMBER}, {PICK}) = {TOTAL_COMBINATIONS:,} 通り")
    print()
    header = (
        pad("等級", 6)
        + pad("条件", 40)
        + pad("確率", 16, ">")
        + pad("当せん金(目安)", 18, ">")
    )
    print(header)
    print("-" * display_width(header))
    expected_value = 0.0
    for tier in TIERS:
        p = probabilities[tier.rank]
        prize = TYPICAL_PRIZES[tier.rank]
        expected_value += p * prize
        print(
            pad(tier.label, 6)
            + pad(tier.description, 40)
            + pad("1/" + format(1 / p, ",.0f"), 16, ">")
            + pad(f"{prize:,}円", 18, ">")
        )
    print("-" * display_width(header))
    print(f"1 口 {TICKET_PRICE} 円あたりの期待値（概算）: {expected_value:,.1f} 円 "
          f"= 回収率 {expected_value / TICKET_PRICE * 100:.1f}%")
    print()
    print("※ 当せん金はパリミュチュエル方式のため回ごとに変動します（上表は目安値）。")
    print("※ 期待値は購入額を下回ります。これはどの買い方を選んでも変わりません。")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    draws = _load(args.data)
    target = next((d for d in draws if d.round == args.round), None) if args.round else draws[-1]
    if target is None:
        raise SystemExit(f"第{args.round}回のデータがありません。")
    ticket = validate_selection(_parse_numbers(args.numbers))
    rank = classify(ticket, target.numbers, target.bonus)
    matched = sorted(set(ticket) & set(target.numbers))
    bonus_matched = sorted(set(ticket) & set(target.bonus))

    print(f"抽せん: {target}")
    print(f"購入口: {' '.join(f'{n:2d}' for n in ticket)}")
    print(f"一致  : 本数字 {len(matched)} 個 {matched} / ボーナス {len(bonus_matched)} 個 {bonus_matched}")
    if rank is None:
        print("結果  : はずれ")
    else:
        tier = next(t for t in TIERS if t.rank == rank)
        print(f"結果  : {tier.label}（{tier.description}）")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from .fetch import FetchError, fetch, merge

    try:
        incoming = fetch(timeout=args.timeout)
    except FetchError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    existing = load_csv(args.data) if args.data.exists() else []
    merged = merge(existing, incoming)
    save_csv(merged, args.data)
    print(f"{len(incoming)} 回分を取得し、{args.data} を更新しました（合計 {len(merged)} 回分）。")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    from .fetch import merge

    incoming = load_csv(args.source)
    existing = load_csv(args.data) if args.data.exists() else []
    merged = merge(existing, incoming)
    save_csv(merged, args.data)
    print(
        f"{args.source} から {len(incoming)} 回分を取り込み、"
        f"{args.data} を更新しました（合計 {len(merged)} 回分）。"
    )
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loto7",
        description="ロト7 の過去当せん番号を分析し、買い目候補を生成・検証するツール",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA, help="抽せん結果 CSV（既定: %(default)s）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_model_options(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--model",
            default="ensemble",
            choices=[*sorted(MODELS), "ensemble"],
            help="スコアリングモデル（既定: %(default)s）",
        )
        p.add_argument("--temperature", type=float, default=1.0, help="大きいほどランダムに近づく")
        p.add_argument("--seed", type=int, default=None, help="乱数シード（再現用）")
        p.add_argument("--no-constraints", action="store_true", help="出目の形の制約を使わない")

    p_predict = sub.add_parser("predict", help="次回の買い目候補を生成する")
    add_model_options(p_predict)
    p_predict.add_argument("-n", "--count", type=int, default=5, help="生成する口数")
    p_predict.add_argument("--include", help="必ず含める数字（例: \"7 13\"）")
    p_predict.add_argument("--exclude", help="除外する数字")
    p_predict.add_argument("--quantile", type=float, default=0.05, help="制約の裾切り分位点")
    p_predict.add_argument(
        "--diversity",
        type=float,
        default=0.15,
        help="口同士の数字の重なりを減らす強さ（0 で純粋なスコア順）",
    )
    p_predict.set_defaults(func=cmd_predict)

    p_stats = sub.add_parser("stats", help="出現回数・間隔・一様性検定を表示する")
    p_stats.add_argument(
        "--model", default="ensemble", choices=[*sorted(MODELS), "ensemble"], help="表示するスコア"
    )
    p_stats.add_argument("--sort", choices=["number", "score"], default="number")
    p_stats.set_defaults(func=cmd_stats)

    def add_backtest_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--tickets", type=int, default=5, help="1 回あたりの購入口数")
        p.add_argument("--warmup", type=int, default=100, help="学習に使う最低回数")
        p.add_argument("--window", type=int, default=None, help="直近 N 回だけで学習する")
        p.add_argument("--temperature", type=float, default=1.0)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--no-constraints", action="store_true")

    p_backtest = sub.add_parser("backtest", help="ウォークフォワード検証を実行する")
    p_backtest.add_argument(
        "--model", default="ensemble", choices=[*sorted(MODELS), "ensemble"]
    )
    add_backtest_options(p_backtest)
    p_backtest.set_defaults(func=cmd_backtest)

    p_compare = sub.add_parser("compare", help="複数モデルを一括で検証して比較する")
    p_compare.add_argument("--models", nargs="+", choices=[*sorted(MODELS), "ensemble"])
    add_backtest_options(p_compare)
    p_compare.set_defaults(func=cmd_compare)

    p_odds = sub.add_parser("odds", help="理論当せん確率と期待値を表示する")
    p_odds.set_defaults(func=cmd_odds)

    p_check = sub.add_parser("check", help="買い目を指定回の抽せん結果と照合する")
    p_check.add_argument("numbers", help="購入した 7 個の数字（例: \"1 5 9 14 22 30 33\"）")
    p_check.add_argument("--round", type=int, default=None, help="照合する回号（既定: 最新回）")
    p_check.set_defaults(func=cmd_check)

    p_fetch = sub.add_parser("fetch", help="公式サイトから抽せん結果を取得する")
    p_fetch.add_argument("--timeout", type=float, default=20.0)
    p_fetch.set_defaults(func=cmd_fetch)

    p_import = sub.add_parser("import", help="CSV を取り込んでデータを更新する")
    p_import.add_argument("source", type=Path, help="取り込む CSV")
    p_import.set_defaults(func=cmd_import)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
