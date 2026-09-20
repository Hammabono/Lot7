"""抽せん結果データの読み書き。"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .rules import BONUS_COUNT, MAX_NUMBER, MIN_NUMBER, PICK

HEADER = [
    "round",
    "date",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "b1",
    "b2",
]


@dataclass(frozen=True)
class Draw:
    """1 回分の抽せん結果。"""

    round: int
    date: dt.date | None
    numbers: tuple[int, ...]
    bonus: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.numbers) != PICK or len(set(self.numbers)) != PICK:
            raise ValueError(f"第{self.round}回: 本数字は重複なし {PICK} 個必要です")
        if len(self.bonus) != BONUS_COUNT or len(set(self.bonus)) != BONUS_COUNT:
            raise ValueError(f"第{self.round}回: ボーナス数字は {BONUS_COUNT} 個必要です")
        if set(self.numbers) & set(self.bonus):
            raise ValueError(f"第{self.round}回: 本数字とボーナス数字が重複しています")
        for n in self.numbers + self.bonus:
            if not (MIN_NUMBER <= n <= MAX_NUMBER):
                raise ValueError(f"第{self.round}回: 範囲外の数字 {n}")
        if tuple(sorted(self.numbers)) != self.numbers:
            object.__setattr__(self, "numbers", tuple(sorted(self.numbers)))
        if tuple(sorted(self.bonus)) != self.bonus:
            object.__setattr__(self, "bonus", tuple(sorted(self.bonus)))

    @property
    def all_numbers(self) -> tuple[int, ...]:
        return tuple(sorted(self.numbers + self.bonus))

    def __str__(self) -> str:
        date = self.date.isoformat() if self.date else "----------"
        nums = " ".join(f"{n:2d}" for n in self.numbers)
        bonus = " ".join(f"{n:2d}" for n in self.bonus)
        return f"第{self.round:>4}回 {date}  {nums}  (B: {bonus})"


def _parse_date(value: str) -> dt.date | None:
    value = value.strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日"):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"日付を解釈できません: {value!r}")


def load_csv(path: str | Path) -> list[Draw]:
    """CSV から抽せん結果を読み込み、回号昇順で返す。"""
    path = Path(path)
    draws: list[Draw] = []
    with path.open(encoding="utf-8-sig", newline="") as fp:
        rows = csv.DictReader(_strip_comments(fp))
        missing = set(HEADER) - set(rows.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: 列が不足しています: {sorted(missing)}")
        for row in rows:
            draws.append(
                Draw(
                    round=int(row["round"]),
                    date=_parse_date(row["date"]),
                    numbers=tuple(int(row[f"n{i}"]) for i in range(1, PICK + 1)),
                    bonus=tuple(int(row[f"b{i}"]) for i in range(1, BONUS_COUNT + 1)),
                )
            )
    draws.sort(key=lambda d: d.round)
    seen: set[int] = set()
    for draw in draws:
        if draw.round in seen:
            raise ValueError(f"回号が重複しています: 第{draw.round}回")
        seen.add(draw.round)
    return draws


def _strip_comments(lines: Iterable[str]) -> Iterator[str]:
    for line in lines:
        if line.lstrip().startswith("#"):
            continue
        yield line


def save_csv(draws: Sequence[Draw], path: str | Path, header_note: str = "") -> None:
    """抽せん結果を CSV に書き出す。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        for line in header_note.splitlines():
            fp.write(f"# {line}\n")
        writer = csv.writer(fp)
        writer.writerow(HEADER)
        for draw in sorted(draws, key=lambda d: d.round):
            writer.writerow(
                [
                    draw.round,
                    draw.date.isoformat() if draw.date else "",
                    *draw.numbers,
                    *draw.bonus,
                ]
            )
