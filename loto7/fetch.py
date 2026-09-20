"""公式サイトから抽せん結果を取得する（ベストエフォート）。

みずほ銀行のページ構造は予告なく変わるため、取得に失敗した場合は
手元の CSV を ``loto7 import`` で取り込む運用にフォールバックできるようにしている。
"""

from __future__ import annotations

import datetime as dt
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser
from typing import Sequence

from .data import Draw
from .rules import BONUS_COUNT, MAX_NUMBER, MIN_NUMBER, PICK

DEFAULT_URLS: tuple[str, ...] = (
    "https://www.mizuhobank.co.jp/retail/takarakuji/loto/loto7/index.html",
    "https://www.mizuhobank.co.jp/retail/takarakuji/loto/loto7/backnumber/index.html",
    "https://www.mizuhobank.co.jp/retail/takarakuji/loto/loto7/backnumber/detail.html",
)

USER_AGENT = "loto7-analyzer/1.0 (+local analysis tool)"

_ROUND_RE = re.compile(r"第\s*([0-9,]+)\s*回")
_DATE_RE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_NUMBER_RE = re.compile(r"^\s*(\d{1,2})\s*$")


class FetchError(RuntimeError):
    """ネットワーク取得またはページ解析に失敗した。"""


class _CellCollector(HTMLParser):
    """テーブルセルのテキストを出現順に集める。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cells: list[str] = []
        self._depth = 0
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in ("td", "th"):
            self._depth += 1
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._depth:
            self._depth -= 1
            self.cells.append(" ".join("".join(self._buffer).split()))
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._depth:
            self._buffer.append(data)


def parse_backnumber_html(html: str) -> list[Draw]:
    """HTML から抽せん結果を抽出する。

    「第N回」→（日付）→ 本数字 7 個 → ボーナス数字 2 個 の並びを走査する
    ゆるいパーサ。ページ構造が変わると 0 件を返すことがある。
    """
    collector = _CellCollector()
    collector.feed(html)
    cells = collector.cells

    draws: list[Draw] = []
    index = 0
    while index < len(cells):
        round_match = _ROUND_RE.search(cells[index])
        if not round_match:
            index += 1
            continue
        round_no = int(round_match.group(1).replace(",", ""))

        date: dt.date | None = None
        numbers: list[int] = []
        cursor = index + 1
        while cursor < len(cells) and len(numbers) < PICK + BONUS_COUNT:
            cell = cells[cursor]
            if _ROUND_RE.search(cell):
                break
            if date is None:
                date_match = _DATE_RE.search(cell)
                if date_match:
                    date = dt.date(*(int(g) for g in date_match.groups()))
                    cursor += 1
                    continue
            value_match = _NUMBER_RE.match(cell)
            if value_match:
                value = int(value_match.group(1))
                if MIN_NUMBER <= value <= MAX_NUMBER:
                    numbers.append(value)
            cursor += 1

        if len(numbers) == PICK + BONUS_COUNT:
            try:
                draws.append(
                    Draw(
                        round=round_no,
                        date=date,
                        numbers=tuple(numbers[:PICK]),
                        bonus=tuple(numbers[PICK:]),
                    )
                )
            except ValueError:
                pass  # 解析ミスで不正な組み合わせになった行は捨てる
        index = max(cursor, index + 1)

    return draws


def download(url: str, timeout: float = 20.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise FetchError(f"{url} を取得できません: {exc}") from exc
    for encoding in ("utf-8", "cp932", "euc-jp"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def fetch(urls: Sequence[str] = DEFAULT_URLS, timeout: float = 20.0) -> list[Draw]:
    """複数 URL から取得し、回号でマージして返す。"""
    merged: dict[int, Draw] = {}
    errors: list[str] = []
    for url in urls:
        try:
            html = download(url, timeout=timeout)
        except FetchError as exc:
            errors.append(str(exc))
            continue
        for draw in parse_backnumber_html(html):
            merged.setdefault(draw.round, draw)
    if not merged:
        detail = "\n  ".join(errors) if errors else "ページから抽せん結果を抽出できませんでした"
        raise FetchError(
            "抽せん結果を取得できませんでした。\n  "
            + detail
            + "\n公式サイトから CSV を保存し `loto7 import <file.csv>` で取り込んでください。"
        )
    return [merged[r] for r in sorted(merged)]


def merge(existing: Sequence[Draw], incoming: Sequence[Draw]) -> list[Draw]:
    """既存データに新しい抽せん結果をマージする（既存を優先しない = 上書き更新）。"""
    merged: dict[int, Draw] = {d.round: d for d in existing}
    merged.update({d.round: d for d in incoming})
    return [merged[r] for r in sorted(merged)]
