"""全角文字を含む表を端末で揃えるためのユーティリティ。"""

from __future__ import annotations

import unicodedata


def display_width(text: str) -> int:
    """端末上の表示幅（全角文字は 2 とみなす）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, width: int, align: str = "<") -> str:
    """表示幅を基準に空白で埋める。``align`` は ``<``（左）/ ``>``（右）/ ``^``（中央）。"""
    padding = max(0, width - display_width(text))
    if align == ">":
        return " " * padding + text
    if align == "^":
        left = padding // 2
        return " " * left + text + " " * (padding - left)
    return text + " " * padding
