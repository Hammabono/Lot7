import datetime as dt

import pytest

from loto7.data import Draw
from loto7.fetch import FetchError, fetch, merge, parse_backnumber_html

SAMPLE_HTML = """
<html><body>
<table>
  <tr><th>回別</th><td>第100回</td><td>第101回</td></tr>
  <tr><th>抽せん日</th><td>2015年3月6日</td><td>2015年3月13日</td></tr>
</table>
<table>
  <tr><th>第100回</th></tr>
  <tr><th>抽せん日</th><td>2015年3月6日</td></tr>
  <tr><th>本数字</th><td>3</td><td>9</td><td>14</td><td>21</td><td>25</td><td>30</td><td>34</td></tr>
  <tr><th>ボーナス数字</th><td>7</td><td>19</td></tr>
</table>
<table>
  <tr><th>第101回</th></tr>
  <tr><th>抽せん日</th><td>2015年3月13日</td></tr>
  <tr><th>本数字</th><td>1</td><td>4</td><td>11</td><td>18</td><td>22</td><td>29</td><td>37</td></tr>
  <tr><th>ボーナス数字</th><td>12</td><td>26</td></tr>
</table>
</body></html>
"""


def test_parse_backnumber_html():
    draws = parse_backnumber_html(SAMPLE_HTML)
    rounds = {d.round: d for d in draws}
    assert 100 in rounds and 101 in rounds
    assert rounds[100].numbers == (3, 9, 14, 21, 25, 30, 34)
    assert rounds[100].bonus == (7, 19)
    assert rounds[100].date == dt.date(2015, 3, 6)
    assert rounds[101].numbers == (1, 4, 11, 18, 22, 29, 37)
    assert rounds[101].bonus == (12, 26)


def test_parse_handles_comma_separated_round_numbers():
    html = """
    <table>
      <tr><th>第1,234回</th></tr>
      <tr><td>2013年4月5日</td></tr>
      <tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr>
      <tr><td>8</td><td>9</td></tr>
    </table>
    """
    assert parse_backnumber_html(html)[0].round == 1234


def test_parse_ignores_unparseable_markup():
    assert parse_backnumber_html("<html><body><p>メンテナンス中</p></body></html>") == []
    assert parse_backnumber_html("") == []


def test_fetch_reports_a_usable_error(monkeypatch):
    """取得に失敗したら CSV 取り込みへ誘導するメッセージを出す。"""
    import loto7.fetch as fetch_module

    def boom(url, timeout=20.0):
        raise FetchError(f"{url} を取得できません: blocked")

    monkeypatch.setattr(fetch_module, "download", boom)
    with pytest.raises(FetchError, match="import"):
        fetch(urls=("https://example.invalid/",))


def test_merge_prefers_incoming_and_sorts():
    old = Draw(round=1, date=None, numbers=(1, 2, 3, 4, 5, 6, 7), bonus=(8, 9))
    fixed = Draw(round=1, date=None, numbers=(1, 2, 3, 4, 5, 6, 10), bonus=(8, 9))
    new = Draw(round=2, date=None, numbers=(2, 3, 4, 5, 6, 7, 8), bonus=(9, 10))
    merged = merge([old], [new, fixed])
    assert [d.round for d in merged] == [1, 2]
    assert merged[0].numbers == fixed.numbers
