import datetime as dt

import pytest

from loto7.data import Draw, load_csv, save_csv


def test_draw_sorts_numbers():
    draw = Draw(round=1, date=None, numbers=(7, 1, 5, 3, 9, 11, 13), bonus=(20, 2))
    assert draw.numbers == (1, 3, 5, 7, 9, 11, 13)
    assert draw.bonus == (2, 20)
    assert draw.all_numbers == (1, 2, 3, 5, 7, 9, 11, 13, 20)


@pytest.mark.parametrize(
    "numbers, bonus, message",
    [
        ((1, 2, 3, 4, 5, 6), (8, 9), "本数字"),
        ((1, 1, 2, 3, 4, 5, 6), (8, 9), "本数字"),
        ((1, 2, 3, 4, 5, 6, 7), (8,), "ボーナス"),
        ((1, 2, 3, 4, 5, 6, 7), (7, 9), "重複"),
        ((1, 2, 3, 4, 5, 6, 38), (8, 9), "範囲外"),
    ],
)
def test_draw_validation(numbers, bonus, message):
    with pytest.raises(ValueError, match=message):
        Draw(round=1, date=None, numbers=numbers, bonus=bonus)


def test_csv_roundtrip(tmp_path, random_draws):
    path = tmp_path / "draws.csv"
    save_csv(random_draws, path, header_note="テスト用\n2 行目")
    assert path.read_text(encoding="utf-8").startswith("# テスト用\n# 2 行目\n")
    assert load_csv(path) == random_draws


def test_load_csv_accepts_japanese_dates(tmp_path):
    path = tmp_path / "draws.csv"
    path.write_text(
        "round,date,n1,n2,n3,n4,n5,n6,n7,b1,b2\n"
        "1,2013年4月5日,1,2,3,4,5,6,7,8,9\n"
        "2,2013/04/12,2,3,4,5,6,7,8,9,10\n"
        "3,,3,4,5,6,7,8,9,10,11\n",
        encoding="utf-8",
    )
    draws = load_csv(path)
    assert draws[0].date == dt.date(2013, 4, 5)
    assert draws[1].date == dt.date(2013, 4, 12)
    assert draws[2].date is None


def test_load_csv_rejects_duplicate_rounds(tmp_path):
    path = tmp_path / "draws.csv"
    path.write_text(
        "round,date,n1,n2,n3,n4,n5,n6,n7,b1,b2\n"
        "1,,1,2,3,4,5,6,7,8,9\n"
        "1,,2,3,4,5,6,7,8,9,10\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="回号が重複"):
        load_csv(path)


def test_load_csv_rejects_missing_columns(tmp_path):
    path = tmp_path / "draws.csv"
    path.write_text("round,date,n1\n1,,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="列が不足"):
        load_csv(path)
