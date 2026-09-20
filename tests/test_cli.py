import pytest

from loto7.cli import DEFAULT_DATA, main
from loto7.data import load_csv, save_csv

DATA = ["--data", str(DEFAULT_DATA)]


@pytest.fixture
def small_data(tmp_path, random_draws) -> list[str]:
    path = tmp_path / "draws.csv"
    save_csv(random_draws, path)
    return ["--data", str(path)]


def test_sample_data_is_present_and_labelled():
    assert DEFAULT_DATA.exists()
    head = DEFAULT_DATA.read_text(encoding="utf-8")[:400]
    assert "シミュレーションデータ" in head
    assert "実際のロト7の当せん番号ではない" in head


def test_predict_prints_candidates(capsys, small_data):
    assert main([*small_data, "predict", "-n", "3", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "買い目候補" in out
    assert out.count("候補") >= 3
    assert "1/10,295,472" in out  # 免責が必ず出る


def test_predict_include_and_exclude(capsys, small_data):
    assert main([*small_data, "predict", "-n", "3", "--seed", "2",
                 "--include", "7 13", "--exclude", "1,2,3", "--no-constraints"]) == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip().startswith("候補")]
    assert lines
    for line in lines:
        numbers = {int(tok) for tok in line.split(":")[1].split("[")[0].split()}
        assert {7, 13} <= numbers
        assert not numbers & {1, 2, 3}


def test_predict_rejects_conflicting_filters(small_data):
    with pytest.raises(SystemExit, match="重複"):
        main([*small_data, "predict", "--include", "5", "--exclude", "5"])


def test_predict_rejects_out_of_range_numbers(small_data):
    with pytest.raises(SystemExit, match="範囲"):
        main([*small_data, "predict", "--include", "40"])


def test_stats_reports_uniformity_test(capsys, small_data):
    assert main([*small_data, "stats"]) == 0
    out = capsys.readouterr().out
    assert "カイ二乗統計量" in out
    assert "p 値" in out
    assert "出目の形" in out


def test_backtest_reports_z_score(capsys, small_data):
    assert main([*small_data, "backtest", "--tickets", "2", "--warmup", "150"]) == 0
    out = capsys.readouterr().out
    assert "平均一致数" in out
    assert "ランダムの期待値" in out
    assert "z 値" in out


def test_compare_lists_models(capsys, small_data):
    assert main([*small_data, "compare", "--models", "uniform", "ensemble",
                 "--tickets", "1", "--warmup", "180"]) == 0
    out = capsys.readouterr().out
    assert "uniform" in out and "ensemble" in out and "理論値" in out


def test_odds_shows_official_probabilities(capsys):
    assert main(["odds"]) == 0
    out = capsys.readouterr().out
    for denominator in ("10,295,472", "735,391", "52,528", "1,127"):
        assert denominator in out
    assert "回収率" in out


def test_check_identifies_the_jackpot(capsys, tmp_path, random_draws):
    path = tmp_path / "draws.csv"
    save_csv(random_draws, path)
    winning = " ".join(str(n) for n in random_draws[-1].numbers)
    assert main(["--data", str(path), "check", winning]) == 0
    assert "1等" in capsys.readouterr().out


def test_check_reports_a_loss(capsys, tmp_path, random_draws):
    path = tmp_path / "draws.csv"
    save_csv(random_draws, path)
    last = set(random_draws[-1].all_numbers)
    losing = sorted(n for n in range(1, 38) if n not in last)[:7]
    assert main(["--data", str(path), "check", " ".join(map(str, losing))]) == 0
    assert "はずれ" in capsys.readouterr().out


def test_check_specific_round(capsys, tmp_path, random_draws):
    path = tmp_path / "draws.csv"
    save_csv(random_draws, path)
    target = random_draws[10]
    winning = " ".join(str(n) for n in target.numbers)
    assert main(["--data", str(path), "check", winning, "--round", str(target.round)]) == 0
    out = capsys.readouterr().out
    assert f"第{target.round:>4}回" in out
    assert "1等" in out


def test_check_missing_round(tmp_path, random_draws):
    path = tmp_path / "draws.csv"
    save_csv(random_draws, path)
    with pytest.raises(SystemExit, match="第9999回"):
        main(["--data", str(path), "check", "1 2 3 4 5 6 7", "--round", "9999"])


def test_import_merges_into_dataset(capsys, tmp_path, random_draws):
    target = tmp_path / "dataset.csv"
    source = tmp_path / "incoming.csv"
    save_csv(random_draws[:100], target)
    save_csv(random_draws[100:], source)
    assert main(["--data", str(target), "import", str(source)]) == 0
    assert len(load_csv(target)) == len(random_draws)
    assert "取り込み" in capsys.readouterr().out


def test_missing_data_file_is_reported(tmp_path):
    with pytest.raises(SystemExit, match="見つかりません"):
        main(["--data", str(tmp_path / "nope.csv"), "predict"])


def test_fetch_failure_returns_error_code(capsys, tmp_path, monkeypatch):
    import loto7.fetch as fetch_module

    def boom(url, timeout=20.0):
        raise fetch_module.FetchError("blocked")

    monkeypatch.setattr(fetch_module, "download", boom)
    assert main(["--data", str(tmp_path / "d.csv"), "fetch"]) == 2
    assert "import" in capsys.readouterr().err
