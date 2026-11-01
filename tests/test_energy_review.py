import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src.energy_review import assign_bins, bin_labels, prepare, score, wilson

EDGES = [-70.0, -60.0, -50.0, -40.0, -30.0]
WORDS = ["down", "go", "left", "no", "right", "stop", "up", "yes"]
RULE = {"unusable_rate_threshold": 0.5, "control_max_unusable_rate": 0.2}


@pytest.fixture()
def pkg(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for b, centre in enumerate([-80, -65, -55, -45, -35, -22]):     # one cluster per bin
        for i in range(10):
            w = WORDS[(b * 10 + i) % 8]
            fn = f"{b:02d}{i:06x}_nohash_{i}.wav"
            (tmp_path / "raw" / w).mkdir(parents=True, exist_ok=True)
            sf.write(tmp_path / "raw" / w / fn, 0.05 * rng.standard_normal(16000), 16000, subtype="PCM_16")
            rows.append({"Word_Class": w, "Filename": fn, "RMS_dBFS": centre + rng.uniform(-3, 3)})
    df = pd.DataFrame(rows)
    sel = prepare(df, tmp_path / "raw", tmp_path / "rev", EDGES, 6, 42, 0.7)
    return tmp_path, df, sel


def test_bins():
    assert list(assign_bins([-80, -70, -65, -30, -10], EDGES)) == [0, 1, 1, 5, 5]
    assert bin_labels(EDGES)[0] == "<-70" and bin_labels(EDGES)[-1] == ">=-30" and len(bin_labels(EDGES)) == 6


def test_blinding_and_determinism(pkg):
    tmp, df, sel = pkg
    sheet = pd.read_csv(tmp / "rev" / "review_sheet.csv", encoding="utf-8-sig")
    assert list(sheet.columns) == ["clip_id", "heard_word", "notes"]          # no label / level leaks
    assert len(sheet) == 36 and sheet["clip_id"].is_unique
    assert len(list((tmp / "rev" / "clips").glob("*.wav"))) == 36
    x, _ = sf.read(tmp / "rev" / "clips" / "clip_001.wav")
    assert abs(np.max(np.abs(x)) - 0.7) < 1e-3                                  # peak-normalised
    again = prepare(df, tmp / "raw", tmp / "rev", EDGES, 6, 42, 0.7, overwrite=True)
    assert again["Filename"].tolist() == sel["Filename"].tolist()               # same seed -> same package
    with pytest.raises(FileExistsError):
        prepare(df, tmp / "raw", tmp / "rev", EDGES, 6, 42, 0.7)               # never clobber answers


def _fill(tmp, heard_fn):
    key = pd.read_csv(tmp / "rev" / "review_key.csv")
    sheet = pd.DataFrame({"clip_id": key["Clip_ID"],
                          "heard_word": [heard_fn(r) for _, r in key.iterrows()], "notes": ""})
    sheet.to_csv(tmp / "rev" / "review_sheet.csv", index=False)


def test_rule_excludes_up_to_minus50(pkg):
    tmp, *_ = pkg
    _fill(tmp, lambda r: "?" if r["Bin"] <= 2 else r["Word_Class"])           # bins <-70, -70..-60, -60..-50 unusable
    res = score(tmp / "rev" / "review_sheet.csv", tmp / "rev" / "review_key.csv", WORDS, EDGES, RULE)
    assert res["protocol_valid"] and res["recommended_exclude_low_energy"]
    assert res["recommended_threshold_dbfs"] == -50.0
    assert "| بازه‌ی RMS" in res["table_md"]


def test_rule_keeps_everything_when_all_heard(pkg):
    tmp, *_ = pkg
    _fill(tmp, lambda r: r["Word_Class"])
    res = score(tmp / "rev" / "review_sheet.csv", tmp / "rev" / "review_key.csv", WORDS, EDGES, RULE)
    assert res["protocol_valid"] and not res["recommended_exclude_low_energy"]


def test_invalid_when_control_fails_and_bad_input_rejected(pkg):
    tmp, *_ = pkg
    _fill(tmp, lambda r: "?")
    res = score(tmp / "rev" / "review_sheet.csv", tmp / "rev" / "review_key.csv", WORDS, EDGES, RULE)
    assert not res["protocol_valid"] and not res["recommended_exclude_low_energy"]
    _fill(tmp, lambda r: "")
    with pytest.raises(ValueError):
        score(tmp / "rev" / "review_sheet.csv", tmp / "rev" / "review_key.csv", WORDS, EDGES, RULE)
    _fill(tmp, lambda r: "banana")
    with pytest.raises(ValueError):
        score(tmp / "rev" / "review_sheet.csv", tmp / "rev" / "review_key.csv", WORDS, EDGES, RULE)


def test_wilson():
    lo, hi = wilson(3, 6)
    assert 0.1 < lo < 0.5 < hi < 0.9
    assert wilson(0, 0)[0] != wilson(0, 0)[0]                                   # nan
