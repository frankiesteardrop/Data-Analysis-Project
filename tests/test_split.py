import numpy as np
import pandas as pd
import pytest

from src.splits import make_speaker_split, trivial_baselines, verify_split


def _toy(n_speakers=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_speakers):
        for _ in range(rng.integers(1, 20)):
            rows.append({"Speaker_ID": f"{s:08x}", "Word_Class": rng.choice(list("abcdefgh")),
                         "Filename": f"{s:08x}_nohash_{len(rows)}.wav", "Audio_MD5": f"md5{len(rows)}"})
    return pd.DataFrame(rows)


def test_no_speaker_overlap_and_deterministic():
    df = _toy()
    r = {"train": .8, "val": .1, "test": .1}
    a, b = make_speaker_split(df, r, 42), make_speaker_split(df, r, 42)
    assert a["Split"].equals(b["Split"])
    verify_split(a)                                            # must not raise
    assert abs(a["Split"].value_counts(normalize=True)["train"] - 0.8) < 0.03


def test_verify_split_detects_leakage():
    df = make_speaker_split(_toy(), {"train": .8, "val": .1, "test": .1}, 1)
    df.loc[df.index[0], "Split"] = "test" if df.loc[df.index[0], "Split"] != "test" else "train"
    with pytest.raises(AssertionError):
        verify_split(df)


def test_balanced_majority_baseline_formula():
    # balanced 8 classes -> majority accuracy 1/8, macro-F1 = (2p/(1+p))/8 ~ 0.0278
    rows = [{"Speaker_ID": f"{i:08x}", "Word_Class": c, "Filename": f"{i}_{c}", "Audio_MD5": f"{i}{c}"}
            for i in range(400) for c in "abcdefgh"]
    df = make_speaker_split(pd.DataFrame(rows), {"train": .5, "val": .0, "test": .5}, 3)
    b = trivial_baselines(df, 3)
    assert b["majority"]["accuracy"] == pytest.approx(0.125, abs=1e-9)
    assert b["majority"]["macro_f1"] == pytest.approx((2 * .125 / 1.125) / 8, abs=1e-6)
