"""Speaker-disjoint train/val/test split + verification + trivial baselines.

The split unit is the SPEAKER, the prediction unit is the CLIP. Speakers are shuffled with a
fixed seed and assigned to train/val/test by cumulative FILE share, so file proportions
approximate the requested ratios while no speaker is ever shared between splits.
The test set is locked: no tuning of any kind may use it.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def make_speaker_split(df: pd.DataFrame, ratios: dict, seed: int) -> pd.DataFrame:
    assert abs(sum(ratios.values()) - 1.0) < 1e-9, "ratios must sum to 1"
    counts = df["Speaker_ID"].value_counts().sort_index()          # sorted -> deterministic
    rng = np.random.default_rng(seed)
    order = counts.index.to_numpy()[rng.permutation(len(counts))]
    cum = np.cumsum(counts.loc[order].to_numpy()) / counts.sum()
    t_tr, t_va = ratios["train"], ratios["train"] + ratios["val"]
    assign = np.where(cum <= t_tr, "train", np.where(cum <= t_va, "val", "test"))
    spk2split = dict(zip(order, assign))
    out = df.copy()
    out["Split"] = out["Speaker_ID"].map(spk2split)
    return out


def verify_split(df: pd.DataFrame) -> dict:
    """Hard checks. Raises AssertionError on any speaker overlap."""
    sets = {s: set(g["Speaker_ID"]) for s, g in df.groupby("Split")}
    for a, b in combinations(sorted(sets), 2):
        inter = sets[a] & sets[b]
        assert not inter, f"Speaker leakage between {a} and {b}: {len(inter)} shared speakers"
    assert df["Split"].notna().all(), "unassigned rows"
    # exact duplicates of audio must not straddle splits either
    if "Audio_MD5" in df:
        spl = df.groupby("Audio_MD5")["Split"].nunique()
        assert (spl <= 1).all(), "identical audio appears in more than one split"
    rep = {
        "files_per_split": df["Split"].value_counts().to_dict(),
        "speakers_per_split": {s: len(v) for s, v in sets.items()},
        "file_share_per_split": df["Split"].value_counts(normalize=True).round(4).to_dict(),
        "class_counts_per_split": pd.crosstab(df["Split"], df["Word_Class"]).to_dict("index"),
        "pairwise_speaker_intersections": {f"{a}&{b}": 0 for a, b in combinations(sorted(sets), 2)},
    }
    return rep


def save_split(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for s, g in df.groupby("Split"):
        (out_dir / f"speakers_{s}.txt").write_text("\n".join(sorted(g["Speaker_ID"].unique())) + "\n")
    cols = [c for c in ["Split", "Word_Class", "Speaker_ID", "Filename", "Exclude", "Exclude_Reason"] if c in df]
    df[cols].sort_values(["Split", "Word_Class", "Filename"]).to_csv(out_dir / "split_manifest.csv", index=False)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: list) -> float:
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def trivial_baselines(df: pd.DataFrame, seed: int, n_rep: int = 200) -> dict:
    """Majority-class (train majority) and uniform-random baselines evaluated on the test split."""
    tr, te = df[df["Split"] == "train"], df[df["Split"] == "test"]
    classes = sorted(df["Word_Class"].unique())
    maj = tr["Word_Class"].value_counts().idxmax()
    y = te["Word_Class"].to_numpy()
    y_maj = np.full(len(y), maj, dtype=object)
    rng = np.random.default_rng(seed)
    sims = [macro_f1(y, rng.choice(classes, size=len(y)), classes) for _ in range(n_rep)]
    acc_rand = [np.mean(rng.choice(classes, size=len(y)) == y) for _ in range(n_rep)]
    return {
        "train_majority_class": maj,
        "majority": {"accuracy": float(np.mean(y_maj == y)), "macro_f1": macro_f1(y, y_maj, classes)},
        "uniform_random": {"accuracy_mean": float(np.mean(acc_rand)),
                           "macro_f1_mean": float(np.mean(sims)), "macro_f1_std": float(np.std(sims)),
                           "n_repetitions": n_rep},
        "n_test_files": int(len(y)),
    }
