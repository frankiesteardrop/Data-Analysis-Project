"""Phase 2 data preparation: manifest -> fixed-length 16 kHz mono waveforms.

Re-runnable chain (all parameters in configs/config.yaml -> phase2):
  manifest (Phase-1 split, exclusions already decided) -> drop excluded rows -> read wav
  -> mono -> resample to 16 kHz if needed -> pad (zeros, end) / crop (first 1 s) to 16000 samples
  -> keep `n_valid` (real samples before padding) so features can ignore the padded tail.

"Missing data" in this dataset is not NaN cells: it is (a) files that are too short (-> padding),
(b) files that are unreadable (none), (c) near-silent files (excluded in Phase 1 by the blinded listening
test; they are NOT imputed). Padding is documented per clip, never hidden.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

log = logging.getLogger(__name__)
EPS = 1e-12


def load_manifest(cfg: dict, splits=("train", "val")) -> pd.DataFrame:
    """Phase-1 manifest restricted to the requested splits, excluded rows removed.

    The test split must be requested explicitly (it is locked in Phase 2)."""
    m = pd.read_csv(Path(cfg["_root"]) / cfg["split"]["out_dir"] / "split_manifest.csv",
                    dtype={"Speaker_ID": str, "Filename": str, "Word_Class": str})
    m = m[m["Split"].isin(splits)]
    m = m[~m["Exclude"].astype(bool)].copy()
    return m.sort_values(["Split", "Word_Class", "Filename"]).reset_index(drop=True)


def fix_length(x: np.ndarray, target: int) -> tuple[np.ndarray, int, str]:
    """Pad with zeros at the end / keep the first `target` samples. Returns (clip, n_valid, action)."""
    n = len(x)
    if n == target:
        return x.astype(np.float32), n, "none"
    if n < target:
        out = np.zeros(target, dtype=np.float32)
        out[:n] = x
        return out, n, "padded"
    return x[:target].astype(np.float32), target, "cropped"


def read_clip(path: Path, sr_target: int, target_len: int):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    resampled = False
    if sr != sr_target:                               # never happens in this dataset (checked in Phase 1)
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=sr_target)
        resampled = True
    clip, n_valid, action = fix_length(x, target_len)
    return clip, n_valid, action, resampled


def load_waveforms(cfg: dict, manifest: pd.DataFrame):
    """Return (X float32 [N, target_len], meta DataFrame with n_valid / action / rms_dbfs)."""
    p2 = cfg["phase2"]
    root = Path(cfg["_root"]) / cfg["data"]["base_dir"] / "mini_speech_commands"
    X = np.zeros((len(manifest), p2["target_len"]), dtype=np.float32)
    nv, act, rs = [], [], []
    for i, r in enumerate(manifest.itertuples()):
        clip, n_valid, action, resampled = read_clip(root / r.Word_Class / r.Filename, p2["sr"], p2["target_len"])
        X[i] = clip
        nv.append(n_valid)
        act.append(action)
        rs.append(20 * np.log10(np.sqrt(np.mean(clip[:n_valid] ** 2)) + EPS))
    meta = manifest.copy()
    meta["n_valid"] = nv
    meta["length_action"] = act
    meta["rms_dbfs"] = rs
    if not np.isfinite(X).all():
        raise ValueError("non-finite samples after loading")
    return X, meta


def preprocessing_summary(meta: pd.DataFrame, target_len: int) -> dict:
    out = {}
    for split, g in meta.groupby("Split"):
        out[split] = {
            "n_clips": int(len(g)),
            "n_padded": int((g.length_action == "padded").sum()),
            "n_cropped": int((g.length_action == "cropped").sum()),
            "frac_padded": float((g.length_action == "padded").mean()),
            "valid_sec_median": float(g.n_valid.median() / 16000),
            "valid_sec_min": float(g.n_valid.min() / 16000),
            "valid_sec_p5": float(g.n_valid.quantile(0.05) / 16000),
            "median_pad_samples": float((target_len - g.n_valid[g.length_action == "padded"]).median())
            if (g.length_action == "padded").any() else 0.0,
        }
    return out
