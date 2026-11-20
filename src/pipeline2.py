"""Phase 2 stage 1: waveforms -> features (train, val, train-augmented, val noise conditions). Cached as .npz."""
from __future__ import annotations

import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .augment import assert_train_only, augment_train_clip, noise_condition_set
from .features import clip_features
from .preprocess import load_manifest, load_waveforms, preprocessing_summary

log = logging.getLogger(__name__)
_CFG = None


def _init(cfg):
    global _CFG
    _CFG = cfg


def _one(args):
    x, start, n_valid, want_logmel = args
    o = clip_features(x, int(start), int(n_valid), _CFG, want_logmel)
    return o


def extract(cfg, X, starts, n_valid, want_logmel=True, procs=2):
    jobs = [(X[i], starts[i], n_valid[i], want_logmel) for i in range(len(X))]
    with Pool(procs, initializer=_init, initargs=(cfg,)) as pool:
        res = pool.map(_one, jobs, chunksize=64)
    out = {"mfcc": np.stack([r["mfcc_vec"] for r in res]),
           "eng": np.stack([r["eng_vec"] for r in res]),
           "logmel_ms": np.stack([r["logmel_ms"] for r in res]),
           "diag": np.stack([r["diag"] for r in res])}
    if want_logmel:
        out["logmel"] = np.stack([r["logmel"] for r in res])
    return out


def build(cfg, include_test=False, procs=2, force=False):
    p2 = cfg["phase2"]
    fdir = Path(cfg["_root"]) / p2["paths"]["features_dir"]
    fdir.mkdir(parents=True, exist_ok=True)
    splits = ("train", "val") + (("test",) if include_test else ())
    done = fdir / "build_done.txt"
    if done.exists() and not force:
        log.info("features cached in %s (use --force to rebuild)", fdir)
        return
    man = load_manifest(cfg, splits)
    log.info("manifest: %s", man.Split.value_counts().to_dict())
    X, meta = load_waveforms(cfg, man)
    meta.to_csv(fdir / "meta.csv", index=False)
    pre = preprocessing_summary(meta, p2["target_len"])
    pd.Series(pre).to_json(fdir / "preprocessing_summary.json")
    zeros = np.zeros(len(X), dtype=int)
    nv = meta.n_valid.to_numpy()
    # ---- clean features for every requested split
    for sp in splits:
        idx = np.flatnonzero(meta.Split.to_numpy() == sp)
        log.info("features %s (%d clips)", sp, len(idx))
        f = extract(cfg, X[idx], zeros[idx], nv[idx], True, procs)
        np.savez_compressed(fdir / f"feat_{sp}.npz", idx=idx, **f)

    # ---- augmentation: TRAIN ONLY
    tr = np.flatnonzero(meta.Split.to_numpy() == "train")
    assert_train_only(meta.Split.iloc[tr].unique())
    rng = np.random.default_rng(p2["seed"])
    Xtr, nvtr = X[tr], nv[tr]
    Xa, starts, nva, rows = [], [], [], []
    for j in range(len(tr)):
        y, st, n, prm = augment_train_clip(Xtr[j], int(nvtr[j]), Xtr, nvtr, j, p2["augmentation"], p2["sr"], rng)
        Xa.append(y); starts.append(st); nva.append(n); rows.append(prm)
    log.info("features train-augmented")
    fa = extract(cfg, np.stack(Xa), np.array(starts), np.array(nva), False, procs)
    alog = pd.DataFrame(rows)
    alog.insert(0, "train_row", np.arange(len(tr)))
    alog.to_csv(fdir / "augmentation_log.csv", index=False)
    np.savez_compressed(fdir / "feat_train_aug.npz", **fa)

    # ---- noise conditions: VAL ONLY
    va = np.flatnonzero(meta.Split.to_numpy() == "val")
    names, kinds, snrs, feats = [], [], [], []
    for name, t, snr, Y in noise_condition_set(X[va], nv[va], p2["noise_conditions"], p2["augmentation"]["babble_n"], p2["seed"]):
        log.info("features val noise %s", name)
        f = extract(cfg, Y, zeros[va], nv[va], False, procs)
        feats.append(f); names.append(name); kinds.append(t); snrs.append(snr)
    np.savez_compressed(fdir / "feat_val_noise.npz",
                        names=np.array(names), kinds=np.array(kinds), snrs=np.array(snrs),
                        mfcc=np.stack([f["mfcc"] for f in feats]), eng=np.stack([f["eng"] for f in feats]),
                        logmel_ms=np.stack([f["logmel_ms"] for f in feats]), diag=np.stack([f["diag"] for f in feats]))
    # save a few waveforms for the figure of the preprocessing step
    np.savez_compressed(fdir / "examples.npz", X=X[tr[:: max(1, len(tr) // 400)]][:400],
                        nv=nv[tr[:: max(1, len(tr) // 400)]][:400])
    done.write_text("ok")
    log.info("feature build finished")
