"""Controlled augmentation / noise injection. Pure functions + explicit RNG => reproducible.

* `augment_train_clip`  : gain, time shift, optional noise. TRAIN ONLY (see `assert_train_only`).
* `add_noise`           : additive noise at an exact SNR (power ratio measured over the clip's valid samples).
* `noise_condition_set` : fixed-seed clean/noisy copies of VAL clips (diagnostic noise conditions).
Every call returns the drawn intensities so they can be logged (reports/phase2/tables/augmentation_log.csv).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def assert_train_only(splits) -> None:
    bad = set(map(str, splits)) - {"train"}
    if bad:
        raise ValueError(f"augmentation is allowed on the train split only, got {sorted(bad)}")


def white_noise(n: int, rng) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)


def pink_noise(n: int, rng) -> np.ndarray:
    """1/f power spectrum (amplitude ~ 1/sqrt(f)) via FFT shaping."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    spec = spec / np.sqrt(f)
    spec[0] = 0.0
    y = np.fft.irfft(spec, n)
    return (y / (np.std(y) + EPS)).astype(np.float32)


def babble_noise(pool: np.ndarray, n_valid: np.ndarray, n: int, k: int, rng, exclude: int | None = None) -> np.ndarray:
    """Sum of k other clips (their valid parts) from the same split pool, each randomly rolled."""
    idx = rng.choice(len(pool), size=min(k + 1, len(pool)), replace=False)
    idx = [i for i in idx if i != exclude][:k]
    y = np.zeros(n, dtype=np.float64)
    for i in idx:
        c = np.roll(pool[i, :n_valid[i]], int(rng.integers(0, max(1, n_valid[i]))))
        reps = int(np.ceil(n / len(c)))
        y += np.tile(c, reps)[:n]
    return (y / (np.std(y) + EPS)).astype(np.float32)


def add_noise(x: np.ndarray, n_valid: int, noise: np.ndarray, snr_db: float, start: int = 0) -> np.ndarray:
    """x + g*noise with  P_signal(valid samples) / P_noise(all samples) = 10^(snr/10). Noise covers the full
    1 s (background noise is also present in the silent/padded part)."""
    p_sig = float(np.mean(x[start:start + n_valid] ** 2)) + EPS
    p_noi = float(np.mean(noise ** 2)) + EPS
    g = np.sqrt(p_sig / (p_noi * 10 ** (snr_db / 10)))
    return (x + g * noise).astype(np.float32)


def measured_snr_db(clean: np.ndarray, noisy: np.ndarray, n_valid: int, start: int = 0) -> float:
    p_sig = np.mean(clean[start:start + n_valid] ** 2) + EPS
    p_noi = np.mean((noisy - clean) ** 2) + EPS
    return float(10 * np.log10(p_sig / p_noi))


def shift_zero_fill(x: np.ndarray, shift: int) -> np.ndarray:
    out = np.zeros_like(x)
    if shift >= 0:
        out[shift:] = x[:len(x) - shift] if shift else x
    else:
        out[:shift] = x[-shift:]
    return out


def augment_train_clip(x, n_valid, pool, pool_nvalid, i, cfg_aug, sr, rng):
    """One augmented copy of train clip i. Returns (x_aug, start, n_valid_aug, params dict)."""
    g_db = float(rng.uniform(*cfg_aug["gain_db"]))
    shift = int(round(rng.uniform(*cfg_aug["shift_ms"]) / 1000 * sr))
    y = shift_zero_fill(x, shift) * 10 ** (g_db / 20)
    start, end = max(0, shift), min(len(x), n_valid + shift)       # where the speech samples live now
    y_before = y.copy()
    ntype, snr = "none", np.nan
    if rng.random() < cfg_aug["noise_prob"]:
        ntype = str(rng.choice(cfg_aug["noise_types"]))
        snr = float(rng.choice(cfg_aug["noise_snr_db"]))
        if ntype == "white":
            nz = white_noise(len(x), rng)
        elif ntype == "pink":
            nz = pink_noise(len(x), rng)
        else:
            nz = babble_noise(pool, pool_nvalid, len(x), cfg_aug["babble_n"], rng, exclude=i)
        y = add_noise(y, end - start, nz, snr, start=start)
    params = {"gain_db": g_db, "shift_samples": shift, "noise_type": ntype, "snr_db": snr,
              "measured_snr_db": measured_snr_db(y_before, y, end - start, start) if ntype != "none" else np.nan}
    return y.astype(np.float32), start, end - start, params


def noise_condition_set(X, n_valid, cfg_nc, babble_k, seed):
    """Yield (name, noise_type, snr, X_noisy) for each VAL noise condition; fixed seed per condition."""
    for ti, t in enumerate(cfg_nc["types"]):
        for si, snr in enumerate(cfg_nc["snr_db"]):
            rng = np.random.default_rng(seed * 1000 + ti * 10 + si)
            Y = np.empty_like(X)
            for i in range(len(X)):
                if t == "white":
                    nz = white_noise(X.shape[1], rng)
                elif t == "pink":
                    nz = pink_noise(X.shape[1], rng)
                else:
                    nz = babble_noise(X, n_valid, X.shape[1], babble_k, rng, exclude=i)
                Y[i] = add_noise(X[i], int(n_valid[i]), nz, float(snr))
            yield f"{t}_{int(snr)}dB", t, float(snr), Y
