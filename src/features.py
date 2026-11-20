"""Feature extraction. Every feature is defined here (FEATURE_SPEC) BEFORE any train/val analysis.

Sets
  MFCC-summary  (52): mean+std over valid frames of 13 MFCC and of their 13 deltas       [PROJECT_CARD §12]
  Engineered    (16): domain-motivated spectral/temporal descriptors (FEATURE_SPEC below)
  Log-mel       (64 x 101): kept as a tensor for the Phase-3 CNN, also summarised for the ablation
  Diagnostics   (not used as inputs): rms_dbfs, duration, f0 -> only for colouring / channel diagnostics
"""
from __future__ import annotations

import librosa
import numpy as np

EPS = 1e-10

ENGINEERED_SPEC = [
    # name, domain logic (why it should carry information about the WORD, not the speaker)
    ("zcr_mean", "Zero-crossing rate: high for fricatives/aspiration (the /s/ of 'yes', 'stop'), low for vowels/nasals."),
    ("zcr_std", "Variation of ZCR in time: words that mix a fricative and a vowel ('yes', 'stop') vs. purely voiced ones ('no', 'down')."),
    ("centroid_mean", "Spectral centroid: brightness; separates fricative/burst-rich words from low-formant words."),
    ("centroid_std", "Centroid movement: formant glides (e.g. the diphthong in 'no', 'go')."),
    ("rolloff_mean", "85% spectral roll-off: where the energy ends (fricatives push it above 4 kHz)."),
    ("bandwidth_mean", "Spectral bandwidth: spread of energy around the centroid (noise-like vs. harmonic)."),
    ("flatness_mean", "Spectral flatness (log): noise-like (near 0 dB) vs. tonal/harmonic frames."),
    ("flux_mean", "Spectral flux: frame-to-frame change; plosive bursts ('d' in 'down', 't' in 'stop', 'p' in 'up')."),
    ("low_high_db", "Energy < 1 kHz vs. > 4 kHz (dB): voiced vowel/nasal energy vs. fricative energy."),
    ("spectral_tilt", "Slope of the long-term log spectrum: voice quality / channel colouring."),
    ("frame_dyn_db", "p95 - p10 of the frame level (dB) over valid frames: speech vs. steady noise (Phase-1 descriptor)."),
    ("active_frac", "Share of valid frames within 35 dB of the loudest frame."),
    ("active_dur_s", "Duration of the active span (s): word length at the acoustic level, independent of padding."),
    ("onset_s", "Time of the first active frame (s): reaction/trim position; a recording-protocol cue, not content."),
    ("temporal_centroid", "Energy-weighted time centre of the active span (0..1): early- vs. late-stressed words."),
    ("n_energy_peaks", "Number of local maxima of the smoothed energy envelope (syllable-count proxy)."),
]
ENGINEERED_NAMES = [n for n, _ in ENGINEERED_SPEC]


def mfcc_names(n=13):
    base = [f"mfcc{i}" for i in range(n)]
    return ([f"{b}_mean" for b in base] + [f"{b}_std" for b in base] +
            [f"d_{b}_mean" for b in base] + [f"d_{b}_std" for b in base])


MFCC_NAMES = mfcc_names()
LOGMEL_MS_NAMES = [f"lm{i}_mean" for i in range(64)] + [f"lm{i}_std" for i in range(64)]
DIAG_NAMES = ["rms_dbfs", "duration_s", "f0_median", "f0_voiced_frac"]


def _frame_mask(n_frames: int, hop: int, start: int, end: int) -> np.ndarray:
    centres = np.arange(n_frames) * hop
    m = (centres >= start) & (centres < end)
    if m.sum() < 3:                                     # degenerate clip: fall back to all frames
        m = np.ones(n_frames, bool)
    return m


def _smooth(v, w=5):
    k = np.ones(w) / w
    return np.convolve(v, k, mode="same")


def clip_features(x: np.ndarray, start: int, n_valid: int, cfg: dict, want_logmel=True) -> dict:
    """All features of ONE clip. x: float32 [target_len]; speech samples live in [start, start+n_valid)."""
    f = cfg["phase2"]["features"]
    sr = cfg["phase2"]["sr"]
    n_fft, hop = f["n_fft"], f["hop"]
    S = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop, window="hann", center=True)) ** 2   # power
    T = S.shape[1]
    mask = _frame_mask(T, hop, start, start + n_valid)

    mel40 = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=f["n_mels_mfcc"], fmin=f["fmin"], fmax=f["fmax"])
    logmel40 = librosa.power_to_db(mel40, ref=1.0, amin=1e-10, top_db=None)
    mf = librosa.feature.mfcc(S=logmel40, n_mfcc=f["n_mfcc"])
    dm = librosa.feature.delta(mf, width=f["delta_width"], mode="nearest")
    mfv, dmv = mf[:, mask], dm[:, mask]
    out = {"mfcc_vec": np.concatenate([mfv.mean(1), mfv.std(1), dmv.mean(1), dmv.std(1)]).astype(np.float64)}

    mel64 = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=f["n_mels_logmel"], fmin=f["fmin"], fmax=f["fmax"])
    lm = librosa.power_to_db(mel64, ref=1.0, amin=1e-10, top_db=None)
    out["logmel_ms"] = np.concatenate([lm[:, mask].mean(1), lm[:, mask].std(1)])      # 128-d summary (ablation)
    if want_logmel:
        out["logmel"] = lm.astype(np.float16)

    # ---- engineered
    mag = np.sqrt(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    zcr = librosa.feature.zero_crossing_rate(x, frame_length=n_fft, hop_length=hop, center=True)[0][:T]
    cen = librosa.feature.spectral_centroid(S=mag, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(S=mag, sr=sr)[0]
    roll = librosa.feature.spectral_rolloff(S=mag, sr=sr, roll_percent=f["rolloff_pct"])[0]
    flat = librosa.feature.spectral_flatness(S=mag)[0]
    flux = np.r_[0.0, np.sqrt((np.diff(np.log1p(mag * 1e3), axis=1) ** 2).sum(0))]
    lowb, highb = freqs < f["low_band_hz"], freqs >= f["high_band_hz"]
    e_low, e_high = S[lowb][:, mask].sum(), S[highb][:, mask].sum()
    low_high = 10 * np.log10((e_low + EPS) / (e_high + EPS))
    ltas = np.log10(S[:, mask].mean(1) + EPS)
    sel = (freqs >= 100) & (freqs <= 7500)
    tilt = np.polyfit(np.log10(freqs[sel]), ltas[sel], 1)[0]

    lev = 10 * np.log10(S.sum(0) / n_fft + EPS)                    # frame level (dB, arbitrary ref)
    lv = lev[mask]
    ref = lv.max()
    act = (lev >= ref - f["active_thresh_db"]) & mask
    idx = np.flatnonzero(act)
    if len(idx) == 0:
        idx = np.flatnonzero(mask)
        act = mask
    span = (idx.max() - idx.min() + 1)
    e = 10 ** (lev / 10) * act
    tc = (np.sum(np.arange(T) * e) / (e.sum() + EPS) - idx.min()) / max(span - 1, 1)
    env = _smooth(10 ** (lev / 10) * mask, 7)
    peaks = ((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > 10 ** ((ref - 20) / 10))).sum()

    def m(v):
        return float(np.mean(v[mask]))

    def s(v):
        return float(np.std(v[mask]))
    eng = {
        "zcr_mean": m(zcr), "zcr_std": s(zcr), "centroid_mean": m(cen), "centroid_std": s(cen),
        "rolloff_mean": m(roll), "bandwidth_mean": m(bw), "flatness_mean": float(np.mean(np.log10(flat[mask] + EPS))),
        "flux_mean": m(flux), "low_high_db": float(low_high), "spectral_tilt": float(tilt),
        "frame_dyn_db": float(np.percentile(lv, 95) - np.percentile(lv, 10)),
        "active_frac": float(act.sum() / mask.sum()), "active_dur_s": float(span * hop / sr),
        "onset_s": float(max(0, idx.min() * hop - start) / sr),
        "temporal_centroid": float(np.clip(tc, 0, 1)), "n_energy_peaks": float(peaks),
    }
    out["eng_vec"] = np.array([eng[n] for n in ENGINEERED_NAMES], dtype=np.float64)

    # ---- diagnostics (never inputs)
    seg = x[start:start + n_valid]
    rms = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)
    try:
        f0 = librosa.yin(seg, fmin=f["f0"]["fmin"], fmax=f["f0"]["fmax"], sr=sr,
                         frame_length=f["f0"]["frame_length"], hop_length=hop)
        # voiced proxy: frames in the active set (YIN has no voicing decision) and inside the f0 range
        lv_seg = lev[: len(f0)] if len(f0) <= T else lev
        ok = (lv_seg[:len(f0)] >= ref - 20) & (f0 > f["f0"]["fmin"] * 1.02) & (f0 < f["f0"]["fmax"] * 0.98)
        f0_med = float(np.median(f0[ok])) if ok.sum() >= 3 else np.nan
        f0_frac = float(ok.mean())
    except Exception:
        f0_med, f0_frac = np.nan, 0.0
    out["diag"] = np.array([rms, n_valid / sr, f0_med, f0_frac])
    return out
