"""Blinded listening test used to decide what to do with low-energy clips (Phase 1, step 5).

    python -m src.energy_review prepare     # builds the blind review package under data/energy_review/
    ... you listen and fill data/energy_review/review_sheet.csv ...
    python -m src.energy_review score       # applies the FROZEN rule from configs/config.yaml

Design (frozen in the config before any listening):
  * clips are sampled per RMS level bin with a fixed seed (stratified; the top bin is a control of
    normal-level clips);
  * every clip is peak-normalised (loudness is not a cue), renamed clip_001.. and shuffled, so the
    listener sees neither the label nor the energy level;
  * for each clip the listener writes one of the 8 words or '?' (cannot tell);
  * a clip is "unusable" if the heard word differs from the label ('?' included);
  * rule: if some bin below the control has an unusable rate >= unusable_rate_threshold, the highest such
    bin's upper edge becomes the exclusion threshold; otherwise nothing is excluded for low energy.
    If the control bin's unusable rate exceeds control_max_unusable_rate the test is invalid.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

UNSURE = "?"


# ------------------------------------------------------------------------------ helpers
def bin_labels(edges) -> list[str]:
    return [f"<{edges[0]:g}"] + [f"{lo:g}..{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])] + [f">={edges[-1]:g}"]


def assign_bins(rms_dbfs, edges) -> np.ndarray:
    """Bin i covers [edges[i-1], edges[i]); bin 0 is below edges[0]; the last bin is >= edges[-1]."""
    return np.digitize(np.asarray(rms_dbfs, dtype=float), np.asarray(edges, dtype=float))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _read_table(path: Path) -> pd.DataFrame:
    # sep=None: tolerate ';' (Excel in some locales); utf-8-sig: tolerate the BOM Excel adds
    t = pd.read_csv(path, dtype=str, sep=None, engine="python", encoding="utf-8-sig")
    t.columns = [c.strip().lower() for c in t.columns]
    return t


# ------------------------------------------------------------------------------ prepare
def prepare(df: pd.DataFrame, extracted_dir: Path, out_dir: Path, edges, n_per_bin: int, seed: int,
            peak: float = 0.7, overwrite: bool = False) -> pd.DataFrame:
    out_dir, extracted_dir = Path(out_dir), Path(extracted_dir)
    sheet_p, key_p, clips = out_dir / "review_sheet.csv", out_dir / "review_key.csv", out_dir / "clips"
    if sheet_p.exists() and not overwrite:
        raise FileExistsError(f"{sheet_p} exists. It may hold your answers; use --overwrite only to start over.")
    if clips.exists():
        shutil.rmtree(clips)
    clips.mkdir(parents=True, exist_ok=True)

    d = df.copy()
    d["Bin"] = assign_bins(d["RMS_dBFS"], edges)
    pop = d["Bin"].value_counts().to_dict()
    rng = np.random.default_rng(seed)
    picks = []
    for b in range(len(edges) + 1):
        pool = d[d["Bin"] == b].sort_values(["Word_Class", "Filename"]).reset_index(drop=True)
        k = min(n_per_bin, len(pool))
        if k:
            picks.append(pool.iloc[np.sort(rng.choice(len(pool), size=k, replace=False))])
    sel = pd.concat(picks).reset_index(drop=True)
    sel = sel.iloc[rng.permutation(len(sel))].reset_index(drop=True)      # blind order
    sel["Clip_ID"] = [f"clip_{i + 1:03d}" for i in range(len(sel))]
    sel["Bin_Population"] = sel["Bin"].map(pop).astype(int)

    for _, r in sel.iterrows():
        x, sr = sf.read(extracted_dir / r["Word_Class"] / r["Filename"], dtype="float64", always_2d=True)
        x = x.mean(axis=1)
        m = float(np.max(np.abs(x))) if len(x) else 0.0
        if m > 0:
            x = x / m * peak
        sf.write(clips / f"{r['Clip_ID']}.wav", x, sr, subtype="PCM_16")

    pd.DataFrame({"clip_id": sel["Clip_ID"], "heard_word": "", "notes": ""}).to_csv(
        sheet_p, index=False, encoding="utf-8-sig")
    keep = ["Clip_ID", "Word_Class", "Filename", "RMS_dBFS", "Bin", "Bin_Population"]
    if "Frame_Dyn_dB" in sel:
        keep.insert(4, "Frame_Dyn_dB")
    sel[keep].to_csv(key_p, index=False)
    return sel[keep]


# ------------------------------------------------------------------------------ score
def score(sheet_path: Path, key_path: Path, words: list[str], edges, rule: dict) -> dict:
    thr = float(rule["unusable_rate_threshold"])
    ctrl_max = float(rule["control_max_unusable_rate"])
    sheet, key = _read_table(Path(sheet_path)), pd.read_csv(key_path, dtype={"Clip_ID": str, "Filename": str})
    if not {"clip_id", "heard_word"} <= set(sheet.columns):
        raise ValueError("review_sheet.csv must keep the columns clip_id, heard_word")
    sheet["clip_id"] = sheet["clip_id"].str.strip()
    heard = sheet["heard_word"].fillna("").str.strip().str.lower().replace({"؟": UNSURE})
    ok_vals = set(w.lower() for w in words) | {UNSURE}
    bad = sheet.loc[~heard.isin(ok_vals), "clip_id"].tolist()
    if bad:
        raise ValueError(f"{len(bad)} clip(s) have an empty/invalid heard_word (allowed: the 8 words or '?'): {bad[:15]}")
    sheet["heard_word"] = heard
    t = key.merge(sheet[["clip_id", "heard_word"]], left_on="Clip_ID", right_on="clip_id", how="left")
    if t["heard_word"].isna().any():
        raise ValueError("some clips from the key are missing in the sheet")
    t["Unusable"] = t["heard_word"] != t["Word_Class"].str.lower()
    t["Unsure"] = t["heard_word"] == UNSURE

    labels = bin_labels(edges)
    ctrl = len(edges)
    rows = []
    for b in range(len(edges) + 1):
        g = t[t["Bin"] == b]
        n, k = len(g), int(g["Unusable"].sum())
        lo, hi = wilson(k, n)
        rows.append({"bin": b, "label": labels[b] + (" (control)" if b == ctrl else ""),
                     "population": int(g["Bin_Population"].iloc[0]) if n else 0,
                     "n_listened": n, "n_unusable": k, "n_unsure": int(g["Unsure"].sum()),
                     "rate": (k / n) if n else float("nan"), "ci_lo": lo, "ci_hi": hi})
    bins = pd.DataFrame(rows)

    c = bins.loc[bins["bin"] == ctrl].iloc[0]
    control_rate = float(c["rate"]) if c["n_listened"] > 0 else float("nan")
    valid = bool(c["n_listened"] > 0 and control_rate <= ctrl_max)

    exclude, thr_dbfs = False, None
    if valid:
        cand = bins[(bins["bin"] < ctrl) & (bins["n_listened"] > 0) & (bins["rate"] >= thr)]
        if len(cand):
            exclude, thr_dbfs = True, float(edges[int(cand["bin"].max())])

    if not valid:
        text = (f"نرخ نامطابقت در بازه‌ی کنترل ({control_rate:.0%}) بیش از سقف {ctrl_max:.0%} است یا کنترل نمونه ندارد؛ "
                "آزمون شنیدن معتبر نیست و باید دوباره انجام شود. هنوز هیچ فایلی به دلیل انرژی حذف نمی‌شود.")
    elif exclude:
        text = (f"طبق قاعده‌ی پیش‌ثبت‌شده، در بالاترین بازه‌ی دارای نرخ غیرقابل‌استفاده ≥ {thr:.0%} (لبه‌ی بالایی {thr_dbfs:g} dBFS) "
                f"دست‌کم نیمی از کلیپ‌ها برای شنونده قابل‌تشخیص نبودند؛ بنابراین همه‌ی فایل‌های با RMS کمتر از {thr_dbfs:g} dBFS "
                "از train/val/test حذف می‌شوند.")
    else:
        text = (f"در هیچ بازه‌ای زیر کنترل نرخ غیرقابل‌استفاده به {thr:.0%} نرسید؛ طبق قاعده‌ی پیش‌ثبت‌شده هیچ فایلی به‌خاطر "
                "انرژی حذف نمی‌شود و پرچم Low_Energy فقط برای تحلیل حساسیت (فاز ۴) می‌ماند.")

    md = ["| بازه‌ی RMS (dBFS) | فایل‌های داده | کلیپ‌های شنیده‌شده | غیرقابل‌استفاده | نرخ (فاصله‌ی اطمینان ۹۵٪ ویلسون) |",
          "|---|---|---|---|---|"]
    for _, r in bins.iterrows():
        rate = "—" if r["n_listened"] == 0 else f"{r['rate']:.0%} ({r['ci_lo']:.0%}–{r['ci_hi']:.0%})"
        md.append(f"| {r['label']} | {r['population']} | {r['n_listened']} | {r['n_unusable']} | {rate} |")

    return {
        "n_listened": int(len(t)),
        "protocol_valid": valid,
        "control_unusable_rate": control_rate,
        "unusable_rate_threshold": thr,
        "recommended_exclude_low_energy": exclude,
        "recommended_threshold_dbfs": thr_dbfs,
        "decision_text": text,
        "table_md": "\n".join(md),
        "bins": bins.to_dict("records"),
        "_joined": t,
    }


# ------------------------------------------------------------------------------ CLI
def main():
    from .config import load_config, resolve

    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "score"])
    ap.add_argument("--overwrite", action="store_true", help="prepare: start over (destroys existing answers)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    er = cfg["energy_review"]
    out_dir = resolve(cfg, er["out_dir"] + "/x").parent
    edges = er["bin_edges_dbfs"]

    if a.action == "prepare":
        df = pd.read_csv(resolve(cfg, cfg["paths"]["audit_csv"]),
                         dtype={"Speaker_ID": str, "Filename": str, "Word_Class": str})
        extracted = Path(cfg["_root"]) / cfg["data"]["base_dir"] / "mini_speech_commands"
        sel = prepare(df, extracted, out_dir, edges, er["n_per_bin"], er["seed"], er["peak_normalize_to"], a.overwrite)
        print(f"\nPrepared {len(sel)} blind clips in: {out_dir / 'clips'}")
        print(f"Fill the heard_word column of: {out_dir / 'review_sheet.csv'}")
        print("Allowed values: down go left no right stop up yes   or   ?  (cannot tell).")
        print("Do NOT open review_key.csv until you have finished.")
    else:
        res = score(out_dir / "review_sheet.csv", out_dir / "review_key.csv",
                    cfg["data"]["expected_classes"], edges, er)
        joined = res.pop("_joined")
        resolve(cfg, cfg["paths"]["tables_dir"] + "/x")
        joined.to_csv(resolve(cfg, cfg["paths"]["tables_dir"] + "/listening_test.csv"), index=False)
        with open(resolve(cfg, cfg["paths"]["listening_json"]), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        print(res["table_md"], "\n")
        print(res["decision_text"], "\n")
        if res["protocol_valid"]:
            print("Edit configs/config.yaml (section audit) to:")
            if res["recommended_exclude_low_energy"]:
                print(f"  low_energy_rms_dbfs: {res['recommended_threshold_dbfs']}")
                print("  exclude_low_energy: true")
            else:
                print("  exclude_low_energy: false     # keep low_energy_rms_dbfs as is (descriptive flag only)")
            print("then run: python main.py   (no --force needed)")


if __name__ == "__main__":
    main()
