"""Signal-level audit: per-file features, data-quality counts, exclusion flags, speaker statistics."""
import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

logger = logging.getLogger(__name__)

# Speech Commands naming scheme: <speaker hash>_nohash_<utterance index>.wav
FILENAME_RE = re.compile(r"^([0-9a-f]+)_nohash_(\d+)\.wav$")
EPS = 1e-12
FLOOR_DB = -100.0        # frame levels are floored here so exact-zero frames cannot blow up the dynamics


class SpeechDataAuditor:
    def __init__(self, cfg: dict, extracted_dir: Path):
        self.cfg = cfg
        self.acfg = cfg["audit"]
        self.extracted_dir = Path(extracted_dir)
        if not self.extracted_dir.is_dir():
            raise FileNotFoundError(f"{self.extracted_dir} not found; run the data loader first.")
        # sorted -> deterministic order on every OS
        self.words = sorted(p.name for p in self.extracted_dir.iterdir() if p.is_dir())
        if not self.words:
            raise ValueError("No class folders found.")
        self.corrupt = pd.DataFrame(columns=["Word_Class", "Filename", "Error"])

    # ------------------------------------------------------------ per file
    def _frame_dyn_db(self, x: np.ndarray) -> float:
        """Frame dynamics = p95 - p10 of the frame RMS level (dBFS). Speech has large dynamics
        (loud syllables vs. pauses); stationary noise / digital hiss has small dynamics."""
        L, H = int(self.acfg.get("frame_len", 400)), int(self.acfg.get("frame_hop", 160))
        if len(x) < L:
            return 0.0
        n = 1 + (len(x) - L) // H
        idx = np.arange(L)[None, :] + H * np.arange(n)[:, None]
        lev = 20 * np.log10(np.sqrt((x[idx] ** 2).mean(axis=1)) + EPS)
        lev = np.maximum(lev, FLOOR_DB)
        return float(np.percentile(lev, 95) - np.percentile(lev, 10))

    def _features(self, path: Path) -> dict:
        info = sf.info(path)
        x, sr = sf.read(path, dtype="float64", always_2d=True)   # (n, channels)
        channels = x.shape[1]
        x = x.mean(axis=1)                                       # mono mix (data should be mono)
        n = len(x)
        m = FILENAME_RE.match(path.name)
        row = {
            "Word_Class": path.parent.name,
            "Filename": path.name,
            "Speaker_ID": m.group(1) if m else path.name.split("_")[0],
            "Utt_Index": int(m.group(2)) if m else -1,
            "Name_OK": bool(m),
            "Sample_Rate": sr,
            "Channels": channels,
            "Subtype": info.subtype,
            "N_Samples": n,
            "Duration_sec": n / sr if sr else 0.0,
            "Audio_MD5": hashlib.md5(x.astype(np.float32).tobytes()).hexdigest(),
        }
        if n == 0:
            row.update(Peak=0.0, RMS=0.0, RMS_dBFS=20 * np.log10(EPS), Clip_Frac=0.0, DC_Offset=0.0,
                       ZCR=0.0, Spec_Centroid_Hz=0.0, Frame_Dyn_dB=0.0)
            return row
        a = np.abs(x)
        rms = float(np.sqrt(np.mean(x ** 2)))
        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        row.update(
            Peak=float(a.max()),
            RMS=rms,
            RMS_dBFS=float(20 * np.log10(rms + EPS)),
            Clip_Frac=float(np.mean(a >= self.acfg["clip_threshold"])),
            DC_Offset=float(x.mean()),
            ZCR=float(np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))) if n > 1 else 0.0,
            Spec_Centroid_Hz=float((freqs * spec).sum() / (spec.sum() + EPS)),
            Frame_Dyn_dB=self._frame_dyn_db(x),
        )
        return row

    def run(self) -> pd.DataFrame:
        rows, corrupt = [], []
        for w in self.words:
            for p in sorted((self.extracted_dir / w).glob("*.wav")):
                try:
                    rows.append(self._features(p))
                except Exception as e:                       # noqa: BLE001 - we log every failure
                    corrupt.append({"Word_Class": w, "Filename": p.name, "Error": repr(e)})
        df = pd.DataFrame(rows)
        self.corrupt = pd.DataFrame(corrupt, columns=["Word_Class", "Filename", "Error"])
        df = self.add_flags(df)
        logger.info("Audited %d files, %d unreadable.", len(df), len(self.corrupt))
        return df

    def add_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """(Re)compute threshold-based flags and the exclusion decision from the CURRENT config.
        Also used on a cached audit table, so changing a threshold never needs a full re-audit."""
        a = self.acfg
        df = df.copy()
        df["Low_Energy"] = df["RMS_dBFS"] < a["low_energy_rms_dbfs"]
        df["Clipped_File"] = df["Clip_Frac"] > a["clipped_file_frac"]
        df["Very_Short"] = df["Duration_sec"] < a["very_short_sec"]

        # duplicates: identical waveform. Cross-class duplicates = label conflict -> all members excluded.
        # Within-class duplicates: keep the first file (sorted by class, file name), exclude the rest.
        df["Dup_CrossClass"] = df.groupby("Audio_MD5")["Word_Class"].transform("nunique") > 1
        srt = df.sort_values(["Word_Class", "Filename"])
        extra = srt.duplicated(["Audio_MD5", "Word_Class"], keep="first").reindex(df.index)
        df["Dup_Extra"] = extra & ~df["Dup_CrossClass"]

        conds = pd.DataFrame({
            "zero_length": df["N_Samples"] == 0,
            "digital_silence": df["Peak"] == 0,
            "dup_cross_class": df["Dup_CrossClass"],
            "dup_extra": df["Dup_Extra"],
            "low_energy": df["Low_Energy"] & bool(a.get("exclude_low_energy", False)),
        })
        df["Exclude_Reason"] = ["|".join(k for k, v in r.items() if v) for r in conds.to_dict("records")]
        df["Exclude"] = df["Exclude_Reason"] != ""
        return df

    # ------------------------------------------------------------- summary
    def summarize(self, df: pd.DataFrame) -> dict:
        a = self.acfg
        s: dict = {}
        s["n_files"] = int(len(df))
        s["n_unreadable"] = int(len(self.corrupt))
        s["n_classes"] = int(df["Word_Class"].nunique())
        s["files_per_class"] = df["Word_Class"].value_counts().sort_index().to_dict()
        s["sample_rates"] = df["Sample_Rate"].value_counts().to_dict()
        s["channels"] = df["Channels"].value_counts().to_dict()
        s["subtypes"] = df["Subtype"].value_counts().to_dict()
        s["n_nonstandard_filenames"] = int((~df["Name_OK"]).sum())
        s["missing_values_per_column"] = df.isna().sum().astype(int).to_dict()
        s["n_missing_values_total"] = int(df.isna().sum().sum())

        # durations
        d = df["Duration_sec"]
        s["duration"] = {k: float(v) for k, v in d.describe().items()}
        s["n_shorter_than_1s"] = int((df["N_Samples"] < a["target_len_samples"]).sum())
        s["n_longer_than_1s"] = int((df["N_Samples"] > a["target_len_samples"]).sum())
        s["n_exactly_1s"] = int((df["N_Samples"] == a["target_len_samples"]).sum())
        s["n_very_short"] = int(df["Very_Short"].sum())
        s["n_zero_length"] = int((df["N_Samples"] == 0).sum())
        s["frac_shorter_than_1s_per_class"] = (
            (df["N_Samples"] < a["target_len_samples"]).groupby(df["Word_Class"]).mean().round(4).to_dict())

        # energy / clipping
        fin = df.loc[np.isfinite(df["RMS_dBFS"]), "RMS_dBFS"]
        s["rms_dbfs"] = {k: float(v) for k, v in fin.describe().items()}
        s["n_digital_silence"] = int((df["Peak"] == 0).sum())
        s["low_energy_threshold_dbfs"] = a["low_energy_rms_dbfs"]
        s["exclude_low_energy"] = bool(a.get("exclude_low_energy", False))
        s["n_low_energy"] = int(df["Low_Energy"].sum())
        s["low_energy_sweep"] = {str(t): int((df["RMS_dBFS"] < t).sum()) for t in a["low_energy_sweep_dbfs"]}
        s["low_energy_per_class"] = df.groupby("Word_Class")["Low_Energy"].sum().astype(int).to_dict()
        s["frame_dyn_db_median_low_energy"] = float(df.loc[df["Low_Energy"], "Frame_Dyn_dB"].median()) \
            if df["Low_Energy"].any() else float("nan")
        s["frame_dyn_db_median_other"] = float(df.loc[~df["Low_Energy"], "Frame_Dyn_dB"].median())
        s["n_files_with_any_clipping"] = int((df["Clip_Frac"] > 0).sum())
        s["n_clipped_files"] = int(df["Clipped_File"].sum())
        s["clipped_per_class"] = df.groupby("Word_Class")["Clipped_File"].sum().astype(int).to_dict()
        s["abs_dc_offset_max"] = float(df["DC_Offset"].abs().max())

        # Tukey outliers (1.5*IQR) on the two main continuous variables
        def tukey(series):
            q1, q3 = series.quantile([0.25, 0.75])
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            return {"lower_fence": float(lo), "upper_fence": float(hi),
                    "n_low": int((series < lo).sum()), "n_high": int((series > hi).sum())}
        s["tukey_outliers"] = {"RMS_dBFS": tukey(fin), "Duration_sec": tukey(d)}

        # duplicates (identical waveform)
        dup = df[df.duplicated("Audio_MD5", keep=False)]
        s["n_files_in_duplicate_groups"] = int(len(dup))
        s["n_duplicate_groups"] = int(dup["Audio_MD5"].nunique())
        s["n_duplicate_groups_across_classes"] = int(
            (dup.groupby("Audio_MD5")["Word_Class"].nunique() > 1).sum()) if len(dup) else 0

        # exclusions (decisions frozen in the config; applied identically to every split)
        s["n_excluded"] = int(df["Exclude"].sum())
        reasons = df.loc[df["Exclude"], "Exclude_Reason"].str.split("|").explode().value_counts()
        s["exclude_reason_counts"] = reasons.to_dict() if len(reasons) else {"none": 0}

        # speakers
        sc = df["Speaker_ID"].value_counts()
        classes_per_spk = df.groupby("Speaker_ID")["Word_Class"].nunique()
        s["n_speakers"] = int(sc.size)
        s["files_per_speaker"] = {k: float(v) for k, v in sc.describe().items()}
        s["n_speakers_with_1_file"] = int((sc == 1).sum())
        s["n_speakers_in_ge2_classes"] = int((classes_per_spk >= 2).sum())
        s["frac_speakers_in_ge2_classes"] = float((classes_per_spk >= 2).mean())
        s["n_speakers_in_all_classes"] = int((classes_per_spk == df["Word_Class"].nunique()).sum())
        s["frac_files_from_speakers_in_ge2_classes"] = float(
            df["Speaker_ID"].map(classes_per_spk).ge(2).mean())
        s["top5_speaker_share_of_files"] = float(sc.head(5).sum() / len(df))
        s["max_files_one_speaker"] = int(sc.max())
        s["speakers_repeating_same_word"] = int(
            (df.groupby(["Speaker_ID", "Word_Class"]).size() > 1).groupby("Speaker_ID").any().sum())

        # is low energy a property of a few speakers/recordings? (descriptive; informs leakage risk #7)
        le = df[df["Low_Energy"]]
        top4 = le["Speaker_ID"].value_counts().head(4)
        s["low_energy_n_speakers"] = int(le["Speaker_ID"].nunique())
        s["low_energy_top4_speakers_files"] = int(top4.sum())
        s["low_energy_top4_speakers_share"] = float(top4.sum() / len(le)) if len(le) else float("nan")
        spk = df.groupby("Speaker_ID")["RMS_dBFS"].agg(["mean", "size"])
        s["rms_between_speaker_sd_db"] = float(spk.loc[spk["size"] >= 5, "mean"].std())
        s["rms_overall_sd_db"] = float(df["RMS_dBFS"].std())

        # univariate table (centre, spread, shape); excess kurtosis
        cols = [("Duration_sec", "طول (ثانیه)"), ("RMS_dBFS", "RMS (dBFS)"), ("Peak", "Peak"),
                ("Clip_Frac", "Clip_Frac"), ("ZCR", "ZCR"), ("Spec_Centroid_Hz", "مرکز طیفی (Hz)"),
                ("Frame_Dyn_dB", "Frame_Dyn_dB")]
        rows = ["| متغیر | میانگین | میانه | انحراف معیار | IQR | چولگی | کشیدگی (اضافی) |", "|---|---|---|---|---|---|---|"]
        for c, lab in cols:
            x = df[c][np.isfinite(df[c])]
            q1, q3 = x.quantile([0.25, 0.75])
            rows.append(f"| {lab} | {x.mean():.4g} | {x.median():.4g} | {x.std():.4g} | {q3 - q1:.4g} | "
                        f"{x.skew():.2f} | {x.kurt():.2f} |")
        s["univariate_md"] = "\n".join(rows)

        # how much of each variable's variance is between classes (eta^2 = SS_between / SS_total)?
        eta = {}
        for c in ["Duration_sec", "RMS_dBFS", "ZCR", "Spec_Centroid_Hz"]:
            x = df[c][np.isfinite(df[c])]
            ssb = sum(len(g) * (g.mean() - x.mean()) ** 2 for _, g in x.groupby(df["Word_Class"]))
            eta[c] = float(ssb / ((x - x.mean()) ** 2).sum())
        s["eta_squared_by_class"] = eta
        pc = df.groupby("Word_Class").agg(zcr=("ZCR", "median"), cen=("Spec_Centroid_Hz", "median"),
                                          rms=("RMS_dBFS", "median"), dur=("Duration_sec", "mean"))
        rows = ["| کلاس | میانه‌ی ZCR | میانه‌ی مرکز طیفی (Hz) | میانه‌ی RMS (dBFS) | میانگین طول (ثانیه) |", "|---|---|---|---|---|"]
        rows += [f"| {w} | {r.zcr:.3f} | {r.cen:.0f} | {r.rms:.1f} | {r.dur:.3f} |" for w, r in pc.iterrows()]
        s["per_class_md"] = "\n".join(rows)

        # trivial baselines on the whole (unsplit) data; recomputed on the real test split later
        p = df["Word_Class"].value_counts(normalize=True)
        k = len(p)
        pm = float(p.max())
        s["majority_class"] = p.idxmax()
        s["majority_accuracy_full_data"] = pm
        s["majority_macro_f1_full_data"] = float((2 * pm / (1 + pm)) / k)
        return s

    # -------------------------------------------------------------- tables
    def save_tables(self, df: pd.DataFrame, summary: dict, tables_dir: Path, summary_json: Path) -> None:
        tables_dir.mkdir(parents=True, exist_ok=True)
        df.groupby("Word_Class").agg(
            n=("Filename", "size"), n_speakers=("Speaker_ID", "nunique"),
            dur_mean=("Duration_sec", "mean"), dur_min=("Duration_sec", "min"),
            rms_dbfs_median=("RMS_dBFS", "median"), n_low_energy=("Low_Energy", "sum"),
            n_clipped=("Clipped_File", "sum"), n_excluded=("Exclude", "sum"),
        ).round(4).to_csv(tables_dir / "per_class_summary.csv")
        df[df["Low_Energy"]].sort_values("RMS_dBFS")[
            ["Word_Class", "Filename", "Duration_sec", "Peak", "RMS_dBFS", "Frame_Dyn_dB"]
        ].to_csv(tables_dir / "low_energy_files.csv", index=False)
        df[df["Clipped_File"]].sort_values("Clip_Frac", ascending=False)[
            ["Word_Class", "Filename", "Peak", "Clip_Frac"]
        ].to_csv(tables_dir / "clipped_files.csv", index=False)
        df[df["Exclude"]][["Word_Class", "Filename", "Speaker_ID", "Exclude_Reason"]].to_csv(
            tables_dir / "excluded_files.csv", index=False)
        dups = df[df["Dup_CrossClass"] | df.duplicated("Audio_MD5", keep=False)].sort_values(
            ["Audio_MD5", "Word_Class", "Filename"])
        dups[["Audio_MD5", "Word_Class", "Filename", "Speaker_ID", "Dup_CrossClass"]].to_csv(
            tables_dir / "duplicate_files.csv", index=False)
        self.corrupt.to_csv(tables_dir / "unreadable_files.csv", index=False)
        pd.crosstab(df["Speaker_ID"], df["Word_Class"]).to_csv(tables_dir / "speaker_by_class_counts.csv")

        def conv(o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)): return bool(o)
            raise TypeError(type(o))
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=conv)
