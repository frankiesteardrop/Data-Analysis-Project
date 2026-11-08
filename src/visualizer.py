"""All Phase-1 figures. Every function SAVES a PNG (no plt.show) so figures are reproducible files."""
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

DPI = 200
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


class SpeechDataVisualizer:
    def __init__(self, cfg: dict, df: pd.DataFrame, extracted_dir: Path, fig_dir: Path):
        self.cfg, self.df = cfg, df
        self.e = cfg["eda"]
        self.a = cfg["audit"]
        self.root = Path(extracted_dir)
        self.fig_dir = Path(fig_dir)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.words = sorted(df["Word_Class"].unique())
        self.seed = cfg["seed"]

    # ----------------------------------------------------------------- utils
    def _save(self, fig, name: str) -> Path:
        p = self.fig_dir / name
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        return p

    def _load(self, word: str, fn: str):
        y, sr = sf.read(self.root / word / fn, dtype="float32")
        return (y if y.ndim == 1 else y.mean(axis=1)), sr

    def _logmel(self, y, sr, ref=np.max):
        m = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.e["n_fft"], hop_length=self.e["hop_length"],
                                           n_mels=self.e["n_mels"], fmax=self.e["fmax"])
        return librosa.power_to_db(m, ref=ref, top_db=80.0 if ref is np.max else None, amin=1e-10)

    # ------------------------------------------------------------- figure 1
    def plot_data_flow(self):
        fig, ax = plt.subplots(figsize=(14, 3)); ax.axis("off"); ax.grid(False)
        labels = ["TensorFlow\nmini_speech_commands.zip\n(SHA-256 verified)",
                  "Extracted data\n8 class folders\n<speaker>_nohash_<k>.wav",
                  "Per-file audit\nduration, SR, RMS dBFS,\nclipping, MD5",
                  "audit_table.csv +\naudit_summary.json",
                  "Speaker-disjoint split\ntrain / val / test\n+ leakage check"]
        xs = [0.10, 0.30, 0.50, 0.70, 0.90]
        for txt, x in zip(labels, xs):
            ax.text(x, 0.5, txt, ha="center", va="center", fontsize=8.5, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", fc="#eef3fb", ec="#4a6fa5"))
        for x in xs[:-1]:
            ax.annotate("", xy=(x + 0.125, 0.5), xytext=(x + 0.10, 0.5), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#4a6fa5"))
        ax.set_title("Phase 1 data flow", fontweight="bold")
        return self._save(fig, "fig01_data_flow.png")

    # ------------------------------------------------------------- figure 2
    def plot_class_counts(self):
        c = self.df["Word_Class"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(c.index, c.values, color="#4a6fa5")
        for i, v in enumerate(c.values): ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Number of clips"); ax.set_title("Class counts (full dataset, before any split)")
        ax.set_ylim(0, c.max() * 1.12)
        return self._save(fig, "fig02_class_counts.png")

    # ------------------------------------------------------------- figure 3
    def plot_duration(self):
        d = self.df
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(d["Duration_sec"], bins=50, color="#4a6fa5")
        axes[0].axvline(1.0, color="red", ls="--", label="1.0 s target")
        axes[0].set_xlabel("Duration (s)"); axes[0].set_ylabel("Clips"); axes[0].set_yscale("log")
        axes[0].set_title("Clip duration (log y)"); axes[0].legend()
        frac = (d["N_Samples"] < self.a["target_len_samples"]).groupby(d["Word_Class"]).mean().sort_index()
        axes[1].bar(frac.index, frac.values * 100, color="#c0504d")
        axes[1].set_ylabel("% of clips shorter than 1 s"); axes[1].set_title("Share of short clips per class")
        return self._save(fig, "fig03_duration.png")

    # ------------------------------------------------------------- figure 4
    def plot_energy(self):
        d = self.df[np.isfinite(self.df["RMS_dBFS"])]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].hist(d["RMS_dBFS"], bins=80, color="#4a6fa5")
        for t in self.a["low_energy_sweep_dbfs"]:
            axes[0].axvline(t, color="grey", ls=":", lw=1)
        axes[0].axvline(self.a["low_energy_rms_dbfs"], color="red", ls="--",
                        label=f"descriptive flag / listening-test cut-off: {self.a['low_energy_rms_dbfs']:.0f} dBFS")
        axes[0].set_xlabel("RMS level (dBFS)"); axes[0].set_ylabel("Clips"); axes[0].legend()
        axes[0].set_title("Energy distribution with threshold sweep (dotted)")
        data = [d.loc[d["Word_Class"] == w, "RMS_dBFS"] for w in self.words]
        axes[1].boxplot(data, tick_labels=self.words, showfliers=True, flierprops=dict(markersize=2))
        axes[1].set_ylabel("RMS level (dBFS)"); axes[1].set_title("Energy by class")
        return self._save(fig, "fig04_energy.png")

    def plot_energy_dynamics(self):
        """RMS level vs. frame dynamics: speech has large dynamics, noise-only/hiss has small dynamics."""
        if "Frame_Dyn_dB" not in self.df:
            return None
        d = self.df[np.isfinite(self.df["RMS_dBFS"]) & np.isfinite(self.df["Frame_Dyn_dB"])]
        lo, ok = d[d["Low_Energy"]], d[~d["Low_Energy"]]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.scatter(ok["RMS_dBFS"], ok["Frame_Dyn_dB"], s=4, alpha=0.3, color="#4a6fa5", label="at/above threshold")
        ax.scatter(lo["RMS_dBFS"], lo["Frame_Dyn_dB"], s=9, alpha=0.7, color="red", label="Low_Energy flag")
        ax.axvline(self.a["low_energy_rms_dbfs"], color="red", ls="--", lw=1)
        ax.set_xlabel("RMS level (dBFS)"); ax.set_ylabel("Frame dynamics: p95 - p10 of 25 ms frame RMS (dB)")
        ax.set_title("Energy vs. dynamics (low dynamics at low level = noise/hiss rather than speech)")
        ax.legend()
        return self._save(fig, "fig11_energy_vs_dynamics.png")

    # ------------------------------------------------------------- figure 5
    def plot_speakers(self):
        sc = self.df["Speaker_ID"].value_counts()
        cps = self.df.groupby("Speaker_ID")["Word_Class"].nunique().value_counts().sort_index()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].hist(sc, bins=range(1, sc.max() + 2), color="#4a6fa5", edgecolor="white")
        axes[0].axvline(sc.median(), color="darkorange", ls="--", label=f"median = {sc.median():.0f}")
        axes[0].axvline(sc.mean(), color="firebrick", ls="--", label=f"mean = {sc.mean():.2f}")
        axes[0].set_yscale("log"); axes[0].legend()
        axes[0].set_xlabel("Clips per speaker"); axes[0].set_ylabel("Speakers (log)")
        axes[0].set_title("Clips per speaker")
        axes[1].bar(cps.index.astype(str), cps.values, color="#4a6fa5")
        axes[1].set_xlabel("Number of different keywords a speaker uttered"); axes[1].set_ylabel("Speakers")
        axes[1].set_title("Keyword coverage per speaker")
        return self._save(fig, "fig05_speakers.png")

    def plot_speaker_class_heatmap(self, top_n: int = 30):
        ct = pd.crosstab(self.df["Speaker_ID"], self.df["Word_Class"])
        top = ct.loc[self.df["Speaker_ID"].value_counts().head(top_n).index]
        fig, ax = plt.subplots(figsize=(7, 8))
        im = ax.imshow(top.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(top.shape[1]), top.columns, rotation=45); ax.set_yticks([])
        ax.set_ylabel(f"Top-{top_n} speakers by clip count (IDs hidden)"); ax.grid(False)
        fig.colorbar(im, ax=ax, label="clips"); ax.set_title("Speaker x keyword counts")
        return self._save(fig, "fig06_speaker_class_heatmap.png")

    # ------------------------------------------------------------- figure 7
    def plot_multivariate(self):
        d = self.df[np.isfinite(self.df["RMS_dBFS"])]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        axes[0, 0].boxplot([d.loc[d["Word_Class"] == w, "Spec_Centroid_Hz"] for w in self.words],
                           tick_labels=self.words, flierprops=dict(markersize=2))
        axes[0, 0].set_ylabel("Spectral centroid (Hz)"); axes[0, 0].set_title("Spectral centroid by class")
        axes[0, 1].boxplot([d.loc[d["Word_Class"] == w, "ZCR"] for w in self.words],
                           tick_labels=self.words, flierprops=dict(markersize=2))
        axes[0, 1].set_ylabel("Zero-crossing rate"); axes[0, 1].set_title("ZCR by class")
        for w in self.words:
            s = d[d["Word_Class"] == w]
            axes[1, 0].scatter(s["RMS_dBFS"], s["Spec_Centroid_Hz"], s=4, alpha=0.4, label=w)
        axes[1, 0].set_xlabel("RMS (dBFS)"); axes[1, 0].set_ylabel("Spectral centroid (Hz)")
        axes[1, 0].legend(markerscale=3, fontsize=7, ncol=2); axes[1, 0].set_title("Energy vs. spectral centroid")
        cols = ["Duration_sec", "RMS_dBFS", "Peak", "Clip_Frac", "ZCR", "Spec_Centroid_Hz"]
        corr = d[cols].corr(method="spearman")
        im = axes[1, 1].imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm"); axes[1, 1].grid(False)
        axes[1, 1].set_xticks(range(len(cols)), cols, rotation=45, ha="right"); axes[1, 1].set_yticks(range(len(cols)), cols)
        for i in range(len(cols)):
            for j in range(len(cols)):
                axes[1, 1].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        axes[1, 1].set_title("Spearman correlation"); fig.colorbar(im, ax=axes[1, 1])
        fig.tight_layout()
        return self._save(fig, "fig07_multivariate.png")

    # ------------------------------------------------------------- figure 8
    def plot_class_examples(self):
        """For each class: n random clips (seeded) -> waveform + log-mel. Padding region shaded."""
        n = self.e["n_samples_per_class"]
        paths = []
        for w in self.words:
            sub = self.df[(self.df["Word_Class"] == w) & (~self.df["Low_Energy"])]
            samp = sub.sample(n=n, random_state=self.seed)
            fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 6), squeeze=False)
            fig.suptitle(f"'{w}' - {n} random clips (seed {self.seed}); low-energy files excluded from sampling",
                         fontweight="bold")
            for j, (_, r) in enumerate(samp.iterrows()):
                y, sr = self._load(w, r["Filename"]); end = len(y) / sr
                t = np.arange(len(y)) / sr
                axes[0, j].plot(t, y, lw=0.6, color="teal"); axes[0, j].set_xlim(0, 1); axes[0, j].set_ylim(-1, 1)
                axes[0, j].set_title(f"dur {end:.2f}s | RMS {r['RMS_dBFS']:.1f} dBFS", fontsize=9)
                lm = self._logmel(y, sr)
                librosa.display.specshow(lm, sr=sr, hop_length=self.e["hop_length"], x_axis="time", y_axis="mel",
                                         fmax=self.e["fmax"], ax=axes[1, j]); axes[1, j].set_xlim(0, 1)
                axes[1, j].grid(False)
                if end < 1.0:
                    for ax in axes[:, j]:
                        ax.axvspan(end, 1.0, color="lightgray", alpha=0.5); ax.axvline(end, color="red", ls="--", lw=1.2)
                axes[0, j].set_xlabel("")
            axes[0, 0].set_ylabel("Amplitude"); axes[1, 0].set_ylabel("Hz (mel)")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            paths.append(self._save(fig, f"fig08_examples_{w}.png"))
        return paths

    def plot_mean_logmel(self):
        """Class-mean log-mel after zero-padding to 1 s (absolute dB, shared colour scale)."""
        T = self.a["target_len_samples"]
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
        mats = {}
        for w in self.words:
            sub = self.df[(self.df["Word_Class"] == w) & (~self.df["Low_Energy"])]
            samp = sub.sample(n=min(100, len(sub)), random_state=self.seed)
            acc = []
            for fn in samp["Filename"]:
                y, sr = self._load(w, fn)
                y = np.pad(y[:T], (0, max(0, T - len(y))))
                acc.append(self._logmel(y, sr, ref=1.0))
            mats[w] = np.mean(acc, axis=0)
        vmin = min(m.min() for m in mats.values()); vmax = max(m.max() for m in mats.values())
        for ax, w in zip(axes.ravel(), self.words):
            im = librosa.display.specshow(mats[w], sr=16000, hop_length=self.e["hop_length"], x_axis="time",
                                          y_axis="mel", fmax=self.e["fmax"], ax=ax, vmin=vmin, vmax=vmax)
            ax.set_title(w); ax.grid(False)
        fig.colorbar(im, ax=axes, label="mean log-mel power (dB, abs.)")
        fig.suptitle("Class-mean log-mel spectrogram (100 random clips per class, zero-padded to 1 s)", fontweight="bold")
        return self._save(fig, "fig09_mean_logmel_per_class.png")

    # ------------------------------------------------------------ figure 10
    def plot_anomalies(self, k: int = 4):
        quiet = self.df.nsmallest(k, "RMS")
        clipped = self.df.nlargest(k, "Clip_Frac")
        fig, axes = plt.subplots(4, k, figsize=(4.4 * k, 12), squeeze=False)
        fig.suptitle("Anomalies: lowest-energy clips (rows 1-2) and most-clipped clips (rows 3-4)", fontweight="bold")
        for r, (name, sub) in enumerate([("lowest RMS", quiet), ("most clipped", clipped)]):
            for j, (_, row) in enumerate(sub.iterrows()):
                y, sr = self._load(row["Word_Class"], row["Filename"])
                wa, sa = axes[2 * r, j], axes[2 * r + 1, j]
                wa.plot(np.arange(len(y)) / sr, y, lw=0.6, color="darkred" if r else "navy")
                lim = max(0.01, float(np.abs(y).max()) * 1.2) if r == 0 else 1.05
                wa.set_ylim(-lim, lim)
                wa.set_title(f"{row['Word_Class']} | peak {row['Peak']:.4f}\nRMS {row['RMS_dBFS']:.1f} dBFS | clip {row['Clip_Frac']:.3f}", fontsize=8)
                librosa.display.specshow(self._logmel(y, sr), sr=sr, hop_length=self.e["hop_length"],
                                         x_axis="time", y_axis="mel", fmax=self.e["fmax"], ax=sa); sa.grid(False)
            axes[2 * r, 0].set_ylabel(f"{name}\namplitude (note y-scale)")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        return self._save(fig, "fig10_anomalies.png")

    def run_all(self):
        self.plot_data_flow(); self.plot_class_counts(); self.plot_duration(); self.plot_energy(); self.plot_energy_dynamics()
        self.plot_speakers(); self.plot_speaker_class_heatmap(); self.plot_multivariate()
        self.plot_class_examples(); self.plot_mean_logmel(); self.plot_anomalies()
