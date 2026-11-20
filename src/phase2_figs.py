"""Phase 2 figures. Saved to figures/phase2/, never shown. Speaker IDs are shown only as anonymous letters."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _scatter(ax, P, labels, order, title, colors=None, s=4, alpha=0.6):
    colors = colors or {l: PAL[i % len(PAL)] for i, l in enumerate(order)}
    for l in order:
        m = np.asarray(labels) == l
        ax.scatter(P[m, 0], P[m, 1], s=s, alpha=alpha, color=colors[l], label=f"{l} ({m.sum()})", linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def fig01_preprocessing(c):
    X, nv = c["examples"]["X"], c["examples"]["nv"]
    pre = c["pre"]["train"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    ax[0].hist(c["mtr"].n_valid / 16000, bins=40, color="#4a6fa5")
    ax[0].axvline(1.0, color="k", ls="--", lw=0.8)
    ax[0].set_xlabel("valid length before padding (s)"); ax[0].set_ylabel("train clips")
    ax[0].set_title(f"{100 * pre['frac_padded']:.1f}% of train clips are padded; {pre['n_cropped']} cropped", fontsize=9)
    i = int(np.argmin(nv))
    j = int(np.argsort(nv)[len(nv) // 2])
    for a, idx, t in ((ax[1], i, "shortest of the sample"), (ax[2], j, "median clip")):
        a.plot(np.arange(len(X[idx])) / 16000, X[idx], lw=0.5, color="#333")
        a.axvspan(nv[idx] / 16000, 1.0, color="#f4c542", alpha=0.35, label="zero padding" if nv[idx] < 16000 else None)
        a.set_title(f"{t}: {nv[idx] / 16000:.2f} s valid", fontsize=9); a.set_xlabel("time (s)"); a.set_xlim(0, 1)
        if nv[idx] < 16000:
            a.legend(frameon=False, fontsize=8)
    _save(fig, c["figs"] / "fig01_preprocessing.png")


def fig02_features(c):
    lm = c["tr"]["logmel"]; w = c["word"]
    words = c["words"]
    fig, ax = plt.subplots(2, 4, figsize=(12, 5))
    for a, wd in zip(ax.ravel(), words):
        i = int(np.flatnonzero(w == wd)[0])
        a.imshow(lm[i].astype(float), origin="lower", aspect="auto", cmap="magma", extent=[0, 1, 0, 8])
        a.set_title(wd, fontsize=9); a.set_xlabel("time (s)"); a.set_ylabel("kHz")
    fig.suptitle("log-mel (64 bands, 25 ms window, 10 ms hop, 0-8 kHz): one training example per word", fontsize=10)
    _save(fig, c["figs"] / "fig02_logmel_examples.png")


def fig03_pca(c):
    Z, w, words = c["Z"], c["word"], c["words"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].bar(range(1, 21), c["S"]["pca"] and np.r_[c["pcs"][:20]], color="#4a6fa5")
    ax[0].plot(range(1, 21), np.cumsum(c["pcs"][:20]), "k.-", lw=0.8)
    ax[0].set_xlabel("PC"); ax[0].set_ylabel("explained variance ratio (bars) / cumulative (line)")
    ax[0].set_title("PCA scree (52 MFCC-summary features, train)", fontsize=9)
    _scatter(ax[1], Z[:, :2], w, words, "PC1 vs PC2, coloured by word")
    ax[1].legend(fontsize=6, markerscale=3, frameon=False, ncol=2)
    _scatter(ax[2], Z[:, [2, 3]], w, words, "PC3 vs PC4, coloured by word")
    _save(fig, c["figs"] / "fig03_pca_word.png")


def fig04_pca_confounds(c):
    Z = c["Z"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    _scatter(ax[0], Z[:, :2], c["cov"]["loudness"], ["low", "mid", "high"], "PC1/PC2 by loudness tertile (RMS dBFS)",
             colors={"low": "#d62728", "mid": "#bbbbbb", "high": "#1f77b4"})
    ax[0].legend(fontsize=7, markerscale=3, frameon=False)
    _scatter(ax[1], Z[:, :2], c["cov"]["duration"], ["low", "mid", "high"], "PC1/PC2 by valid-duration tertile",
             colors={"low": "#d62728", "mid": "#bbbbbb", "high": "#1f77b4"})
    ax[1].legend(fontsize=7, markerscale=3, frameon=False)
    letters = sorted(set(c["spk_letter"]) - {"other"})
    sl = c["spk_letter"]
    ax[2].scatter(Z[sl == "other", 0], Z[sl == "other", 1], s=3, color="#dddddd", linewidths=0, rasterized=True)
    for i, l in enumerate(letters):
        m = sl == l
        ax[2].scatter(Z[m, 0], Z[m, 1], s=9, color=PAL[i % 8], label=f"speaker {l} ({m.sum()})", linewidths=0)
    ax[2].set_title("PC1/PC2: the 8 largest training speakers (anonymous letters)", fontsize=9)
    ax[2].legend(fontsize=6, frameon=False, markerscale=1.5, ncol=2); ax[2].set_xticks([]); ax[2].set_yticks([])
    _save(fig, c["figs"] / "fig04_pca_confounds.png")


def fig05_umap(c):
    E, w, words = c["E"], c["word"], c["words"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    _scatter(ax[0, 0], E, w, words, "UMAP (train), coloured by word", s=3)
    ax[0, 0].legend(fontsize=6, markerscale=3, frameon=False, ncol=2)
    _scatter(ax[0, 1], E, c["cov"]["loudness"], ["low", "mid", "high"], "coloured by loudness tertile",
             colors={"low": "#d62728", "mid": "#bbbbbb", "high": "#1f77b4"}, s=3)
    ax[0, 1].legend(fontsize=7, markerscale=3, frameon=False)
    sl = c["spk_letter"]
    ax[1, 0].scatter(E[sl == "other", 0], E[sl == "other", 1], s=2, color="#e0e0e0", linewidths=0, rasterized=True)
    for i, l in enumerate(sorted(set(sl) - {"other"})):
        m = sl == l
        ax[1, 0].scatter(E[m, 0], E[m, 1], s=7, color=PAL[i % 8], label=f"speaker {l}", linewidths=0)
    ax[1, 0].set_title("coloured by the 8 largest speakers (anonymous)", fontsize=9)
    ax[1, 0].legend(fontsize=6, frameon=False, ncol=2); ax[1, 0].set_xticks([]); ax[1, 0].set_yticks([])
    km = c["km_sel"].labels_
    ks = sorted(set(km))
    _scatter(ax[1, 1], E, km, ks, f"k-means clusters (k={c['k_sel']}, selected by silhouette)", s=3,
             colors={k: plt.cm.tab20(i % 20) for i, k in enumerate(ks)})
    ax[1, 1].legend(fontsize=6, markerscale=3, frameon=False, ncol=2)
    _save(fig, c["figs"] / "fig05_umap.png")


def fig06_k_selection(c):
    k = c["ktab"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(k.k, k.silhouette, "o-", label="k-means"); ax[0].plot(k.k, k.silhouette_ward, "s--", label="Ward")
    ax[0].axvline(c["k_sel"], color="k", ls=":", lw=0.8); ax[0].axvline(8, color="gray", ls=":", lw=0.8)
    ax[0].set_xlabel("k"); ax[0].set_ylabel("silhouette"); ax[0].legend(frameon=False)
    ax[0].set_title("silhouette (dotted: selected k and k=8)", fontsize=9)
    ax[1].plot(k.k, k.seed_ari, "o-", label="seed"); ax[1].plot(k.k, k.subsample_ari, "s-", label="80% subsample")
    ax[1].plot(k.k, k.scaling_ari_robust, "^-", label="robust scaling")
    ax[1].plot(k.k, k.scaling_ari_minmax, "v-", label="min-max scaling")
    ax[1].plot(k.k, k.scaling_ari_no_c0, "d-", label="drop c0 (loudness)")
    ax[1].axhline(c["stable_thr"], color="r", ls="--", lw=0.8)
    ax[1].set_xlabel("k"); ax[1].set_ylabel("ARI vs. reference clustering"); ax[1].legend(frameon=False, fontsize=7)
    ax[1].set_title("stability (red: 'stable' threshold)", fontsize=9)
    ax[2].plot(k.k, k.nmi_word, "o-", label="NMI(cluster, word)"); ax[2].plot(k.k, k.nmi_loudness, "s-", label="NMI(cluster, loudness tertile)")
    ax[2].set_xlabel("k"); ax[2].legend(frameon=False); ax[2].set_title("what do the clusters align with?", fontsize=9)
    _save(fig, c["figs"] / "fig06_k_selection.png")


def fig07_cluster_word(c):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={"width_ratios": [1.0, 1.0, 0.8]})
    for a, (lab, t) in zip(ax[:2], ((c["km_sel"].labels_, f"k-means k={c['k_sel']}"), (c["km8"].labels_, "k-means k=8"))):
        ct = pd.crosstab(lab, c["word"], normalize="index")
        a.imshow(ct.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        a.set_xticks(range(len(ct.columns))); a.set_xticklabels(ct.columns, rotation=45)
        a.set_yticks(range(len(ct))); a.set_yticklabels([f"{i} (n={n})" for i, n in zip(ct.index, np.bincount(lab))], fontsize=7)
        a.set_title(f"{t}: word composition of each cluster", fontsize=9)
        for (i, j), v in np.ndenumerate(ct.values):
            if v >= 0.25:
                a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.6 else "black")
    ct = pd.crosstab(c["km8"].labels_, c["cov"]["loudness"], normalize="index")[["low", "mid", "high"]]
    ax[2].imshow(ct.values, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["low", "mid", "high"]); ax[2].set_yticks(range(len(ct)))
    ax[2].set_title("k=8: loudness-tertile composition", fontsize=9)
    for (i, j), v in np.ndenumerate(ct.values):
        ax[2].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
    _save(fig, c["figs"] / "fig07_cluster_word.png")


def fig08_variance_sources(c):
    pe = c["pc_eta"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(pe)); w = 0.2
    ax[0].bar(x - 1.5 * w, pe.eta2_word, w, label="word", color="#1f77b4")
    ax[0].bar(x - 0.5 * w, pe.eta2_loudness, w, label="loudness tertile", color="#d62728")
    ax[0].bar(x + 0.5 * w, pe.eta2_speaker, w, label="speaker (>= 8 clips)", color="#2ca02c")
    ax[0].bar(x + 1.5 * w, pe.eta2_speaker_shuffled, w, label="speaker, labels shuffled (chance)", color="#bbbbbb")
    ax[0].set_xticks(x); ax[0].set_xticklabels(pe.pc, rotation=45); ax[0].set_ylabel("eta-squared (share of PC variance)")
    ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("What explains each principal component?", fontsize=9)
    et = c["eng_tab"].sort_values("eta2_word")
    y = np.arange(len(et))
    ax[1].barh(y + 0.2, et.eta2_word, 0.4, label="word", color="#1f77b4")
    ax[1].barh(y - 0.2, et.eta2_speaker, 0.4, label="speaker", color="#2ca02c")
    ax[1].set_yticks(y); ax[1].set_yticklabels(et.feature, fontsize=7); ax[1].legend(frameon=False, fontsize=7)
    ax[1].set_xlabel("eta-squared"); ax[1].set_title("Engineered features: word vs speaker variance share", fontsize=9)
    _save(fig, c["figs"] / "fig08_variance_sources.png")


def fig09_feature_sets(c):
    cm = c["comp"].set_index("feature_set")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].bar(cm.index, cm.kmeans8_nmi_word_train, color="#1f77b4", label="train")
    ax[0].bar(cm.index, cm.kmeans8_nmi_word_val, width=0.35, color="#ff7f0e", label="val (speaker-disjoint)")
    ax[0].set_title("NMI(k-means k=8, word)", fontsize=9); ax[0].legend(frameon=False, fontsize=7)
    ax[1].bar(cm.index, cm.knn10_word_purity, color="#2ca02c"); ax[1].axhline(1 / 8, color="k", ls=":", lw=0.8)
    ax[1].set_title("10-NN word purity (dotted: chance)", fontsize=9)
    ax[2].bar(cm.index, cm.kmeans8_nmi_loudness, color="#d62728"); ax[2].set_title("NMI(k-means k=8, loudness tertile)", fontsize=9)
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    _save(fig, c["figs"] / "fig09_feature_sets.png")


def fig10_corr(c):
    cr = c["corr"]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cr))); ax.set_xticklabels(cr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cr))); ax.set_yticklabels(cr.index, fontsize=7)
    fig.colorbar(im, fraction=0.04); ax.set_title("Spearman correlation of the 16 engineered features (train)", fontsize=9)
    _save(fig, c["figs"] / "fig10_engineered_corr.png")


def fig11_augmentation(c):
    a = c["alog"]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    ax[0].hist(a.gain_db, bins=30, color="#4a6fa5"); ax[0].set_xlabel("gain (dB)"); ax[0].set_title("gain", fontsize=9)
    ax[1].hist(a.shift_samples / 16, bins=30, color="#4a6fa5"); ax[1].set_xlabel("shift (ms)"); ax[1].set_title("time shift", fontsize=9)
    nz = a[a.noise_type != "none"]
    for i, t in enumerate(sorted(nz.noise_type.unique())):
        d = nz[nz.noise_type == t]
        ax[2].scatter(d.snr_db + np.random.default_rng(0).uniform(-.8, .8, len(d)), d.measured_snr_db, s=3, alpha=.4, color=PAL[i], label=t)
    ax[2].plot([0, 22], [0, 22], "k--", lw=0.7); ax[2].set_xlabel("requested SNR (dB)"); ax[2].set_ylabel("measured SNR (dB)")
    ax[2].legend(frameon=False, fontsize=7, markerscale=3); ax[2].set_title("noise: requested vs measured", fontsize=9)
    order = ["none", "white", "pink", "babble"]
    ax[3].boxplot([c["disp"][(a.noise_type == t).to_numpy()] for t in order], tick_labels=order, showfliers=False)
    ax[3].set_ylabel("distance to the original clip in PCA-10"); ax[3].set_title("feature displacement caused by augmentation", fontsize=9)
    _save(fig, c["figs"] / "fig11_augmentation.png")


def fig12_noise(c):
    nt = c["ntab"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for i, t in enumerate(["white", "pink", "babble"]):
        d = nt[nt.noise == t].sort_values("snr_db")
        ax[0].plot(d.snr_db, d.nmi_word_k8, "o-", color=PAL[i], label=t)
        ax[1].plot(d.snr_db, d.knn_word_purity, "o-", color=PAL[i], label=t)
    clean = c["S"]["noise"]["clean"]
    ax[0].axhline(clean["nmi_k8"], color="k", ls=":", lw=0.8, label="clean val")
    ax[1].axhline(clean["knn_word"], color="k", ls=":", lw=0.8, label="clean val")
    ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("NMI(train-fitted k-means k=8, word)"); ax[0].legend(frameon=False, fontsize=7)
    ax[1].set_xlabel("SNR (dB)"); ax[1].set_ylabel("share of 10 nearest TRAIN clips with the same word"); ax[1].legend(frameon=False, fontsize=7)
    ax[0].set_title("do clusters still align with words under noise? (val)", fontsize=9)
    ax[1].set_title("neighbourhood word purity under noise (val)", fontsize=9)
    npl = c["noise_plot"]
    cols = {"inf": "#bbbbbb", "white": PAL[0], "pink": PAL[1], "babble": PAL[2]}
    kinds = npl["kind"]
    for t in ["clean", "white", "pink", "babble"]:
        m = kinds == t
        ax[2].scatter(npl["E"][m, 0], npl["E"][m, 1], s=3, alpha=0.5, linewidths=0, rasterized=True,
                      color={"clean": "#999999", "white": PAL[0], "pink": PAL[1], "babble": PAL[2]}[t], label=f"{t} ({m.sum()})")
    ax[2].legend(frameon=False, fontsize=7, markerscale=3); ax[2].set_xticks([]); ax[2].set_yticks([])
    ax[2].set_title("val projected into the train UMAP: clean + 5 dB + 0 dB noise", fontsize=9)
    _save(fig, c["figs"] / "fig12_noise_conditions.png")


def make_all(c):
    for f in (fig01_preprocessing, fig02_features, fig03_pca, fig04_pca_confounds, fig05_umap, fig06_k_selection,
              fig07_cluster_word, fig08_variance_sources, fig09_feature_sets, fig10_corr, fig11_augmentation, fig12_noise):
        f(c)
