"""Phase 2 stage 2: representation + unsupervised pattern discovery + diagnostics.

Questions (PROJECT_CARD §12): is the structure of the feature space driven by the WORD, or by
channel / speaker / loudness factors? How stable is it across seed, subsample and scaling?
All fitting (scalers, PCA, UMAP, k-means) uses TRAIN rows only; val rows are projected / assigned.
Speaker IDs are used only as anonymous colour groups; no speaker identification is attempted.
"""
from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import (adjusted_rand_score as ARI, davies_bouldin_score, normalized_mutual_info_score as NMI,
                             silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .features import ENGINEERED_NAMES, ENGINEERED_SPEC, LOGMEL_MS_NAMES, MFCC_NAMES

log = logging.getLogger(__name__)
SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}


# ----------------------------------------------------------------------------- loading
def load_all(cfg):
    fdir = Path(cfg["_root"]) / cfg["phase2"]["paths"]["features_dir"]
    meta = pd.read_csv(fdir / "meta.csv", dtype={"Speaker_ID": str, "Filename": str, "Word_Class": str})
    D = {}
    for sp in ("train", "val"):
        z = np.load(fdir / f"feat_{sp}.npz")
        m = meta.iloc[z["idx"]].reset_index(drop=True)
        D[sp] = {"meta": m, "mfcc": z["mfcc"], "eng": z["eng"], "logmel_ms": z["logmel_ms"], "diag": z["diag"],
                 "logmel": z["logmel"]}
    za = np.load(fdir / "feat_train_aug.npz")
    D["aug"] = {k: za[k] for k in ("mfcc", "eng", "logmel_ms", "diag")}
    D["aug"]["log"] = pd.read_csv(fdir / "augmentation_log.csv")
    zn = np.load(fdir / "feat_val_noise.npz")
    D["noise"] = {k: zn[k] for k in zn.files}
    D["examples"] = np.load(fdir / "examples.npz")
    return D


def feature_matrix(part, name):
    c0 = [i for i, n in enumerate(MFCC_NAMES) if n.startswith(("mfcc0_", "d_mfcc0_"))]
    if name == "mfcc":
        return part["mfcc"]
    if name == "mfcc_noc0":
        return np.delete(part["mfcc"], c0, axis=1)
    if name == "eng":
        return part["eng"]
    if name == "mfcc+eng":
        return np.hstack([part["mfcc"], part["eng"]])
    if name == "logmel_ms":
        return part["logmel_ms"]
    raise KeyError(name)


# ----------------------------------------------------------------------------- statistics helpers
def eta2(x: np.ndarray, labels) -> float:
    """Between-group share of variance (0..1) of a 1-D variable."""
    lab = pd.Series(labels).reset_index(drop=True)
    x = pd.Series(x).reset_index(drop=True)
    tot = ((x - x.mean()) ** 2).sum()
    if tot <= 0:
        return 0.0
    gm = x.groupby(lab).agg(["mean", "count"])
    return float((gm["count"] * (gm["mean"] - x.mean()) ** 2).sum() / tot)


def tertile(v):
    v = pd.Series(v)
    return pd.qcut(v.rank(method="first"), 3, labels=["low", "mid", "high"]).astype(str).to_numpy()


def nmi_with_chance(a, b, rng, n_perm=20):
    obs = NMI(a, b)
    ch = np.mean([NMI(a, rng.permutation(b)) for _ in range(n_perm)])
    return float(obs), float(ch)


def knn_diagnostics(Z, word, spk, k=10):
    """Share of the k nearest neighbours (Euclid, excluding self) that share the word; and, among
    same-word neighbours, the share that also share the speaker, with its chance level."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)[:, 1:]
    word, spk = np.asarray(word), np.asarray(spk)
    same_w = word[idx] == word[:, None]
    same_s = spk[idx] == spk[:, None]
    word_purity = float(same_w.mean())
    sw = same_w.sum()
    spk_given_w = float((same_w & same_s).sum() / sw) if sw else np.nan
    # chance: P(two random clips of the same word share a speaker)
    df = pd.DataFrame({"w": word, "s": spk})
    num = den = 0.0
    for _, g in df.groupby("w"):
        n = len(g)
        c = g.s.value_counts().to_numpy()
        num += (c * (c - 1)).sum()
        den += n * (n - 1)
    chance = float(num / den)
    return {"word_purity": word_purity, "word_chance": float(np.mean([(df.w == w).mean() for w in df.w.unique()])),
            "spk_given_word": spk_given_w, "spk_given_word_chance": chance,
            "spk_enrichment": float(spk_given_w / chance) if chance > 0 else np.nan}


def true_label_silhouette(Z, labels, seed, n=4000):
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(Z), size=min(n, len(Z)), replace=False)
    return float(silhouette_score(Z[sel], np.asarray(labels)[sel]))


# ----------------------------------------------------------------------------- representation
class Space:
    """scaler + PCA fitted on TRAIN only."""

    def __init__(self, Xtr, scaler="standard", n_pc=10, seed=42):
        self.sc = SCALERS[scaler]().fit(Xtr)
        self.pca_full = PCA(random_state=seed).fit(self.sc.transform(Xtr))
        self.pca = PCA(n_components=n_pc, random_state=seed).fit(self.sc.transform(Xtr))

    def z(self, X, n=None):
        Z = self.pca.transform(self.sc.transform(X))
        return Z if n is None else Z[:, :n]


def kmeans_labels(Z, k, seed, n_init=10):
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Z)


def stability_for_k(Z, Xtr_by_scaling, k, cfg_d, seed):
    """ARI-based stability of k-means for one k: seeds, subsamples, scalings."""
    ref = kmeans_labels(Z, k, seed).labels_
    s_lab = [KMeans(k, n_init=1, random_state=seed + 1 + i).fit(Z).labels_ for i in range(cfg_d["n_seeds"])]
    seeds = float(np.mean([ARI(a, b) for a, b in combinations(s_lab, 2)]))
    rng = np.random.default_rng(seed)
    sub = []
    for _ in range(cfg_d["n_subsamples"]):
        s = rng.choice(len(Z), size=int(cfg_d["subsample_frac"] * len(Z)), replace=False)
        km = KMeans(k, n_init=3, random_state=int(rng.integers(1e6))).fit(Z[s])
        sub.append(ARI(ref, km.predict(Z)))
    scal = {}
    for name, Zs in Xtr_by_scaling.items():
        scal[name] = float(ARI(ref, kmeans_labels(Zs, k, seed).labels_))
    return {"seed_ari": seeds, "subsample_ari": float(np.mean(sub)), "subsample_ari_sd": float(np.std(sub)),
            **{f"scaling_ari_{n}": v for n, v in scal.items()}}


# ----------------------------------------------------------------------------- main
def run_analysis(cfg):
    from . import phase2_figs as F
    p2 = cfg["phase2"]
    dcfg = p2["discovery"]
    seed = p2["seed"]
    rng = np.random.default_rng(seed)
    root = Path(cfg["_root"])
    figs = root / p2["paths"]["figures_dir"]; figs.mkdir(parents=True, exist_ok=True)
    tabs = root / p2["paths"]["tables_dir"]; tabs.mkdir(parents=True, exist_ok=True)
    D = load_all(cfg)
    tr, va = D["train"], D["val"]
    mtr, mva = tr["meta"], va["meta"]
    S = {}                                                       # summary json

    # ---------------- 1. preprocessing summary + missing data
    pre = json.loads((root / p2["paths"]["features_dir"] / "preprocessing_summary.json").read_text())
    S["preprocessing"] = pre
    nan_counts = {"train_nan_mfcc": int(np.isnan(tr["mfcc"]).sum()), "train_nan_eng": int(np.isnan(tr["eng"]).sum()),
                  "val_nan_mfcc": int(np.isnan(va["mfcc"]).sum()), "val_nan_eng": int(np.isnan(va["eng"]).sum()),
                  "train_nan_f0": int(np.isnan(tr["diag"][:, 2]).sum()), "val_nan_f0": int(np.isnan(va["diag"][:, 2]).sum()),
                  "train_inf": int((~np.isfinite(np.nan_to_num(tr["mfcc"], nan=0))).sum() + (~np.isfinite(np.nan_to_num(tr["eng"], nan=0))).sum())}
    S["missing"] = nan_counts
    f0_med_train = float(np.nanmedian(tr["diag"][:, 2]))           # train statistic only
    S["missing"]["f0_impute_value_hz"] = f0_med_train
    for part in (tr, va, D["aug"]):                                # f0 is a DIAGNOSTIC only: impute for colouring
        part["diag"] = part["diag"].copy()
        bad = np.isnan(part["diag"][:, 2])
        part["diag"][bad, 2] = f0_med_train
    S["excluded_in_phase1"] = int(115)

    # ---------------- 2. covariates (diagnostics, never inputs)
    rms = tr["diag"][:, 0]
    cov_tr = {"loudness": tertile(rms), "duration": tertile(tr["diag"][:, 1]), "f0": tertile(tr["diag"][:, 2])}
    spk_counts = mtr.Speaker_ID.value_counts()
    big = spk_counts[spk_counts >= dcfg["min_clips_per_speaker_group"]].index
    spk_mask = mtr.Speaker_ID.isin(big).to_numpy()
    top = spk_counts.index[: dcfg["n_speaker_groups_plot"]]
    letters = {s: chr(65 + i) for i, s in enumerate(top)}
    spk_letter = mtr.Speaker_ID.map(letters).fillna("other").to_numpy()
    S["speakers"] = {"n_train_speakers": int(mtr.Speaker_ID.nunique()), "n_speakers_ge_min": int(len(big)),
                     "clips_in_speakers_ge_min": int(spk_mask.sum()),
                     "top_group_sizes": [int(spk_counts[s]) for s in top]}
    word = mtr.Word_Class.to_numpy()
    words = sorted(set(word))

    # ---------------- 3. engineered-feature diagnostics (train)
    eng = pd.DataFrame(tr["eng"], columns=ENGINEERED_NAMES)
    rows = []
    for (n, logic), j in zip(ENGINEERED_SPEC, range(len(ENGINEERED_SPEC))):
        v = eng[n].to_numpy()
        rows.append({"feature": n, "logic": logic, "eta2_word": eta2(v, word),
                     "eta2_speaker": eta2(v[spk_mask], mtr.Speaker_ID[spk_mask]),
                     "eta2_speaker_shuffled": eta2(v[spk_mask], rng.permutation(mtr.Speaker_ID[spk_mask].to_numpy())),
                     "rho_rms": float(spearmanr(v, rms)[0]), "rho_duration": float(spearmanr(v, tr["diag"][:, 1])[0]),
                     "median": float(np.median(v)), "iqr": float(np.subtract(*np.percentile(v, [75, 25])))})
    eng_tab = pd.DataFrame(rows)
    eng_tab.to_csv(tabs / "engineered_features.csv", index=False)
    corr = eng.corr(method="spearman")
    iu = np.triu_indices(len(corr), 1)
    redundant = [(corr.index[i], corr.columns[j], float(corr.values[i, j]))
                 for i, j in zip(*iu) if abs(corr.values[i, j]) >= p2["features"]["max_feature_corr"]]
    S["engineered"] = {"n": len(ENGINEERED_NAMES), "redundant_pairs": redundant,
                       "mean_eta2_word": float(eng_tab.eta2_word.mean()),
                       "mean_eta2_speaker": float(eng_tab.eta2_speaker.mean()),
                       "mean_eta2_speaker_shuffled": float(eng_tab.eta2_speaker_shuffled.mean()),
                       "top3_word": eng_tab.sort_values("eta2_word", ascending=False).feature.head(3).tolist(),
                       "top3_rms": eng_tab.reindex(eng_tab.rho_rms.abs().sort_values(ascending=False).index).feature.head(3).tolist()}

    # ---------------- 4. feature-set comparison (+ PCA/eta2 per PC for the primary set)
    sets = ["mfcc", "mfcc_noc0", "eng", "mfcc+eng", "logmel_ms"]
    spaces, comp = {}, []
    for name in sets:
        sp = Space(feature_matrix(tr, name), "standard", dcfg["pca_components_for_clustering"], seed)
        spaces[name] = sp
        Z = sp.z(feature_matrix(tr, name))
        Zv = sp.z(feature_matrix(va, name))
        km = kmeans_labels(Z, 8, seed)
        kd = knn_diagnostics(Z, word, mtr.Speaker_ID.to_numpy())
        nm_w, nm_w_ch = nmi_with_chance(km.labels_, word, rng)
        nm_l, nm_l_ch = nmi_with_chance(km.labels_, cov_tr["loudness"], rng)
        val_nmi = NMI(km.predict(Zv), mva.Word_Class.to_numpy())
        comp.append({"feature_set": name, "n_features": feature_matrix(tr, name).shape[1],
                     "var_explained_pc10": float(sp.pca.explained_variance_ratio_.sum()),
                     "silhouette_true_words": true_label_silhouette(Z, word, seed),
                     "knn10_word_purity": kd["word_purity"], "knn10_spk_enrichment": kd["spk_enrichment"],
                     "kmeans8_nmi_word_train": nm_w, "kmeans8_nmi_word_chance": nm_w_ch,
                     "kmeans8_nmi_word_val": float(val_nmi),
                     "kmeans8_nmi_loudness": nm_l, "kmeans8_nmi_loudness_chance": nm_l_ch})
    comp = pd.DataFrame(comp)
    comp.to_csv(tabs / "feature_set_comparison.csv", index=False)
    S["feature_sets"] = comp.set_index("feature_set").to_dict("index")

    # ---------------- 5. primary representation: MFCC-summary (frozen in PROJECT_CARD §12)
    X_tr, X_va = feature_matrix(tr, "mfcc"), feature_matrix(va, "mfcc")
    sp = spaces["mfcc"]
    Z = sp.z(X_tr)
    Zv = sp.z(X_va)
    pcs = sp.pca_full.explained_variance_ratio_
    S["pca"] = {"var_ratio_pc1_5": [float(v) for v in pcs[:5]], "cum_var_pc2": float(pcs[:2].sum()),
                "cum_var_pc10": float(pcs[:10].sum()),
                "n_pc_80": int(np.searchsorted(np.cumsum(pcs), 0.8) + 1), "n_pc_95": int(np.searchsorted(np.cumsum(pcs), 0.95) + 1)}
    pc_eta = pd.DataFrame({
        "pc": [f"PC{i + 1}" for i in range(10)],
        "eta2_word": [eta2(Z[:, i], word) for i in range(10)],
        "eta2_speaker": [eta2(Z[spk_mask, i], mtr.Speaker_ID[spk_mask]) for i in range(10)],
        "eta2_speaker_shuffled": [eta2(Z[spk_mask, i], rng.permutation(mtr.Speaker_ID[spk_mask].to_numpy())) for i in range(10)],
        "eta2_loudness": [eta2(Z[:, i], cov_tr["loudness"]) for i in range(10)],
        "eta2_duration": [eta2(Z[:, i], cov_tr["duration"]) for i in range(10)],
        "var_explained": pcs[:10]})
    pc_eta.to_csv(tabs / "pca_eta2.csv", index=False)
    S["pc_eta2"] = {c: [float(v) for v in pc_eta[c]] for c in pc_eta.columns if c != "pc"}
    kd = knn_diagnostics(Z, word, mtr.Speaker_ID.to_numpy())
    S["knn_primary"] = kd
    # separate "which PCs carry word vs loudness" summary
    S["pc_summary"] = {"max_eta2_word_pc": int(pc_eta.eta2_word.idxmax() + 1), "max_eta2_word": float(pc_eta.eta2_word.max()),
                       "max_eta2_loudness_pc": int(pc_eta.eta2_loudness.idxmax() + 1),
                       "max_eta2_loudness": float(pc_eta.eta2_loudness.max()),
                       "max_eta2_speaker_pc": int(pc_eta.eta2_speaker.idxmax() + 1),
                       "max_eta2_speaker": float(pc_eta.eta2_speaker.max())}
    # PC1 relation to c0 (loudness) -> c0 loading
    load = pd.Series(sp.pca.components_[0], index=MFCC_NAMES).abs().sort_values(ascending=False)
    S["pc1_top_loadings"] = load.head(5).index.tolist()

    # ---------------- 6. k selection (k-means, PCA-10) + stability
    k_lo, k_hi = dcfg["k_range"]
    scal_spaces = {}
    for sc in ("robust", "minmax"):
        s2 = Space(X_tr, sc, dcfg["pca_components_for_clustering"], seed)
        scal_spaces[sc] = s2.z(X_tr)
    s3 = Space(feature_matrix(tr, "mfcc_noc0"), "standard", dcfg["pca_components_for_clustering"], seed)
    scal_spaces["no_c0"] = s3.z(feature_matrix(tr, "mfcc_noc0"))
    rows = []
    sil_idx = np.random.default_rng(seed).choice(len(Z), size=min(dcfg["silhouette_sample"], len(Z)), replace=False)
    for k in range(k_lo, k_hi + 1):
        km = kmeans_labels(Z, k, seed)
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Z)
        st = stability_for_k(Z, scal_spaces, k, dcfg, seed)
        rows.append({"k": k, "inertia": float(km.inertia_),
                     "silhouette": float(silhouette_score(Z[sil_idx], km.labels_[sil_idx])),
                     "davies_bouldin": float(davies_bouldin_score(Z, km.labels_)),
                     "silhouette_ward": float(silhouette_score(Z[sil_idx], hc.labels_[sil_idx])),
                     "ari_kmeans_vs_ward": float(ARI(km.labels_, hc.labels_)),
                     "nmi_word": float(NMI(km.labels_, word)), "nmi_loudness": float(NMI(km.labels_, cov_tr["loudness"])),
                     **st})
        log.info("k=%d sil=%.3f seedARI=%.2f", k, rows[-1]["silhouette"], st["seed_ari"])
    ktab = pd.DataFrame(rows)
    ktab.to_csv(tabs / "k_selection.csv", index=False)
    k_sel = int(ktab.loc[ktab.silhouette.idxmax(), "k"])
    S["k_selection"] = {"k_selected": k_sel, "rule": dcfg["k_selection_rule"],
                        "silhouette_at_k": float(ktab.set_index("k").loc[k_sel, "silhouette"]),
                        "silhouette_at_8": float(ktab.set_index("k").loc[8, "silhouette"]),
                        "stability_at_k": ktab.set_index("k").loc[k_sel, ["seed_ari", "subsample_ari", "scaling_ari_robust",
                                                                       "scaling_ari_minmax", "scaling_ari_no_c0"]].to_dict(),
                        "stability_at_8": ktab.set_index("k").loc[8, ["seed_ari", "subsample_ari", "scaling_ari_robust",
                                                                    "scaling_ari_minmax", "scaling_ari_no_c0"]].to_dict(),
                        "stable_threshold": dcfg["stable_ari_threshold"],
                        "k_stable": [int(k) for k in ktab[(ktab[["seed_ari", "subsample_ari", "scaling_ari_robust",
                                                                  "scaling_ari_minmax", "scaling_ari_no_c0"]].min(axis=1)
                                                          >= dcfg["stable_ari_threshold"])].k],
                        "k_best_stability": int(ktab.loc[ktab[["seed_ari", "subsample_ari", "scaling_ari_robust", "scaling_ari_minmax", "scaling_ari_no_c0"]].mean(axis=1).idxmax(), "k"])}
    # cluster quality at k_sel and 8
    for k in sorted({k_sel, 8}):
        km = kmeans_labels(Z, k, seed)
        lv = km.predict(Zv)
        lab_l = tertile(rms)
        ctab = pd.crosstab(km.labels_, word)
        purity = float(ctab.max(axis=1).sum() / len(word))
        ctab_l = pd.crosstab(km.labels_, lab_l)
        purity_l = float(ctab_l.max(axis=1).sum() / len(word))
        nm_w, nm_w_c = nmi_with_chance(km.labels_, word, rng)
        nm_s, nm_s_c = nmi_with_chance(km.labels_[spk_mask], mtr.Speaker_ID[spk_mask].to_numpy(), rng)
        nm_l, nm_l_c = nmi_with_chance(km.labels_, lab_l, rng)
        S[f"clusters_k{k}"] = {"purity_word": purity, "purity_loudness_tertile": purity_l, "nmi_word": nm_w, "nmi_word_chance": nm_w_c,
                               "ari_word": float(ARI(km.labels_, word)), "nmi_speaker": nm_s, "nmi_speaker_chance": nm_s_c,
                               "nmi_loudness": nm_l, "nmi_loudness_chance": nm_l_c,
                               "nmi_word_val": float(NMI(lv, mva.Word_Class.to_numpy())),
                               "ari_word_val": float(ARI(lv, mva.Word_Class.to_numpy())),
                               "sizes": [int(v) for v in np.bincount(km.labels_)]}
        ctab.to_csv(tabs / f"cluster_by_word_k{k}.csv")
        ctab_l.to_csv(tabs / f"cluster_by_loudness_k{k}.csv")
    km_sel = kmeans_labels(Z, k_sel, seed)
    km8 = kmeans_labels(Z, 8, seed)
    hc8 = AgglomerativeClustering(n_clusters=8, linkage="ward").fit(Z)
    S["ward8"] = {"nmi_word": float(NMI(hc8.labels_, word)), "ari_vs_kmeans8": float(ARI(hc8.labels_, km8.labels_))}

    # ---------------- 7. UMAP
    import umap
    u = dcfg["umap"]
    um = umap.UMAP(n_neighbors=u["n_neighbors"], min_dist=u["min_dist"], metric=u["metric"], random_state=seed, n_components=2)
    E = um.fit_transform(Z)
    Ev = um.transform(Zv)
    E_seeds = []
    for s in range(4):
        e = umap.UMAP(n_neighbors=u["n_neighbors"], min_dist=u["min_dist"], metric=u["metric"], random_state=seed + 1 + s).fit_transform(Z)
        E_seeds.append(KMeans(8, n_init=10, random_state=seed).fit_predict(e))
    S["umap"] = {"trustworthiness_pca2": float(trustworthiness(Z, Z[:, :2], n_neighbors=15)),
                 "trustworthiness_umap2": float(trustworthiness(Z, E, n_neighbors=15)),
                 "kmeans8_on_umap_nmi_word": float(NMI(KMeans(8, n_init=10, random_state=seed).fit_predict(E), word)),
                 "seed_ari_kmeans8_on_umap": float(np.mean([ARI(a, b) for a, b in combinations(E_seeds, 2)])),
                 "knn10_word_purity_umap": knn_diagnostics(E, word, mtr.Speaker_ID.to_numpy())["word_purity"],
                 "knn10_word_purity_pca2": knn_diagnostics(Z[:, :2], word, mtr.Speaker_ID.to_numpy())["word_purity"],
                 "val_knn_word_purity": None}
    # val transfer: val points' nearest train neighbours (in PCA-10) share the word?
    nn = NearestNeighbors(n_neighbors=10).fit(Z)
    vi = nn.kneighbors(Zv, return_distance=False)
    S["umap"]["val_knn_word_purity"] = float((word[vi] == mva.Word_Class.to_numpy()[:, None]).mean())

    # ---------------- 8. augmentation diagnostics (train only)
    alog = D["aug"]["log"]
    Za = sp.z(feature_matrix(D["aug"], "mfcc"))
    disp = np.linalg.norm(Za - Z, axis=1)
    nn1 = NearestNeighbors(n_neighbors=2).fit(Z)
    j = nn1.kneighbors(Za, return_distance=False)
    own_is_nn = (j[:, 0] == np.arange(len(Z)))
    # nearest ORIGINAL train clip (other than the source clip itself) has the same word?
    nn_other = np.where(own_is_nn, j[:, 1], j[:, 0])
    S["augmentation"] = {
        "n_aug": int(len(alog)), "frac_with_noise": float((alog.noise_type != "none").mean()),
        "noise_counts": alog.noise_type.value_counts().to_dict(),
        "gain_db_min": float(alog.gain_db.min()), "gain_db_max": float(alog.gain_db.max()),
        "shift_ms_min": float(alog.shift_samples.min() / 16), "shift_ms_max": float(alog.shift_samples.max() / 16),
        "measured_snr_abs_err_max": float((alog.measured_snr_db - alog.snr_db).abs().max()),
        "source_is_nn_frac": float(own_is_nn.mean()),
        "nn_other_same_word": float((word[nn_other] == word).mean()),
        "nmi_word_kmeans8_aug": float(NMI(km8.predict(Za), word)),
        "nmi_word_kmeans8_orig": float(NMI(km8.labels_, word)),
        "median_displacement": float(np.median(disp)),
        "median_displacement_by_noise": {k: float(np.median(disp[(alog.noise_type == k).to_numpy()])) for k in alog.noise_type.unique()},
        "median_displacement_by_snr": {str(k): float(np.median(disp[(alog.snr_db == k).to_numpy()])) for k in sorted(alog.snr_db.dropna().unique())},
        "spearman_disp_gain": float(spearmanr(disp, alog.gain_db.abs())[0]),
        "train_only_check": "augmentation executed on rows with Split == train only (asserted in src/augment.py)",
    }
    alog.groupby("noise_type").agg(n=("gain_db", "size"), gain_mean=("gain_db", "mean"), shift_mean=("shift_samples", "mean"),
                                   measured_snr_mean=("measured_snr_db", "mean")).to_csv(tabs / "augmentation_summary.csv")

    # ---------------- 9. noise conditions on VAL (diagnostic)
    nz = D["noise"]
    names, kinds, snrs = nz["names"], nz["kinds"], nz["snrs"]
    vword = mva.Word_Class.to_numpy()
    zc = Zv
    clean_c = {"clean": {"nmi_k8": float(NMI(km8.predict(zc), vword)), "nmi_ksel": float(NMI(km_sel.predict(zc), vword)),
                         "knn_word": float((word[nn.kneighbors(zc, return_distance=False)] == vword[:, None]).mean())}}
    nrows = []
    allZ, allk, allsnr = [zc], ["clean"] * len(zc), [np.inf] * len(zc)
    cent_clean = zc[:, :2].mean(0)
    for i, (nm, kd_, snr) in enumerate(zip(names, kinds, snrs)):
        Zn = sp.z(nz["mfcc"][i])
        lab8, labs = km8.predict(Zn), km_sel.predict(Zn)
        nnw = (word[nn.kneighbors(Zn, return_distance=False)] == vword[:, None]).mean()
        nrows.append({"condition": nm, "noise": str(kd_), "snr_db": float(snr), "nmi_word_k8": float(NMI(lab8, vword)),
                      "nmi_word_ksel": float(NMI(labs, vword)), "knn_word_purity": float(nnw),
                      "centroid_shift_pc12": float(np.linalg.norm(Zn[:, :2].mean(0) - cent_clean)),
                      "mean_displacement": float(np.linalg.norm(Zn - zc, axis=1).mean()),
                      "frac_same_cluster_k8": float((lab8 == km8.predict(zc)).mean())})
        allZ.append(Zn); allk += [str(kd_)] * len(Zn); allsnr += [float(snr)] * len(Zn)
    ntab = pd.DataFrame(nrows)
    ntab.to_csv(tabs / "noise_conditions.csv", index=False)
    allZ = np.vstack(allZ); allk = np.array(allk); allsnr = np.array(allsnr)
    lab_all = km8.predict(allZ)
    S["noise"] = {"clean": clean_c["clean"],
                  "worst_condition": ntab.sort_values("nmi_word_k8").iloc[0].to_dict(),
                  "nmi_word_k8_at_0db": {t: float(ntab[(ntab.noise == t) & (ntab.snr_db == 0)].nmi_word_k8.iloc[0]) for t in sorted(set(kinds))},
                  "nmi_word_k8_at_20db": {t: float(ntab[(ntab.noise == t) & (ntab.snr_db == 20)].nmi_word_k8.iloc[0]) for t in sorted(set(kinds))},
                  "knn_word_at_0db": {t: float(ntab[(ntab.noise == t) & (ntab.snr_db == 0)].knn_word_purity.iloc[0]) for t in sorted(set(kinds))},
                  "knn_word_at_20db": {t: float(ntab[(ntab.noise == t) & (ntab.snr_db == 20)].knn_word_purity.iloc[0]) for t in sorted(set(kinds))},
                  # does noise TYPE form clusters? (restrict to 5 dB and 0 dB conditions)
                  "nmi_noise_type_vs_cluster_k8_low_snr": float(NMI(lab_all[np.isin(allsnr, [5.0, 0.0])], allk[np.isin(allsnr, [5.0, 0.0])])),
                  "nmi_word_vs_cluster_k8_low_snr": float(NMI(lab_all[np.isin(allsnr, [5.0, 0.0])],
                                                              np.concatenate([vword] * (1 + len(names)))[np.isin(allsnr, [5.0, 0.0])])),
                  "nmi_snr_vs_cluster_k8": float(NMI(lab_all, allsnr.astype(str))),
                  "nmi_word_vs_cluster_k8_all": float(NMI(lab_all, np.concatenate([vword] * (1 + len(names)))))}
    Eall = um.transform(allZ[np.isin(allsnr, [np.inf, 5.0, 0.0])])
    noise_plot = {"E": Eall, "kind": allk[np.isin(allsnr, [np.inf, 5.0, 0.0])], "snr": allsnr[np.isin(allsnr, [np.inf, 5.0, 0.0])]}

    # ---------------- 10. figures
    ctx = dict(cfg=cfg, figs=figs, words=words, word=word, Z=Z, E=E, Ev=Ev, pcs=pcs, km_sel=km_sel, km8=km8, hc8=hc8,
               k_sel=k_sel, ktab=ktab, comp=comp, pc_eta=pc_eta, eng_tab=eng_tab, corr=corr, cov=cov_tr, spk_letter=spk_letter,
               rms=rms, tr=tr, va=va, mtr=mtr, alog=alog, Za=Za, disp=disp, ntab=ntab, noise_plot=noise_plot,
               examples=D["examples"], pre=pre, Zv=Zv, stable_thr=dcfg["stable_ari_threshold"], spk_mask=spk_mask, S=S)
    F.make_all(ctx)

    out = root / p2["paths"]["summary_json"]
    out.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        raise TypeError(type(o))
    out.write_text(json.dumps(S, indent=2, default=conv, ensure_ascii=False), encoding="utf-8")
    log.info("summary -> %s", out)
    return S
