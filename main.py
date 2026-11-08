"""Phase 1 pipeline:  python main.py [--force] [--skip-figures]

download/verify -> audit -> summary tables -> speaker-disjoint split + checks -> figures
"""
import argparse
import json
import logging

import pandas as pd

from src.auditor import SpeechDataAuditor
from src.config import load_config, resolve
from src.data_loader import SpeechDataLoader
from src.splits import make_speaker_split, save_split, trivial_baselines, verify_split
from src.visualizer import SpeechDataVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="recompute the audit even if the CSV exists")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    extracted = SpeechDataLoader(cfg).download_and_extract()
    audit = SpeechDataAuditor(cfg, extracted)
    csv_path = resolve(cfg, cfg["paths"]["audit_csv"])
    tables = resolve(cfg, cfg["paths"]["tables_dir"] + "/x").parent
    summary_path = resolve(cfg, cfg["paths"]["summary_json"])

    df = None
    if csv_path.exists() and not args.force and summary_path.exists():
        # dtype=str matters: hex speaker IDs such as "12e45678" would otherwise be parsed as floats
        cached = pd.read_csv(csv_path, dtype={"Speaker_ID": str, "Filename": str, "Audio_MD5": str,
                                              "Word_Class": str})
        if "Frame_Dyn_dB" in cached.columns:
            log.info("Using cached audit table (use --force to recompute).")
            unread = tables / "unreadable_files.csv"
            if unread.exists():
                audit.corrupt = pd.read_csv(unread)
            df = audit.add_flags(cached)           # thresholds/exclusions always follow the CURRENT config
        else:
            log.info("Cached audit table comes from an older version; recomputing.")
    if df is None:
        df = audit.run()
        df.to_csv(csv_path, index=False)
    summary = audit.summarize(df)

    # speaker-disjoint split (locked test set) + hard verification
    split_df = make_speaker_split(df, cfg["split"]["ratios"], cfg["seed"])
    summary["split"] = verify_split(split_df)          # leakage check sees ALL rows, excluded or not
    clean = split_df[~split_df["Exclude"]]            # exclusions drop rows only; speaker assignment is unchanged
    summary["split"]["files_per_split_after_exclusion"] = clean["Split"].value_counts().to_dict()
    summary["split"]["class_counts_per_split_after_exclusion"] = pd.crosstab(
        clean["Split"], clean["Word_Class"]).to_dict("index")
    summary["split"]["baselines_on_test"] = trivial_baselines(clean, cfg["seed"])
    # descriptive: where do low-energy / very short files fall (all rows, before exclusion)?
    summary["split"]["low_energy_per_split"] = split_df.groupby("Split")["Low_Energy"].sum().astype(int).to_dict()
    summary["split"]["very_short_per_split"] = split_df.groupby("Split")["Very_Short"].sum().astype(int).to_dict()
    summary["split"]["median_rms_dbfs_per_split"] = split_df.groupby("Split")["RMS_dBFS"].median().round(2).to_dict()
    save_split(split_df, resolve(cfg, cfg["split"]["out_dir"] + "/x").parent)

    audit.save_tables(df, summary, tables, summary_path)
    log.info("Key numbers:\n%s", json.dumps(
        {k: summary[k] for k in ["n_files", "n_unreadable", "n_speakers", "n_low_energy", "n_clipped_files",
                                 "n_shorter_than_1s", "n_duplicate_groups", "n_duplicate_groups_across_classes",
                                 "n_excluded", "frac_speakers_in_ge2_classes"]},
        indent=2))

    if not args.skip_figures:
        SpeechDataVisualizer(cfg, df, extracted, resolve(cfg, cfg["paths"]["figures_dir"] + "/x").parent).run_all()
    log.info("Phase 1 pipeline finished.")


if __name__ == "__main__":
    main()
