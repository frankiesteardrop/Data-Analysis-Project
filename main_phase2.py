"""Phase 2 pipeline:  python main_phase2.py [--force] [--include-test] [--procs 2]

  build   : manifest -> fixed-length waveforms -> features (+ train-only augmentation, val noise conditions)
  analyse : scaling (train-fitted) -> PCA/UMAP -> k-means/hierarchical -> stability -> diagnostics -> figures/tables
The test split is NOT read unless --include-test is given, and the Phase-2 report never uses it.
"""
import argparse
import logging

from src.config import load_config
from src.pipeline2 import build

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="rebuild cached features")
    ap.add_argument("--include-test", action="store_true", help="also extract clean test features (not analysed)")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--skip-analysis", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config) if a.config else load_config()
    build(cfg, include_test=a.include_test, procs=a.procs, force=a.force)
    if not a.skip_analysis:
        from src.discovery import run_analysis
        run_analysis(cfg)


if __name__ == "__main__":
    main()
