# Keyword Spotting on Google Speech Commands (Data Analysis course, Fall 2026)

Individual project, **Path D** (audio / keyword spotting). This repository covers **Phase 1** (problem understanding, data audit, EDA, leakage design, tag `phase-1`) and
**Phase 2** (data preparation, feature engineering, pattern discovery, tag `phase-2`). Later phases get their own tags.

**Question (Q1):** under a *speaker-disjoint* split, how much does macro-F1 degrade as additive-noise SNR
decreases, and does a small CNN on log-mel features degrade less than a classical MFCC baseline?
Full registration: [`PROJECT_CARD.md`](PROJECT_CARD.md). Report: `reports/phase1.pdf`.

## Data
`mini_speech_commands` (8 keywords: down, go, left, no, right, stop, up, yes; 1000 clips each). It has **no**
Silence/Unknown class and no background-noise recordings, so Phase 1 is a closed-set 8-class problem and
noise robustness will use *synthetic* noise injection in later phases. Source, version, license and
checksum: [`data/README.md`](data/README.md). Raw data is never committed.

## Reproduce Phase 1
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                 # download+verify -> audit -> split+checks -> figures
python -m pytest tests         # split/leakage unit tests
python -m src.energy_review prepare   # blind listening package for low-energy clips (see report section 4)
#   ...listen, fill data/energy_review/review_sheet.csv...
python -m src.energy_review score     # applies the frozen decision rule -> reports/phase1/listening_summary.json
python tools/build_report.py --pdf   # fill report_template.md from audit_summary.json -> reports/phase1.pdf
```
`python main.py --force` recomputes the audit; `--skip-figures` skips plotting. All thresholds and the seed
(`seed: 42`) live in [`configs/config.yaml`](configs/config.yaml) and were fixed before any modelling.

## Phase 1 outputs
| Output | Location |
|---|---|
| Report | `reports/phase1.pdf` (source: `reports/phase1/report_template.md`) |
| Project card | `PROJECT_CARD.md` |
| Audit table (derived, not committed) | `data/audit_table.csv` |
| Numbers used in the report | `reports/phase1/audit_summary.json`, `reports/phase1/tables/` |
| Figures | `figures/phase1/` |
| Speaker-disjoint split (locked test set) | `configs/splits/` (`speakers_*.txt`, `split_manifest.csv`) |
| AI usage disclosure | `AI_USAGE.md` |

## Reproduce Phase 2
```bash
pip install -r requirements.txt
python main_phase2.py            # waveforms -> features (+train-only augmentation, val noise) -> PCA/UMAP/clustering -> figures
python -m pytest tests
python tools/build_report.py --phase 2 --pdf     # -> reports/phase2.pdf
```
Config: `configs/config.yaml` -> `phase2` (frozen before the first run). The **test split is never read** in Phase 2
(`--include-test` only extracts clean test features; nothing analyses them).

| Phase 2 output | Location |
|---|---|
| Report | `reports/phase2.pdf` (source `reports/phase2/report_template.md`) |
| Pipeline code | `src/preprocess.py`, `src/augment.py`, `src/features.py`, `src/discovery.py`, `src/phase2_figs.py`, `main_phase2.py` |
| Engineered features + logic | report section 3, `reports/phase2/tables/engineered_features.csv` |
| Figures (DR + clustering + diagnostics) | `figures/phase2/` |
| Numbers | `reports/phase2/phase2_summary.json`, `reports/phase2/tables/` |

## Layout
```
configs/config.yaml     seed, thresholds, paths      src/data_loader.py   download, SHA-256, validation
configs/splits/         split artefacts              src/auditor.py       per-file audit + data-quality summary
data/README.md          data provenance              src/splits.py        speaker-disjoint split + verification
figures/phase1/         generated figures            src/visualizer.py    all figures (saved, never shown)
reports/phase1/         report source + tables       tests/               split/leakage tests
tools/build_report.py   template -> HTML/PDF         main.py              Phase 1 entry point
```

## Notes
* The test split is locked: it is not used for tuning, feature selection or model choice.
* Speaker IDs are hashed identifiers used only to build splits and analyse generalisation; no attempt is made
  to identify speakers.
