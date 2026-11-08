# Data provenance

Last checked against public documentation: 2026-11-01.

| Item | Value |
|---|---|
| Dataset | `mini_speech_commands`: the small subset used in the TensorFlow "Simple audio recognition" tutorial, which describes it as a portion of the Speech Commands dataset (Warden, 2018) |
| Download URL | http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip (the form used by the tutorial; `configs/config.yaml` uses the https form of the same path) |
| Archive SHA-256 | `49650f2341b26d886b46b3f4fb8fed59e30300b17550f1ee4a768b3106cf93a0` (also stored in `configs/config.yaml`, key `data.zip_sha256`) |
| Archive size | The tutorial's download log reports 182,082,353 bytes. Local check: see the checklist below |
| Source release (v0.01 / v0.02) | **v0.01**, per the archive's own `README.md` (read from the extracted archive): it tells readers to see the original dataset `speech_commands_v0.01.tar.gz` for documentation and license information. It does not literally say "cut from v0.01"; this is the only release it names. The tutorial's description of the full dataset (105,000+ files, 35 words) matches v0.02 figures instead, so that sentence is NOT used to characterise this subset (v0.01: 64,727 files, released 2017-08-03; v0.02: 105,829 files, 2018-04-11) |
| Classes | down, go, left, no, right, stop, up, yes (1000 clips each, 8000 total; verified by `main.py`, which fails if the class folders or counts differ) |
| Not included | Silence / Unknown classes and `_background_noise_` recordings (the loader's structure check passes with exactly the 8 class folders), and official train/validation/test lists |
| Archive contents | 8 class folders plus a `README.md` inside `mini_speech_commands/` (the tutorial filters this file out of the class list). Extraction may also create an unrelated `__MACOSX` folder |
| Format | WAV; sample rate, channels and sample type are measured per file by the audit (see `reports/phase1/audit_summary.json`: `sample_rates`, `channels`, `subtypes`) |
| File name | `<speaker hash>_nohash_<k>.wav`; the label is the parent folder name |
| Metadata available to us | hashed speaker ID and utterance index (from the file name) and the keyword label only. The full dataset's published feature list contains no age, gender, accent or device fields, and the subset has no timestamps |
| Collection | Crowd-sourced. Volunteers were prompted for individual words (not sentences) during a roughly five-minute session. English only. Prompted recording, not a controlled experiment and not natural speech |
| Intended purpose | Training and evaluating small keyword-spotting models |
| Unique speakers in this subset | 1750 (from our audit, `n_speakers`). Totals for the full dataset are not used because secondary sources disagree |
| License | **CC BY 4.0** (Creative Commons Attribution 4.0). Attribution is required: cite Warden (2018) |
| Privacy | The dataset consists of voices donated online, and its dataset card states that users agree not to attempt to determine speakers' identities. We comply: speaker IDs are used only to build speaker-disjoint splits and to analyse generalisation |

## Verification checklist (done / still to do)
- [x] Full-dataset description in the tutorial (105,000+ files, 35 words) read; it matches v0.02 figures and is therefore not used for this subset
- [x] Which release the mini archive derives from: the archive's own README names v0.01 (see table)
- [x] License (CC BY 4.0) and required citation checked against the tutorial and the dataset card
- [x] No demographic fields in the published feature list of the full dataset (we checked the dataset card, not every file of the mini archive)
- [x] Unique speakers in this subset = 1750 (our audit)
- [ ] Local archive size equals 182,082,353 bytes (PowerShell: `(Get-Item data\mini_speech_commands.zip).Length`)
- [x] Contents of the archive's own `README.md` read

## How to obtain
`python main.py` downloads the archive, checks its SHA-256, extracts it to `data/mini_speech_commands/` and validates the folder structure. Everything under `data/` except this file is git-ignored.

## Derived files
- `data/audit_table.csv`: one row per clip (duration, RMS dBFS, clipping, MD5, ...); regenerate with `python main.py --force`.
- Speaker-disjoint split artefacts are committed under `configs/splits/`.

## Sources
- TensorFlow tutorial: https://www.tensorflow.org/tutorials/audio/simple_audio
- Dataset card (license, collection, privacy note, versions): https://huggingface.co/datasets/google/speech_commands
- TFDS catalog entry: https://www.tensorflow.org/datasets/catalog/speech_commands
- Paper: https://arxiv.org/abs/1804.03209

## Citation
P. Warden, "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition", arXiv:1804.03209, 2018.