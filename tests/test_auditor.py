import numpy as np
import pandas as pd
import soundfile as sf

from src.auditor import SpeechDataAuditor

CFG = {"audit": {"target_len_samples": 16000, "clip_threshold": 0.99, "clipped_file_frac": 0.01,
                 "very_short_sec": 0.5, "frame_len": 400, "frame_hop": 160, "low_energy_rms_dbfs": -50.0,
                 "exclude_low_energy": False, "low_energy_sweep_dbfs": [-40, -50, -60]}}


def _speech_like(rng, n=16000, level=0.2):
    t = np.arange(n) / 16000
    env = np.clip(np.sin(2 * np.pi * 3 * t), 0, None)                           # bursts and pauses
    return level * env * np.sin(2 * np.pi * 300 * t) + 1e-4 * rng.standard_normal(n)


def _make(tmp_path):
    rng = np.random.default_rng(1)
    for w in ["a", "b"]:
        (tmp_path / w).mkdir()
    sp = lambda i: f"{i:08x}_nohash_0.wav"
    sf.write(tmp_path / "a" / sp(1), _speech_like(rng), 16000, subtype="PCM_16")             # normal
    sf.write(tmp_path / "a" / sp(2), 1e-4 * rng.standard_normal(16000), 16000, subtype="PCM_16")  # low-energy hiss
    sf.write(tmp_path / "a" / sp(3), np.zeros(16000), 16000, subtype="PCM_16")               # digital silence
    x = _speech_like(rng)
    sf.write(tmp_path / "a" / sp(4), x, 16000, subtype="PCM_16")                             # same audio ...
    sf.write(tmp_path / "a" / sp(5), x, 16000, subtype="PCM_16")                             # ... within-class dup
    y = _speech_like(rng)
    sf.write(tmp_path / "a" / sp(6), y, 16000, subtype="PCM_16")
    sf.write(tmp_path / "b" / sp(7), y, 16000, subtype="PCM_16")                             # cross-class dup of sp(6)


def test_flags_exclusions_and_dynamics(tmp_path):
    _make(tmp_path)
    aud = SpeechDataAuditor(CFG, tmp_path)
    df = aud.run().set_index("Speaker_ID")
    assert df.loc["00000002", "Low_Energy"] and df.loc["00000002", "Frame_Dyn_dB"] < 10
    assert df.loc["00000001", "Frame_Dyn_dB"] > 20                              # bursty speech-like signal
    reasons = df["Exclude_Reason"].to_dict()
    assert "digital_silence" in reasons["00000003"]
    assert reasons["00000004"] == "" and reasons["00000005"] == "dup_extra"     # keep first, drop the extra
    assert reasons["00000006"] == "dup_cross_class" and reasons["00000007"] == "dup_cross_class"
    assert reasons["00000002"] == ""                                            # low energy NOT excluded by default
    summ = aud.summarize(aud.run())
    assert summ["n_duplicate_groups_across_classes"] == 1 and summ["n_excluded"] == 4


def test_exclude_low_energy_follows_config(tmp_path):
    _make(tmp_path)
    cfg = {"audit": {**CFG["audit"], "exclude_low_energy": True}}
    df = SpeechDataAuditor(cfg, tmp_path).run().set_index("Speaker_ID")
    assert "low_energy" in df.loc["00000002", "Exclude_Reason"]
