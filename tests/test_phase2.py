"""Phase 2 unit tests: padding/cropping, augmentation SNR + train-only guard, train-only scaling, determinism."""
import numpy as np
import pytest

from src.augment import (add_noise, assert_train_only, augment_train_clip, babble_noise, measured_snr_db,
                         pink_noise, shift_zero_fill, white_noise)
from src.config import load_config
from src.features import ENGINEERED_NAMES, clip_features
from src.preprocess import fix_length, load_manifest


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_fix_length_pad_crop_exact():
    x = np.ones(100, dtype=np.float32)
    y, n, a = fix_length(x, 160)
    assert len(y) == 160 and n == 100 and a == "padded" and y[100:].sum() == 0 and y[:100].sum() == 100
    y, n, a = fix_length(np.arange(200, dtype=np.float32), 160)
    assert len(y) == 160 and n == 160 and a == "cropped" and y[-1] == 159
    y, n, a = fix_length(x, 100)
    assert a == "none" and n == 100


def test_manifest_excludes_flagged_and_hides_test(cfg):
    m = load_manifest(cfg)
    assert set(m.Split) == {"train", "val"}            # test is locked unless requested explicitly
    assert not m.Exclude.astype(bool).any()


def test_speaker_disjoint_after_loading(cfg):
    m = load_manifest(cfg)
    assert not (set(m[m.Split == "train"].Speaker_ID) & set(m[m.Split == "val"].Speaker_ID))


@pytest.mark.parametrize("kind", ["white", "pink", "babble"])
@pytest.mark.parametrize("snr", [20.0, 5.0, 0.0])
def test_noise_hits_requested_snr(kind, snr):
    rng = np.random.default_rng(0)
    x = (0.1 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)).astype(np.float32)
    pool = np.stack([x * (0.5 + i / 10) for i in range(8)]); nv = np.full(8, 16000)
    nz = {"white": lambda: white_noise(16000, rng), "pink": lambda: pink_noise(16000, rng),
          "babble": lambda: babble_noise(pool, nv, 16000, 6, rng)}[kind]()
    y = add_noise(x, 16000, nz, snr)
    assert abs(measured_snr_db(x, y, 16000) - snr) < 0.05


def test_pink_noise_has_1_over_f_slope():
    n = pink_noise(2 ** 16, np.random.default_rng(1)).astype(float)
    p = np.abs(np.fft.rfft(n)) ** 2
    f = np.arange(len(p))[10:-10]
    slope = np.polyfit(np.log(f), np.log(p[10:-10]), 1)[0]
    assert -1.3 < slope < -0.7


def test_shift_zero_fill():
    x = np.arange(1, 11, dtype=np.float32)
    assert list(shift_zero_fill(x, 3)) == [0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
    assert list(shift_zero_fill(x, -3)) == [4, 5, 6, 7, 8, 9, 10, 0, 0, 0]


def test_augmentation_is_train_only_guard():
    assert_train_only(["train"])
    with pytest.raises(ValueError):
        assert_train_only(["train", "val"])
    with pytest.raises(ValueError):
        assert_train_only(["test"])


def test_augmentation_deterministic_and_within_bounds(cfg):
    a = cfg["phase2"]["augmentation"]
    x = (0.05 * np.random.default_rng(3).standard_normal(16000)).astype(np.float32)
    pool = np.stack([x] * 8); nv = np.full(8, 16000)
    r1 = augment_train_clip(x, 12000, pool, nv, 0, a, 16000, np.random.default_rng(7))
    r2 = augment_train_clip(x, 12000, pool, nv, 0, a, 16000, np.random.default_rng(7))
    assert np.array_equal(r1[0], r2[0]) and r1[3] == r2[3]
    for s in range(200):
        _, _, _, p = augment_train_clip(x, 12000, pool, nv, 0, a, 16000, np.random.default_rng(s))
        assert a["gain_db"][0] <= p["gain_db"] <= a["gain_db"][1]
        assert abs(p["shift_samples"]) <= 1600
        assert p["noise_type"] in ["none"] + a["noise_types"]
        if p["noise_type"] != "none":
            assert p["snr_db"] in a["noise_snr_db"] and abs(p["measured_snr_db"] - p["snr_db"]) < 0.1


def test_features_finite_shapes_and_padding_invariance(cfg):
    t = np.arange(16000) / 16000
    x = np.zeros(16000, dtype=np.float32)
    x[:9000] = (0.2 * np.sin(2 * np.pi * 300 * t[:9000]) * np.hanning(9000)).astype(np.float32)
    o = clip_features(x, 0, 9000, cfg)
    assert o["mfcc_vec"].shape == (52,) and o["eng_vec"].shape == (len(ENGINEERED_NAMES),) and o["logmel"].shape == (64, 101)
    assert np.isfinite(o["mfcc_vec"]).all() and np.isfinite(o["eng_vec"]).all()
    # features are computed over valid frames only: extra zero padding must not change the MFCC summary much
    x2 = np.concatenate([x[:9000], np.zeros(7000, dtype=np.float32)])
    o2 = clip_features(x2, 0, 9000, cfg)
    assert np.allclose(o["mfcc_vec"], o2["mfcc_vec"])


def test_scaler_and_pca_fit_on_train_only():
    from src.discovery import Space
    rng = np.random.default_rng(0)
    Xtr = rng.standard_normal((300, 12))
    Xva_a = rng.standard_normal((50, 12)) * 100        # wildly different val data
    sp = Space(Xtr, "standard", 5, 0)
    mu, sd = sp.sc.mean_.copy(), sp.sc.scale_.copy()
    sp.z(Xva_a)                                        # transforming val must not change the fitted state
    assert np.array_equal(mu, sp.sc.mean_) and np.array_equal(sd, sp.sc.scale_)
    assert np.allclose(mu, Xtr.mean(0))


def test_eta2_properties():
    from src.discovery import eta2
    x = np.r_[np.zeros(50), np.ones(50)]
    assert eta2(x, ["a"] * 50 + ["b"] * 50) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(2000)
    assert eta2(v, rng.integers(0, 5, 2000)) < 0.01
