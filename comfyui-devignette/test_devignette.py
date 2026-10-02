"""Tests for the de-vignette core. Pure numpy/PIL, no ComfyUI, no torch:

    python test_devignette.py [path/to/folder/of/photos]

The unit tests run anywhere. The accuracy benchmark needs a folder of real
photographs; point it at one and it will measure the estimator against known
vignettes. Without one it is skipped.

The benchmark uses a DIFFERENTIAL metric, which is the only fair way to score
a blind estimator on real photos: measure the frame, add a known falloff,
measure again, and check the difference equals what was added. Any vignetting
the photograph already carried cancels out, so the number reflects the
estimator rather than an unknown baseline.
"""
from __future__ import annotations

import os
import sys
import glob

import numpy as np

import devignette_core as core


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def apply_synthetic_vignette(img, stops, shape=(0.30, 0.70, 0.0)):
    """Darken an image with a known monotone falloff, in linear light."""
    h, w = img.shape[:2]
    c = np.array(shape, dtype=float) * stops * np.log(2)
    gain = core.build_gain_map(h, w, c, 1.0, strength=1.0, max_gain=100.0)
    lin = core.srgb_to_linear(img.astype(np.float64))
    return core.linear_to_srgb(np.clip(lin / gain[..., None], 0, 1)).astype(np.float32)


def _checkerboard(h=600, w=900, seed=0):
    """Synthetic 'scene' with plenty of texture but no radial structure."""
    rng = np.random.default_rng(seed)
    base = rng.random((h // 20 + 1, w // 20 + 1, 3)) * 0.5 + 0.25
    img = np.repeat(np.repeat(base, 20, axis=0), 20, axis=1)[:h, :w]
    return np.clip(img + rng.normal(0, 0.01, img.shape), 0.02, 0.98).astype(np.float32)


FAILURES = []


def check(cond, label, detail=""):
    status = "PASS" if cond else "FAIL"
    print("  [%s] %s%s" % (status, label, ("  -- " + detail) if detail and not cond else ""))
    if not cond:
        FAILURES.append(label)


# ---------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------

def test_gain_is_monotone_and_never_darkens():
    """The bug this exists to catch: an unconstrained polynomial fit will
    happily produce a gain map that DARKENS a mid-frame annulus while
    brightening the corners. In darken_only mode the gain must be >= 1
    everywhere and never decrease outward."""
    print("\ngain map shape guarantees")
    img = _checkerboard()
    vig = apply_synthetic_vignette(img, 0.8)
    res = core.compute_devignette_gain(vig, strength=1.0, max_gain=2.0,
                                       correction_mode="darken_only")
    g = res["gain_map"]
    check(g.min() >= 1.0 - 1e-6, "gain never dips below 1.0 in darken_only",
          "min gain %.4f" % g.min())

    c = res["c"]
    s = np.linspace(0, 1, 200)
    d = c[0] * s + c[1] * s ** 2 + c[2] * s ** 3
    check(np.all(np.diff(d) >= -1e-12), "falloff is monotonic in r^2")
    check(abs(d[0]) < 1e-12, "falloff is exactly unity at the optical center")

    h, w = vig.shape[:2]
    check(abs(g[h // 2, w // 2] - 1.0) < 1e-3, "center pixel gain == 1.0",
          "got %.4f" % g[h // 2, w // 2])
    check(g[0, 0] >= g[h // 2, 0] >= g[h // 2, w // 2], "gain increases toward the corner")


def test_max_gain_is_respected():
    """max_gain must be both a hard ceiling AND reachable -- a user who asks
    for 1.5x on a heavily vignetted frame should get 1.5x, not 1.41x because
    a roll-off quietly ate the top of the range."""
    print("\nmax_gain ceiling")
    img = _checkerboard()
    vig = apply_synthetic_vignette(img, 2.5)          # far more than any cap
    for mg in (1.2, 1.5, 2.0):
        res = core.compute_devignette_gain(vig, strength=1.0, max_gain=mg)
        peak = res["gain_map"].max()
        check(peak <= mg + 1e-4, "max_gain=%.1f is a hard ceiling" % mg,
              "peak %.4f" % peak)
        check(peak > mg - 0.02, "max_gain=%.1f is reachable" % mg,
              "peak only %.4f" % peak)

    # strength > 1 deliberately overshoots; the ceiling must still hold.
    res = core.compute_devignette_gain(vig, strength=2.0, max_gain=1.5)
    check(res["gain_map"].max() <= 1.5 + 1e-4, "ceiling holds at strength=2.0",
          "peak %.4f" % res["gain_map"].max())


def test_threshold_cannot_become_unreachable():
    """A threshold above what max_gain can even express would make the node
    silently do nothing forever: max_gain 1.2 tops out at 16.6% falloff, so a
    20% threshold could never be met. The gate has to fall back to what is
    measurable."""
    print("\nthreshold vs ceiling interaction")
    vig = apply_synthetic_vignette(_checkerboard(), 2.5)
    res = core.compute_devignette_gain(vig, max_gain=1.2, min_correction_pct=20.0)
    check(res["applied"], "heavy vignette still corrected when max_gain < threshold")
    check(res["gain_map"].max() > 1.19, "and the correction reaches the ceiling",
          "peak %.4f" % res["gain_map"].max())


def test_flat_image_is_left_alone():
    """The estimator has a small positive noise floor and it is structural:
    because c >= 0, noise can only ever push a reading UP, never down, so a
    genuinely vignette-free frame reads slightly positive on average. What
    matters is not that it never trips, but that when it does the correction
    it applies is negligible."""
    print("\nno meaningful false positive on a vignette-free frame")
    raw, peak = [], []
    for seed in range(8):
        img = _checkerboard(seed=seed)
        res = core.compute_devignette_gain(img, strength=1.0, max_gain=1.5)
        raw.append(res["corner_atten_stops"])
        peak.append(res["gain_map"].max())
    check(max(raw) < 0.25, "noise floor stays under 0.25 stops",
          "worst %.3f" % max(raw))
    check(max(peak) < 1.12, "any spurious correction is imperceptible",
          "worst peak gain %.3fx" % max(peak))


def test_strength_zero_is_identity():
    print("\nstrength / passthrough")
    img = _checkerboard(seed=5)
    vig = apply_synthetic_vignette(img, 0.8)
    res = core.compute_devignette_gain(vig, strength=0.0, max_gain=1.5)
    out = core.apply_gain(vig, res["gain_map"])
    check(np.abs(out - vig).max() < 2e-3, "strength=0 leaves the image unchanged",
          "max delta %.5f" % np.abs(out - vig).max())


def test_alpha_channel_is_not_scaled():
    print("\nextra channels")
    img = _checkerboard(seed=7)
    rgba = np.concatenate([img, np.full(img.shape[:2] + (1,), 0.5, np.float32)], axis=2)
    gain = np.full(img.shape[:2], 1.5, np.float32)
    out = core.apply_gain(rgba, gain)
    check(np.allclose(out[..., 3], 0.5), "alpha passes through unscaled")


def test_center_offset_moves_the_center():
    print("\noptical center offset")
    g0 = core.build_gain_map(400, 400, np.array([0.0, 0.5, 0.0]), 1.0, max_gain=4.0)
    g1 = core.build_gain_map(400, 400, np.array([0.0, 0.5, 0.0]), 1.0, max_gain=4.0,
                             center_x_offset=0.2)
    check(abs(g0[200, 200] - 1.0) < 1e-4, "unshifted center sits at the middle")
    check(abs(g1[200, 280] - 1.0) < 1e-3, "shifted center follows center_x_offset",
          "gain there %.4f" % g1[200, 280])


def test_downsample_averages_in_linear_light():
    """Averaging gamma-encoded values biases every block by an amount that
    depends on its local contrast -- noise injected straight into the
    measurement. Check we average the light, not the encoding."""
    print("\nanalysis downsample")
    img = np.zeros((64, 64, 3), np.float32)
    img[:, ::2] = 1.0                       # alternating black/white columns
    luma, valid, _, _ = core.downsampled_luma(img, 8, linear_light=True)
    check(abs(luma.mean() - 0.5) < 0.02,
          "half-black/half-white averages to 0.5 in linear light",
          "got %.4f" % luma.mean())
    check(valid.mean() < 0.05, "fully clipped pixels are marked invalid",
          "valid fraction %.3f" % valid.mean())


def test_auto_mode_catches_reverse_vignette():
    print("\nauto mode")
    img = _checkerboard(seed=11)
    h, w = img.shape[:2]
    gain = core.build_gain_map(h, w, np.array([0.0, 0.55, 0.0]), 1.0, max_gain=4.0)
    bright = np.clip(img * gain[..., None], 0, 1).astype(np.float32)   # corners BRIGHTER
    r_dark = core.compute_devignette_gain(bright, correction_mode="darken_only",
                                          max_gain=1.5)
    r_auto = core.compute_devignette_gain(bright, correction_mode="auto", max_gain=1.5)
    check(not r_dark["applied"], "darken_only ignores brightened corners")
    check(r_auto["applied"] and r_auto["sign"] < 0,
          "auto detects brightened corners", "sign %.0f" % r_auto["sign"])
    check(r_auto["gain_map"].max() <= 1.0 + 1e-6,
          "reverse correction darkens rather than brightens")


# ---------------------------------------------------------------------------
# accuracy benchmark on real photographs
# ---------------------------------------------------------------------------

def benchmark(folder, long_edge=900, limit=16):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(glob.glob(os.path.join(folder, ext)))
    files = sorted(files)[:limit]
    if not files:
        print("\nno images found in %s -- benchmark skipped" % folder)
        return

    def load(p):
        im = Image.open(p).convert("RGB")
        w, h = im.size
        s = long_edge / max(w, h)
        return np.asarray(im.resize((max(8, int(w * s)), max(8, int(h * s))),
                                    Image.LANCZOS)).astype(np.float32) / 255.0

    print("\n" + "=" * 70)
    print("ACCURACY BENCHMARK -- %d photos, differential metric" % len(files))
    print("=" * 70)

    # Generous cap so this measures the estimator, not the ceiling.
    CAP_STOPS = 3.0

    def detect(im):
        return core.compute_devignette_gain(im, max_gain=2.0 ** CAP_STOPS,
                                            min_correction_pct=0.0)["corner_atten_stops"]

    # A differential score is only meaningful while the estimator is in its
    # linear range. Two ways out of it, both of which make the difference
    # mean nothing rather than indicate error:
    #   floored  -- the frame's corners are genuinely BRIGHTER than its
    #               center, so c >= 0 pins the reading at zero and adding a
    #               vignette just moves it back toward, not past, zero.
    #   capped   -- the reading is already at sum(c) <= cap, so it has no
    #               room to rise no matter what we add.
    errs, skipped = [], 0
    for p in files:
        img = load(p)
        d0 = detect(img)
        for applied in (0.3, 0.6, 1.0):
            d1 = detect(apply_synthetic_vignette(img, applied))
            if d1 < 0.05 or max(d0, d1) > 0.95 * CAP_STOPS:
                skipped += 1
                continue
            errs.append((d1 - d0) - applied)

    errs = np.array(errs)
    print("  n = %d  (%d saturated samples excluded)" % (errs.size, skipped))
    print("  bias           %+.3f stops" % errs.mean())
    print("  mean abs error  %.3f stops" % np.abs(errs).mean())
    print("  90th pct error  %.3f stops" % np.percentile(np.abs(errs), 90))
    check(abs(errs.mean()) < 0.15, "benchmark bias within 0.15 stops")
    check(np.abs(errs).mean() < 0.25, "benchmark mean abs error within 0.25 stops")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_gain_is_monotone_and_never_darkens()
    test_max_gain_is_respected()
    test_threshold_cannot_become_unreachable()
    test_flat_image_is_left_alone()
    test_strength_zero_is_identity()
    test_alpha_channel_is_not_scaled()
    test_center_offset_moves_the_center()
    test_downsample_averages_in_linear_light()
    test_auto_mode_catches_reverse_vignette()

    if len(sys.argv) > 1:
        benchmark(sys.argv[1])
    else:
        print("\n(pass a folder of photos as an argument to run the accuracy benchmark)")

    print("\n" + ("ALL PASS" if not FAILURES else "FAILED: " + ", ".join(FAILURES)))
    sys.exit(1 if FAILURES else 0)
