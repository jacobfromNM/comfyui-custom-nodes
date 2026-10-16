"""
Core de-vignetting math. Deliberately framework-agnostic (numpy + PIL only,
no torch) so it can be imported and unit-tested completely outside of
ComfyUI, and so the ComfyUI node itself stays a thin wrapper.

The model:
    Vignetting is a MULTIPLICATIVE, RADIALLY-SYMMETRIC light falloff:
        observed(x, y) = true_scene(x, y) * falloff(r)
    where r is distance from the optical center, normalized so r = 1 at the
    image corner farthest from that center.

    Real scenes are almost never radially symmetric on their own (a sky-to-
    ground gradient, a window's light falling off left-to-right, a bright
    band of sunlit rock across the middle of the frame). Vignetting, in
    contrast, is rotationally symmetric around the lens axis -- it darkens
    all four corners in a comparable way *because it comes from geometry
    that doesn't know anything about your scene content*.

    So we fit both components at once, on a heavily smoothed version of the
    image (to strip out local detail/texture), then only ever invert the
    RADIAL term. The scene term is fit purely so it doesn't contaminate the
    radial estimate -- it is discarded, never corrected. That's what keeps
    this from "fixing" a bright sky or a dark foreground into mush.

How the scene/vignette split is actually made
---------------------------------------------
Everything is done in log-luminance, where the multiplication above becomes
an addition, and expanded in ANGULAR HARMONICS about the optical center:

    log L(r, t) = sum over m of  [ radial polynomial in r ] * cos(m t), sin(m t)

The m = 0 (rotationally invariant) terms are the only place vignetting can
live, by definition. Every m >= 1 term is scene structure. We fit both and
keep only m = 0.

Two things about that are easy to get wrong, and both were measured rather
than assumed (see _scene_basis):

* The scene model must stay DELIBERATELY POOR -- here, a directional tilt
  (m = 1) and nothing more. Make it richer and it eventually spans the
  isotropic functions itself, at which point the two models are degenerate
  and the split is decided by numerical noise. Bases rich enough to include
  saddle-shaped (m = 2) terms were measured reporting 3-4 stops of
  "vignetting" on perfectly ordinary photographs.
* Scene and vignette must be fit JOINTLY. Over a rectangular frame a tilt
  and a radial bowl are partly correlated, and arbitrating that overlap is
  the whole job. Orthogonalizing the bases to remove the overlap is
  self-defeating: it leaves the scene block unable to influence the radial
  estimate at all.

The one thing this decomposition genuinely cannot separate is scene content
that is itself rotationally symmetric about the frame center -- most often
a sun/flare glow near the middle, which reads as "bright center, dark
corners" no matter how you slice it. The constraints below are what keep
that case from running away; it is still worth eyeballing via the gain
preview.

Why the radial term is CONSTRAINED
----------------------------------
Physical vignetting is monotonic: brightness never goes back up as you move
away from the optical axis. An unconstrained r^2/r^4 polynomial has no idea
about that, and on a busy frame it will happily land on a shape that
*darkens* a mid-frame annulus in order to fit scene content -- which, in
"darken_only" mode, is the exact opposite of what the user asked for.

So the radial falloff is parameterized as

    log falloff(s) = -(c1*s + c2*s^2 + c3*s^3),   s = r^2,   c1,c2,c3 >= 0

With all c >= 0 the falloff is guaranteed to be <= 1, to equal exactly 1 at
the center, and to be monotonically decreasing outward -- structurally, for
any data, rather than as something we hope the fit lands on.

A second constraint, sum(c) <= log(max_gain), caps how much falloff the fit
is allowed to claim. Blind single-image estimation systematically over-reads
on frames that are legitimately bright in the middle, so this ceiling is
doing real work as a physical-plausibility prior -- not just trimming the
output. Applying it inside the fit (rather than clipping the finished gain
map) is what keeps the recovered falloff a smooth curve instead of one whose
outer field is a flat clipped plateau with a visible seam.

Both constraints are handled by a small exact solve nested inside the robust
(Huber IRLS) fit.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

EPS = 1e-6

# How expressive the scene-lighting model is allowed to be. Both numbers were
# chosen by measurement, not taste -- see _scene_basis for why bigger is worse.
SCENE_MAX_M = 1      # angular harmonics: a directional tilt, nothing more
SCENE_DEGREE = 3     # radial reach of those harmonics

# Ceiling on the "auto" mode's reverse correction (darkening corners that came
# out brighter than the middle). Deliberately much tighter than max_gain --
# see the comment in fit_illumination_model for why the two directions cannot
# share an allowance.
REVERSE_MAX_GAIN = 1.35
# Number of radial (vignette) basis functions: s, s^2, s^3.
_N_VIG = 3


# ---------------------------------------------------------------------------
# Color space helpers
# ---------------------------------------------------------------------------

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB (gamma-encoded, 0..1) -> linear light. Vignetting physically
    happens to light hitting the sensor, i.e. in linear space, so that's
    where the multiplicative correction belongs for the least color skew."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


# ---------------------------------------------------------------------------
# Coordinate grid
# ---------------------------------------------------------------------------

def axis_coords(h: int, w: int, cx_off: float = 0.0, cy_off: float = 0.0,
                dtype=np.float32):
    """Returns 1-D (rx_row, ry_col) normalized signed coordinates.

    Kept separable on purpose: r^2 = ry^2[:, None] + rx^2[None, :] builds the
    full grid by broadcasting, so a 24 MP image never needs the two full-size
    float64 meshgrids the old np.mgrid path allocated.

    Normalization puts r^2 == 1.0 exactly at the image corner farthest from
    the chosen center, which makes every downstream coefficient directly
    interpretable as "log-attenuation at the corner", regardless of
    resolution or aspect ratio.

    cx_off / cy_off shift the assumed optical center as a fraction of
    width/height (e.g. 0.05 = 5% of width to the right), for images where
    the lens axis isn't at the geometric center (asymmetric crops, stitched
    panoramas, etc).
    """
    cx = w / 2.0 + cx_off * w
    cy = h / 2.0 + cy_off * h

    # Farthest corner distance, using pixel-center coordinates.
    corners_x = np.array([0.0, w - 1.0]) - cx
    corners_y = np.array([0.0, h - 1.0]) - cy
    r_max = np.sqrt((corners_y[:, None] ** 2 + corners_x[None, :] ** 2)).max()
    r_max = max(float(r_max), 1e-6)

    rx = ((np.arange(w, dtype=np.float64) - cx) / r_max).astype(dtype)
    ry = ((np.arange(h, dtype=np.float64) - cy) / r_max).astype(dtype)
    return rx, ry


def build_coord_grid(h: int, w: int, cx_off: float = 0.0, cy_off: float = 0.0):
    """Full (rx, ry, r2) grids. Only used at analysis resolution (a few
    hundred pixels a side) -- the full-resolution path stays separable."""
    rx1, ry1 = axis_coords(h, w, cx_off, cy_off, dtype=np.float64)
    rx = np.broadcast_to(rx1[None, :], (h, w))
    ry = np.broadcast_to(ry1[:, None], (h, w))
    r2 = ry ** 2 + rx ** 2
    return rx, ry, r2


# ---------------------------------------------------------------------------
# Design matrices
# ---------------------------------------------------------------------------

def _radial_basis(rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
    """The rotationally invariant powers s, s^2, s^3 (s = r^2).

    Only even powers of r appear, because a physical falloff is smooth
    through the optical axis."""
    s = (rx * rx + ry * ry).reshape(-1)
    return np.stack([s, s * s, s * s * s], axis=-1)


def _scene_basis(rx: np.ndarray, ry: np.ndarray, max_m: int = SCENE_MAX_M,
                 degree: int = SCENE_DEGREE) -> np.ndarray:
    """What we allow the SCENE's own lighting to look like: a constant plus
    angular harmonics m = 1..max_m at radial orders m, m+2, ... <= degree.

    r^n*cos(m t) and r^n*sin(m t) are just the real and imaginary parts of
    (rx + i*ry)^m * s^k, which is why this is four lines.

    Why the scene model is deliberately POOR
    ----------------------------------------
    It is tempting to make this basis as expressive as possible so it soaks
    up more scene structure. That reasoning breaks down completely, and it
    is worth spelling out why, because the failure is silent.

    A vignette is an isotropic function of r. A rich enough polynomial
    scene basis CONTAINS the isotropic functions -- a complete degree-6
    2-D polynomial spans s, s^2 and s^3 exactly -- so the two models become
    literally degenerate and the split is decided by nothing but numerical
    noise. Orthogonalizing the scene basis against the radial one doesn't
    rescue it either: that makes the scene block exactly orthogonal to the
    radial block, so it stops influencing the radial estimate at all and
    you are back to fitting a bare radial average of the frame.

    So the scene basis has to be a genuinely restricted family, and where
    to draw that line is an empirical question, not an aesthetic one. It
    was settled by measuring, over a spread of real photographs: apply a
    KNOWN vignette, re-estimate, and check the recovered magnitude, while
    separately watching what each candidate reports on untouched frames.
    Bases with m >= 2 (saddle-shaped terms) scored slightly better on the
    known-vignette test but reported 3-4 stops of "vignetting" on ordinary
    photographs -- they had become expressive enough to start trading with
    the radial term. m = 1 through radial order 3 recovers a known falloff
    to about 0.14 stops while keeping its readings on real frames bounded,
    so that is what ships.
    """
    z = rx + 1j * ry
    s = rx * rx + ry * ry
    cols = [np.ones_like(rx)]
    for m in range(1, max_m + 1):
        base = z ** m
        for k in range(0, (degree - m) // 2 + 1):
            t = base * (s ** k)
            cols.append(t.real)
            cols.append(t.imag)
    return np.stack(cols, axis=-1).reshape(-1, len(cols))


# ---------------------------------------------------------------------------
# Constrained radial solve
# ---------------------------------------------------------------------------

def _solve_constrained(G: np.ndarray, h: np.ndarray, n_scene: int,
                       cap: float) -> np.ndarray:
    """Minimize x'Gx - 2x'h where the first n_scene entries (the scene
    lighting coefficients) are free and the last three -- the radial
    falloff -- are confined to {c >= 0, sum(c) <= cap}.

    That feasible region is a small simplex in 3-D, so we enumerate its
    faces: every subset of the c's pinned to zero, crossed with the cap
    being active or slack. Each face is one equality-constrained least
    squares solve, and every point of the polytope lies on some enumerated
    face, so the best feasible answer found is the exact global optimum --
    16 tiny linear solves, no iteration, no tolerance to tune.

    The two constraints carry all the physics:
      c >= 0        the falloff is monotonic outward and exactly 1.0 at the
                    optical center, structurally, for any data. Without it
                    the polynomial happily darkens a mid-frame annulus in
                    order to chase scene content, which is the opposite of
                    what "remove the vignette" means.
      sum(c) <= cap the corner attenuation cannot exceed what the user
                    declared physically plausible (max_gain). Enforcing it
                    HERE rather than clipping the finished gain map is the
                    difference between a smooth, gentler falloff and one
                    whose whole outer field is a flat clipped plateau with a
                    visible seam. It is also the main guard against
                    over-reading a scene that happens to be bright in the
                    middle, which no amount of fitting can rule out.
    """
    n = G.shape[0]
    base = list(range(n_scene))
    best_x, best_obj = None, np.inf
    for bits in range(1 << _N_VIG):
        free = base + [n_scene + k for k in range(_N_VIG) if (bits >> k) & 1]
        nf = len(free)
        for cap_active in (False, True):
            if cap_active and nf == n_scene:
                continue           # no c free: sum(c) = 0 can't meet a positive cap
            sub = G[np.ix_(free, free)].copy()
            sub.flat[:: nf + 1] += 1e-12 * max(np.trace(sub), 1.0)
            try:
                if cap_active:
                    e = np.array([0.0] * n_scene + [1.0] * (nf - n_scene))
                    K = np.zeros((nf + 1, nf + 1))
                    K[:nf, :nf] = sub
                    K[:nf, nf] = e
                    K[nf, :nf] = e
                    sol = np.linalg.solve(K, np.concatenate([h[free], [cap]]))[:nf]
                else:
                    sol = np.linalg.solve(sub, h[free])
            except np.linalg.LinAlgError:
                continue
            x = np.zeros(n)
            x[free] = sol
            if np.any(x[n_scene:] < -1e-10) or x[n_scene:].sum() > cap + 1e-9:
                continue
            x[n_scene:] = np.maximum(x[n_scene:], 0.0)
            obj = float(x @ G @ x - 2.0 * x @ h)
            if obj < best_obj:
                best_obj, best_x = obj, x
    return np.zeros(n) if best_x is None else best_x


def fit_illumination_model(luma: np.ndarray, rx: np.ndarray, ry: np.ndarray,
                           r2: np.ndarray | None = None,
                           weights: np.ndarray | None = None,
                           allow_brightening: bool = False,
                           cap: float = np.log(1.5),
                           reverse_cap: float = np.log(REVERSE_MAX_GAIN),
                           huber_c: float = 1.345, iters: int = 4) -> dict:
    """Fits, in log-luminance:

        log(L) = scene(rx, ry) . beta  -  sign * (c1*s + c2*s^2 + c3*s^3)

    with c >= 0, sum(c) <= cap, s = r^2, and sign fixed at +1 (corners
    darker than center) unless allow_brightening lets it also try -1 and
    keep whichever explains the frame better.

    Scene and radial blocks are fit JOINTLY, not one after the other. They
    overlap -- a directional tilt across a rectangular frame is partly
    correlated with a radial bowl -- and it is precisely that overlap the
    joint solve has to arbitrate. (Removing the overlap by orthogonalizing
    the two bases sounds tidier and is in fact fatal: it leaves the scene
    block with no influence on the radial estimate whatsoever.)

    The outer loop is iteratively-reweighted least squares with Huber
    weights rather than a plain lstsq: even after heavy downsampling, a
    stray bright window or dark doorway can still tilt an ordinary
    least-squares fit, and Huber weighting caps the influence any single
    region of the frame can have.

    Returns dict with c (3,), sign, and the weighted residual scale.
    """
    y = np.log(np.maximum(luma.astype(np.float64).ravel(), EPS))
    rx = rx.astype(np.float64)
    ry = ry.astype(np.float64)
    scene = _scene_basis(rx, ry)
    vig = _radial_basis(rx, ry)
    n_scene = scene.shape[1]

    base_w = np.ones_like(y) if weights is None else weights.astype(np.float64).ravel()

    signs = (1.0, -1.0) if allow_brightening else (1.0,)
    best = None
    for sign in signs:
        # The two directions do NOT get the same allowance. Correcting darker
        # corners is the job, so it gets the user's full max_gain. Correcting
        # BRIGHTER corners means darkening them, i.e. adding a vignette, and a
        # large allowance there is actively dangerous: a sunset glow in one
        # corner reads as a bright edge, and with a loose cap the node will
        # commit to it and crush the corners to a third of their brightness.
        # Genuine reverse-vignette artifacts (sensor hot spots and the like)
        # are mild, so a mild ceiling loses nothing real.
        sign_cap = cap if sign > 0 else min(cap, reverse_cap)
        A = np.concatenate([scene, -sign * vig], axis=1)
        w = base_w.copy()
        c = np.zeros(_N_VIG)
        resid = y
        for _ in range(iters):
            Aw = A * w[:, None]
            G = A.T @ Aw
            hvec = Aw.T @ y
            G.flat[:: G.shape[0] + 1] += 1e-9 * np.trace(G) / G.shape[0]
            x = _solve_constrained(G, hvec, n_scene, sign_cap)
            c = x[n_scene:]

            resid = y - A @ x
            sigma = np.median(np.abs(resid[base_w > 0])) * 1.4826 + 1e-6
            u = np.abs(resid) / (huber_c * sigma)
            w = base_w * np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))

        cost = float(np.sum(base_w * resid ** 2) / max(base_w.sum(), 1.0))
        if best is None or cost < best["cost"]:
            best = {"c": c, "sign": float(sign), "cost": cost}

    c = best["c"]
    return {
        "c": c,
        "sign": best["sign"],
        "cost": best["cost"],
        # log-attenuation at the corner; positive = corners darker than center
        "corner_log_atten": float(best["sign"] * c.sum()),
    }


# ---------------------------------------------------------------------------
# Analysis-resolution luma extraction
# ---------------------------------------------------------------------------

def _box_matrix(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) area-averaging matrix. Exact box/area resampling for any
    ratio, including non-integer ones, as a plain matrix so the whole
    downsample is two BLAS calls."""
    edges = np.linspace(0.0, n_in, n_out + 1)
    M = np.zeros((n_out, n_in), dtype=np.float64)
    for j in range(n_out):
        lo, hi = edges[j], edges[j + 1]
        i0, i1 = int(np.floor(lo)), int(np.ceil(hi))
        for i in range(i0, min(i1, n_in)):
            M[j, i] = max(0.0, min(hi, i + 1.0) - max(lo, float(i)))
        tot = M[j].sum()
        if tot > 0:
            M[j] /= tot
    return M


def downsampled_luma(img_float: np.ndarray, target_long_edge: int = 256,
                     linear_light: bool = True, chunk_rows: int = 512):
    """Area-averaged downsample that doubles as a strong low-pass filter --
    it's what strips out texture/detail so the fit only ever sees the smooth
    illumination field, not the scene's actual content.

    Two things it does that the old PIL path did not:

    * It averages in LINEAR light (when linear_light is set). Vignetting is
      multiplicative on linear radiance; averaging gamma-encoded values
      first biases every block by an amount that depends on how much local
      contrast happens to be in it, which is noise injected straight into
      the thing we're trying to measure.
    * It never round-trips through uint8, so shadow detail keeps its
      precision instead of being quantized to ~1/255 right before we take
      its logarithm.

    Also returns a per-cell validity weight in 0..1: pixels that were
    clipped at white or crushed at black carry no usable multiplicative
    information (you cannot tell how much light a blown highlight *would*
    have had), so cells made mostly of them are down-weighted out of the
    fit.

    Returns (luma, valid_weight, small_h, small_w).
    """
    h, w = img_float.shape[:2]
    scale = target_long_edge / max(h, w)
    small_h = int(max(8, min(h, round(h * scale))))
    small_w = int(max(8, min(w, round(w * scale))))

    Rh = _box_matrix(h, small_h)
    Rw = _box_matrix(w, small_w).T          # (w, small_w)

    luma_acc = np.zeros((small_h, small_w), dtype=np.float64)
    valid_acc = np.zeros((small_h, small_w), dtype=np.float64)

    rgb = img_float[..., :3]
    for i in range(0, h, chunk_rows):
        j = min(i + chunk_rows, h)
        blk = np.clip(rgb[i:j].astype(np.float32), 0.0, 1.0)

        mx = blk.max(axis=-1)
        ok = ((mx < 0.996) & (mx > 0.004)).astype(np.float64)

        lin = srgb_to_linear(blk) if linear_light else blk.astype(np.float64)
        luma = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]

        luma_acc += Rh[:, i:j] @ (luma @ Rw)
        valid_acc += Rh[:, i:j] @ (ok @ Rw)

    return luma_acc, valid_acc, small_h, small_w


# ---------------------------------------------------------------------------
# Gain map construction
# ---------------------------------------------------------------------------

def _soft_limit_log(lg: np.ndarray, limit: float, knee_frac: float = 0.6) -> np.ndarray:
    """Squashes |lg| toward `limit` with a smooth exponential roll-off
    instead of a hard clip, in place.

    A hard np.clip on the gain map turns the outer part of the falloff into
    a flat plateau with a visible circular seam where the clip kicks in --
    exactly the kind of ring artifact that reads as "processed". Rolling off
    keeps the first derivative continuous, so the ceiling is approached
    rather than hit.
    """
    if limit <= 0:
        lg[:] = 0.0
        return lg
    knee = knee_frac * limit
    span = max(limit - knee, 1e-6)
    a = np.abs(lg)
    over = a > knee
    if over.any():
        excess = a[over] - knee
        a_over = knee + span * (1.0 - np.exp(-excess / span))
        lg[over] = np.sign(lg[over]) * a_over
    return lg


def build_gain_map(h: int, w: int, c: np.ndarray, sign: float, *,
                   strength: float = 1.0, max_gain: float = 1.5,
                   center_x_offset: float = 0.0, center_y_offset: float = 0.0,
                   chunk_rows: int = 512) -> np.ndarray:
    """Evaluates the fitted falloff at full resolution, row-chunked and in
    float32.

    The whole map is a function of s = r^2 alone, and s is separable, so
    each row block only needs one (rows, w) temporary. On a 24 MP image this
    is the difference between ~100 MB and well over a gigabyte of float64
    meshgrids and intermediates.
    """
    rx1, ry1 = axis_coords(h, w, center_x_offset, center_y_offset, dtype=np.float32)
    rx2 = (rx1 ** 2)[None, :]
    ry2 = (ry1 ** 2)

    c1, c2, c3 = (float(v) for v in c)
    log_limit = float(np.log(max(max_gain, 1.0 + 1e-6)))

    # The falloff is monotonic, so its extreme is at the corner (s = 1) and we
    # know the peak in closed form. The fit already enforces sum(c) <= the same
    # limit, so at strength <= 1 the map cannot breach the ceiling and the
    # limiter would only rob the correction of its top end -- max_gain=1.5
    # would quietly deliver 1.41x. Engage it only when it is actually needed,
    # which in practice means only when strength > 1 deliberately overshoots.
    peak = abs(strength) * (c1 + c2 + c3)
    needs_limit = peak > log_limit + 1e-9

    out = np.empty((h, w), dtype=np.float32)
    for i in range(0, h, chunk_rows):
        j = min(i + chunk_rows, h)
        s = ry2[i:j, None] + rx2                       # (rows, w) float32
        # Horner: d = ((c3*s + c2)*s + c1)*s  -- the log-attenuation at s
        d = s * c3 + c2
        d *= s
        d += c1
        d *= s
        # gain that exactly undoes it, then blended by strength in log space
        # so strength scales the falloff shape rather than the gain values.
        np.multiply(d, np.float32(sign * strength), out=d)
        if needs_limit:
            _soft_limit_log(d, log_limit)
        np.exp(d, out=d)
        out[i:j] = d
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_devignette_gain(img_float: np.ndarray, *, strength: float = 1.0,
                            max_gain: float = 3.0,
                            correction_mode: str = "auto",
                            min_correction_pct: float = 20.0,
                            center_x_offset: float = 0.0,
                            center_y_offset: float = 0.0,
                            analysis_size: int = 256,
                            linear_light: bool = True) -> dict:
    """img_float: H,W,C array, values in [0, 1].

    Returns a dict with:
      gain_map          (H, W) float32, the per-pixel multiplicative gain,
                        already blended by strength and soft-limited.
      applied           bool, whether any correction was applied (False if
                        the detector decided the image has no vignette).
      corner_atten_stops  raw detected corner falloff in stops (log2);
                        positive = corners darker than center.
      c, sign           the fitted radial coefficients, for diagnostics.
    """
    h, w = img_float.shape[:2]

    luma_s, valid_s, sh, sw = downsampled_luma(img_float, analysis_size,
                                               linear_light=linear_light)
    rx_s, ry_s, _ = build_coord_grid(sh, sw, center_x_offset, center_y_offset)

    # Cells that are mostly clipped tell us nothing about how much light was
    # really there, so they only get a vote proportional to how much of them
    # survived. Squaring sharpens the rejection of half-blown cells.
    wgt = np.clip(valid_s, 0.0, 1.0) ** 2
    if wgt.sum() < 0.05 * wgt.size:
        wgt = np.ones_like(wgt)   # degenerate (nearly all clipped): don't starve the fit

    # max_gain is a statement about what falloff is physically plausible, so
    # it belongs in the fit as a constraint -- not as a clip on the finished
    # gain map, which would just flatten the outer field into a plateau.
    cap = float(np.log(max(max_gain, 1.0 + 1e-6)))
    fit = fit_illumination_model(
        luma_s, rx_s, ry_s, weights=wgt,
        allow_brightening=(correction_mode != "darken_only"),
        cap=cap,
    )
    c = fit["c"]
    sign = fit["sign"]
    corner_log_atten = fit["corner_log_atten"]   # +ve = corners darker

    # Threshold on the magnitude of the detected falloff.
    min_log_atten = -np.log(max(1e-6, 1.0 - min_correction_pct / 100.0))  # positive

    # The fit can never report more falloff than its own ceiling allows, so a
    # threshold set above that ceiling would be permanently unreachable and the
    # node would silently do nothing forever -- max_gain 1.2 can only ever
    # express 16.6%, so a 20% threshold would never once be met. Cap the
    # threshold at what is measurable. A fit sitting exactly on its ceiling
    # means "at least this much, possibly far more", which is ample evidence.
    sign_cap = cap if sign > 0 else min(cap, float(np.log(REVERSE_MAX_GAIN)))
    effective_min = min(min_log_atten, sign_cap)

    magnitude = corner_log_atten if correction_mode == "darken_only" else abs(corner_log_atten)
    applied = magnitude >= effective_min - 1e-9 and magnitude > 0.0

    if applied and strength > 0.0:
        gain_map = build_gain_map(
            h, w, c, sign, strength=strength, max_gain=max_gain,
            center_x_offset=center_x_offset, center_y_offset=center_y_offset,
        )
    else:
        gain_map = np.ones((h, w), dtype=np.float32)

    return {
        "gain_map": gain_map,
        "applied": bool(applied),
        "corner_atten_stops": float(corner_log_atten / np.log(2)),
        "c": np.asarray(c, dtype=float),
        "sign": float(sign),
        "residual": float(fit["cost"]),
    }


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def _soft_knee_compress(m: np.ndarray, knee: float, ceiling: float) -> np.ndarray:
    """Below `knee`, passes through unchanged. Above it, rolls off smoothly
    (exponentially) toward `ceiling` instead of hard-clipping at 1.0. This
    is what avoids a discontinuity at the clip point -- a hard clamp turns a
    gradient into a flat plateau with a visible edge; the soft knee keeps it
    looking like a highlight rather than a scanner error."""
    span = max(ceiling - knee, 1e-6)
    rolled = knee + span * (1.0 - np.exp(-(m - knee) / span))
    out = np.where(m > knee, rolled, m)
    return np.minimum(out, ceiling)


def apply_gain(img_float: np.ndarray, gain_map: np.ndarray, linear_light: bool = True,
               protect_highlights: bool = True, highlight_knee: float = 0.7,
               highlight_ceiling: float = 0.97, chunk_rows: int = 512) -> np.ndarray:
    """Multiplies the image by gain_map and converts back to display space.

    Runs in float32, row-chunked, so a 24 MP frame doesn't need several
    600 MB float64 copies of itself live at once.

    Any channels past the first three (e.g. alpha) are passed through
    untouched -- an alpha channel is not light and must not be scaled.

    protect_highlights matters more than it looks like it should, and it's
    worth understanding *why* rather than just tuning it blind:

    A pixel that's already near-white in a vignetted corner (a blown sky, a
    bright cloud) has little or no real headroom. Multiply it by a
    corner-strength gain and you don't recover anything -- you just slam it
    into a flat, textureless clip, and if you clip R, G, and B by different
    amounts (because they weren't equally close to 1.0 to start with), you
    also get a visible color shift in the process.

    So this works per-pixel, not with one global ceiling: it looks at each
    pixel's brightest channel, and if the *gain-corrected* value would blow
    past `highlight_knee`, it compresses just the overshoot with a smooth
    exponential roll-off toward `highlight_ceiling` -- then rescales all
    three channels by the same factor, which preserves hue/saturation
    instead of clipping channels independently. Mid-tone and shadow pixels
    (which is most of what actually needs correcting in a vignetted corner)
    sit below the knee and get the full, untouched gain.
    """
    h = img_float.shape[0]
    n_ch = img_float.shape[2] if img_float.ndim == 3 else 1
    out = np.empty(img_float.shape, dtype=np.float32)
    gm = gain_map.astype(np.float32, copy=False)

    for i in range(0, h, chunk_rows):
        j = min(i + chunk_rows, h)
        blk = np.clip(img_float[i:j, ..., :3].astype(np.float32), 0.0, 1.0)

        lin = srgb_to_linear(blk).astype(np.float32) if linear_light else blk
        desired = lin * gm[i:j, :, None]   # can legitimately exceed 1.0 pre-compression

        if protect_highlights:
            m = desired.max(axis=-1, keepdims=True)
            m_c = _soft_knee_compress(m, highlight_knee, highlight_ceiling)
            desired *= (m_c / np.maximum(m, 1e-6)).astype(np.float32)

        np.clip(desired, 0.0, 1.0, out=desired)
        out[i:j, ..., :3] = linear_to_srgb(desired) if linear_light else desired
        if n_ch > 3:
            out[i:j, ..., 3:] = img_float[i:j, ..., 3:]

    return out



# ---------------------------------------------------------------------------
# De-blotch: evening out repeating soft patches
# ---------------------------------------------------------------------------
#
# A different problem from lens vignetting, but the same family. When an image
# model outpaints or extends a photo, it tends to copy any soft darkening it
# sees (a leftover vignette, a cloud shadow) and repeat it as a motif across
# the new area: big dark smudges every few hundred pixels.
#
# Those smudges vary side to side, while real landscape lighting mostly varies
# top to bottom (bright sky, darker ground). So each spot is compared against
# the average of its own height band across the full width, and big soft
# departures from it get pulled back. All of it is done in log-luminance on a
# small blurred copy, so fine detail (texture, small bright objects) never
# drives the correction -- it only rides along with a very smooth gain map.

def _gauss_kernel(sigma: float) -> np.ndarray:
    r = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _blur_axis(a: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    """Separable Gaussian along one axis, mirrored at the edges so the border
    column isn't smeared inward. Only ever run on analysis-size arrays."""
    if sigma <= 0:
        return a
    k = _gauss_kernel(sigma)
    r = len(k) // 2
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode="symmetric")
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), axis, p)


def compute_deblotch_gain(img_float: np.ndarray, *, strength: float = 0.85,
                          patch_size: float = 0.15, max_lift: float = 2.5,
                          max_dim: float = 1.6, mask: np.ndarray | None = None,
                          analysis_size: int = 256) -> np.ndarray:
    """Returns a full-resolution (H, W) float32 gain map in linear light.

    patch_size is the width of the patch finder as a fraction of image height;
    its vertical reach is half that, since smudges tend to be tall and the
    sky-to-ground gradient must not leak into the comparison. The local
    average and the row reference share the exact same vertical weights, so
    any purely vertical gradient cancels to a gain of 1.0 everywhere.

    mask (any resolution, 1 = correct, 0 = leave alone) limits the fix to an
    area such as the outpainted sides. It is softened horizontally by the
    same amount as the patch finder so there's no seam at its edge.
    """
    h, w = img_float.shape[:2]
    luma, _, sh, sw = downsampled_luma(img_float, analysis_size)
    log_l = np.log(np.maximum(luma, EPS))

    sx = 0.5 * patch_size * sh
    sy = 0.25 * patch_size * sh
    local = _blur_axis(_blur_axis(log_l, sx, 1), sy, 0)
    row = _blur_axis(log_l.mean(axis=1, keepdims=True), sy, 0)

    lg = strength * (row - local)
    np.clip(lg, -np.log(max(max_dim, 1.0)), np.log(max(max_lift, 1.0)), out=lg)

    if mask is not None:
        m = _box_matrix(mask.shape[0], sh) @ mask.astype(np.float64) @ _box_matrix(mask.shape[1], sw).T
        lg *= np.clip(_blur_axis(m, 0.5 * sx, 1), 0.0, 1.0)

    # Bilinear upsample of a field this smooth is indistinguishable from
    # anything fancier, and it's done in log space so the curve stays even
    up = Image.fromarray(lg.astype(np.float32)).resize((w, h), Image.BILINEAR)
    return np.exp(np.asarray(up, dtype=np.float32))

#     _
# .__(.)< (meow.)
#  \__)
# ~~~~~~~~~~~~~~
