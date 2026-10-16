# ComfyUI De-Vignette

A custom ComfyUI node that **detects and corrects lens vignetting**
(darkened corners/edges) automatically, while doing its best to leave real
scene lighting -- sky-to-ground gradients, one-sided golden-hour glow, fog,
foreground shadow -- alone.

## Install

Drop this whole folder into your `ComfyUI/custom_nodes/` directory, e.g.:

```
ComfyUI/custom_nodes/comfyui-devignette/
    __init__.py
    nodes.py
    devignette_core.py
    test_devignette.py
    README.md
```

Restart ComfyUI. You'll find both nodes under **image/postprocessing**:
**De-Vignette (Auto Detect + Correct)** and **De-Blotch (Even Out Repeating
Patches)**.

No extra Python packages required -- it only uses `numpy`, `Pillow`, and
`torch`, all of which ComfyUI already depends on.

## How it works (short version)

Lens vignetting is a *multiplicative, radially-symmetric* falloff centered
on the optical axis -- it darkens all four corners by a comparable amount
because it comes from lens/sensor geometry, not scene content. Real scene
lighting is usually *directional*: brighter sky at top, darker water at
bottom, more light on one side.

The node works in log-luminance on a heavily downsampled copy of the image
(which strips out detail and texture, leaving just the smooth lighting
field) and fits two things at once: a directional scene term and a radial
falloff term. It then throws away the scene term and inverts only the
radial one.

Three properties are enforced structurally rather than hoped for:

- **The correction is monotonic and never darkens.** The falloff is
  parameterized as `-(c1*s + c2*s² + c3*s³)` with `s = r²` and all
  `c >= 0`, so the gain is exactly 1.0 at the center and rises smoothly
  outward, for any input. An unconstrained polynomial fit will cheerfully
  darken a mid-frame annulus to chase scene content, which is the opposite
  of de-vignetting.
- **The correction is bounded by `max_gain`, inside the fit.** The detector
  is told up front what falloff is physically plausible and finds the best
  curve obeying it, rather than fitting freely and getting clipped
  afterward (which flattens the whole outer field into a plateau with a
  visible seam).
- **Highlights are protected per-pixel**, with all three channels rescaled
  by the same factor so hue is preserved instead of channels clipping
  independently.

The full reasoning, including what was tried and rejected, is at the top of
`devignette_core.py`.

## Parameters

| Parameter | Default | What it does |
|---|---|---|
| `strength` | 1.0 | How much of the fix to apply. 1.0 = full, 0.5 = half, 0 = nothing. Above 1.0 overshoots for a stylised look |
| `max_gain` | 3.0 | The most the corners may be brightened, **and** the node's sense of how bad vignetting can plausibly be. Turn this down first if results feel strong |
| `correction_mode` | auto | `darken_only` only brightens dark corners (the safe choice); `auto` also dims corners that came out too bright |
| `min_correction_pct` | 20.0 | How obvious the dark corners must be before the node acts. At 20 only clearly vignetted photos are touched |
| `protect_highlights` | True | Keeps bright edges (sky, cloud) from being pushed to flat white |
| `linear_light` | True | Brightens the way real light behaves, so colours don't shift |
| `analysis_size` | 256 | How closely the photo is inspected when measuring. Default suits almost everything |
| `center_x_offset` / `center_y_offset` | 0.0 | Move the spot the darkening is measured around, for off-centre crops |

The defaults are tuned for running over a mixed folder: the 20% threshold
means only clearly vignetted photos are touched at all, and the generous
`max_gain` then lets those be fully corrected. For a gentler, more
conservative pass, use `max_gain` 1.5 with `min_correction_pct` 8.

Two interactions worth knowing:

- `max_gain` bounds what the detector can even *measure*, so a threshold
  above that bound would be unreachable. The node clamps the threshold to
  what is measurable rather than silently never firing.
- `auto` mode's *darkening* direction is capped much lower than `max_gain`
  (at 1.35x), because a large allowance there is dangerous -- see below.

The second output, `correction_preview`, shows what was actually applied:
mid-grey is no change, brighter is more lift. Check it whenever a result
looks off; it makes over-reads obvious at a glance.

## Accuracy

Measured over a set of real photographs, using a differential protocol:
estimate the frame, apply a *known* vignette, estimate again, and check the
difference matches what was added. (That cancels whatever vignetting the
photograph already carried, so it scores the estimator rather than an
unknown baseline.)

| | |
|---|---|
| bias | +0.04 stops |
| mean absolute error | 0.17 stops |
| 90th percentile error | 0.38 stops |

Run it yourself -- the tests need no ComfyUI and no torch:

```
python test_devignette.py                     # unit tests only
python test_devignette.py /path/to/photos     # + accuracy benchmark
```

Cost on a 24 MP (4000x6000) frame is roughly 2 s to detect and build the
gain map, 3 s to apply it. Everything runs in float32 and is chunked by
rows, so peak memory stays in the hundreds of MB rather than the several
gigabytes a naive float64 implementation needs at that size.

## Known limitations (read before trusting it blindly)

- **Scene content that is itself radially symmetric is genuinely
  ambiguous.** A sun or flare glow near the middle of the frame reads as
  "bright center, dark corners" and no amount of fitting can distinguish
  that from vignetting -- the two are the same function. `max_gain` is the
  guard, and the preview output is worth a look on backlit shots.
- **`auto` mode can dim a corner that was legitimately bright.** A sunset
  glow reaching into one corner reads as "corners brighter than middle" and
  auto will treat it as a fault. The reverse direction is deliberately
  capped at 1.35x (about 0.43 stops) so the worst case stays mild rather
  than crushing corners to a third of their brightness, but if you shoot a
  lot of backlit or corner-lit work, `darken_only` is the safer mode.
- **The detector has a small positive noise floor, by construction.**
  Because it can only ever report darkening, noise pushes readings up and
  never down, so a vignette-free frame reads slightly positive on average
  (measured up to ~0.15 stops). `min_correction_pct` is the guard; the
  correction implied at that level is imperceptible anyway.
- **The scene model is deliberately simple** -- a directional tilt, nothing
  more. That is not an oversight: a richer scene model eventually spans the
  radial functions too, at which point the two are degenerate and the split
  becomes numerically arbitrary. Measured alternatives with saddle-shaped
  terms reported 3-4 stops of "vignetting" on ordinary photographs. The
  cost of the simple model is that strongly non-linear scene lighting can
  still leak into the estimate; the constraints bound the damage.
- **Extreme vignette plus already-near-white corner content has no clean
  fix.** `protect_highlights` minimizes the damage but cannot manufacture
  detail the capture didn't retain.
- **Very fast falloff (mustache-shaped vignette from extreme wide-angle or
  fisheye lenses)** isn't well modeled by the r²/r⁴/r⁶ terms used here and
  may under-correct near the very edge.

## De-Blotch (Even Out Repeating Patches)

A companion node for a related problem. When an image model outpaints or
extends a photo, it tends to copy any soft darkening it sees (a leftover
vignette, a cloud shadow) and repeat it across the new area as big dark
smudges every few hundred pixels.

Those smudges vary side to side, while real landscape lighting mostly varies
top to bottom. So De-Blotch compares each spot with the average brightness of
its own height band across the full width, and pulls big soft departures back
toward it. It works in log-luminance on a small blurred copy, so fine detail
never drives the correction. The local average and the row reference use the
same vertical weighting, so a purely vertical gradient (sky over ground) is
left exactly as it is.

Connect a mask (for example the outpaint mask) to limit it to the generated
areas. The mask edge is blended softly, so there's no seam against the
original photo.

| Parameter | Default | What it does |
|---|---|---|
| `strength` | 0.85 | How far patches are pulled toward their surroundings. 1.0 = fully, 0 = nothing |
| `patch_size` | 0.15 | Patch width as a fraction of image height. Smaller catches tighter smudges but starts evening out real shapes |
| `max_lift` | 2.5 | The most a dark patch can be brightened (linear light) |
| `max_dim` | 1.6 | The most a bright patch can be darkened. 1.0 = only ever brighten |
| `protect_highlights` | True | Same highlight roll-off as De-Vignette |
| `mask` | optional | Only correct the white areas |

Limitation: it can't tell a smudge from real darkness that changes side to
side, such as a single large cloud shadow or one dark hill. Those get softened
too. `max_lift` and a larger `patch_size` are the guards. About 0.5 s on a
2560x720 image, about 5 s at 24 MP.
