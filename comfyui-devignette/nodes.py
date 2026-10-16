"""
ComfyUI node registration + torch<->numpy bridge.

ComfyUI passes IMAGE tensors as torch.Tensor, shape (B, H, W, C), float32,
values in [0, 1]. All the actual math lives in devignette_core.py (pure
numpy/PIL) so it stays trivially unit-testable outside of a running
ComfyUI instance -- this file's only job is looping over the batch and
converting types at the boundary.
"""
import numpy as np
import torch

from .devignette_core import compute_devignette_gain, compute_deblotch_gain, apply_gain


class DeVignette:
    """Detects radially-symmetric lens vignetting and corrects it, while
    leaving directional scene lighting (sky-to-ground gradients, sunset
    glow, one-sided window light, etc) alone. See devignette_core.py for
    the full explanation of how that distinction is made.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "How much of the fix to apply.\n\n"
                               "1.0 = the full correction. 0.5 = half of it. 0 = leave the image "
                               "alone.\n\nAbove 1.0 goes further than the photo needs and makes the "
                               "corners brighter than the middle, which is a look rather than a fix."
                }),
                "max_gain": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 4.0, "step": 0.1,
                    "tooltip": "The most you'll let the corners be brightened, and the main dial "
                               "worth touching.\n\n"
                               "3.0 means a corner can end up 3x brighter. This also tells the node "
                               "how bad vignetting is allowed to get, so it doubles as the safety "
                               "limit: it's what stops the node going overboard on photos that are "
                               "simply bright in the middle, like a sunset or a sun in frame.\n\n"
                               "If results ever feel too strong, turn this down before anything else. "
                               "Around 1.5 gives a gentle, very safe correction."
                }),
                "correction_mode": (["auto", "darken_only"], {
                    "default": "auto",
                    "tooltip": "Which problem to fix.\n\n"
                               "darken_only: only brightens corners that came out too dark. This is "
                               "the only thing lenses actually do, so it's the safe choice.\n\n"
                               "auto: also handles the opposite case, where the corners came out "
                               "brighter than the middle, by darkening them slightly. Handy, but on "
                               "photos with a bright sky or sunset glow reaching into a corner it can "
                               "read that as a fault and dim it. Switch to darken_only if you see "
                               "that. (Darkening is always kept gentle, whatever max_gain says.)"
                }),
                "min_correction_pct": ("FLOAT", {
                    "default": 20.0, "min": 0.0, "max": 30.0, "step": 0.5,
                    "tooltip": "How obvious the dark corners have to be before the node bothers.\n\n"
                               "At 20, only clearly vignetted photos get touched and everything else "
                               "passes through completely untouched, so you can run this over a whole "
                               "folder safely.\n\nLower it to catch milder vignetting, at the risk of "
                               "it occasionally nudging photos that were already fine."
                }),
                "protect_highlights": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Stops bright areas near the edges, like sky or clouds, from being "
                               "pushed all the way to flat white when the corners get brightened.\n\n"
                               "Leave this on."
                }),
                "linear_light": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Does the brightening the way real light behaves, which keeps colours "
                               "from shifting as the corners come up.\n\nLeave this on."
                }),
                "analysis_size": ("INT", {
                    "default": 256, "min": 64, "max": 512, "step": 32,
                    "tooltip": "How closely the node inspects the photo when measuring the "
                               "vignetting.\n\nIt works from a small, blurred copy on purpose, so "
                               "that trees, rocks and other detail don't get mistaken for shading. "
                               "The default is right for almost everything."
                }),
            },
            "optional": {
                "center_x_offset": ("FLOAT", {
                    "default": 0.0, "min": -0.4, "max": 0.4, "step": 0.01,
                    "tooltip": "Moves the spot the darkening is measured around, left or right.\n\n"
                               "Only needed if the photo was cropped off-centre, so the dark corners "
                               "aren't evenly placed. Leave at 0 for normal photos."
                }),
                "center_y_offset": ("FLOAT", {
                    "default": 0.0, "min": -0.4, "max": 0.4, "step": 0.01,
                    "tooltip": "Same as above, but up and down."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "correction_preview")
    FUNCTION = "run"
    CATEGORY = "image/postprocessing"
    DESCRIPTION = (
        "Evens out dark corners caused by the lens, without flattening the photo's own "
        "lighting.\n\n"
        "Lenses tend to let less light reach the corners of the frame, so photos come out "
        "slightly darker at the edges. This finds that darkening and lifts it back, while "
        "leaving real lighting alone: a bright sky over dark ground, sun coming from one "
        "side, a shadowed foreground. The middle of the image is never changed.\n\n"
        "Photos that don't need it are passed straight through untouched.\n\n"
        "The second output shows what was done: grey means unchanged, brighter means more "
        "lift. Worth a look if a result ever seems off."
    )

    def run(self, image, strength, max_gain, correction_mode, min_correction_pct,
            protect_highlights, linear_light, analysis_size,
            center_x_offset=0.0, center_y_offset=0.0):

        # Stay in float32 end to end. At 4000x6000 a single float64 copy of the
        # batch is ~576 MB per image, and the math gains nothing from it -- the
        # fit itself is done at analysis resolution in float64 where it's free.
        batch = image.detach().cpu().numpy().astype(np.float32, copy=False)
        out = np.empty(batch.shape, dtype=np.float32)
        previews = np.empty(batch.shape[:3] + (3,), dtype=np.float32)

        log_span = float(np.log(max(max_gain, 1.0 + 1e-6)))

        for i in range(batch.shape[0]):
            img = batch[i]  # (H, W, C)

            result = compute_devignette_gain(
                img,
                strength=strength,
                max_gain=max_gain,
                correction_mode=correction_mode,
                min_correction_pct=min_correction_pct,
                center_x_offset=center_x_offset,
                center_y_offset=center_y_offset,
                analysis_size=analysis_size,
                linear_light=linear_light,
            )
            gain_map = result["gain_map"]

            out[i] = apply_gain(
                img, gain_map,
                linear_light=linear_light,
                protect_highlights=protect_highlights,
            )

            # Normalize the gain map to 0..1 for viewing: 0.5 = no change,
            # above 0.5 = brightened. Mapped in LOG gain so the scale is
            # symmetric -- 1/max_gain and max_gain land on 0.0 and 1.0
            # respectively, instead of the old linear mapping that clipped
            # the bright end while leaving the dark end short of black.
            preview = 0.5 + 0.5 * np.log(np.maximum(gain_map, 1e-6)) / log_span
            np.clip(preview, 0.0, 1.0, out=preview)
            previews[i] = preview[..., None]

            # Report in plain percentages rather than stops: "the corner is 40%
            # dimmer than the middle" is something anyone can picture.
            pct = (1.0 - 2.0 ** -abs(result["corner_atten_stops"])) * 100.0
            if not result["applied"]:
                print(f"[De-Vignette] image {i}: corners look fine ({pct:.0f}% off the middle, "
                      f"under the {min_correction_pct:.0f}% threshold) -- left unchanged.")
            elif result["sign"] < 0:
                print(f"[De-Vignette] image {i}: corners were BRIGHTER than the middle; "
                      f"darkened them by up to {(1 - gain_map.min()) * 100:.0f}% "
                      f"(auto mode -- use darken_only if that's not wanted).")
            else:
                print(f"[De-Vignette] image {i}: corners were {pct:.0f}% dimmer than the middle; "
                      f"brightened them by up to {gain_map.max():.2f}x.")

        return (torch.from_numpy(out), torch.from_numpy(previews))


class DeBlotch:
    """Evens out big soft patches that repeat side to side, the kind image
    models leave behind when they outpaint and copy a vignette or shadow
    across the new area. See the de-blotch section of devignette_core.py.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much of the fix to apply.\n\n"
                               "1.0 pulls every patch all the way to the brightness of its "
                               "surroundings. 0.85 leaves a hint of natural variation so the "
                               "result doesn't look flat. 0 = leave the image alone."
                }),
                "patch_size": ("FLOAT", {
                    "default": 0.15, "min": 0.05, "max": 0.5, "step": 0.01,
                    "tooltip": "Roughly how wide the patches are, compared with the image "
                               "height.\n\nSmaller catches tighter smudges but starts to even out "
                               "real shapes like a single dark hill. Larger only goes after broad "
                               "patches."
                }),
                "max_lift": ("FLOAT", {
                    "default": 2.5, "min": 1.0, "max": 4.0, "step": 0.1,
                    "tooltip": "The most a dark patch can be brightened.\n\n"
                               "Turn this down if real shadows, like a dark mountain, are getting "
                               "lifted too much."
                }),
                "max_dim": ("FLOAT", {
                    "default": 1.6, "min": 1.0, "max": 4.0, "step": 0.1,
                    "tooltip": "The most a bright patch can be darkened.\n\n"
                               "Kept gentler than max_lift on purpose so a real sun glow or bright "
                               "break in the clouds isn't dimmed much. 1.0 = only ever brighten."
                }),
                "protect_highlights": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Stops bright areas, like sky or clouds, from being pushed all "
                               "the way to flat white when they get brightened.\n\nLeave this on."
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Only fix the white areas, for example just the outpainted "
                               "sides, so the original photo stays exactly as it was. The edge "
                               "is blended softly. Leave unconnected to fix the whole image."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "correction_preview")
    FUNCTION = "run"
    CATEGORY = "image/postprocessing"
    DESCRIPTION = (
        "Evens out big soft dark or bright patches that repeat across an image, like the "
        "smudges an image model leaves when it outpaints and copies a vignette around the "
        "scene.\n\n"
        "Each spot is compared with the rest of the image at the same height, so a bright "
        "sky over darker ground is left alone, while a dark smudge sitting in the middle of "
        "an open field is lifted. Fine detail is never touched; only the broad brightness "
        "shifts.\n\n"
        "Connect a mask to limit it to the generated areas.\n\n"
        "The second output shows what was done: grey means unchanged, brighter means "
        "lifted, darker means dimmed."
    )

    def run(self, image, strength, patch_size, max_lift, max_dim, protect_highlights, mask=None):
        batch = image.detach().cpu().numpy().astype(np.float32, copy=False)
        out = np.empty(batch.shape, dtype=np.float32)
        previews = np.empty(batch.shape[:3] + (3,), dtype=np.float32)

        masks = None
        if mask is not None:
            masks = mask.detach().cpu().numpy().astype(np.float32, copy=False)
            if masks.ndim == 2:
                masks = masks[None]

        log_span = float(np.log(max(max_lift, max_dim, 1.0 + 1e-6)))

        for i in range(batch.shape[0]):
            img = batch[i]
            # One mask for the whole batch is the common case; reuse the last
            m = None if masks is None else masks[min(i, masks.shape[0] - 1)]

            gain_map = compute_deblotch_gain(
                img, strength=strength, patch_size=patch_size,
                max_lift=max_lift, max_dim=max_dim, mask=m,
            )
            out[i] = apply_gain(img, gain_map, protect_highlights=protect_highlights)

            preview = 0.5 + 0.5 * np.log(np.maximum(gain_map, 1e-6)) / log_span
            np.clip(preview, 0.0, 1.0, out=preview)
            previews[i] = preview[..., None]

        return (torch.from_numpy(out), torch.from_numpy(previews))


NODE_CLASS_MAPPINGS = {
    "DeVignette": DeVignette,
    "DeBlotch": DeBlotch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeVignette": "De-Vignette (Auto Detect + Correct)",
    "DeBlotch": "De-Blotch (Even Out Repeating Patches)",
}

#     _
# .__(.)< (meow.)
#  \__)
# ~~~~~~~~~~~~~~
