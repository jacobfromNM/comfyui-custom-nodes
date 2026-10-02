import os

import comfy.sd
import folder_paths
import nodes

SLOTS = 4


class ModelCompareLoader:
    # The JS side queues one job per filled slot and blanks the rest, so by the
    # time we run there's (normally) exactly one model to load. Nice and lazy.
    @classmethod
    def INPUT_TYPES(s):
        unets = ["None"] + folder_paths.get_filename_list("diffusion_models")
        required = {f"unet_{i}": (unets, ) for i in range(1, SLOTS + 1)}
        required["weight_dtype"] = (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], )
        required["seed"] = ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True})
        # A wired-in model beats the dropdown in the same slot (merges, LoRA'd models, etc.)
        optional = {f"model_{i}": ("MODEL", ) for i in range(1, SLOTS + 1)}
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("MODEL", "INT", "STRING")
    RETURN_NAMES = ("model", "seed", "model_name")
    FUNCTION = "load"
    CATEGORY = "advanced/loaders"

    def load(self, weight_dtype, seed, **kwargs):
        for i in range(1, SLOTS + 1):
            if kwargs.get(f"model_{i}") is not None:
                return (kwargs[f"model_{i}"], seed, f"model_input_{i}")
            name = kwargs[f"unet_{i}"]
            if name != "None":
                return (nodes.UNETLoader().load_unet(name, weight_dtype)[0], seed, name.rsplit(".", 1)[0])
        raise ValueError("Model Compare: all 4 slots are empty, pick at least one model.")


class CheckpointCompareLoader:
    # Same trick as above, but for full checkpoints (Pony, Illustrious, SDXL...).
    # Wired-in model/clip replace that part of the slot's checkpoint, which then
    # only has to cough up whatever's still missing (usually just the VAE).
    @classmethod
    def INPUT_TYPES(s):
        ckpts = ["None"] + folder_paths.get_filename_list("checkpoints")
        required = {f"ckpt_name_{i}": (ckpts, ) for i in range(1, SLOTS + 1)}
        required["seed"] = ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True})
        optional = {}
        for i in range(1, SLOTS + 1):
            optional[f"model_{i}"] = ("MODEL", )
            optional[f"clip_{i}"] = ("CLIP", )
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "INT", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "seed", "model_name")
    FUNCTION = "load"
    CATEGORY = "advanced/loaders"

    def load(self, seed, **kwargs):
        for i in range(1, SLOTS + 1):
            model, clip, name = kwargs.get(f"model_{i}"), kwargs.get(f"clip_{i}"), kwargs[f"ckpt_name_{i}"]
            if name == "None":
                if model is None and clip is None:
                    continue
                raise ValueError(f"Checkpoint Compare: slot {i} has a wired-in model/clip but no checkpoint picked, pick one so it can supply the VAE.")
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", name)
            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=clip is None, output_model=model is None,
                                                        embedding_directory=folder_paths.get_folder_paths("embeddings"))
            label = f"model_input_{i}" if model is not None else os.path.splitext(name)[0]
            return (out[0] if model is None else model, out[1] if clip is None else clip, out[2], seed, label)
        raise ValueError("Checkpoint Compare: all 4 slots are empty, pick at least one checkpoint.")


WEB_DIRECTORY = "./js"
NODE_CLASS_MAPPINGS = {"ModelCompareLoader": ModelCompareLoader, "CheckpointCompareLoader": CheckpointCompareLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"ModelCompareLoader": "Model Compare Loader (x4)", "CheckpointCompareLoader": "Checkpoint Compare Loader (x4)"}

#     _
# .__(.)< (meow.)
#  \__)
# ~~~~~~~~~~~~~~
