# ComfyUI Model Compare

A/B test up to 16 models at the same seed without babysitting the queue. Pick your models, hit Run once, and one job per filled slot gets queued, all sharing the same seed. The seed's fixed / increment / decrement / randomize setting then advances once, ready for the next round.

Each job loads only its own model, so comparing a pile of large models never needs them all in memory at once.

## Adding and removing slots

Each node starts with 2 slots. Click **+ Add model** / **+ Add checkpoint** under the last slot to add another (up to 16), or **- Remove** to drop the last one (down to 1). Removing a slot clears its pick and unhooks anything wired into it. The slot count is saved with the workflow, and workflows saved with the old 4-slot nodes load with all 4 slots intact.

## Nodes

Both live under **advanced/loaders**.

### Model Compare Loader

For standalone diffusion models (the `diffusion_models` folder, same as the stock Load Diffusion Model node).

- `unet_1` .. `unet_N`: model dropdowns. `None` skips the slot.
- `model_1` .. `model_N` (optional): wire in a MODEL (merges, LoRA-patched models, etc). A wired input beats the dropdown in the same slot.
- `weight_dtype`: same options as Load Diffusion Model.
- `seed`: shared seed, can also be driven by a link.
- Outputs: `model`, `seed`, `model_name`.

### Checkpoint Compare Loader

For full checkpoints (SD1.5, SDXL, Pony, Illustrious, ...).

- `ckpt_name_1` .. `ckpt_name_N`: checkpoint dropdowns. `None` skips the slot.
- `model_N` / `clip_N` (optional): wire in a merged MODEL and/or CLIP for that slot. The slot's checkpoint then only loads what isn't wired in (usually just the VAE), so a slot with a wired input still needs a checkpoint picked.
- `seed`: shared seed, can also be driven by a link.
- Outputs: `model`, `clip`, `vae`, `seed`, `model_name`.

## Tips

- Feed `seed` into your sampler and `model_name` into your image saver's filename so each comparison set is easy to tell apart (e.g. `%seed_%basemodelname`).
- Slots with a wired-in model report `model_name` as `model_input_N`.
- Only the loaders feeding the current job's slot actually run, so merge chains for the other slots don't cost anything.
- The workflow embedded in each saved image shows every slot, not just the one that produced it.
- After a compare run, the node shows a side-by-side gallery of each job's final image, captioned with its model name. Click an image to open it full size. Saved images win over previews, videos (e.g. from SaveVideo) play inline, and only output nodes downstream of the compare node count. Fully cached reruns still refresh the gallery. On the next run the old images dim until each slot's new image lands. The gallery isn't saved with the workflow.

## How it works

A small frontend extension wraps the queue call: when the prompt contains a compare node, it sends one copy per filled slot with the other slots blanked out. Queued via the API without the frontend, the node just uses its first filled slot.
