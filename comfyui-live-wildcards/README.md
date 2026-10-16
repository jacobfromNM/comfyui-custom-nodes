# Live Wildcards

A wildcard prompt node for ComfyUI that picks up edits to your wildcard files right away. No restart needed.

- **Edit a file and save it:** the next queued prompt uses the new contents, even when the seed hasn't changed.
- **Add or delete a file:** press `r` in ComfyUI and the dropdown updates.

## Install

Copy `comfyui-live-wildcards` into `ComfyUI/custom_nodes` and restart ComfyUI once.

## Wildcard files

Put your files in `ComfyUI/models/wildcards`. The folder is created on first launch. You can add more folders under a `wildcards:` entry in `extra_model_paths.yaml`.

| File | Wildcard name |
| --- | --- |
| `color.txt` | `__color__` (one option per line; blank lines and `#` comments are skipped) |
| `characters/hair.txt` | `__characters/hair__`, or `__hair__` for short |
| `.yaml` / `.json` | Taken from the keys: `animals: {cats: [...]}` becomes `__animals/cats__` |

## Syntax

| Syntax | What it does |
| --- | --- |
| `__color__` | Line number `seed` from `color.txt` (seed 0 = first line, wraps around at the end) |
| `__animals/*__` | Same, over all matching wildcards combined, in file order |
| `{a\|b\|c}` | Picks one option |
| `{3::a\|b}` | Weighted pick (`a` is 3x as likely) |
| `{2$$a\|b\|c}` | Picks 2 different options, joined by spaces |
| `{1-3$$, $$a\|b\|c}` | Picks 1 to 3 options, joined by `, ` |

Wildcards and options can be nested inside each other and inside wildcard files. Every wildcard in a prompt uses the same line number, while `{...}` options are picked randomly from the seed. The same seed always gives the same result. Blank lines and `#` comments don't count as lines. If a wildcard has no matching file, it's left in the text as-is and a warning is logged.

The **add_wildcard** dropdown inserts the wildcard you pick at the text cursor (replacing any selected text), with no separator added.
