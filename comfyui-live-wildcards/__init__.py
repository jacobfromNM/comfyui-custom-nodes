import json
import logging
import os
import random
import re

import yaml

import folder_paths

# Wildcard files live in ComfyUI/models/wildcards, or any extra "wildcards" path from extra_model_paths.yaml
WILDCARD_DIR = os.path.join(folder_paths.models_dir, "wildcards")
folder_paths.add_model_folder_path("wildcards", WILDCARD_DIR, is_default=True)
os.makedirs(WILDCARD_DIR, exist_ok=True)

EXTENSIONS = (".txt", ".yaml", ".yml", ".json")
PICK = "(pick a wildcard to add it)"
WILDCARD_RE = re.compile(r"__([\w\s.\-+/*\\]+?)__")
OPTION_RE = re.compile(r"{([^{}]*)}")
MAX_DEPTH = 100

wildcards = {}
wildcard_stamp = None


def normalize(key):
    return str(key).replace("\\", "/").lower()


def add_tree(key, value):
    # yaml/json can nest: {colors: {warm: [...]}} becomes __colors/warm__
    if isinstance(value, dict):
        for k, v in value.items():
            add_tree(f"{key}/{k}" if key else k, v)
    elif isinstance(value, list):
        wildcards[normalize(key)] = [str(v) for v in value]


def load_file(base, path):
    key, ext = os.path.splitext(os.path.relpath(path, base))
    ext = ext.lower()
    with open(path, encoding="utf-8", errors="ignore") as f:
        if ext == ".txt":
            # Blank lines and # comments are skipped, everything else is fair game
            wildcards[normalize(key)] = [l for l in f.read().splitlines() if l.strip() and not l.lstrip().startswith("#")]
            return
        try:
            add_tree("", yaml.safe_load(f) if ext in (".yaml", ".yml") else json.load(f))
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            # Probably caught mid-edit, it'll get picked up again on the next save
            logging.warning(f"[Live Wildcards] skipping {path}, couldn't parse it: {e}")


def refresh_wildcards():
    # Stat every file (cheap) and only re-read them all when something actually changed.
    # Returns the folder fingerprint, which doubles as the node's IS_CHANGED value.
    global wildcard_stamp
    files = []
    for base in folder_paths.get_folder_paths("wildcards"):
        for root, _, names in os.walk(base, followlinks=True):
            for name in names:
                if name.lower().endswith(EXTENSIONS):
                    path = os.path.join(root, name)
                    st = os.stat(path)
                    files.append((base, path, st.st_mtime_ns, st.st_size))
    stamp = hash(tuple(sorted(files)))
    if stamp != wildcard_stamp:
        wildcards.clear()  # so deleted or renamed files don't haunt the dropdown
        for base, path, _, _ in sorted(files):
            load_file(base, path)
        wildcard_stamp = stamp
    return wildcard_stamp


def lookup(key):
    key = normalize(key)
    if key in wildcards:
        return wildcards[key]
    if "*" in key:
        pattern = re.compile(re.escape(key).replace(r"\*", ".*"))
        return [v for k, values in wildcards.items() if pattern.fullmatch(k) for v in values]
    if "/" not in key:
        # __hair__ also finds characters/hair.txt
        return lookup(f"*/{key}")
    return []


def pick_options(body, rng):
    # {a|b|c}, weighted {3::a|b}, multi-pick {2$$a|b|c}, {1-3$$a|b|c}, custom separator {2$$, $$a|b|c}
    options = body.split("|")
    count, sep = (1, 1), " "
    head = options[0].split("$$")
    if len(head) in (2, 3):
        m = re.fullmatch(r"\s*(\d*)\s*(?:-\s*(\d+))?\s*", head[0])
        if m and (m[1] or m[2]):
            lo = int(m[1] or 1)
            count = (lo, int(m[2] or lo))
            sep = head[1] if len(head) == 3 else sep
            options[0] = head[-1]

    pool = []
    for option in options:
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)::(.*)", option, re.S)
        pool.append((float(m[1]), m[2]) if m else (1.0, option))

    # Weighted pick without repeats, one at a time
    picked = []
    for _ in range(min(rng.randint(*count), len(pool))):
        i = rng.choices(range(len(pool)), weights=[w for w, _ in pool])[0]
        picked.append(pool.pop(i)[1])
    return sep.join(picked)


def expand(text, seed):
    rng = random.Random(seed)  # our own RNG, so we don't stomp on anyone else's global random state
    missing = set()

    def pick_wildcard(m):
        values = lookup(m[1])
        if not values:
            missing.add(m[0])
            return m[0]
        return rng.choice(values)

    # Wildcards can contain options and other wildcards, so keep unwrapping until nothing changes
    for _ in range(MAX_DEPTH):
        new = WILDCARD_RE.sub(pick_wildcard, OPTION_RE.sub(lambda m: pick_options(m[1], rng), text))
        if new == text:
            break
        text = new
    if missing:
        logging.warning(f"[Live Wildcards] no wildcard file found for {', '.join(sorted(missing))}")
    return text


class LiveWildcards:
    @classmethod
    def INPUT_TYPES(s):
        # Called whenever the frontend refreshes node defs (the 'r' key), so the dropdown stays current
        refresh_wildcards()
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False,
                                    "tooltip": "Your prompt. __name__ pulls a random line from name.txt, {a|b|c} picks one option."}),
                "add_wildcard": ([PICK] + [f"__{k}__" for k in sorted(wildcards)],
                                 {"tooltip": "Pick one to add it to the end of the text. Files live in models/wildcards."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # Edited a wildcard file? New fingerprint, so the cached output gets tossed even with the same seed
        return refresh_wildcards()

    def run(self, text, add_wildcard, seed):
        refresh_wildcards()
        return (expand(text, seed),)


WEB_DIRECTORY = "./js"
NODE_CLASS_MAPPINGS = {"LiveWildcards": LiveWildcards}
NODE_DISPLAY_NAME_MAPPINGS = {"LiveWildcards": "Live Wildcards"}

#     _
# .__(.)< (meow.)
#  \__)
# ~~~~~~~~~~~~~~
