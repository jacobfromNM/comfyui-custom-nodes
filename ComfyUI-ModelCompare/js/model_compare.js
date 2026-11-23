import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Python declares all of these (keep in sync with SLOTS in __init__.py), the
// +/- buttons just decide how many are actually on show
const SLOTS = 16;
const DEFAULT_SLOTS = 2; // fresh nodes start as a plain A/B test
const LEGACY_SLOTS = 4; // workflows saved before +/- existed had exactly 4

// Per node type: the dropdown name, the linkable inputs in each slot (and their
// types, so removed sockets can be rebuilt), and what to call a slot on the buttons
const COMPARE_NODES = {
  ModelCompareLoader: { combo: "unet", links: ["model"], types: { model: "MODEL" }, noun: "model" },
  CheckpointCompareLoader: { combo: "ckpt_name", links: ["model", "clip"], types: { model: "MODEL", clip: "CLIP" }, noun: "checkpoint" },
};

// Show the first n slots and tuck the rest away. Dropdowns get hidden (not
// removed) so widgets_values keeps its positions, while link-only sockets are
// genuinely removed/re-added since inputs save by name anyway.
function applySlots(node, cfg, n, resize = false) {
  n = Math.max(1, Math.min(SLOTS, n));
  node.properties.compareSlots = n;
  for (let i = 1; i <= SLOTS; i++) {
    const show = i <= n;
    const w = node.widgets?.find((w) => w.name === `${cfg.combo}_${i}`);
    if (w) {
      if (!show && !w.hidden) {
        // A removed slot forgets its pick, otherwise it'd sneak into the next Run
        w.value = "None";
        const s = node.inputs?.findIndex((inp) => inp.widget?.name === w.name) ?? -1;
        if (s !== -1 && node.inputs[s].link != null) node.disconnectInput(s);
      }
      w.hidden = !show;
    }
    for (const l of cfg.links) {
      const idx = node.findInputSlot(`${l}_${i}`);
      if (!show && idx !== -1) node.removeInput(idx);
      else if (show && idx === -1) node.addInput(`${l}_${i}`, cfg.types[l]);
    }
  }
  // Only snap the size when a button was clicked, so loading a workflow keeps
  // whatever size you dragged the node to
  if (resize) node.setSize([node.size[0], node.computeSize()[1]]);
  node.setDirtyCanvas(true, true);
}

// Which slots on this node actually have something in them. Linked inputs show
// up in the API prompt as [nodeId, outputIndex] arrays, dropdowns as strings.
function activeSlots(inputs, { combo, links }) {
  const slots = [];
  for (let i = 1; i <= SLOTS; i++) {
    if (inputs[`${combo}_${i}`] !== "None" || links.some((l) => Array.isArray(inputs[`${l}_${i}`]))) slots.push(i);
  }
  return slots;
}

// prompt_id -> { targets: [{ node, job, round, label, downstream, saved }] }, so finished images can find their way home
const pending = new Map();
// A fully cached job can finish before /prompt even answers us, so its events
// wait here (prompt_id -> [detail]) until we know who they belong to
const early = new Map();
// prompt_ids that already wrapped up, in case they beat their own registration
const finished = new Set();
const BUFFER_CAP = 64; // nobody needs a museum of stale prompt ids

// Keep the little side buffers from growing forever if a prompt never claims them
function trim(coll) {
  while (coll.size > BUFFER_CAP) coll.delete(coll.keys().next().value);
}

// Every node in the API prompt that (eventually) eats this node's output. Only
// output nodes in here get to put pictures in the gallery, so an unrelated
// reference-image preview can't photobomb the comparison.
function downstreamOf(prompt, rootId) {
  const kids = new Map();
  for (const [id, n] of Object.entries(prompt)) {
    for (const v of Object.values(n.inputs ?? {})) {
      if (Array.isArray(v) && v.length === 2) {
        const src = String(v[0]);
        if (!kids.has(src)) kids.set(src, []);
        kids.get(src).push(id);
      }
    }
  }
  const seen = new Set();
  const todo = [String(rootId)];
  while (todo.length) {
    for (const k of kids.get(todo.pop()) ?? []) {
      if (!seen.has(k)) {
        seen.add(k);
        todo.push(k);
      }
    }
  }
  return seen;
}

// Videos (SaveVideo and friends report them under "images" too) need a <video>, an <img> just shows a sad broken icon
const isVideo = (img) => /\.(mp4|webm|mov|mkv|m4v|avi)$/i.test(img.filename) || img.format?.startsWith?.("video/");

// What the Python side would call this slot, minus folders so captions stay short
function slotLabel(inputs, { combo, links }, i) {
  if (links.includes("model") && Array.isArray(inputs[`model_${i}`])) return `model_input_${i}`;
  return inputs[`${combo}_${i}`].split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
}

// Lazily bolt a little gallery onto the node the first time it has something to show
function gallery(node) {
  if (node.compareGallery) return node.compareGallery;
  const el = document.createElement("div");
  el.style.cssText = "display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:4px;overflow:auto;";
  node.addDOMWidget("compare_preview", "compare_preview", el, { serialize: false, getMinHeight: () => 160 });
  node.setSize([node.size[0], Math.max(node.size[1], node.computeSize()[1])]);
  node.compareGallery = { el, cells: [] };
  return node.compareGallery;
}

function showResult(node, { job, round, label }, images) {
  const g = gallery(node);
  // A slow job from an older Run must not stomp on a newer Run's picture
  if (g.cells[job] && round < g.cells[job].round) return;
  // Slot each job into its own cell so out-of-order finishes don't shuffle the grid
  if (!g.cells[job]) {
    const cell = document.createElement("figure");
    cell.style.cssText = "margin:0;display:flex;flex-direction:column;min-width:0;";
    cell.innerHTML = '<a target="_blank"></a><figcaption style="font-size:11px;text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></figcaption>';
    g.cells[job] = cell;
    g.el.replaceChildren(...g.cells.filter(Boolean));
  }
  const img = images[0];
  const cell = g.cells[job];
  cell.round = round;
  // Only a result from the latest Run counts as fresh, older ones stay dimmed
  cell.style.opacity = round === node.compareRound ? 1 : 0.5;
  // subfolder can be missing entirely, and "undefined" is not a folder anyone has.
  // The rand param stops the browser serving a stale copy when a filename gets reused.
  const params = new URLSearchParams({ filename: img.filename, subfolder: img.subfolder ?? "", type: img.type ?? "output", rand: Math.random() });
  const url = api.apiURL(`/view?${params}`);
  // Swap in the right kind of media element, a video in an <img> is how we got broken icons
  const tag = isVideo(img) ? "VIDEO" : "IMG";
  const a = cell.querySelector("a");
  let media = a.firstElementChild;
  if (media?.tagName !== tag) {
    media = document.createElement(tag);
    media.style.cssText = "width:100%;height:100%;object-fit:contain;min-height:0;display:block";
    if (tag === "VIDEO") Object.assign(media, { autoplay: true, loop: true, muted: true, playsInline: true });
    a.replaceChildren(media);
  }
  a.href = url;
  media.src = url;
  cell.querySelector("figcaption").textContent = cell.title = label;
}

// One "executed" event -> hand it to every gallery it belongs to
function deliver(entry, detail) {
  const images = detail.output?.images;
  if (!images?.length) return;
  // Output node finish order isn't fixed, so a saved file beats any preview,
  // and among equals the last one wins
  const saved = images[0].type === "output";
  for (const t of entry.targets) {
    if (!t.downstream.has(String(detail.node)) && !t.downstream.has(String(detail.display_node))) continue;
    if (t.saved && !saved) continue;
    t.saved ||= saved;
    showResult(t.node, t, images);
  }
}

app.registerExtension({
  name: "ModelCompare.QueuePerModel",
  beforeRegisterNodeDef(nodeType, nodeData) {
    const cfg = COMPARE_NODES[nodeData.name];
    if (!cfg) return;

    // Every new node (loaded ones too, configure runs right after) gets its
    // +/- buttons parked right under the last slot
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const count = () => this.properties.compareSlots ?? DEFAULT_SLOTS;
      const plus = this.addWidget("button", `+ Add ${cfg.noun}`, null, () => applySlots(this, cfg, count() + 1, true), { serialize: false });
      const minus = this.addWidget("button", `- Remove ${cfg.noun}`, null, () => applySlots(this, cfg, count() - 1, true), { serialize: false });
      // Belt and braces: buttons must never take a seat in widgets_values
      plus.serialize = minus.serialize = false;
      const last = this.widgets.findIndex((w) => w.name === `${cfg.combo}_${SLOTS}`);
      if (last !== -1) {
        this.widgets.splice(this.widgets.indexOf(plus), 1);
        this.widgets.splice(this.widgets.indexOf(minus), 1);
        this.widgets.splice(last + 1, 0, plus, minus);
      }
      applySlots(this, cfg, DEFAULT_SLOTS, true);
      return r;
    };

    // Old 4-slot workflows: no slot count saved, and their widgets_values are
    // laid out for 4 dropdowns. Pad in the 12 missing "None"s so the seed and
    // dtype don't end up crammed into unet_5 and friends.
    const configure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      if (info && info.properties?.compareSlots == null) {
        const vals = info.widgets_values;
        info = { ...info, properties: { ...info.properties, compareSlots: LEGACY_SLOTS } };
        if (Array.isArray(vals) && vals.length < SLOTS) {
          info.widgets_values = [...vals.slice(0, LEGACY_SLOTS), ...Array(SLOTS - LEGACY_SLOTS).fill("None"), ...vals.slice(LEGACY_SLOTS)];
        }
      }
      return configure.call(this, info);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      applySlots(this, cfg, this.properties.compareSlots ?? LEGACY_SLOTS);
      return r;
    };
  },
  setup() {
    const queuePrompt = api.queuePrompt.bind(api);

    // Cached output nodes get their results re-sent as "executed" for THIS prompt,
    // and those are the right pictures for this job, so they count too. (Ignoring
    // them is why re-running the same seed/models never refreshed anything.)
    api.addEventListener("executed", ({ detail }) => {
      const entry = pending.get(detail.prompt_id);
      if (entry) return deliver(entry, detail);
      // Too early to know who this is for, park it until the job is registered
      if (!early.has(detail.prompt_id)) early.set(detail.prompt_id, []);
      early.get(detail.prompt_id).push(detail);
      trim(early);
    });
    for (const ev of ["execution_success", "execution_error", "execution_interrupted"]) {
      api.addEventListener(ev, ({ detail }) => {
        const id = detail?.prompt_id;
        if (pending.delete(id)) return;
        // Finished before we even registered it, leave a note for the registration
        finished.add(id);
        trim(finished);
      });
    }
    // Wire a freshly queued job up, then replay anything that raced ahead of us
    const register = (id, targets) => {
      const entry = { targets };
      for (const detail of early.get(id) ?? []) deliver(entry, detail);
      early.delete(id);
      if (finished.delete(id)) return; // already done, nothing more is coming
      pending.set(id, entry);
    };

    // One Run click -> one job per filled slot, all sharing the same seed.
    // The seed widget's fixed/increment/decrement/randomize only kicks in once
    // per click, which is exactly what we want here.
    api.queuePrompt = async (number, data, ...rest) => {
      const compare = Object.entries(data.output)
        .filter(([, n]) => n.class_type in COMPARE_NODES)
        .map(([id, n]) => {
          const cfg = COMPARE_NODES[n.class_type];
          return { node: app.graph.getNodeById(id), inputs: n.inputs, cfg, slots: activeSlots(n.inputs, cfg), downstream: downstreamOf(data.output, id) };
        })
        .filter((n) => n.slots.length);
      const jobs = Math.max(0, ...compare.map((n) => n.slots.length));
      if (jobs < 2) return queuePrompt(number, data, ...rest);

      const originals = compare.map((n) => ({ ...n.inputs }));
      // New Run: bump the round, drop cells for slots that went away, and dim the
      // rest until their new picture lands
      for (const { node } of compare) {
        if (!node) continue;
        node.compareRound = (node.compareRound ?? 0) + 1;
        const g = node.compareGallery;
        if (!g) continue;
        g.cells.length = Math.min(g.cells.length, jobs);
        g.el.replaceChildren(...g.cells.filter(Boolean));
        for (const cell of g.cells) if (cell) cell.style.opacity = 0.5;
      }
      let res;
      for (let job = 0; job < jobs; job++) {
        const targets = [];
        compare.forEach((n, k) => {
          // Extra compare nodes with fewer slots just stick on their last model
          const keep = n.slots[Math.min(job, n.slots.length - 1)];
          Object.assign(n.inputs, originals[k]);
          for (let i = 1; i <= SLOTS; i++) {
            if (i === keep) continue;
            // Unlinking means loaders feeding the other slots don't even run
            for (const l of n.cfg.links) delete n.inputs[`${l}_${i}`];
            n.inputs[`${n.cfg.combo}_${i}`] = "None";
          }
          // Nodes tucked inside subgraphs won't resolve here, they just go without a gallery
          if (n.node) targets.push({ node: n.node, job, round: n.node.compareRound, label: slotLabel(originals[k], n.cfg, keep), downstream: n.downstream, saved: false });
        });
        // queuePrompt stringifies right away, so tweaking in place between jobs is safe
        res = await queuePrompt(number, data, ...rest);
        if (res?.prompt_id && targets.length) register(res.prompt_id, targets);
      }
      return res;
    };
  },
});

//     _
// .__(.)< (meow.)
//  \__)
// ~~~~~~~~~~~~~~
