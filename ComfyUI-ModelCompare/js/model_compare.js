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

// prompt_id -> { targets: [{ node, job, round, label }], saved }, so finished images can find their way home
const pending = new Map();
// prompt_id -> cached node ids. Kept apart since it can show up before /prompt even answers us
const cachedNodes = new Map();

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
    cell.innerHTML = '<a target="_blank"><img style="width:100%;height:100%;object-fit:contain;min-height:0"></a><figcaption style="font-size:11px;text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></figcaption>';
    g.cells[job] = cell;
    g.el.replaceChildren(...g.cells.filter(Boolean));
  }
  const img = images[0];
  g.cells[job].round = round;
  // Only a result from the latest Run counts as fresh, older ones stay dimmed
  g.cells[job].style.opacity = round === node.compareRound ? 1 : 0.5;
  const url = api.apiURL(`/view?${new URLSearchParams({ filename: img.filename, subfolder: img.subfolder, type: img.type })}`);
  g.cells[job].querySelector("a").href = url;
  g.cells[job].querySelector("img").src = url;
  g.cells[job].querySelector("figcaption").textContent = g.cells[job].title = label;
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

    // Cached nodes get their old images replayed as "executed" too, so remember which ones to ignore
    api.addEventListener("execution_cached", ({ detail }) => cachedNodes.set(detail.prompt_id, new Set(detail.nodes)));
    // Output node finish order isn't fixed, so a saved image beats any preview,
    // and among equals the last one wins
    api.addEventListener("executed", ({ detail }) => {
      const entry = pending.get(detail.prompt_id);
      const images = detail.output?.images;
      if (!entry || !images?.length || cachedNodes.get(detail.prompt_id)?.has(detail.node)) return;
      const saved = images[0].type === "output";
      if (entry.saved && !saved) return;
      entry.saved ||= saved;
      for (const t of entry.targets) showResult(t.node, t, images);
    });
    for (const ev of ["execution_success", "execution_error", "execution_interrupted"]) {
      api.addEventListener(ev, ({ detail }) => {
        pending.delete(detail?.prompt_id);
        cachedNodes.delete(detail?.prompt_id);
      });
    }

    // One Run click -> one job per filled slot, all sharing the same seed.
    // The seed widget's fixed/increment/decrement/randomize only kicks in once
    // per click, which is exactly what we want here.
    api.queuePrompt = async (number, data, ...rest) => {
      const compare = Object.entries(data.output)
        .filter(([, n]) => n.class_type in COMPARE_NODES)
        .map(([id, n]) => {
          const cfg = COMPARE_NODES[n.class_type];
          return { node: app.graph.getNodeById(id), inputs: n.inputs, cfg, slots: activeSlots(n.inputs, cfg) };
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
          if (n.node) targets.push({ node: n.node, job, round: n.node.compareRound, label: slotLabel(originals[k], n.cfg, keep) });
        });
        // queuePrompt stringifies right away, so tweaking in place between jobs is safe
        res = await queuePrompt(number, data, ...rest);
        if (res?.prompt_id && targets.length) pending.set(res.prompt_id, { targets, saved: false });
      }
      return res;
    };
  },
});

//     _
// .__(.)< (meow.)
//  \__)
// ~~~~~~~~~~~~~~
