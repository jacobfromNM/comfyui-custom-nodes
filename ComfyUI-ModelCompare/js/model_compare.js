import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const SLOTS = 4;

// Per node type: the dropdown name and the linkable inputs in each slot
const COMPARE_NODES = {
  ModelCompareLoader: { combo: "unet", links: ["model"] },
  CheckpointCompareLoader: { combo: "ckpt_name", links: ["model", "clip"] },
};

// Which slots on this node actually have something in them. Linked inputs show
// up in the API prompt as [nodeId, outputIndex] arrays, dropdowns as strings.
function activeSlots(inputs, { combo, links }) {
  const slots = [];
  for (let i = 1; i <= SLOTS; i++) {
    if (inputs[`${combo}_${i}`] !== "None" || links.some((l) => Array.isArray(inputs[`${l}_${i}`]))) slots.push(i);
  }
  return slots;
}

// prompt_id -> [{ node, label }], so finished images can find their way home
const pending = new Map();

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

function showResult(node, job, label, images) {
  const g = gallery(node);
  // Slot each job into its own cell so out-of-order finishes don't shuffle the grid
  if (!g.cells[job]) {
    const cell = document.createElement("figure");
    cell.style.cssText = "margin:0;display:flex;flex-direction:column;min-width:0;";
    cell.innerHTML = '<a target="_blank"><img style="width:100%;height:100%;object-fit:contain;min-height:0"></a><figcaption style="font-size:11px;text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></figcaption>';
    g.cells[job] = cell;
    g.el.replaceChildren(...g.cells.filter(Boolean));
  }
  const img = images[0];
  const url = api.apiURL(`/view?${new URLSearchParams({ filename: img.filename, subfolder: img.subfolder, type: img.type })}`);
  g.cells[job].querySelector("a").href = url;
  g.cells[job].querySelector("img").src = url;
  g.cells[job].querySelector("figcaption").textContent = g.cells[job].title = label;
}

app.registerExtension({
  name: "ModelCompare.QueuePerModel",
  setup() {
    const queuePrompt = api.queuePrompt.bind(api);

    // Last image-producing node in each job wins, which is usually the final save/preview
    api.addEventListener("executed", ({ detail }) => {
      const targets = pending.get(detail.prompt_id);
      if (!targets || !detail.output?.images?.length) return;
      for (const t of targets) showResult(t.node, t.job, t.label, detail.output.images);
    });
    for (const ev of ["execution_success", "execution_error", "execution_interrupted"]) {
      api.addEventListener(ev, ({ detail }) => pending.delete(detail?.prompt_id));
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
      // Fresh run, fresh gallery
      for (const n of compare) if (n.node?.compareGallery) n.node.compareGallery.cells = [];
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
          if (n.node) targets.push({ node: n.node, job, label: slotLabel(originals[k], n.cfg, keep) });
        });
        // queuePrompt stringifies right away, so tweaking in place between jobs is safe
        res = await queuePrompt(number, data, ...rest);
        if (res?.prompt_id && targets.length) pending.set(res.prompt_id, targets);
      }
      return res;
    };
  },
});

//     _
// .__(.)< (meow.)
//  \__)
// ~~~~~~~~~~~~~~
