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

app.registerExtension({
  name: "ModelCompare.QueuePerModel",
  setup() {
    const queuePrompt = api.queuePrompt.bind(api);

    // One Run click -> one job per filled slot, all sharing the same seed.
    // The seed widget's fixed/increment/decrement/randomize only kicks in once
    // per click, which is exactly what we want here.
    api.queuePrompt = async (number, data, ...rest) => {
      const compare = Object.values(data.output)
        .filter((n) => n.class_type in COMPARE_NODES)
        .map((n) => {
          const cfg = COMPARE_NODES[n.class_type];
          return { inputs: n.inputs, cfg, slots: activeSlots(n.inputs, cfg) };
        })
        .filter((n) => n.slots.length);
      const jobs = Math.max(0, ...compare.map((n) => n.slots.length));
      if (jobs < 2) return queuePrompt(number, data, ...rest);

      const originals = compare.map((n) => ({ ...n.inputs }));
      let res;
      for (let job = 0; job < jobs; job++) {
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
        });
        // queuePrompt stringifies right away, so tweaking in place between jobs is safe
        res = await queuePrompt(number, data, ...rest);
      }
      return res;
    };
  },
});

//     _
// .__(.)< (meow.)
//  \__)
// ~~~~~~~~~~~~~~
