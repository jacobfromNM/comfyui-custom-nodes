import { app } from "../../scripts/app.js";

// Must match PICK in __init__.py
const PICK = "(pick a wildcard to add it)";

app.registerExtension({
  name: "LiveWildcards.Picker",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LiveWildcards") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      const text = this.widgets.find((w) => w.name === "text");
      const picker = this.widgets.find((w) => w.name === "add_wildcard");
      const callback = picker.callback;

      // The dropdown is really a button: append the pick to the text, then snap back to the placeholder
      picker.callback = function (value) {
        callback?.apply(this, arguments);
        if (value === PICK) return;
        text.value = text.value.trim() ? `${text.value.trimEnd()}, ${value}` : value;
        picker.value = PICK;
        app.graph.setDirtyCanvas(true);
      };
    };
  },
});

//     _
// .__(.)< (meow.)
//  \__)
// ~~~~~~~~~~~~~~
