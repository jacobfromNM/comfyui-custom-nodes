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

      // The dropdown is really a button: drop the pick at the cursor, then snap back to the placeholder
      picker.callback = function (value) {
        callback?.apply(this, arguments);
        if (value === PICK) return;
        // The textarea remembers its caret even after the dropdown steals focus - no textarea, no caret, so the end it is
        const el = text.inputEl ?? text.element;
        const at = el?.selectionStart ?? text.value.length;
        const end = el?.selectionEnd ?? at;
        text.value = text.value.slice(0, at) + value + text.value.slice(end);
        // Park the caret after the new wildcard so back-to-back picks line up instead of piling up in reverse
        el?.setSelectionRange(at + value.length, at + value.length);
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
