// Checks that web/lstm.js reproduces the PyTorch forecaster output.
// Run after the pipeline's forecast stage:  node tests/test_js_lstm.mjs
import fs from "fs";
import { forecast } from "../web/lstm.js";

const model = JSON.parse(fs.readFileSync(new URL("../web/data/forecast_model.json", import.meta.url)));
const tc = JSON.parse(fs.readFileSync(new URL("../outputs/models/forecast_js_testcase.json", import.meta.url)));
const days = tc.dates.map((d, i) => {
  const o = { date: d };
  for (const k in tc.inputs) o[k] = tc.inputs[k][i];
  return o;
});
const { pred } = forecast(model, days);
const diff = Math.max(...pred.map((v, i) => Math.abs(v - tc.expected[i])));
console.log("JS     :", pred.map((v) => v.toFixed(3)).join(" "));
console.log("PyTorch:", tc.expected.map((v) => v.toFixed(3)).join(" "));
if (diff > 0.02) {
  console.error(`FAIL: max difference ${diff}`);
  process.exit(1);
}
console.log(`OK (max difference ${diff.toExponential(2)} C)`);
