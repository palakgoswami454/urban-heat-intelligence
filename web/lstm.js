// In-browser inference for the trained Attention-LSTM heat forecaster.
// Mirrors uhi.models.AttnLSTMForecaster (PyTorch gate order: input, forget, cell, output).

function matvec(W, x) {
  const out = new Float64Array(W.length);
  for (let i = 0; i < W.length; i++) {
    const row = W[i];
    let s = 0;
    for (let j = 0; j < x.length; j++) s += row[j] * x[j];
    out[i] = s;
  }
  return out;
}
const sigmoid = (v) => 1 / (1 + Math.exp(-v));

function dayOfYear(dateStr) {
  const d = new Date(dateStr + "T00:00:00Z");
  const start = Date.UTC(d.getUTCFullYear(), 0, 0);
  return Math.round((d.getTime() - start) / 86400000);
}

/**
 * @param model  forecast_model.json contents
 * @param days   array of {date, <feature>: value} for the last `window` days (oldest first)
 * @returns {pred: number[] (deg C, days +1..+horizon), attention: number[]}
 */
export function forecast(model, days) {
  const T = model.window, H = model.hidden, w = model.weights;
  if (days.length < T) throw new Error(`need ${T} days of data, got ${days.length}`);
  const seq = days.slice(-T).map((d) => {
    const doy = dayOfYear(d.date);
    const raw = model.features.map((f) =>
      f === "doy_sin" ? Math.sin((2 * Math.PI * doy) / 365.25)
        : f === "doy_cos" ? Math.cos((2 * Math.PI * doy) / 365.25) : d[f]);
    return raw.map((v, i) => (v - model.f_mean[i]) / model.f_std[i]);
  });

  let h = new Float64Array(H), c = new Float64Array(H);
  const hs = [];
  for (const x of seq) {
    const gi = matvec(w["lstm.weight_ih_l0"], x);
    const gh = matvec(w["lstm.weight_hh_l0"], h);
    const nh = new Float64Array(H), nc = new Float64Array(H);
    for (let k = 0; k < H; k++) {
      const z = (o) => gi[o + k] + gh[o + k] + w["lstm.bias_ih_l0"][o + k] + w["lstm.bias_hh_l0"][o + k];
      const ig = sigmoid(z(0)), fg = sigmoid(z(H)), gg = Math.tanh(z(2 * H)), og = sigmoid(z(3 * H));
      nc[k] = fg * c[k] + ig * gg;
      nh[k] = og * Math.tanh(nc[k]);
    }
    h = nh; c = nc; hs.push(h);
  }
  // additive attention over time steps
  const scores = hs.map((ht) => {
    const a = matvec(w["att_w.weight"], ht);
    let s = 0;
    for (let k = 0; k < H; k++) s += w["att_v.weight"][0][k] * Math.tanh(a[k] + w["att_w.bias"][k]);
    return s;
  });
  const mx = Math.max(...scores);
  const ex = scores.map((s) => Math.exp(s - mx));
  const sum = ex.reduce((p, q) => p + q, 0);
  const att = ex.map((e) => e / sum);
  const ctx = new Float64Array(H);
  hs.forEach((ht, t) => { for (let k = 0; k < H; k++) ctx[k] += att[t] * ht[k]; });
  const z = Float64Array.from([...ctx, ...hs[hs.length - 1]]);
  const l1 = matvec(w["head.0.weight"], z).map((v, i) => Math.max(0, v + w["head.0.bias"][i]));
  const out = matvec(w["head.2.weight"], l1).map((v, i) => v + w["head.2.bias"][i]);
  const lastTmax = days[days.length - 1].temperature_2m_max;
  const pred = Array.from(out, (v, i) => lastTmax + v * model.d_std[i] + model.d_mean[i]);
  return { pred, attention: att };
}
