# Chaotic Time Series ESN (MSLP)

**Architecture:** InputConv(1→256) → FC(256→1)
**Status:** 📋 Analysis complete; extends ecoPrimals ESN readout
**Task:** Predict next step in chaotic MSLP (Mean Sea Level Pressure) time series
**Source:** NeuroBench chaotic prediction benchmark; Lorenz-like atmospheric data

---

## Connection to ecoPrimals ESN

This is architecturally identical to `models/physics/esn_readout.md`:

| | MSLP ESN | hotSpring ESN |
|--|---------|--------------|
| Architecture | InputConv(1→256) → FC(256→1) | InputConv(50→128) → FC(128→1) |
| Input | 1 value (current step) | 50 values (plaquette history) |
| Output | 1 prediction (next step) | 1 flag (thermalization) |
| Reservoir | 256 NPs | 128 NPs |
| Task | Atmosphere prediction | QCD monitoring |
| Training data | MSLP atmospheric pressure | SU(3) lattice plaquette |

The difference is only the training domain and reservoir size.
The hardware execution path is identical — same Akida program structure.

---

## Architecture

```
CPU reservoir (fixed random W_res, 256×256):
  x(t+1) = tanh(W_res × x(t) + W_in × u(t))
  u(t) = current MSLP value (scalar)

Akida readout:
Input: float[256]  (reservoir activations at time t)
  │
  ▼
InputConv(1→256, kernel=1)    ← maps 256 activations to 256 NPs
  │
  ▼
FC(256→1)                     ← linear readout W_out (trained by least squares)
  │
  ▼
Output: float[1]  (predicted MSLP at t+1)
```

---

## NeuroBench Results

| Metric | Value |
|--------|-------|
| sMAPE | 3.8% |
| Throughput | ~18,000 Hz |
| Energy | ~1.4 µJ |

sMAPE (symmetric mean absolute percentage error) below 4% on chaotic prediction
is competitive with full LSTM models while running at 18,000× less energy.

---

## The Reservoir Computing Advantage for Akida

Why ESN maps so naturally to Akida:

1. **Reservoir is random and fixed** — trained once, never changes.
   The chip only runs the readout, not the reservoir dynamics.

2. **Readout is linear** — FC layer with least-squares weights.
   Int4 quantization loses very little precision for linear regression.

3. **Input history is compressed** — 256 reservoir activations summarize
   the entire temporal history of the input. Akida sees 256 floats, not
   a time series.

4. **Speed matters** — at 18,000 Hz, the readout keeps up with any
   real-time input stream (sensors, simulation, audio).

This makes reservoir computing the **ideal neuromorphic workload**:
the compute-intensive (reservoir) runs on CPU/GPU, the fast readout on NPU.

---

## ecoPrimals Extensions

| Extension | Description |
|-----------|-------------|
| Multi-step prediction | FC(256→10) for 10-step ahead forecast |
| Multi-variable | FC(256→3) for {MSLP, temperature, humidity} |
| Online adaptation | set_variable() to swap W_out at 86 µs (Discovery 6) |
| Ensemble readout | 3 classifiers hot-swapped, majority vote |
| Physics hybrid | Same reservoir, different readouts per domain |

The ensemble readout (3 classifiers) is demonstrated in hotSpring Exp 022
and wetSpring Exp 193–195 (NPU sentinel, online evolution at 136 gen/sec).
