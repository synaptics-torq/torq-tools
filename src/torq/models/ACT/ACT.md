# ACT — Action Chunking Transformer

**What is it:** A robotics model. Takes in a VGA image, spits out the next 100 states of the robot. Running at 30Hz, this means about 3.3s of actions.

**Goal:** Get it supported on torq (compiled to `.vmfb`, running on the Astra SL2610 NSS).

---

## TL;DR — what "supported" looks like

Starting from a `.safetensors` checkpoint + the LeRobot export wrapper, we end at a
**hybrid pipeline that runs on the board**:

```
image[1,3,480,640] ─▶ ResNet-18 backbone (torq NSS) ─▶ permute_1[15,20,1,512]
                                                          │  (+ state[1,6])
                                                          ▼
                                  Transformer piece A (torq NSS) ─▶ layer_norm_3[302,1,512]
                                                          ▼
                                  Transformer piece B (torq NSS) ─▶ action[1,100,6]
```

Steady-state board runtime — 20-loop mean, 2 warmup discarded, host profiler, per-inference (excludes one-time
module load). std ≤ 1.3 ms on every stage. The final fully-folded pieces:

| stage | precision | execute | dispatch | provenance |
|---|---|---|---|---|
| ResNet-18 backbone (BatchNorm-folded) | bf16 | 482 ms | 501 ms | measured (20-loop) |
| ResNet-18 backbone | int8 | 118 ms | 133 ms | measured (representative rebuild, see Stage 8) |
| Transformer piece A (enc L1+L2, folded) | bf16 | 207 ms | 236 ms | measured (20-loop) |
| Transformer piece B (enc L3+L4 + decoder, folded) | bf16 | 242 ms | 274 ms | measured (20-loop) |
| **end-to-end (bf16 backbone)** | | **~931 ms** | **~1011 ms** | sum of measured stages |
| **end-to-end (int8 backbone)** | | **~567 ms** | **~643 ms** | sum of measured stages |

vs. the whole model on CPU/ONNX-Runtime ≈ 3.35 s → **~3.5× (bf16) / ~5–6× (int8)**.

> The backbone time is **with BatchNorm folded into the convs** (Stage 4b) — that's ~17% faster than the unfolded
> 585/605 ms. **Accuracy:** the fully-folded compiled bf16 pipeline matches the original fp32 graph to **1.44%
> relL2 / 0.99990 cosine** (board, randomized input-sensitive weights — see "Numerical verification").

> Why two transformer pieces and not one? The single bf16 transformer module is 714 MB and
> **OOMs** the 2 GB board mid-run (exit 137). Splitting the encoder in half keeps each module
> ~330–390 MB, which runs fine. Same total compute, dispatched in two halves.

---

## Prerequisites

- `torq-compile` and `tosa-converter-for-tflite` (built in `work-dev/iree-build/.../tools`, also in the venv).
- `iree.compiler.tools.import_onnx` (ONNX → torch-MLIR), `onnx`, `ml_dtypes`, `tensorflow`/`keras` (for the int8 backbone).
- A board reachable over ssh (`root@10.3.10.62`) with `torq-run-module`.
- The LeRobot package importable (for the export step only).

All scripts referenced below live in **this directory**. Surgeries that already had robust,
proven implementations are reused from `work-dev/lerobot/` and cited inline.

> **Env note:** the export (Stage 1) and the bf16 converter (Stage 2) need the LeRobot + ONNX-GS
> packages — the `work-dev/venv_lerobot` environment has both. The surgeries/split need only
> `onnx`/`ml_dtypes`. This pipeline was run end-to-end from `model.safetensors` and verified:
> `piece_A.vmfb` (327 MB) compiled and **ran on the board** producing a valid `layer_norm_3` output.

---

## Stage 1 — Export `.safetensors` → fp32 ONNX  (`01_export.py`)

The ACT policy's `select_action()` has an `eval()`/control-flow path that `torch.export` can't
trace, so we wrap the underlying model and call `.forward()` with a fixed batch dict.

```bash
python 01_export.py lerobot_model_safetensor/pretrained_model -o lerobot_model.onnx
# image_side[1,3,480,640] f32, state[1,6] f32  ->  action[1,100,6] f32
```

(Adapted from `work-dev/lerobot/convert_model.py`.) Export fp32 — SL2610 has no f64, and we cast
to bf16 next.

---

## Stage 2 — fp32 → bf16  (torq-tools `convert_dtype`)

The NSS data path is **int8/bf16 only** (≤ 2-byte; f32 is allowed only as a matmul *accumulator*
inside a kernel). Use the torq-tools converter (handles Cast/dtype edge cases and bumps the opset):
```bash
python -m torq.tools.convert_dtype onnx -i lerobot_model.onnx -o lerobot_model_bf16.onnx -d bf16
# (needs onnx_graphsurgeon; the venv_lerobot env has it. Upgrades opset to 22.)
```

---

## Stage 3 — ONNX surgeries (make it NSS-compilable)

The raw bf16 graph does **not** compile on the NSS. Four surgeries fix four distinct blockers:

### 3a. Collapse unrolled `unbind`/`stack` concats
The export unrolls the 8-head attention into ~1800 `Squeeze`/`Unsqueeze`/`Slice`/`Concat` nodes.
A `Concat` with ≥164 inputs also crashes `iree-run-module` (fixed upstream by iree #24248/#23155).
Collapse them back into strided ops.
```bash
python /home/nchristo/work-dev/lerobot/collapse_unrolled_concat.py lerobot_model_bf16.onnx collapsed.onnx
```

### 3b. Wrap batched-with-size-1 matmuls → 2D  (`wrap_matmuls.py`) ★ the big one
Every linear layer arrives as `[seq,1,512] @ [512,N]`. That spurious size-1 axis makes the NSS
materialize a `[seq,512,N]` **broadcast** — catastrophically slow (≈75× on a single op) and the
main reason a naive compile is unusable. Stripping it to `[seq,512] @ [512,N]` removes the broadcast.
This one pass covers **both** the FFN matmuls (historically "ffnwrap") and the attention Q/K/V/output
projections ("projwrap"). It deliberately leaves the attention *score* matmuls (`Q@Kᵀ`, `scores·V`)
alone — those are activation×activation and compile fine as bf16 batched matmuls.
```bash
python wrap_matmuls.py collapsed.onnx wrapped.onnx
```

### 3c. Strided-MaxPool fix  (backbone)
The ResNet stem's strided MaxPool hits a halo out-of-bounds in NSS tiling (a regression from torq
commit 2cfc3a65). Rewrite stride-2 MaxPool as stride-1 MaxPool + a slice.
```bash
python /home/nchristo/work-dev/lerobot/apply_maxpool_v4_surgery.py wrapped.onnx mpfix.onnx
```

### 3d. Decompose LayerNorm → matmul-by-ones  (`decompose_layernorm_nss.py`)
The NSS can't tile the last-axis reduction that `LayerNormalization` lowers to ("result is not
accessed using a permuted projection"). Re-express mean/variance as a `MatMul` against a constant
**`(1/N)`-vector** (the NSS-supported contraction path) and keep the normalize as a single broadcast
`Div`. Folding `1/N` into the matmul weight (rather than a trailing `Mul(·, 1/N)`) means the mean is
`MatMul` alone and the variance is `MatMul → Add(eps)` — no `matmul→mul→add` chains. It's exact in bf16
when N is a power of two (`1/512 = 2⁻⁹`); verified bit-identical (0.000% relL2) to the `Mul`-by-1/N form.
> Why `Sqrt → Div(xc,std)` and not the faster `rsqrt(var+eps) · xc`: ONNX has no `Rsqrt`, and both ways to
> express it — `Reciprocal(Sqrt(·))` and `Pow(·,-0.5)` — **fail to lower** on the NSS via the ONNX path
> (`'linalg.generic' op unsupported`). Only `Div` compiles, so `Sqrt → Div` is forced (verified). (The
> TFLite→TOSA path *does* have a native `RSQRT` — used in the int8 LN tests — but the ONNX→torq frontend doesn't
> map `Reciprocal`/`Pow` to it. The normalize is also a tiny `[seq,512]` slice of the matmul-dominated layer, so
> the win would be marginal anyway.)
```bash
python decompose_layernorm_nss.py mpfix.onnx surgered.onnx
```
`surgered.onnx` now compiles on the NSS.

> **Two independent things this surgery must get right:**
> 1. **Reduce as matmul-by-ones, NOT `ReduceMean`.** `ReduceMean` fails to tile on the NSS —
>    `'linalg.generic' op ... result is not accessed using a permuted projection`. Re-verified 2026-06-12 *with*
>    the distinct-square fix below still in place: it is the reduce itself, not the square. matmul-by-ones required.
>    (`decompose_layernorm_reducemean.py` kept as a documented negative control.)
> 2. **Square as `Mul(xc, xc+0)`, NOT `Mul(xc, xc)`.** A self-multiply whose result feeds the reduce miscomputes
>    to ~0 on the NSS in bf16 → `std=√eps` → the normalize `Div` explodes to ±3.4e38 → the whole transformer NaNs.
>    Feeding two *distinct* tensors (`xc` and `xc+0`) fixes it. See
>    `gh-issues/self-multiply-into-reduce-miscompile/`. This is the subtle bug; the decomposition does it correctly.

---

## Stage 4 — Hybrid split (backbone | transformer)  (`split_pipeline.py`)

Split at `permute_1` (the ResNet output, `[15,20,1,512]`). torq wins decisively on the conv
backbone; the transformer is the part the surgeries target.
```bash
python split_pipeline.py surgered.onnx --hybrid     # -> backbone.onnx, transformer.onnx
```

### 4b. Fold BatchNorm into the backbone convs  (`fold_conv_bn.py`)
The ResNet backbone exports BatchNorm (eval mode) as `Conv → Mul(per-channel scale) → Add(per-channel bias)`.
That's algebraically one conv: `W' = W·s`, `bias = b`. Fold it (90→50 nodes, ~17% faster: 585→482 ms):
```bash
python fold_conv_bn.py backbone.onnx backbone_folded.onnx     # folds 20 Conv->Mul->Add; residual Adds untouched
```
Fidelity (board, bf16): folded vs unfolded **0.59% relL2 / maxabs 0.0156** — pure bf16 re-rounding of `W·s`.
Compile `backbone_folded.onnx` (Stage 6).

## Stage 5 — Transformer 2-piece split (avoid OOM)  (`split_pipeline.py`)

The full bf16 transformer compiles to 714 MB and OOMs the board. Cut the encoder in half at the
inter-layer residual `layer_norm_3`:
```bash
python split_pipeline.py transformer.onnx --two-piece   # -> piece_A.onnx, piece_B.onnx
```
- `piece_A`: `[permute_1, state] → layer_norm_3`  (encoder L1+L2, ~327 MB)
- `piece_B`: `layer_norm_3 → action`  (encoder L3+L4 + decoder, ~387 MB)

(The decoder's query stream is a constant, so `piece_B` needs only `layer_norm_3`.)

### 5b. Constant-fold each piece  (`const_fold.py`)
The exporter leaves big all-constant islands un-evaluated: the sinusoidal positional embeddings
(`Sin/Cos/…→stack_*`), the per-layer QKV weight `Split`/`Transpose` prep, and — because the decoder's
query stream is a constant — the **entire decoder self-attention** (its `Softmax` and matmuls run on
constants, independent of the input). Precompute them all into initializers:
```bash
python const_fold.py piece_A.onnx piece_A_folded.onnx   # -> 121 nodes
python const_fold.py piece_B.onnx piece_B_folded.onnx   # -> 181 nodes (decoder self-attn folded away)
```
> Why a custom folder and not the built-in `onnx_graphsurgeon.Graph.fold_constants()` (which torq-tools already
> depends on): gs folds via ONNX Runtime, which has no bf16 CPU kernels — on these bf16 pieces it can't even
> materialize the constants (`Could not convert: BFLOAT16 to a corresponding NumPy type`) and folds **0 nodes**.
> torq-tools' own `cleanup()` is dead-node elimination, not constant folding. So `const_fold.py` fills a real gap
> (bf16 constant folding) and is a candidate to upstream as a `graph_edit` pass.

Done in numpy (ORT can't run bf16 on CPU): evaluate the constant subgraph in fp32, store each boundary
tensor as a bf16 initializer, drop the dead nodes, prune unused initializers. Fidelity: full folded pipeline
matches ORT to **1.44% relL2** (vs 1.30% unfolded — the +0.14% is fp32→bf16 rounding of the precomputed
constants; the fp32-computed pos-embeds are if anything more accurate).

### 5c. Fold the attention scale into the Q projection  (`fold_scalar_mul.py`)
The attention Q-scale (`×1/√d = 0.125`) is a scalar `Mul` a few shape-ops after the Q-projection:
`MatMul(Wq)→Reshape→Add(bq)→Reshape→Transpose→Mul(0.125)`. A scalar commutes through the reshapes, so fold it
into the projection weight + bias (`Wq·s, bq·s`) and drop the Mul:
```bash
python fold_scalar_mul.py piece_A_folded.onnx piece_A_folded.onnx   # 121 -> 119 nodes (2 Q-scales)
python fold_scalar_mul.py piece_B_folded.onnx piece_B_folded.onnx   # 181 -> 179 nodes
```
**Lossless** — `0.125 = 2⁻³` is exact in bf16; verified bit-identical (0.000% relL2) on the board.
(The remaining per-channel `Mul`s are the LayerNorm affine `·γ`, which follows `Div` and is an inherent LN
parameter — not foldable into a matmul.) Compile the final `*_folded.onnx`.

---

## Stage 6 — Compile to `.vmfb`  (`compile.sh`)

```bash
TC=/home/nchristo/work-dev/iree-build/third_party/iree/tools/torq-compile ./compile.sh
```
bf16 ONNX → `import_onnx` → `torq-compile`. Flags (NSS only): `--torq-hw=SL2610
--torq-disable-css --torq-disable-host --torq-tile-and-fuse-distance-limit=1
--torq-enable-split-constants-optimization`.

> Gotchas: the `--torq-tile-and-fuse-distance-limit=1` flag is load-bearing — without it the full
> model fails to compile under fusion pressure. And pass flags **inline**; shell variable expansion
> can silently mangle `--torq-hw=...` into a false "COMPILED".

---

## Stage 7 — Run on board  (`run_board.sh`)

```bash
BOARD=root@10.3.10.62 ./run_board.sh
```
Chains backbone → (host int8→bf16 cast) → piece_A → piece_B → `action`. Add
`--torq_profile_host=/tmp/p.csv` to each `torq-run-module` for warm timings (pair
`DISPATCH_BEGIN/END` for dispatch, `DISPATCH_EXECUTE_ACTIONS_BEGIN/END` for pure-NSS execute).

That is the **bf16 pipeline, verified end-to-end (~931 ms execute, with the Stage-4b BatchNorm fold + Stage-5b const fold)**.

### Numerical verification
The compiled bf16 pipeline was checked against the original fp32 graph (`randomize_export.py` → randomize every
weight so the model is non-degenerate/input-sensitive → export `model_original_randomized.onnx` → run that exact
ONNX through this whole pipeline → compare ORT(fp32) vs the torq board chain on the same input):
**relL2 1.44%, cosine 0.99990, max-abs 0.0103** (fully folded pipeline; 1.30% before the Stage-4b/5b folds) —
pure bf16 rounding, no structural error. Random weights matter:
the real checkpoint is input-insensitive, so a match on it could be trivial; random weights exercise every op.
(The int8 backbone is excluded from this check — it's a separate random-weight rebuild, see Stage 8.)

---

## Stage 8 — int8 backbone optimization (~365 ms faster end-to-end)  (`build_int8_backbone.py`)

The backbone is the biggest single chunk; int8 takes it 482 → 118 ms (~4×). The transformer
**cannot** go int8 yet (see Limitations), so this is backbone-only — a mixed int8/bf16 pipeline.

```bash
python build_int8_backbone.py -o resnet18_backbone_int8.tflite
# compile.sh already includes: tosa-converter-for-tflite ... | torq-compile
```

Why TFLite and not ONNX int8: torq has no NSS lowering for ONNX QDQ ops — only the TFLite→TOSA
path, where IREE folds the quantized pattern into a quantized **Conv**, works. See
**`gh-issues/qdq-int8-nss-lowering/`** and **`gh-issues/02-convert-dtypes-const-outline/`**.

Layout matters: build NHWC with **batch=1, sequence/spatial on H/W**. An int8 fully-connected
flattens to batch=N and torq's conv codegen rejects that layout.

Boundary: the int8 backbone emits int8; `piece_A` wants bf16. The int8→bf16 dequant runs on the
**host** (a standalone dequant doesn't lower on the NSS — same QDQ gap). It's 153,600 elements,
sub-ms on CPU. `run_board.sh` does this cast.

> The backbone built here is a *structurally-equivalent* ResNet-18 with random weights — faithful
> for runtime/plumbing, not for outputs. Porting the real exported conv weights in before PTQ is a
> remaining TODO (it's a quantization/accuracy task, not a compile-path one; the path itself works).

---

## Limitations — why the *transformer* stays bf16

A fully int8 transformer is blocked on the NSS. The op-by-op status (TFLite→TOSA→torq):

| transformer op (int8) | status |
|---|---|
| weight matmul (FFN / projections) | ✅ works as **batch=1 seq-on-spatial 1×1 conv** (25 ms FFN) |
| softmax | ✅ compiles |
| LayerNorm (lnnss conv-reduction form) | ✅ compiles (Keras LN does not — `i32 vs i16`) |
| attention BatchMatMul (`Q@Kᵀ`, `scores·V`) | ❌ **no lowering** |
| int8↔bf16 QUANTIZE/DEQUANTIZE boundary ops | ❌ **no lowering** |

The attention score matmuls are activation×activation, so there's no constant operand to fold into
a conv, and the int8 `tosa.matmul` (i32 accumulate) has no NSS lowering — filed as
**`gh-issues/int8-batch-matmul-nss-lowering/`**. Splitting heads doesn't help (batch=1 fails too).
A *mixed* int8/bf16 transformer is also blocked because the int8↔bf16 boundary needs
QUANTIZE/DEQUANTIZE, which don't lower either (**`gh-issues/qdq-int8-nss-lowering/`**). Closing
**either** of those two issues unblocks the int8 transformer (projected ~310 ms vs bf16 461 ms).

### What stands between us and full-model int8 PTQ
Given a trained checkpoint and a calibration set, the remaining gap is **one rebuild + one compiler fix**:

1. **Rebuild the model in TFLite (engineering, surmountable).** int8 only lowers via TFLite→TOSA→torq —
   ONNX QDQ does not (`gh-issues/qdq-int8-nss-lowering/`) and there's no ONNX→TFLite converter. So the ACT
   model must be reconstructed in TF/Keras with the int8-friendly structure baked in (linear layers as
   batch=1 1×1 Conv2D, LayerNorm as lnnss conv-reductions), then standard TFLite PTQ. We already did this for
   the backbone (`build_int8_backbone.py`); the encoder/decoder is the same pattern, just more of it.
2. **One torq backend fix for the attention (NOT frontend-dodgeable).** The score matmuls can't run in int8,
   and neither workaround currently lowers, so torq must add **either**:
   - int8 activation×activation `matmul` lowering → **pure int8**; or
   - `QUANTIZE`/`DEQUANTIZE` lowering → **mixed int8/bf16** (keep attention in bf16).

   The **QDQ fix is higher-leverage**: transformer int8 PTQ usually needs softmax/LayerNorm (and often
   attention) kept in higher precision to hold accuracy, so the accuracy-viable config is the mixed one —
   which is exactly what QDQ-boundary lowering enables. Everything else (weight matmuls, softmax, LayerNorm)
   already lowers in int8 (table above).

Also note: a naive matmul→1×1-conv rewrite of the *whole* transformer shrinks the module ~8× but
runs ~1.6–2× **slower** (the conv kernels don't fuse as well as torq's matmul mega-dispatch), so it's
a size/OOM lever only — not used in the recommended pipeline.

---

## Script index (this directory)

| file | stage |
|---|---|
| `01_export.py` | safetensors → fp32 ONNX |
| `randomize_export.py` | randomize weights + export (for the numerical-fidelity check) |
| `wrap_matmuls.py` | surgery 3b — ffnwrap + projwrap (the key fix) |
| `decompose_layernorm_nss.py` | surgery 3d — LayerNorm → matmul-by-ones (matmul-ones reduce + distinct-tensor square) |
| `decompose_layernorm_reducemean.py` | negative control — ReduceMean form (does NOT compile) |
| `fold_conv_bn.py` | Stage 4b — fold Conv→Mul→Add (BatchNorm) into one Conv |
| `split_pipeline.py` | hybrid split + transformer 2-piece split |
| `const_fold.py` | Stage 5b — precompute constant islands (pos-embeds, QKV prep, decoder self-attn) |
| `fold_scalar_mul.py` | Stage 5c — fold the scalar attention Q-scale (0.125) into the Q projection |
| `build_int8_backbone.py` | int8 backbone (TFLite PTQ → TOSA → torq) |
| `compile.sh` | import_onnx / tosa-convert → torq-compile |
| `run_board.sh` | scp + run the chain on the board, warm timings |

Reused from `work-dev/lerobot/`: `collapse_unrolled_concat.py` (3a),
`apply_maxpool_v4_surgery.py` (3c), `split_model.py` (inspect piece boundaries / op histograms).
(The old `apply_layernorm_pad_workaround.py` pad step is **no longer needed** — verified droppable, the
lnnss decomposition handles all LayerNorms directly.)
