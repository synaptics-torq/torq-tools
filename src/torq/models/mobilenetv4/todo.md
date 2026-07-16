# MobileNetV4 - TODO

## Blocking: torq-compiler bug in NHWC→NCHW conversion pass

Repo: `/home/yhtet/projects/torq-compiler-dev`
File: `compiler/torq/Transforms/Linalg/ConvertNhwcOpToNchwPass.cpp`
Function: `convertGenericOpToNchw()`, lines ~226-248

### Symptom

`torq-compile` fails on the static MobileNetV4-Conv-Large tosa MLIR with:

```
error: 'linalg.generic' op inferred input/output operand #1 has shape's dimension #1 to be 48, but found 96
    %220 = tosa.conv2d %219, %198, %197, %209, %209 {...} :
        (tensor<1x48x48x192xf32>, tensor<96x1x1x192xf32>, tensor<96xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x48x48x96xf32>
...
    %491 = "linalg.generic"(%486, %478, %490) ... :
        (tensor<1x48x48x96xf32>, tensor<1x96x48x48xf32>, tensor<1x96x48x48xf32>) -> tensor<1x96x48x48xf32>
```

One operand of the generic (`%486`) is still NHWC (`1x48x48x96`) while the other
operand and the output are NCHW (`1x96x48x48`).

### Root cause

`ConvertNhwcOpToNchwPass` converts each conv/pool cluster (anchor + fused
bias/rescale generics) from NHWC to NCHW independently. Confirmed via
`--debug-only=torq-convert-nhwc-conv-to-nchw`:

- A pointwise (1x1) conv has no fused bias/rescale, so it's treated as
  "standalone" and immediately transposed back to NHWC after conversion
  (`Standalone NCHW→NHWC: tensor<1x48x48x96xf32>`).
- That NHWC output is then consumed as a residual/skip-connection input by a
  **different** cluster's bias-add `linalg.generic` (the following depthwise
  conv's cluster).
- In `convertGenericOpToNchw`, inputs not found in the current cluster's
  `valMap` fall through to:

  ```cpp
  else {
      // Keep as-is (1D channel tensors, non-zero tensors, etc.)
      newInp = inp;
  }
  ```

  This conflates two different cases: genuine 1D channel vectors (bias/scale,
  layout-agnostic — fine to leave alone) and real 4D cross-cluster activation
  tensors (which still need transposing to NCHW, just like the zero-filled-fill
  case a few lines above already handles). The result is a generic op with
  mismatched operand layouts, which fails MLIR's shape verifier.

### Suggested fix

In `convertGenericOpToNchw`, split the fallback so real (non-zero) 4D inputs
still get transposed:

```cpp
else if (inpType && inpType.getRank() == 4) {
    // Real (non-zero) 4D NHWC activation from outside this cluster
    // (e.g. a residual/skip-connection input already finished in NHWC)
    // still needs transposing to match this generic's NCHW operands.
    newInp = transposeValue(inp, Permutation::nhwc2nchw(), genericOp.getLoc(), builder);
}
else {
    // Genuine 1D channel tensors (bias/scale/zero-point) — layout-agnostic.
    newInp = inp;
}
```

### Next steps

- [ ] Apply the patch above in `torq-compiler-dev`, rebuild, reinstall into
      `torq-tools-dev/.venv`.
- [ ] Re-run: `python -m torq.models.mobilenetv4.compile --models-dir models`
- [ ] Confirm `.vmfb` is produced for `MobileNetV4-Conv-Large-fp32`.
- [ ] Once compiling end-to-end, decide whether to formalize `compile.py` into
      a full model package (matching `moonshine`/`liquid` conventions:
      `__init__.py` args, registration in `export_model.py`/`infer_model.py`,
      `infer.py`/`validate.py` against ImageNet) or keep it as the minimal
      script.
- [ ] `tosa-converter-for-tflite` still needs installing into
      `torq-tools-dev/.venv` itself (currently only run via
      `/home/yhtet/projects/venv/bin` on `PATH`).
