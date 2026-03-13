# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""
Export Customer B ONNX models to int8 TFLite and then to MLIR / VMFB.

Pipeline
--------
1. ONNX  →  TFLite (int8)   via ``onnx2tf`` + TFLite quantiser
2. TFLite →  TOSA bytecode   via ``iree-import-tflite``
3. TOSA   →  text MLIR       via ``iree-opt``
4. MLIR   →  VMFB            via ``torq-compile`` (or ``iree-compile``)

Usage (from repo root)
~~~~~~~~~~~~~~~~~~~~~~
::

    # Activate the onnx2tf venv (special onnx/tf versions required)
    source .venv_cvte/bin/activate

    # Export all components
    python -m torq.models.customer_b.export --models-dir models/customer_b

    # Export only all_fc
    python -m torq.models.customer_b.export --models-dir models/customer_b --component all_fc

    # Skip TFLite step (use pre-existing .tflite files)
    python -m torq.models.customer_b.export --skip-tflite --models-dir models/customer_b
"""

import argparse
import glob
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Final

import numpy as np
import onnx

from torq.compile import (
    process_iree_args,
)
from torq.utils.logging import configure_logging

from . import (
    MODEL_COMPONENTS,
    DEFAULT_MODELS_DIR,
    add_customer_b_export_args,
)
from .quantize_fc_tflite import quantize_fc_ops_in_tflite


_logger = logging.getLogger(__name__)

# Well-known search paths for IREE tools
_IREE_TOOL_SEARCH_PATHS = [
    Path.home() / "synpu_compiler" / "iree-build" / "third_party" / "iree" / "tools",
    Path.home() / "synpu_compiler" / "venv" / "bin",
]


# Path to the torq-compiler-dev repo (sibling of torq-tools-dev)
_TORQ_COMPILER_DEV = Path.home() / "synpu_compiler" / "torq-compiler-dev"
_TOSA_OPS_DIR = _TORQ_COMPILER_DEV / "tests" / "testdata" / "tosa_ops"
_PYTEST_CACHE_DIR = _TORQ_COMPILER_DEV / ".pytest_cache" / "d" / "versioned_fixtures" / "torq_compiled_model_dir"


def _compile_2610(
    mlir_path: Path,
    output_vmfb_path: Path,
    comp_name: str,
) -> None:
    """Compile an MLIR to a VMFB for SL2610 using torq-compile directly.

    Uses the same compiler flags as the pytest-based compilation:
      --torq-convert-dtypes --torq-enable-torq-hl-tiling
      --torq-convert-io-dtype --torq-enable-transpose-optimization
    """
    torq_compile = _find_tool("torq-compile")

    cmd = [
        torq_compile,
        str(mlir_path),
        "--iree-hal-target-backends=torq",
        "--torq-hw=SL2610",
        "--torq-convert-dtypes",
        "--torq-enable-torq-hl-tiling",
        "--torq-convert-io-dtype",
        "--torq-enable-transpose-optimization",
        "-o", str(output_vmfb_path),
    ]

    _logger.info("Compiling %s for SL2610 …", comp_name)
    _logger.debug("torq-compile command: %s", " ".join(cmd))

    output_vmfb_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5000)
    if result.returncode != 0:
        raise RuntimeError(
            f"torq-compile failed for {comp_name} (exit {result.returncode}):\n"
            f"{result.stderr}"
        )

    _logger.info("SL2610 VMFB written: %s (%.1f MB)",
                 output_vmfb_path, output_vmfb_path.stat().st_size / (1024 * 1024))


def _find_tool(name: str) -> str:
    """Locate an IREE CLI tool on PATH or in well-known directories."""
    found = shutil.which(name)
    if found:
        return found
    for d in _IREE_TOOL_SEARCH_PATHS:
        candidate = d / name
        if candidate.exists():
            return str(candidate)
    # Check IREE_BUILD_DIR env var
    build_dir = os.environ.get("IREE_BUILD_DIR")
    if build_dir:
        candidate = Path(build_dir) / "third_party" / "iree" / "tools" / name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"{name} not found on PATH, in well-known directories, or via IREE_BUILD_DIR"
    )


# ---------------------------------------------------------------------------
# Step 1 helpers: ONNX → TFLite (int8)
# ---------------------------------------------------------------------------

def _sanitize_onnx_names(model_path: str | os.PathLike) -> None:
    """Strip ``/`` from ONNX node/input/output names (required by onnx2tf)."""
    import onnx_graphsurgeon as gs

    onnx_model = onnx.load(str(model_path))
    graph = gs.import_onnx(onnx_model)
    for node in graph.nodes:
        node.name = node.name.strip("/")
    for inp in graph.inputs:
        inp.name = inp.name.strip("/")
    for out in graph.outputs:
        out.name = out.name.strip("/")
    graph = graph.cleanup(
        remove_unused_graph_inputs=True,
        remove_unused_node_outputs=True,
    ).toposort()
    onnx.save(gs.export_onnx(graph), str(model_path))


def _get_saved_model_input_details(saved_model_dir: str | os.PathLike) -> list[dict]:
    """Get input shapes/dtypes from a SavedModel without doing a TFLite conversion."""
    import tensorflow as tf

    loaded = tf.saved_model.load(str(saved_model_dir))
    concrete_fn = loaded.signatures["serving_default"]
    details = []
    for name, spec in concrete_fn.structured_input_signature[1].items():
        details.append({
            "name": name,
            "shape": spec.shape.as_list(),
            "dtype": spec.dtype.as_numpy_dtype,
        })
    return details


def _convert_saved_model_to_int8_tflite(
    saved_model_dir: str | os.PathLike,
    output_tflite_path: str | os.PathLike,
    num_calibration_samples: int = 100,
) -> str:
    """Convert a TensorFlow SavedModel to an int8-quantised TFLite model."""
    import tensorflow as tf

    _logger.info("Converting SavedModel → int8 TFLite …")
    _logger.info("  SavedModel : %s", saved_model_dir)
    _logger.info("  Output     : %s", output_tflite_path)

    input_details = _get_saved_model_input_details(saved_model_dir)

    _logger.info("  Inputs: %d", len(input_details))
    for inp in input_details:
        _logger.info("    %s: shape=%s", inp["name"], inp["shape"])

    # Build representative dataset for calibration
    rng = np.random.default_rng(1234)

    def representative_dataset():
        for _ in range(num_calibration_samples):
            yield [
                rng.random(
                    [d if d is not None else 1 for d in inp["shape"]],
                    dtype=np.float32,
                )
                for inp in input_details
            ]

    # Quantised conversion
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    output_path = Path(output_tflite_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    _logger.info("Saved int8 TFLite model: %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return str(output_path)


def _convert_saved_model_to_fp32_tflite(
    saved_model_dir: str | os.PathLike,
    output_tflite_path: str | os.PathLike,
) -> str:
    """Convert a TensorFlow SavedModel to an fp32 (unquantised) TFLite model."""
    import tensorflow as tf

    _logger.info("Converting SavedModel → fp32 TFLite …")
    _logger.info("  SavedModel : %s", saved_model_dir)
    _logger.info("  Output     : %s", output_tflite_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    output_path = Path(output_tflite_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    _logger.info(
        "Saved fp32 TFLite model: %s (%.1f KB)",
        output_path,
        output_path.stat().st_size / 1024,
    )

    # Post-process: fix dynamic batch dims
    _fix_dynamic_batch_size(output_path)

    return str(output_path)


# ---------------------------------------------------------------------------
# all_lstm: ONNX → TF/Keras → int16x8 TFLite → MLIR
# ---------------------------------------------------------------------------

def _onnx_lstm_gates_to_tf(W_onnx: "np.ndarray") -> "np.ndarray":
    """Reorder ONNX LSTM gate weights to TF gate order.

    ONNX gate order (along axis 0, each chunk = hidden_size):
        i (input), o (output), f (forget), c (cell)
    TF/Keras gate order:
        i (input), f (forget), c (cell), o (output)

    Parameters
    ----------
    W_onnx : ndarray
        Weight matrix of shape ``[4*H, ...]`` in ONNX gate order.

    Returns
    -------
    ndarray
        Same shape, gates reordered for TF/Keras.
    """
    H = W_onnx.shape[0] // 4
    i, o, f, c = W_onnx[:H], W_onnx[H:2*H], W_onnx[2*H:3*H], W_onnx[3*H:]
    return np.concatenate([i, f, c, o], axis=0)


def _build_all_lstm_saved_model(
    onnx_model_path: str | os.PathLike,
    output_dir: str | os.PathLike,
) -> Path:
    """Build a TF SavedModel that replicates the all_lstm ONNX graph.

    The ONNX model has the following repeating structure (×4 blocks):

        Slice → Transpose → LSTM(hidden=1024) → Squeeze → Transpose
            → LayerNorm(eps=1e-5) → (next block or output)

    Each LSTM also emits hidden/cell state outputs.

    This function:
    1. Extracts weights from the ONNX model.
    2. Builds an equivalent ``tf.keras.Model``.
    3. Saves as a TF SavedModel for downstream quantisation.
    """
    import tensorflow as tf
    import onnx
    from onnx import numpy_helper as nph

    onnx_model_path = Path(onnx_model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _logger.info("Building TF SavedModel from ONNX: %s", onnx_model_path.name)

    # ------------------------------------------------------------------
    # 1. Extract weights from ONNX
    # ------------------------------------------------------------------
    model_proto = onnx.load(str(onnx_model_path))
    graph = model_proto.graph

    init_map: dict[str, "np.ndarray"] = {}
    for init in graph.initializer:
        init_map[init.name] = np.array(nph.to_array(init))
    # Identity-copied weights (LSTM 4 shares with LSTM 3, LN weights shared)
    for node in graph.node:
        if node.op_type == "Identity" and node.input[0] in init_map:
            init_map[node.output[0]] = init_map[node.input[0]]

    # Collect per-LSTM info
    lstm_nodes = [n for n in graph.node if n.op_type == "LSTM"]
    assert len(lstm_nodes) == 4, f"Expected 4 LSTM nodes, got {len(lstm_nodes)}"

    hidden_size = 1024
    input_size = 1024
    eps = 1e-5

    # LayerNorm weights (all 4 share the same base weight/bias)
    ln_weight = init_map["ft_rnn.inter_ln.weight"]  # (1024,)
    ln_bias = init_map["ft_rnn.inter_ln.bias"]      # (1024,)

    # ------------------------------------------------------------------
    # 2. Build Keras model
    # ------------------------------------------------------------------
    # Inputs: 9 tensors, all [1, 1, 1024]
    #   en_in4.1       → LSTM 0 data input
    #   inter_h_0.1    → LSTM 0 h_state
    #   inter_h_1.1    → LSTM 0 c_state
    #   inter_h_2.1    → LSTM 1 h_state
    #   inter_h_3.1    → LSTM 1 c_state
    #   onnx::Slice_5  → LSTM 2 h_state
    #   onnx::Slice_6  → LSTM 2 c_state
    #   onnx::Slice_7  → LSTM 3 h_state
    #   onnx::Slice_8  → LSTM 3 c_state

    inp_en_in4 = tf.keras.Input(shape=(1, input_size), batch_size=1, name="en_in4_1")
    inp_h0 = tf.keras.Input(shape=(1, hidden_size), batch_size=1, name="inter_h_0_1")
    inp_c0 = tf.keras.Input(shape=(1, hidden_size), batch_size=1, name="inter_h_1_1")
    inp_h1 = tf.keras.Input(shape=(1, hidden_size), batch_size=1, name="inter_h_2_1")
    inp_c1 = tf.keras.Input(shape=(1, hidden_size), batch_size=1, name="inter_h_3_1")
    inp_h2 = tf.keras.Input(shape=(1, hidden_size), batch_size=1, name="onnx__Slice_5")
    inp_c2 = tf.keras.Input(shape=(1, hidden_size), batch_size=1, name="onnx__Slice_6")
    inp_h3 = tf.keras.Input(shape=(1, hidden_size), batch_size=1, name="onnx__Slice_7")
    inp_c3 = tf.keras.Input(shape=(1, hidden_size), batch_size=1, name="onnx__Slice_8")

    all_inputs = [inp_en_in4, inp_h0, inp_c0, inp_h1, inp_c1,
                  inp_h2, inp_c2, inp_h3, inp_c3]

    # State inputs need to be squeezed from [1,1,1024] to [1,1024] for LSTM
    def squeeze_state(x):
        return tf.squeeze(x, axis=1)

    h_states = [inp_h0, inp_h1, inp_h2, inp_h3]
    c_states = [inp_c0, inp_c1, inp_c2, inp_c3]

    # Build 4 LSTM + LayerNorm blocks
    x = inp_en_in4  # [1, 1, 1024]

    output_tensors = {}  # name -> tensor

    for block_idx in range(4):
        lstm_node = lstm_nodes[block_idx]
        W_name = lstm_node.input[1]
        R_name = lstm_node.input[2]
        B_name = lstm_node.input[3]

        # ONNX weights: W=[1, 4H, input], R=[1, 4H, H], B=[1, 8H]
        W_onnx = init_map[W_name][0]  # [4H, input_size]
        R_onnx = init_map[R_name][0]  # [4H, H]
        B_onnx = init_map[B_name][0]  # [8H] = [Wb_i,Wb_o,Wb_f,Wb_c, Rb_i,Rb_o,Rb_f,Rb_c]

        # Reorder gates: ONNX(i,o,f,c) → TF(i,f,c,o)
        # kernel (input weights): [input_size, 4H] — transposed from ONNX's [4H, input_size]
        kernel = _onnx_lstm_gates_to_tf(W_onnx).T  # [input_size, 4H]
        # recurrent_kernel: [H, 4H]
        rec_kernel = _onnx_lstm_gates_to_tf(R_onnx).T  # [H, 4H]
        # bias: ONNX has Wb + Rb separately; TF uses their sum
        Wb = B_onnx[:4 * hidden_size]  # [4H]
        Rb = B_onnx[4 * hidden_size:]  # [4H]
        Wb_gates = Wb.reshape(4, hidden_size)  # [i, o, f, c] each [H]
        Rb_gates = Rb.reshape(4, hidden_size)  # [i, o, f, c] each [H]
        # Sum Wb + Rb
        combined_bias = Wb_gates + Rb_gates  # [4, H], ONNX order
        # Reorder: ONNX(i,o,f,c) → TF(i,f,c,o)
        tf_bias = np.concatenate([
            combined_bias[0],  # i
            combined_bias[2],  # f
            combined_bias[3],  # c
            combined_bias[1],  # o
        ])  # [4H]

        # Split weights per gate (TF order: i, f, c, o)
        # kernel is [input_size, 4H] — split along axis 1
        kernel_i = kernel[:, :hidden_size]          # [input_size, H]
        kernel_f = kernel[:, hidden_size:2*hidden_size]
        kernel_c = kernel[:, 2*hidden_size:3*hidden_size]
        kernel_o = kernel[:, 3*hidden_size:]
        # rec_kernel is [H, 4H] — split along axis 1
        rec_kernel_i = rec_kernel[:, :hidden_size]  # [H, H]
        rec_kernel_f = rec_kernel[:, hidden_size:2*hidden_size]
        rec_kernel_c = rec_kernel[:, 2*hidden_size:3*hidden_size]
        rec_kernel_o = rec_kernel[:, 3*hidden_size:]
        # bias is [4H] — split into 4
        bias_i = tf_bias[:hidden_size]
        bias_f = tf_bias[hidden_size:2*hidden_size]
        bias_c = tf_bias[2*hidden_size:3*hidden_size]
        bias_o = tf_bias[3*hidden_size:]

        # ----------------------------------------------------------
        # Manual LSTM cell using **separate Dense per gate**.
        #
        # Using a single Dense(4H) → StridedSlice to split gates
        # causes a segfault in TF 2.18's TOSA converter (the
        # ``ExperimentalTFLiteToTosaBytecode`` C++ function crashes
        # on int16x8 models with FullyConnected → StridedSlice).
        #
        # Instead, we use 4 separate Dense(H) layers for input
        # projection and 4 for recurrent projection (8 total),
        # each outputting one gate directly.  This avoids the
        # StridedSlice op entirely.
        # ----------------------------------------------------------

        # Squeeze 3-D inputs to 2-D for matmul:
        #   x: [1, 1, 1024] → [1, 1024]
        x_2d = tf.keras.layers.Reshape(
            (input_size,), name=f"squeeze_x_{block_idx}"
        )(x)
        h_prev = tf.keras.layers.Reshape(
            (hidden_size,), name=f"squeeze_h_{block_idx}"
        )(h_states[block_idx])
        c_prev = tf.keras.layers.Reshape(
            (hidden_size,), name=f"squeeze_c_{block_idx}"
        )(c_states[block_idx])

        # Per-gate input projections: x @ W_gate → [1, H]
        gate_names = ["i", "f", "c", "o"]
        gate_inp_kernels = [kernel_i, kernel_f, kernel_c, kernel_o]
        inp_projs = []
        for gn, gk in zip(gate_names, gate_inp_kernels):
            layer = tf.keras.layers.Dense(
                hidden_size, use_bias=False,
                name=f"lstm_inp_{gn}_{block_idx}",
            )
            proj = layer(x_2d)
            layer.set_weights([gk])
            inp_projs.append(proj)

        # Per-gate recurrent projections: h @ R_gate + bias_gate → [1, H]
        gate_rec_kernels = [rec_kernel_i, rec_kernel_f, rec_kernel_c, rec_kernel_o]
        gate_biases = [bias_i, bias_f, bias_c, bias_o]
        rec_projs = []
        for gn, rk, gb in zip(gate_names, gate_rec_kernels, gate_biases):
            layer = tf.keras.layers.Dense(
                hidden_size, use_bias=True,
                name=f"lstm_rec_{gn}_{block_idx}",
            )
            proj = layer(h_prev)
            layer.set_weights([rk, gb])
            rec_projs.append(proj)

        # gates = inp_proj + rec_proj  → 4 × [1, H]
        gate_i = tf.keras.layers.Add(
            name=f"gate_i_{block_idx}"
        )([inp_projs[0], rec_projs[0]])
        gate_f = tf.keras.layers.Add(
            name=f"gate_f_{block_idx}"
        )([inp_projs[1], rec_projs[1]])
        gate_c = tf.keras.layers.Add(
            name=f"gate_c_{block_idx}"
        )([inp_projs[2], rec_projs[2]])
        gate_o = tf.keras.layers.Add(
            name=f"gate_o_{block_idx}"
        )([inp_projs[3], rec_projs[3]])

        # Activations
        sig_i = tf.keras.layers.Activation(
            "sigmoid", name=f"sig_i_{block_idx}"
        )(gate_i)
        sig_f = tf.keras.layers.Activation(
            "sigmoid", name=f"sig_f_{block_idx}"
        )(gate_f)
        tanh_c = tf.keras.layers.Activation(
            "tanh", name=f"tanh_c_{block_idx}"
        )(gate_c)
        sig_o = tf.keras.layers.Activation(
            "sigmoid", name=f"sig_o_{block_idx}"
        )(gate_o)

        # c_new = sig_f * c_prev + sig_i * tanh_c
        fc = tf.keras.layers.Multiply(
            name=f"fc_{block_idx}"
        )([sig_f, c_prev])
        ic = tf.keras.layers.Multiply(
            name=f"ic_{block_idx}"
        )([sig_i, tanh_c])
        c_new = tf.keras.layers.Add(
            name=f"c_new_{block_idx}"
        )([fc, ic])

        # h_new = sig_o * tanh(c_new)
        tanh_c_new = tf.keras.layers.Activation(
            "tanh", name=f"tanh_c_new_{block_idx}"
        )(c_new)
        h_new = tf.keras.layers.Multiply(
            name=f"h_new_{block_idx}"
        )([sig_o, tanh_c_new])

        # lstm_out for LayerNorm: expand h_new [1,H] → [1,1,H]
        lstm_out = tf.keras.layers.Reshape(
            (1, hidden_size), name=f"expand_lstm_out_{block_idx}"
        )(h_new)

        # Store state outputs (expand back to [1,1,1024])
        h_out = tf.keras.layers.Reshape(
            (1, hidden_size), name=f"h_out_{block_idx}"
        )(h_new)
        c_out = tf.keras.layers.Reshape(
            (1, hidden_size), name=f"c_out_{block_idx}"
        )(c_new)

        h_out_names = ["inter_h_0", "inter_h_2", "348", "456"]
        c_out_names = ["inter_h_1", "inter_h_3", "349", "457"]

        output_tensors[h_out_names[block_idx]] = h_out
        output_tensors[c_out_names[block_idx]] = c_out

        # LayerNorm: x = (x - mean) / sqrt(var + eps) * weight + bias
        ln = tf.keras.layers.LayerNormalization(
            epsilon=eps,
            name=f"layer_norm_{block_idx}",
        )
        x = ln(lstm_out)
        ln.set_weights([ln_weight, ln_bias])

    # Last LayerNorm output is en_in4
    output_tensors["en_in4"] = x

    # Build model with named outputs  matching ONNX output order
    onnx_output_names = [
        "en_in4", "inter_h_0", "inter_h_1", "inter_h_2", "inter_h_3",
        "348", "349", "456", "457",
    ]
    outputs = [output_tensors[name] for name in onnx_output_names]

    model = tf.keras.Model(inputs=all_inputs, outputs=outputs)
    _logger.info("Keras model built: %d params", model.count_params())

    # Save as SavedModel
    saved_model_dir = output_dir / "saved_model"
    model.export(str(saved_model_dir))
    _logger.info("SavedModel saved: %s", saved_model_dir)

    return saved_model_dir


def _fix_dynamic_batch_size(tflite_path: Path) -> Path:
    """Replace dynamic batch dimensions with 1 in the TFLite flatbuffer.

    TF's quantisation converter preserves the Keras ``batch_size=None``
    convention in each tensor's ``shapeSignature`` field (using ``-1`` for
    the dynamic axis).  Even though the concrete ``shape`` field already
    reads ``[1, ...]``, the ``-1`` in ``shapeSignature`` causes
    ``iree-import-tflite`` (and ``tflite_to_tosa_bytecode``) to segfault
    because the TOSA lowering pass cannot handle unknown dimensions.

    This helper rewrites **every** tensor so that:

    * ``shapeSignature`` matches the static ``shape`` (no ``-1`` entries).
    * ``shape`` entries that were 0 or -1 are replaced with 1.

    The file is modified **in-place** and its path is returned.
    """
    from tensorflow.lite.python import schema_py_generated as tfl
    import flatbuffers

    with open(tflite_path, "rb") as f:
        buf = bytearray(f.read())

    model = tfl.ModelT.InitFromPackedBuf(buf, 0)

    patched = 0
    for sg in model.subgraphs:
        for t in sg.tensors:
            # Fix shape: replace any <=0 dim with 1
            if t.shape is not None:
                for i, d in enumerate(t.shape):
                    if int(d) <= 0:
                        t.shape[i] = 1
                        patched += 1
            # Fix shapeSignature: either set to match shape or clear
            if t.shapeSignature is not None:
                has_dyn = any(int(d) < 0 for d in t.shapeSignature)
                if has_dyn:
                    if t.shape is not None:
                        t.shapeSignature = np.array(
                            list(t.shape), dtype=t.shapeSignature.dtype
                        )
                    else:
                        t.shapeSignature = None
                    patched += 1

    if patched == 0:
        _logger.info("No dynamic batch dims found in %s", tflite_path.name)
        return tflite_path

    builder = flatbuffers.Builder(len(buf) + 1024)
    packed = model.Pack(builder)
    builder.Finish(packed, b"TFL3")
    with open(tflite_path, "wb") as f:
        f.write(bytes(builder.Output()))
    _logger.info(
        "Fixed %d dynamic dims → batch_size=1 in %s",
        patched, tflite_path.name,
    )
    return tflite_path


def convert_all_lstm_onnx_to_tflite(
    onnx_model_path: str | os.PathLike,
    output_dir: str | os.PathLike,
) -> Path:
    """Export all_lstm: ONNX → TF SavedModel → fp32 TFLite.

    1. Build TF SavedModel from ONNX (manual layer construction since
       onnx2tf fails on the LSTM state shapes).
    2. Convert to fp32 TFLite.

    The fp32 TFLite is then post-processed by ``quantize_fc_ops_in_tflite``
    to add int16×int8 mixed-precision FC ops at the TFLite level.

    Returns
    -------
    Path
        Path to the fp32 TFLite file.
    """
    onnx_model_path = Path(onnx_model_path)
    output_dir = Path(output_dir) / onnx_model_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    _logger.info("=== all_lstm: ONNX → TF → fp32 TFLite ===")

    # Step 1: Build SavedModel
    saved_model_dir = _build_all_lstm_saved_model(onnx_model_path, output_dir)

    # Step 2: Produce fp32 TFLite
    fp32_path = output_dir / f"{onnx_model_path.stem}_fp32.tflite"
    _convert_saved_model_to_fp32_tflite(saved_model_dir, fp32_path)

    return fp32_path


def convert_onnx_to_tflite(
    onnx_model_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    quantize_int8: bool = True,
    num_calibration_samples: int = 100,
) -> Path:
    """
    Convert an ONNX model to TFLite (optionally int8-quantised).

    Returns the path to the resulting ``.tflite`` file.
    """
    from onnx2tf import convert as onnx2tf_convert

    onnx_model_path = Path(onnx_model_path)
    output_dir = Path(output_dir) / onnx_model_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy + sanitize so we don't modify the original
    work_copy = output_dir / onnx_model_path.name
    if not work_copy.exists():
        shutil.copy2(onnx_model_path, work_copy)
    _sanitize_onnx_names(work_copy)

    # ONNX input names (for preserving shapes)
    graph = onnx.load(str(work_copy)).graph
    onnx_inputs = [str(i.name) for i in graph.input]

    _logger.info("ONNX → SavedModel via onnx2tf …")
    onnx2tf_convert(
        str(work_copy),
        output_folder_path=str(output_dir),
        keep_shape_absolutely_input_names=onnx_inputs,
        copy_onnx_input_output_names_to_tflite=True,
    )
    _logger.info("onnx2tf conversion complete → %s", output_dir)

    primary_path: Path | None = None

    # onnx2tf always produces a fp32 TFLite alongside the SavedModel
    fp32_path = output_dir / f"{onnx_model_path.stem}_float32.tflite"
    if not fp32_path.exists():
        candidates = list(output_dir.glob("*.tflite"))
        if candidates:
            fp32_path = candidates[0]
        else:
            raise FileNotFoundError(f"No .tflite file produced in {output_dir}")

    if quantize_int8:
        int8_path = output_dir / f"{onnx_model_path.stem}_int8.tflite"
        try:
            _convert_saved_model_to_int8_tflite(
                output_dir,
                int8_path,
                num_calibration_samples=num_calibration_samples,
            )
            primary_path = int8_path
        except Exception:
            _logger.warning(
                "Int8 quantization failed for %s — falling back to fp32 TFLite. "
                "Mixed-precision export will still work.",
                onnx_model_path.stem,
                exc_info=True,
            )
            primary_path = fp32_path
    else:
        primary_path = fp32_path

    return primary_path


# ---------------------------------------------------------------------------
# Step 2 helpers: TFLite → TOSA → MLIR
# ---------------------------------------------------------------------------

def _clean_env_for_iree_tools() -> dict[str, str]:
    """Return a sanitised copy of ``os.environ`` for iree CLI tools.

    When the export runs inside a virtual-environment that ships its own
    TensorFlow (e.g. ``.venv_customer_b``), the ``VIRTUAL_ENV`` and
    ``PYTHONPATH`` variables can cause ``iree-import-tflite`` to load
    conflicting TF shared libraries and segfault.  Stripping these
    variables lets the tool's own shebang Python resolve its own packages.
    """
    env = os.environ.copy()
    for key in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    # Ensure the tool's own venv bin dir is first on PATH
    return env


def _split_large_transpose_conv(tflite_path: Path, num_tiles: int = 8) -> Path:
    """Split large TransposeConv ops along the output-channel axis.

    The SL2610 has only 512 KB of LRAM.  When a TransposeConv has a weight
    tensor that is too large (e.g. ``[256, 5, 2, 512]`` ≈ 1.3 MB after
    restructuring), the compiler fails with *"unable to free enough space"*.

    This function replaces each such TransposeConv with ``num_tiles`` smaller
    TransposeConv ops (each producing ``OC / num_tiles`` output channels)
    followed by a CONCATENATION along the channel axis.

    Matching criterion: weight IC >= 256 **and** weight OC >= 128.

    The original file is **not** modified.  A new file with
    ``_split_tconv`` suffix is written next to it.
    """
    from tensorflow.lite.python import schema_py_generated as tfl_schema
    import flatbuffers
    import copy

    with open(tflite_path, "rb") as f:
        buf = bytearray(f.read())

    model = tfl_schema.ModelT.InitFromPackedBuf(buf, 0)
    sg = model.subgraphs[0]

    TRANSPOSE_CONV = 67
    CONCATENATION = 2

    # Map from builtin code → opcode index
    def _get_builtin_code(oc):
        return oc.deprecatedBuiltinCode if oc.deprecatedBuiltinCode != 127 else oc.builtinCode

    # --- Find TransposeConv ops that need splitting ---
    ops_to_split = []  # list of operator indices
    for oi, op in enumerate(sg.operators):
        oc = model.operatorCodes[op.opcodeIndex]
        if _get_builtin_code(oc) != TRANSPOSE_CONV:
            continue

        # TransposeConv inputs: [output_shape, weights, input, bias(opt)]
        wt_tidx = int(op.inputs[1])
        wt_tensor = sg.tensors[wt_tidx]
        wt_shape = list(wt_tensor.shape)  # [OC, kH, kW, IC]

        if len(wt_shape) != 4:
            continue

        oc_dim, _kh, _kw, ic_dim = wt_shape
        if ic_dim >= 256 and oc_dim >= 128:
            _logger.info(
                "TransposeConv op %d: weight [%s] (IC=%d, OC=%d) — will split into %d tiles",
                oi, "x".join(str(d) for d in wt_shape), ic_dim, oc_dim, num_tiles,
            )
            ops_to_split.append(oi)

    if not ops_to_split:
        _logger.info("No large TransposeConv ops found — nothing to split")
        return tflite_path

    # --- Ensure CONCATENATION opcode exists ---
    concat_opcode_idx = None
    for idx, oc in enumerate(model.operatorCodes):
        if _get_builtin_code(oc) == CONCATENATION:
            concat_opcode_idx = idx
            break
    if concat_opcode_idx is None:
        new_oc = tfl_schema.OperatorCodeT()
        new_oc.deprecatedBuiltinCode = CONCATENATION
        new_oc.builtinCode = CONCATENATION
        new_oc.version = 1
        model.operatorCodes.append(new_oc)
        concat_opcode_idx = len(model.operatorCodes) - 1

    # --- Find TransposeConv opcode index (for sub-ops) ---
    tconv_opcode_idx = None
    for idx, oc in enumerate(model.operatorCodes):
        if _get_builtin_code(oc) == TRANSPOSE_CONV:
            tconv_opcode_idx = idx
            break

    # --- Numpy dtype from TFLite TensorType ---
    _DTYPE_MAP = {
        tfl_schema.TensorType.FLOAT32: np.float32,
        tfl_schema.TensorType.INT8: np.int8,
        tfl_schema.TensorType.INT16: np.int16,
        tfl_schema.TensorType.INT32: np.int32,
        tfl_schema.TensorType.INT64: np.int64,
    }

    # Process in reverse so indices stay valid
    for oi in reversed(ops_to_split):
        op = sg.operators[oi]

        # --- Parse original tensors ---
        oshape_tidx = int(op.inputs[0])  # output_shape tensor
        wt_tidx = int(op.inputs[1])      # weights
        inp_tidx = int(op.inputs[2])     # input activation
        has_bias = len(op.inputs) >= 4 and int(op.inputs[3]) != -1
        bias_tidx = int(op.inputs[3]) if has_bias else -1
        out_tidx = int(op.outputs[0])

        wt_tensor = sg.tensors[wt_tidx]
        wt_shape = list(wt_tensor.shape)  # [OC, kH, kW, IC]
        total_oc = wt_shape[0]
        tile_oc = total_oc // num_tiles
        assert tile_oc * num_tiles == total_oc, (
            f"OC={total_oc} not divisible by num_tiles={num_tiles}"
        )

        # Read weight data
        wt_dtype = _DTYPE_MAP[wt_tensor.type]
        wt_buf = model.buffers[wt_tensor.buffer].data
        wt_data = np.frombuffer(bytes(wt_buf), dtype=wt_dtype).reshape(wt_shape)

        # Read bias data (if present)
        bias_data = None
        bias_dtype = None
        if has_bias:
            bias_tensor = sg.tensors[bias_tidx]
            bias_dtype = _DTYPE_MAP[bias_tensor.type]
            bias_buf = model.buffers[bias_tensor.buffer].data
            bias_data = np.frombuffer(bytes(bias_buf), dtype=bias_dtype)

        # Read original output shape tensor (constant [N, H, W, OC])
        oshape_tensor = sg.tensors[oshape_tidx]
        oshape_buf = model.buffers[oshape_tensor.buffer].data
        orig_out_shape = np.frombuffer(bytes(oshape_buf), dtype=np.int32).copy()

        # Original output tensor (for quantization params, type, etc.)
        out_tensor = sg.tensors[out_tidx]

        # Parse TransposeConvOptions
        orig_opts = op.builtinOptions  # TransposeConvOptionsT

        # --- Create per-tile ops ---
        tile_out_tidxs = []
        tile_ops = []

        for t in range(num_tiles):
            oc_start = t * tile_oc
            oc_end = oc_start + tile_oc

            # -- Tile weight tensor --
            tile_wt = wt_data[oc_start:oc_end].copy()
            tile_wt_buf = tfl_schema.BufferT()
            tile_wt_buf.data = np.frombuffer(tile_wt.tobytes(), dtype=np.uint8)
            model.buffers.append(tile_wt_buf)

            tile_wt_tensor = tfl_schema.TensorT()
            tile_wt_tensor.name = f"tconv_split_w_tile{t}".encode("utf-8")
            tile_wt_shape = [tile_oc] + wt_shape[1:]
            tile_wt_tensor.shape = np.array(tile_wt_shape, dtype=np.int32)
            tile_wt_tensor.type = wt_tensor.type
            tile_wt_tensor.buffer = len(model.buffers) - 1
            tile_wt_tensor.isVariable = False
            tile_wt_tensor.hasRank = True
            # Copy weight quantization (per-channel: slice scales/zps)
            if wt_tensor.quantization is not None:
                wt_q = copy.deepcopy(wt_tensor.quantization)
                if wt_q.scale is not None and len(wt_q.scale) == total_oc:
                    wt_q.scale = wt_q.scale[oc_start:oc_end]
                if wt_q.zeroPoint is not None and len(wt_q.zeroPoint) == total_oc:
                    wt_q.zeroPoint = wt_q.zeroPoint[oc_start:oc_end]
                tile_wt_tensor.quantization = wt_q
            sg.tensors.append(tile_wt_tensor)
            tile_wt_tidx = len(sg.tensors) - 1

            # -- Tile bias tensor --
            tile_bias_tidx = -1
            if has_bias and bias_data is not None:
                tile_bias = bias_data[oc_start:oc_end].copy()
                tile_bias_buf_obj = tfl_schema.BufferT()
                tile_bias_buf_obj.data = np.frombuffer(tile_bias.tobytes(), dtype=np.uint8)
                model.buffers.append(tile_bias_buf_obj)

                tile_bias_tensor = tfl_schema.TensorT()
                tile_bias_tensor.name = f"tconv_split_b_tile{t}".encode("utf-8")
                tile_bias_tensor.shape = np.array([tile_oc], dtype=np.int32)
                tile_bias_tensor.type = sg.tensors[bias_tidx].type
                tile_bias_tensor.buffer = len(model.buffers) - 1
                tile_bias_tensor.isVariable = False
                tile_bias_tensor.hasRank = True
                if sg.tensors[bias_tidx].quantization is not None:
                    tile_bias_tensor.quantization = copy.deepcopy(
                        sg.tensors[bias_tidx].quantization
                    )
                    bq = tile_bias_tensor.quantization
                    if bq.scale is not None and len(bq.scale) == total_oc:
                        bq.scale = bq.scale[oc_start:oc_end]
                    if bq.zeroPoint is not None and len(bq.zeroPoint) == total_oc:
                        bq.zeroPoint = bq.zeroPoint[oc_start:oc_end]
                sg.tensors.append(tile_bias_tensor)
                tile_bias_tidx = len(sg.tensors) - 1

            # -- Tile output_shape tensor (constant: [N, H, W, tile_oc]) --
            tile_out_shape_arr = orig_out_shape.copy()
            tile_out_shape_arr[-1] = tile_oc  # last dim = OC in NHWC
            tile_oshape_buf = tfl_schema.BufferT()
            tile_oshape_buf.data = np.frombuffer(tile_out_shape_arr.tobytes(), dtype=np.uint8)
            model.buffers.append(tile_oshape_buf)

            tile_oshape_tensor = tfl_schema.TensorT()
            tile_oshape_tensor.name = f"tconv_split_oshape_tile{t}".encode("utf-8")
            tile_oshape_tensor.shape = np.array(
                list(oshape_tensor.shape), dtype=np.int32
            )
            tile_oshape_tensor.type = oshape_tensor.type
            tile_oshape_tensor.buffer = len(model.buffers) - 1
            tile_oshape_tensor.isVariable = False
            tile_oshape_tensor.hasRank = True
            sg.tensors.append(tile_oshape_tensor)
            tile_oshape_tidx = len(sg.tensors) - 1

            # -- Tile output tensor --
            tile_out_tensor = tfl_schema.TensorT()
            tile_out_tensor.name = f"tconv_split_out_tile{t}".encode("utf-8")
            tile_out_shape = list(out_tensor.shape)
            tile_out_shape[-1] = tile_oc  # NHWC: last dim is channels
            tile_out_tensor.shape = np.array(tile_out_shape, dtype=np.int32)
            tile_out_tensor.type = out_tensor.type
            tile_out_tensor.buffer = 0  # runtime-allocated
            tile_out_tensor.isVariable = False
            tile_out_tensor.hasRank = True
            if out_tensor.quantization is not None:
                tile_out_tensor.quantization = copy.deepcopy(out_tensor.quantization)
            sg.tensors.append(tile_out_tensor)
            tile_out_tidx_ = len(sg.tensors) - 1
            tile_out_tidxs.append(tile_out_tidx_)

            # -- Build tiled TransposeConv op --
            tile_op = tfl_schema.OperatorT()
            tile_op.opcodeIndex = tconv_opcode_idx
            inputs = [tile_oshape_tidx, tile_wt_tidx, inp_tidx]
            if has_bias:
                inputs.append(tile_bias_tidx)
            tile_op.inputs = np.array(inputs, dtype=np.int32)
            tile_op.outputs = np.array([tile_out_tidx_], dtype=np.int32)

            # Copy the original TransposeConv options
            tile_opts = tfl_schema.TransposeConvOptionsT()
            tile_opts.padding = orig_opts.padding
            tile_opts.strideW = orig_opts.strideW
            tile_opts.strideH = orig_opts.strideH
            tile_opts.fusedActivationFunction = orig_opts.fusedActivationFunction
            tile_opts.quantizedBiasType = orig_opts.quantizedBiasType
            tile_op.builtinOptionsType = tfl_schema.BuiltinOptions.TransposeConvOptions
            tile_op.builtinOptions = tile_opts

            tile_ops.append(tile_op)

        # -- Build CONCATENATION op --
        concat_op = tfl_schema.OperatorT()
        concat_op.opcodeIndex = concat_opcode_idx
        concat_op.inputs = np.array(tile_out_tidxs, dtype=np.int32)
        concat_op.outputs = np.array([out_tidx], dtype=np.int32)

        concat_opts = tfl_schema.ConcatenationOptionsT()
        concat_opts.axis = len(out_tensor.shape) - 1  # channel axis (NHWC → last)
        concat_opts.fusedActivationFunction = 0  # NONE
        concat_op.builtinOptionsType = tfl_schema.BuiltinOptions.ConcatenationOptions
        concat_op.builtinOptions = concat_opts

        # -- Replace original op: insert tile ops before it, then replace
        #    the (now shifted) original with the concat.
        #    Deleting or disconnecting operators corrupts the flatbuffer
        #    re-serialisation and causes TOSA bytecode assertion failures,
        #    so we use in-place replacement (same pattern as
        #    _replace_scatter_nd_with_concat). ---
        for ti, tile_op in enumerate(tile_ops):
            sg.operators.insert(oi + ti, tile_op)
        sg.operators[oi + num_tiles] = concat_op

    _logger.info(
        "Split %d TransposeConv ops into %d tiles each in %s",
        len(ops_to_split), num_tiles, tflite_path.name,
    )

    # Re-serialize
    builder = flatbuffers.Builder(len(buf) * 2)
    packed = model.Pack(builder)
    builder.Finish(packed, b"TFL3")

    out_path = tflite_path.with_name(
        tflite_path.stem + "_split_tconv" + tflite_path.suffix
    )
    out_path.write_bytes(bytes(builder.Output()))
    _logger.info(
        "Wrote split-TransposeConv TFLite: %s (%.1f KB)",
        out_path.name, out_path.stat().st_size / 1024,
    )
    return out_path


def _replace_scatter_nd_with_concat(tflite_path: Path) -> Path:
    """Replace ScatterND ops that pad a slice with zeros with CONCATENATION.

    The ``all_conv`` model uses ``ScatterND`` to place a ``[1,1,N,1]`` tensor
    into slot 0 or slot 1 of a zero-filled ``[1,2,N,1]`` tensor.  TOSA does
    not support ``tfl.scatter_nd``, so this function replaces each such op
    with a ``CONCATENATION`` of the updates tensor with a constant zero
    tensor of the same shape, in the correct order along axis 1.

    The detection is generic: any ``ScatterND`` whose indices are constant,
    cover a single contiguous slice along some axis, and whose output has
    exactly 2 slices along that axis is a candidate.

    The original file is **not** modified.  A new file with ``_no_scatter``
    suffix is written next to it.

    Returns the path to the patched ``.tflite`` file (or a copy if no
    ``ScatterND`` ops were found).
    """
    from tensorflow.lite.python import schema_py_generated as tfl_schema
    import flatbuffers

    with open(tflite_path, "rb") as f:
        buf = bytearray(f.read())

    model = tfl_schema.ModelT.InitFromPackedBuf(buf, 0)
    sg = model.subgraphs[0]

    SCATTER_ND = 122
    CONCATENATION = 2

    # --- Identify ScatterND ops ---
    scatter_ops = []  # (operator_index, dim1_value)
    for oi, op in enumerate(sg.operators):
        oc = model.operatorCodes[op.opcodeIndex]
        code = oc.deprecatedBuiltinCode if oc.deprecatedBuiltinCode != 127 else oc.builtinCode
        if code != SCATTER_ND:
            continue

        # TFLite scatter_nd: inputs = [indices, updates, shape]
        idx_tidx = int(op.inputs[0])
        idx_tensor = sg.tensors[idx_tidx]
        idx_buf = model.buffers[idx_tensor.buffer].data
        if idx_buf is None:
            continue  # dynamic indices — can't replace

        idx_arr = np.frombuffer(bytes(idx_buf), dtype=np.int32).reshape(list(idx_tensor.shape))
        # Shape is [1, 1, N, 1, ndims]  — last dim is the number of index dims
        # Check that all indices target a single dim1 value
        dim1_vals = sorted(set(idx_arr[:, :, :, :, 1].flatten().tolist()))
        if len(dim1_vals) != 1:
            continue  # indices span multiple dim1 slots — not our pattern

        # Verify the output has exactly 2 slices along dim 1
        out_tidx = int(op.outputs[0])
        out_tensor = sg.tensors[out_tidx]
        if int(out_tensor.shape[1]) != 2:
            continue

        scatter_ops.append((oi, dim1_vals[0]))

    if not scatter_ops:
        out_path = tflite_path.with_name(
            tflite_path.stem + "_no_scatter" + tflite_path.suffix
        )
        import shutil
        shutil.copy2(tflite_path, out_path)
        return out_path

    # Ensure CONCATENATION opcode exists
    concat_opcode_idx = None
    for idx, oc in enumerate(model.operatorCodes):
        c = oc.deprecatedBuiltinCode if oc.deprecatedBuiltinCode != 127 else oc.builtinCode
        if c == CONCATENATION:
            concat_opcode_idx = idx
            break
    if concat_opcode_idx is None:
        new_oc = tfl_schema.OperatorCodeT()
        new_oc.deprecatedBuiltinCode = CONCATENATION
        new_oc.builtinCode = CONCATENATION
        new_oc.version = 1
        model.operatorCodes.append(new_oc)
        concat_opcode_idx = len(model.operatorCodes) - 1

    # Create a single zero-constant tensor [1, 1, N, 1] for padding.
    # We'll share it across all replacements with the same shape.
    zero_tensors = {}  # shape_tuple → tensor_index

    def _get_zero_tensor(shape_tuple):
        if shape_tuple in zero_tensors:
            return zero_tensors[shape_tuple]
        zero_data = np.zeros(shape_tuple, dtype=np.float32)
        zero_buf = tfl_schema.BufferT()
        zero_buf.data = list(zero_data.tobytes())
        model.buffers.append(zero_buf)
        zero_buf_idx = len(model.buffers) - 1

        t = tfl_schema.TensorT()
        t.name = f"scatter_nd_replacement_zeros_{len(zero_tensors)}".encode("utf-8")
        t.shape = np.array(list(shape_tuple), dtype=np.int32)
        t.type = tfl_schema.TensorType.FLOAT32
        t.buffer = zero_buf_idx
        t.isVariable = False
        t.hasRank = True
        sg.tensors.append(t)
        tidx = len(sg.tensors) - 1
        zero_tensors[shape_tuple] = tidx
        return tidx

    # Replace each ScatterND with CONCATENATION
    # Process in reverse order so operator indices stay valid
    for oi, dim1_val in reversed(scatter_ops):
        op = sg.operators[oi]
        upd_tidx = int(op.inputs[1])  # updates tensor [1, 1, N, 1]
        out_tidx = int(op.outputs[0])  # output tensor [1, 2, N, 1]

        upd_tensor = sg.tensors[upd_tidx]
        upd_shape = tuple(int(d) for d in upd_tensor.shape)  # (1, 1, N, 1)

        zero_tidx = _get_zero_tensor(upd_shape)

        # Build CONCATENATION op along axis=1
        concat_op = tfl_schema.OperatorT()
        concat_op.opcodeIndex = concat_opcode_idx

        if dim1_val == 0:
            # Data goes to slot 0, zeros to slot 1
            concat_op.inputs = np.array([upd_tidx, zero_tidx], dtype=np.int32)
        else:
            # Zeros to slot 0, data goes to slot 1
            concat_op.inputs = np.array([zero_tidx, upd_tidx], dtype=np.int32)

        concat_op.outputs = np.array([out_tidx], dtype=np.int32)

        # ConcatenationOptions: axis=1, fused_activation=NONE
        concat_opts = tfl_schema.ConcatenationOptionsT()
        concat_opts.axis = 1
        concat_opts.fusedActivationFunction = 0  # NONE
        concat_op.builtinOptionsType = tfl_schema.BuiltinOptions.ConcatenationOptions
        concat_op.builtinOptions = concat_opts

        # Replace the ScatterND op in place
        sg.operators[oi] = concat_op

    _logger.info(
        "Replaced %d ScatterND ops with CONCATENATION in %s",
        len(scatter_ops),
        tflite_path.name,
    )

    # Re-serialize
    builder = flatbuffers.Builder(len(buf) * 2)
    packed = model.Pack(builder)
    builder.Finish(packed, b"TFL3")

    out_path = tflite_path.with_name(
        tflite_path.stem + "_no_scatter" + tflite_path.suffix
    )
    out_path.write_bytes(bytes(builder.Output()))
    _logger.info("Wrote patched TFLite: %s (%.1f KB)", out_path.name, out_path.stat().st_size / 1024)
    return out_path


def convert_tflite_to_mlir(
    tflite_path: str | os.PathLike,
    output_dir: str | os.PathLike,
) -> Path:
    """
    Convert a TFLite model to text MLIR via TOSA.

    Returns the path to the ``.mlir`` file.
    """
    tflite_path = Path(tflite_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tosa_path = output_dir / tflite_path.with_suffix(".tosa").name
    mlir_path = output_dir / tflite_path.with_suffix(".mlir").name

    # Use a clean environment so iree tools don't pick up a conflicting TF.
    clean_env = _clean_env_for_iree_tools()

    # iree-import-tflite → TOSA bytecode
    _logger.info("TFLite → TOSA: %s", tflite_path.name)
    iree_import_tflite = _find_tool("iree-import-tflite")
    subprocess.check_call(
        [iree_import_tflite, str(tflite_path), "-o", str(tosa_path)],
        timeout=120,
        env=clean_env,
    )

    # iree-opt → text MLIR
    _logger.info("TOSA → MLIR: %s", tosa_path.name)
    iree_opt = _find_tool("iree-opt")

    subprocess.check_call(
        [iree_opt, str(tosa_path), "-o", str(mlir_path)],
        timeout=120,
        env=clean_env,
    )

    _logger.info("MLIR written: %s", mlir_path)
    return mlir_path


# ---------------------------------------------------------------------------
# Full export pipeline
# ---------------------------------------------------------------------------

def export_customer_b(
    models_dir: str | os.PathLike = DEFAULT_MODELS_DIR,
    output_dir: str | os.PathLike = "output_customer_b",
    component: str | None = None,
    quantize_int8: bool = True,
    num_calibration_samples: int = 100,
    skip_tflite: bool = False,
    skip_iree: bool = False,
):
    """
    End-to-end export: ONNX → TFLite → mixed-precision TFLite → MLIR → VMFB.

    Parameters
    ----------
    models_dir : path
        Directory containing the Customer B ONNX files.
    output_dir : path
        Root output directory for all artefacts.
    component : str or None
        If set, export only this component (e.g. ``"all_fc"``).
    quantize_int8 : bool
        Also quantise to int8 TFLite (for comparison).
    skip_tflite : bool
        Skip the ONNX → TFLite step (expect existing .tflite).
    skip_iree : bool
        Skip the MLIR → VMFB compilation step.
    """
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    components = {component: MODEL_COMPONENTS[component]} if component else MODEL_COMPONENTS

    for comp_name, onnx_filename in components.items():
        _logger.info("=" * 60)
        _logger.info("Processing component: %s", comp_name)
        _logger.info("=" * 60)

        onnx_path = models_dir / onnx_filename
        comp_output_dir = output_dir / comp_name

        # Step 1: ONNX → TFLite
        if comp_name == "all_lstm":
            # all_lstm uses a specialised pipeline (manual TF model build)
            mixed_tflite_path_lstm = comp_output_dir / "all_lstm" / f"{onnx_path.stem}_fc_int16x8_mixed.tflite"

            if skip_tflite:
                if not mixed_tflite_path_lstm.exists():
                    _logger.error("No existing mixed TFLite found for %s at %s — run without --skip-tflite first",
                                  comp_name, mixed_tflite_path_lstm)
                    continue
                _logger.info("Using existing mixed TFLite: %s", mixed_tflite_path_lstm)
            else:
                if not onnx_path.exists():
                    _logger.error("ONNX model not found: %s", onnx_path)
                    continue

                fp32_tflite_path = convert_all_lstm_onnx_to_tflite(
                    onnx_path,
                    comp_output_dir,
                )

                # Default approach: quantize FC ops at the TFLite level.
                # This wraps each FC with QUANTIZE/DEQUANTIZE and uses
                # int16 activations + int8 weights + INT64 bias directly
                # in the flatbuffer, then converts the mixed TFLite to MLIR.
                # The INT64 bias avoids the i48 crash in iree-opt.
                mixed_tflite_path_lstm = quantize_fc_ops_in_tflite(
                    fp32_tflite_path,
                    output_path=fp32_tflite_path.parent / f"{onnx_path.stem}_fc_int16x8_mixed.tflite",
                )
                _logger.info("Mixed int16x8 TFLite: %s", mixed_tflite_path_lstm.name)

            mlir_path = convert_tflite_to_mlir(mixed_tflite_path_lstm, comp_output_dir)
            _logger.info("int16x8-FC MLIR: %s", mlir_path.name)

            if not skip_iree:
                vmfb_path = comp_output_dir / mlir_path.with_suffix(".vmfb").name
                _compile_2610(mlir_path, vmfb_path, comp_name)
                _logger.info("VMFB written: %s", vmfb_path)
            else:
                _logger.info("Skipping IREE compilation for %s", comp_name)
            continue

        if skip_tflite:
            suffix = "_int8.tflite" if quantize_int8 else "_float32.tflite"
            tflite_path = comp_output_dir / onnx_path.stem / f"{onnx_path.stem}{suffix}"
            if not tflite_path.exists():
                candidates = list(comp_output_dir.rglob("*.tflite"))
                if not candidates:
                    _logger.error("No existing TFLite found for %s — run without --skip-tflite first", comp_name)
                    continue
                tflite_path = candidates[0]
            _logger.info("Using existing TFLite: %s", tflite_path)
        else:
            if not onnx_path.exists():
                _logger.error("ONNX model not found: %s", onnx_path)
                continue

            tflite_path = convert_onnx_to_tflite(
                onnx_path,
                comp_output_dir,
                quantize_int8=quantize_int8,
                num_calibration_samples=num_calibration_samples,
            )

        # Step 1b: Replace ScatterND ops with CONCATENATION (TOSA compat)
        tflite_path = _replace_scatter_nd_with_concat(Path(tflite_path))

        # Step 1c: Split large TransposeConv ops to fit in LRAM
        tflite_path = _split_large_transpose_conv(Path(tflite_path))

        # Step 2: TFLite → TOSA → MLIR (primary int8 / fp32)
        mlir_path = None
        try:
            mlir_path = convert_tflite_to_mlir(tflite_path, comp_output_dir)
        except subprocess.CalledProcessError:
            _logger.warning(
                "Primary TFLite → TOSA conversion failed for %s. "
                "Will attempt mixed-precision path instead.",
                comp_name,
            )

        # Step 2a: Mixed-precision (int16×int8) path
        # Locate the original fp32 TFLite produced by onnx2tf.
        # tflite_path may have been renamed by _replace_scatter_nd_with_concat,
        # so strip suffixes to find the original.
        base_stem = (tflite_path.stem
                     .replace("_split_tconv", "")
                     .replace("_no_scatter", "")
                     .replace("_int8", "")
                     .replace("_float32", ""))
        fp32_tflite_for_mixed = tflite_path.parent / f"{base_stem}_float32.tflite"
        mixed_mlir_path = None

        # Check if the fully-processed mixed TFLite already exists (e.g.
        # hand-crafted with correct quantisation for all ops including
        # TransposeConv).  When --skip-tflite is set we prefer the existing
        # artefact so we don't regenerate it (quantize_fc_ops_in_tflite only
        # quantises FC/CONV_2D, leaving TransposeConv as fp32 which differs
        # from the hand-built TFLite).
        final_mixed_tflite = tflite_path.parent / f"{base_stem}_fc_int16x8_mixed_no_scatter_split_tconv.tflite"
        if skip_tflite and final_mixed_tflite.exists():
            _logger.info("Using existing mixed TFLite: %s", final_mixed_tflite)
            try:
                mixed_mlir_path = convert_tflite_to_mlir(final_mixed_tflite, comp_output_dir)
                _logger.info("Mixed int16x8 MLIR: %s", mixed_mlir_path.name)
            except subprocess.CalledProcessError:
                _logger.warning(
                    "Mixed TFLite → TOSA conversion failed for %s.",
                    comp_name,
                )
        elif fp32_tflite_for_mixed.exists():
            mixed_tflite_path = quantize_fc_ops_in_tflite(
                fp32_tflite_for_mixed,
                output_path=fp32_tflite_for_mixed.parent / f"{fp32_tflite_for_mixed.stem.replace('_float32', '')}_fc_int16x8_mixed.tflite",
            )
            _logger.info("Mixed int16x8 TFLite: %s", mixed_tflite_path.name)
            # Replace ScatterND ops before TOSA conversion
            mixed_tflite_path = _replace_scatter_nd_with_concat(mixed_tflite_path)
            # Split large TransposeConv ops to fit in LRAM
            mixed_tflite_path = _split_large_transpose_conv(mixed_tflite_path)
            try:
                mixed_mlir_path = convert_tflite_to_mlir(mixed_tflite_path, comp_output_dir)
                _logger.info("Mixed int16x8 MLIR: %s", mixed_mlir_path.name)
            except subprocess.CalledProcessError:
                _logger.warning(
                    "Mixed TFLite → TOSA conversion failed for %s. "
                    "The model may contain ops unsupported by TOSA (e.g. scatter_nd).",
                    comp_name,
                )
        else:
            _logger.info("No fp32 TFLite found at %s — skipping mixed-precision export", fp32_tflite_for_mixed)

        # Step 3: Compile MLIR → VMFB (mixed-precision only)
        if skip_iree:
            _logger.info("Skipping IREE compilation for %s", comp_name)
            continue

        if mixed_mlir_path is not None:
            mixed_vmfb_path = comp_output_dir / mixed_mlir_path.with_suffix(".vmfb").name
            _compile_2610(mixed_mlir_path, mixed_vmfb_path, f"{comp_name}_mixed")
            _logger.info("Mixed int16x8 VMFB written: %s", mixed_vmfb_path)
        else:
            _logger.error("No mixed MLIR available for %s — cannot compile", comp_name)

    _logger.info("Export complete → %s", output_dir)


def export_customer_b_from_args(args: argparse.Namespace):
    """Entry point from CLI."""
    configure_logging(getattr(args, "logging", "info"))
    export_customer_b(
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        component=args.component,
        quantize_int8=args.quantize_int8,
        num_calibration_samples=args.num_calibration_samples,
        skip_tflite=args.skip_tflite,
        skip_iree=args.skip_iree,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export Customer B ONNX models → TFLite (int8) → MLIR → VMFB"
    )
    add_customer_b_export_args(parser)
    args = parser.parse_args()
    export_customer_b_from_args(args)


if __name__ == "__main__":
    main()
