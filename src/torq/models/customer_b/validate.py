# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""
Validate Customer B models by comparing TFLite runtime output against
IREE (llvm-cpu or torq) runtime output.

Usage
~~~~~
::

    # Compare TFLite vs IREE (llvm-cpu) for a single component
    python -m torq.models.customer_b.validate \\
        --tflite output_customer_b/all_fc/all_fc/all_fc_int8.tflite \\
        --vmfb   output_customer_b/all_fc/all_fc_int8.vmfb

    # Validate all components under an output directory
    python -m torq.models.customer_b.validate --output-dir output_customer_b

    # Custom tolerance
    python -m torq.models.customer_b.validate --output-dir output_customer_b --int-tol 2
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from torq.utils.logging import configure_logging

from .infer import _get_tflite_io_details, _run_tflite, _run_iree, _generate_random_inputs
from . import MODEL_COMPONENTS

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def compare_outputs(
    tflite_outputs: list[np.ndarray],
    iree_outputs: list[np.ndarray],
    int_tol: int = 1,
    fp_avg_tol: float = 0.01,
    fp_max_tol: float = 0.05,
) -> tuple[bool, list[dict]]:
    """
    Compare TFLite and IREE outputs element-wise.

    Returns (all_passed, list_of_per_output_results).
    """
    results = []
    all_passed = True

    for idx, (tf_out, iree_out) in enumerate(zip(tflite_outputs, iree_outputs)):
        res = {"index": idx, "passed": True, "details": ""}

        # Handle batch dimension mismatch: TFLite often keeps the leading
        # batch=1 dim while IREE may squeeze it away (or vice-versa).
        if tf_out.shape != iree_out.shape:
            tf_sq = np.squeeze(tf_out)
            ir_sq = np.squeeze(iree_out)
            if tf_sq.shape == ir_sq.shape:
                _logger.debug(
                    "  output[%d]: auto-squeezed shapes TFLite=%s->%s  IREE=%s->%s",
                    idx, tf_out.shape, tf_sq.shape, iree_out.shape, ir_sq.shape,
                )
                tf_out = tf_sq
                iree_out = ir_sq
            else:
                res["passed"] = False
                res["details"] = f"Shape mismatch: TFLite={tf_out.shape} IREE={iree_out.shape}"
                results.append(res)
                all_passed = False
                continue

        tf_f = tf_out.astype(np.float64)
        ir_f = iree_out.astype(np.float64)
        abs_diff = np.abs(tf_f - ir_f)
        max_abs = float(abs_diff.max())
        n_diff = int(np.count_nonzero(abs_diff))
        total = abs_diff.size
        pct = 100.0 * n_diff / total if total else 0.0

        if np.issubdtype(tf_out.dtype, np.integer):
            passed = max_abs <= int_tol
            res["details"] = (
                f"int  max_abs_diff={max_abs:.0f}  diff={n_diff}/{total} [{pct:.2f}%]  "
                f"tol={int_tol}  {'PASS' if passed else 'FAIL'}"
            )
        else:
            denom = np.abs(tf_f) + np.abs(ir_f) + 1e-8
            rel_diff = abs_diff / denom
            avg_rel = float(rel_diff.mean())
            max_rel = float(rel_diff.max())
            passed = avg_rel <= fp_avg_tol and max_rel <= fp_max_tol
            res["details"] = (
                f"fp   avg_rel={avg_rel:.6f}  max_rel={max_rel:.6f}  "
                f"diff={n_diff}/{total} [{pct:.2f}%]  "
                f"avg_tol={fp_avg_tol}  max_tol={fp_max_tol}  {'PASS' if passed else 'FAIL'}"
            )

        res["passed"] = passed
        if not passed:
            all_passed = False
        results.append(res)

    return all_passed, results


# ---------------------------------------------------------------------------
# Validate a single TFLite / VMFB pair
# ---------------------------------------------------------------------------

def validate_pair(
    tflite_path: str | Path,
    vmfb_path: str | Path,
    seed: int = 42,
    int_tol: int = 1,
    fp_avg_tol: float = 0.01,
    fp_max_tol: float = 0.05,
    device: str = "local-task",
) -> bool:
    """Compare a single TFLite model against its VMFB counterpart."""
    tflite_path = Path(tflite_path)
    vmfb_path = Path(vmfb_path)

    _logger.info("Validating:")
    _logger.info("  TFLite : %s", tflite_path)
    _logger.info("  VMFB   : %s", vmfb_path)

    input_details, _ = _get_tflite_io_details(tflite_path)
    inputs = _generate_random_inputs(input_details, seed=seed)

    tflite_outputs = _run_tflite(tflite_path, inputs)
    iree_outputs = _run_iree(vmfb_path, inputs, device=device)

    passed, results = compare_outputs(
        tflite_outputs, iree_outputs,
        int_tol=int_tol, fp_avg_tol=fp_avg_tol, fp_max_tol=fp_max_tol,
    )

    for r in results:
        level = logging.INFO if r["passed"] else logging.ERROR
        _logger.log(level, "  output[%d]: %s", r["index"], r["details"])

    return passed


# ---------------------------------------------------------------------------
# Validate all components in an output directory
# ---------------------------------------------------------------------------

def validate_all(
    output_dir: str | Path,
    seed: int = 42,
    int_tol: int = 1,
    fp_avg_tol: float = 0.01,
    fp_max_tol: float = 0.05,
    device: str = "local-task",
) -> bool:
    """
    Auto-discover TFLite/VMFB pairs under *output_dir* and validate each.

    Expected layout::

        output_dir/
          all_fc/
            all_fc/all_fc_int8.tflite
            all_fc_int8.vmfb           (or nested under all_fc/)
    """
    output_dir = Path(output_dir)
    all_passed = True
    found_any = False

    for comp_name in MODEL_COMPONENTS:
        comp_dir = output_dir / comp_name
        if not comp_dir.exists():
            continue

        tflites = sorted(comp_dir.rglob("*_int8.tflite"))
        vmfbs = sorted(comp_dir.rglob("*.vmfb"))

        if not tflites:
            tflites = sorted(comp_dir.rglob("*.tflite"))
        if not tflites or not vmfbs:
            _logger.warning("Skipping %s: missing tflite=%d vmfb=%d",
                            comp_name, len(tflites), len(vmfbs))
            continue

        tflite_path = tflites[0]
        vmfb_path = vmfbs[0]
        found_any = True

        _logger.info("=" * 60)
        _logger.info("Component: %s", comp_name)
        _logger.info("=" * 60)

        passed = validate_pair(
            tflite_path, vmfb_path,
            seed=seed, int_tol=int_tol,
            fp_avg_tol=fp_avg_tol, fp_max_tol=fp_max_tol,
            device=device,
        )
        if not passed:
            all_passed = False

    if not found_any:
        _logger.error("No TFLite/VMFB pairs found under %s", output_dir)
        return False

    return all_passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate Customer B models: compare TFLite vs IREE runtime output"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--output-dir",
        type=str,
        metavar="DIR",
        help="Auto-discover and validate all components under this directory",
    )
    group.add_argument(
        "--tflite",
        type=str,
        metavar="FILE",
        help="Path to a specific TFLite model (use with --vmfb)",
    )
    parser.add_argument(
        "--vmfb",
        type=str,
        metavar="FILE",
        help="Path to the corresponding VMFB (required when using --tflite)",
    )
    parser.add_argument(
        "--int-tol",
        type=int,
        default=1,
        help="Max absolute diff tolerance for integer outputs (default: %(default)d)",
    )
    parser.add_argument(
        "--fp-avg-tol",
        type=float,
        default=0.01,
        help="Average relative diff tolerance for fp outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--fp-max-tol",
        type=float,
        default=0.05,
        help="Max relative diff tolerance for fp outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for input generation (default: %(default)d)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="local-task",
        help="IREE device to use (default: %(default)s)",
    )
    parser.add_argument(
        "--logging",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: %(default)s)",
    )
    args = parser.parse_args()
    configure_logging(args.logging)

    if args.tflite:
        if not args.vmfb:
            parser.error("--vmfb is required when using --tflite")
        passed = validate_pair(
            args.tflite, args.vmfb,
            seed=args.seed, int_tol=args.int_tol,
            fp_avg_tol=args.fp_avg_tol, fp_max_tol=args.fp_max_tol,
            device=args.device,
        )
    else:
        passed = validate_all(
            args.output_dir,
            seed=args.seed, int_tol=args.int_tol,
            fp_avg_tol=args.fp_avg_tol, fp_max_tol=args.fp_max_tol,
            device=args.device,
        )

    if passed:
        _logger.info("VALIDATION PASSED")
    else:
        _logger.error("VALIDATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
