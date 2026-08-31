# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import json
import logging
import os
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnxruntime.transformers.optimizer import optimize_model

from ..utils.compile import export_torq
from ..utils.onnx import (
    get_model_opset,
    get_model_ops_count,
    print_onnx_model_inputs_outputs_info,
    check_dynamic_shapes,
)
from ..tools.convert_dtype.onnx import (
    convert_model
)
from ..tools.cleanup.onnx import cleanup_onnx_model
from ..graph_edit.harness import EditSpec, GraphEditHarness

__all__ = [
    "FP_EXPORT_DTYPE_MAPPING",
    "ORTOptimizerConfig",
    "OnnxModelExporterBase",
]

FP_EXPORT_DTYPE_MAPPING: Final[dict] = {
    "float": onnx.TensorProto.FLOAT,
    "fp32" : onnx.TensorProto.FLOAT,
    "fp16" : onnx.TensorProto.FLOAT16,
    "bf16" : onnx.TensorProto.BFLOAT16
}


@dataclass
class ORTOptimizerConfig:
    num_heads: int
    hidden_size: int
    model_type: str = "bert"
    verbose: bool = False
    extra_args: dict[str, Any] = field(default_factory=dict)


class OnnxModelExporterBase(ABC):

    def __init__(
        self,
        model_dtype: str,
        static_models: bool,
        config: Mapping[str, Any],
        models_dir: str | os.PathLike,
        *,
        show_model_info: bool = False,
        convert_dtypes: bool = False,
        opt_configs: Mapping[str, ORTOptimizerConfig] | None = None,
        skip_export: list[str] | None = None,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)

        self._harness: GraphEditHarness | None = None
        self._model_dtype = model_dtype
        self._config = config
        self._models_dir = Path(models_dir)
        self._static_models = static_models
        self._show_model_info = show_model_info
        self._convert_dtypes = convert_dtypes
        self._opt_configs = opt_configs
        try:
            self._onnx_export_dtype = FP_EXPORT_DTYPE_MAPPING[self._model_dtype]
        except KeyError:
            raise ValueError(f"Invalid model dtype '{self._model_dtype}', must be one of {list(FP_EXPORT_DTYPE_MAPPING)}")
        self._skip_export = set(skip_export or [])
        # Directory setup (which downloads or optimum-exports the source model)
        # and loading that source ONNX are deferred to _prepare(), so read-only
        # commands like --view-graph-edits can build an exporter without hitting
        # the network or reading a large model.
        self._onnx_dir: Path | None = None
        self._export_dir: Path | None = None
        self._convert_dir: Path | None = None
        self._torq_dir: Path | None = None
        self._export_paths: dict[str, Path] = {}
        self._components: dict[str, onnx.ModelProto] = {}
        self._prepared = False

    def _prepare(self) -> None:
        """Resolve the export directories and load the source ONNX.

        Idempotent, and called by every entry point that needs either. Kept out
        of ``__init__`` because it is the expensive, network-touching part of
        setting up an exporter.
        """
        if self._prepared:
            return
        self._onnx_dir, self._export_dir, self._convert_dir, self._torq_dir = self._setup_dirs()
        self._components = self._load_onnx()
        self._prepared = True

    @property
    def export_dir(self) -> Path:
        self._prepare()
        return self._export_dir

    def set_graph_edit_harness(self, harness: GraphEditHarness | None) -> None:
        """Attach a parsed graph-edit harness used by ``make_static`` / view."""
        self._harness = harness

    def graph_edit_blocks(self) -> dict[str, list[EditSpec]]:
        """Return the exporter's default graph-edit blocks (name -> ordered specs).

        Each block corresponds to a contiguous run of graph edits applied to a
        component/graph.  Override in subclasses that use the declarative
        harness; used both by ``make_static`` (as the default specs fed to
        ``OnnxGraphEditor.apply_specs``) and by ``--view-graph-edits``.
        """
        return {}

    def describe_graph_edits(self) -> dict[str, list[EditSpec]]:
        """Return the finalized (harness-merged) edit blocks for display."""
        harness = self._harness or GraphEditHarness()
        return {
            name: harness.finalize(specs)
            for name, specs in self.graph_edit_blocks().items()
        }

    @abstractmethod
    def _setup_dirs(self) -> list[Path]: ...

    @abstractmethod
    def _load_onnx(self) -> dict[str, onnx.ModelProto]: ...

    def check_model(self, model: onnx.ModelProto, skip_data_prop: bool = False) -> onnx.ModelProto:
        if model.ir_version > 10:
            self._logger.warning(
                "Warning: Model IR version is > 10 (%d), which might be unsupported by onnxruntime",
                model.ir_version
            )
        model = onnx.shape_inference.infer_shapes(
            model, check_type=True, strict_mode=True, data_prop=not skip_data_prop
        )
        onnx.checker.check_model(model, full_check=True)
        return model

    def optimize_model(self, model_path: str | os.PathLike, opt_config: ORTOptimizerConfig):
        optimized = optimize_model(
            str(model_path),
            model_type=opt_config.model_type,
            num_heads=opt_config.num_heads,
            hidden_size=opt_config.hidden_size,
            only_onnxruntime=True,
            verbose=opt_config.verbose,
            **opt_config.extra_args
        )
        optimized.save_model_to_file(str(model_path))
        optimized_model = onnx.load(model_path)
        optimized_model = onnx.shape_inference.infer_shapes(
            optimized_model, check_type=True, strict_mode=True, data_prop=False
        )
        onnx.save(optimized_model, model_path)

    @abstractmethod
    def make_static(self): ...

    @abstractmethod
    def apply_post_static_patches(self, model_path: str | os.PathLike, component: str): ...

    @abstractmethod
    def validate_onnx(self, n_iters: int = 5): ...

    def _allows_dynamic_shapes(self, component: str) -> bool:
        """Whether ``component`` is *intentionally* exported with dynamic shapes.

        A static export normally hard-fails on any leftover dynamic dim, but a
        multi-component model can legitimately ship a dynamic component that is
        never chip-compiled (e.g. an ORT/host-side subgraph). Override to exempt
        those from the static-shape verification.
        """
        return False

    def _export_path_for_component(self, component: str) -> Path:
        return self._export_dir / f"{component}.onnx"

    @staticmethod
    def sanitize_onnx_names(model: onnx.ModelProto) -> onnx.ModelProto:
        """Replace characters illegal in MLIR identifiers in all tensor names."""
        _illegal = re.compile(r'[^a-zA-Z0-9_./]+')

        def _clean(name: str) -> str:
            return _illegal.sub('_', name).strip('_')

        rename_map: dict[str, str] = {}
        for init in model.graph.initializer:
            clean = _clean(init.name)
            if clean != init.name:
                rename_map[init.name] = clean
        for node in model.graph.node:
            for out in node.output:
                clean = _clean(out)
                if clean != out:
                    rename_map[out] = clean
        if not rename_map:
            return model
        for init in model.graph.initializer:
            if init.name in rename_map:
                init.name = rename_map[init.name]
        for node in model.graph.node:
            for i, inp in enumerate(node.input):
                if inp in rename_map:
                    node.input[i] = rename_map[inp]
            for i, out in enumerate(node.output):
                if out in rename_map:
                    node.output[i] = rename_map[out]
        for io in list(model.graph.input) + list(model.graph.output):
            if io.name in rename_map:
                io.name = rename_map[io.name]
        for vi in model.graph.value_info:
            if vi.name in rename_map:
                vi.name = rename_map[vi.name]
        return model

    def export_onnx(self, validate: bool = True, cleanup: bool = True):
        self._prepare()
        # Reset the export dir here rather than in __init__ so that constructing
        # an exporter (e.g. to render --view-graph-edits) never destroys a
        # previous export's artifacts. Mirrors convert_models/export_torq.
        if self._export_dir.exists():
            shutil.rmtree(self._export_dir, ignore_errors=True)
        self._export_dir.mkdir(parents=True, exist_ok=True)

        if self._static_models:
            self.make_static()

        for comp, model in self._components.items():
            if comp in self._skip_export:
                self._logger.info("Skipping export of component %s", comp)
                continue
            self._export_paths[comp] = self._export_path_for_component(comp)
            model = self.sanitize_onnx_names(model)
            if cleanup:
                # Idempotent and cheap (~1 s no-op re-run on a 250-node
                # graph), so safe to run even on already-clean exports. The
                # cleanup is an optimization: a failure must not break an
                # export that works without it.
                self._logger.info("(%s) Cleaning up exporter artifacts...", comp)
                try:
                    model = cleanup_onnx_model(model)
                except Exception:
                    self._logger.warning(
                        "(%s) ONNX cleanup failed; continuing with the "
                        "uncleaned model", comp, exc_info=True,
                    )
            self._logger.info("(%s) Checking model...", comp)
            model = self.check_model(model)
            onnx.save(model, self._export_paths[comp])
            self._logger.info("(%s) Optimizing model...", comp)
            if (opt_config := self._opt_configs.get(comp)):
                self.optimize_model(self._export_paths[comp], opt_config)
            if self._static_models:
                self._logger.info("(%s) Applying post-static conversion patches...", comp)
                self.apply_post_static_patches(self._export_paths[comp], comp)
            self.check_model(onnx.load(self._export_paths[comp]))
            if self._static_models and self._allows_dynamic_shapes(comp):
                self._logger.info(
                    "(%s) Skipping static-shape verification (component is "
                    "exported dynamic by design)", comp
                )
            elif self._static_models:
                self._logger.info("(%s) Verifying static shapes...", comp)
                dynamic_shapes = check_dynamic_shapes(onnx.load(self._export_paths[comp]))
                if dynamic_shapes:
                    raise ValueError(
                        f"Model '{comp}' still has dynamic shapes: {json.dumps(dynamic_shapes)}"
                    )
            if self._show_model_info:
                print(f"\n\nInfo for model '{self._export_paths[comp]}':")
                print_onnx_model_inputs_outputs_info(self._export_paths[comp])
                print(f"\nModel ops summary:")
                print(
                    json.dumps(
                        get_model_ops_count(onnx.load(self._export_paths[comp])), indent=4
                    ),
                    end="\n\n",
                )
            self._logger.info("(%s) Saved model to '%s'", comp, str(self._export_paths[comp]))

        if validate:
            if self._skip_export:
                self._logger.warning("Skipping validation as components %s have not been exported", str(self._skip_export))
            else:
                self.validate_onnx()

    def convert_models(
        self,
        convert_dir: str | os.PathLike | None = None,
        preserve_io: bool = False,
        skip: list[str] | None = None,
        external_data: list[tuple[str | os.PathLike, np.dtype]] = None
    ):
        if not self._convert_dtypes:
            self._logger.warning("Skipping conversion as convert_dtypes==False")
            return
        self._prepare()
        self._convert_dir = Path(convert_dir or self._convert_dir)
        skip = skip or []
        external_data = external_data or []
        if self._convert_dir.exists():
            shutil.rmtree(self._convert_dir, ignore_errors=True)
        self._convert_dir.mkdir(parents=True, exist_ok=True)
        for comp, model_path in self._export_paths.items():
            if comp in skip:
                shutil.copy2(model_path, self._convert_dir)
                continue
            self._logger.info("(ONNX-convert) Converting model '%s' to dtype bf16...", str(model_path))
            converted_model_path = self._convert_dir / model_path.name
            convert_model(model_path, converted_model_path, "bf16", convert_io=not preserve_io)
            self._logger.info("(ONNX-convert) Successfully converted model to dtype bf16 @ '%s'", str(converted_model_path))
            self._logger.info("(ONNX-convert) Converting model '%s' to dtype int32...", str(model_path))
            convert_model(converted_model_path, converted_model_path, "int32", convert_io=not preserve_io)
            self._logger.info("(ONNX-convert) Successfully converted model to dtype int32 @ '%s'", str(converted_model_path))
            self._export_paths[comp] = converted_model_path
            self._logger.debug("(ONNX-convert) Update %s model path to '%s'", comp, str(converted_model_path))
        for (data_orig_npy, target_dtype) in external_data:
            data_orig_npy = Path(data_orig_npy)
            if not data_orig_npy.exists() or data_orig_npy.suffix != ".npy":
                self._logger.warning("(ONNX-convert) Skipping dtype conversion of invalid external data file '%s'", str(data_orig_npy))
                continue
            data_orig: np.ndarray = np.load(data_orig_npy)
            data_converted = data_orig.astype(target_dtype)
            data_converted_npy = converted_model_path.parent / data_orig_npy.name
            np.save(data_converted_npy, data_converted)
            self._logger.debug("(ONNX-convert) Saved converted external data to '%s'", str(data_converted_npy))

    def export_torq(
        self,
        torq_export_dir: str | os.PathLike | None = None,
        torq_compile_args: list[str] | None = None,
        use_binary: bool = False,
        skip: list[str] | None = None,
        local_compile: bool = False,
        compiler_path: str | Path | None = None,
    ):
        self._prepare()
        self._torq_dir = Path(torq_export_dir or self._torq_dir)
        skip = skip or []
        if self._torq_dir.exists():
            shutil.rmtree(self._torq_dir, ignore_errors=True)
        self._torq_dir.mkdir(parents=True, exist_ok=True)
        for comp, onnx_path in self._export_paths.items():
            if comp in skip:
                self._logger.info("(Torq-export) Skipping %s", comp)
                continue
            self._logger.info("(Torq-export) Exporting %s model @ '%s' to Torq...", comp, str(onnx_path))
            model = onnx.load(onnx_path)
            graph = gs.import_onnx(model)
            graph.name = "main"
            graph.cleanup(
                remove_unused_graph_inputs=True, remove_unused_node_outputs=True
            ).toposort()
            model = gs.export_onnx(graph)
            self.check_model(model)
            onnx.save(model, onnx_path)
            export_torq(
                onnx_path,
                self._torq_dir,
                opset=get_model_opset(model),
                compiler_args=torq_compile_args,
                use_binary=use_binary,
                local_compile=local_compile,
                compiler_path=compiler_path,
            )
            self._logger.info("(Torq-export) Successfully exported '%s/%s.vmfb'", str(self._torq_dir), onnx_path.stem)
