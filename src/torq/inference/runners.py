# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import os
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter_ns
from typing import Literal

import numpy as np
import onnxruntime as ort
import numpy.typing as npt

try:
    import ai_edge_litert.interpreter as lite_rt
except ImportError:
    import tensorflow.lite as lite_rt
try:
    import iree.runtime as iree_rt
    IREE_RT_AVAILABLE = True
except ModuleNotFoundError:
    IREE_RT_AVAILABLE = False

logger = logging.getLogger(__name__)


class InferenceRunner(ABC):

    def __init__(
        self,
        model_path: str | os.PathLike,
    ):
        self._model_path: Path = Path(model_path)
        self._infer_time_ms: float = 0.0

    @property
    def model_path(self) -> str | os.PathLike:
        return self._model_path

    @property
    def infer_time_ms(self) -> float:
        return self._infer_time_ms

    @abstractmethod
    def _infer(self, inputs: Sequence[npt.NDArray] | Mapping[str, npt.NDArray]) -> Sequence[npt.NDArray]:
        ...

    def infer(self, inputs: Sequence[npt.NDArray] | Mapping[str, npt.NDArray]) -> Sequence[npt.NDArray]:
        st = perf_counter_ns()
        results = self._infer(inputs)
        self._infer_time_ms = (perf_counter_ns() - st) / 1e6
        return results


class ORTInferenceRunner(InferenceRunner):

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        n_threads: int | None = None
    ):
        super().__init__(model_path)

        self._opts = ort.SessionOptions()
        if isinstance(n_threads, int):
            self._opts.intra_op_num_threads = n_threads
            self._opts.inter_op_num_threads = n_threads
        self._sess = ort.InferenceSession(self._model_path, self._opts, providers=['CPUExecutionProvider'])

    def _infer(self, inputs: list[npt.NDArray] | dict[str, npt.NDArray]) -> list[npt.NDArray]:
        return [np.asarray(o) for o in self._sess.run(None, inputs)]


class TFLiteInferenceRunner(InferenceRunner):

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        n_threads: int | None = None,
    ):
        super().__init__(model_path)

        self._interpreter = lite_rt.Interpreter(
            self._model_path,
            num_threads=n_threads
        )
        self._interpreter.allocate_tensors()
        self._output_details = self._interpreter.get_output_details()
        self._n_outputs = len(self._output_details)

    def _infer(self, inputs: list[npt.NDArray] | dict[str, npt.NDArray]) -> list[npt.NDArray]:
        if isinstance(inputs, dict):
            inputs = inputs.values()
        for i, inp in enumerate(inputs):
            self._interpreter.set_tensor(i, inp)
        self._interpreter.invoke()
        return [self._interpreter.get_tensor(self._output_details[i]["index"]) for i in range(self._n_outputs)]


class VMFBInferenceRunner(InferenceRunner):

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        n_threads: int | None = None,
        function: str = "main",
        device_uri: str = "local-task",
        load_method: Literal["preload", "mmap"] = "mmap"
    ):
        if not IREE_RT_AVAILABLE:
            raise RuntimeError(
                "IREE Runtime python API not available in environment"
            )

        super().__init__(model_path)

        if isinstance(n_threads, int):
            iree_rt.flags.parse_flags(
                f"--task_topology_group_count={n_threads}",
                f"--task_topology_max_group_count={n_threads}"
            )

        if load_method == "mmap":
            module = iree_rt.load_vm_flatbuffer_file(self._model_path, driver=device_uri)
            vm_module = module.vm_module
        else:
            instance = iree_rt.VmInstance()
            device = iree_rt.get_device(device_uri, cache=False)
            hal_module = iree_rt.create_hal_module(instance, device)
            with open(self._model_path, "rb") as f:
                fb = f.read()
            vm_module = iree_rt.VmModule.from_flatbuffer(instance, fb, warn_if_copy=False)
            _ctx = iree_rt.SystemContext(vm_modules=[hal_module, vm_module])
            module = _ctx.modules[vm_module.name]

        if function not in vm_module.function_names:
            raise ValueError(f"Function '{function}' not found in '{self._model_path}'")
        self._invoker = module[function]

    def _infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> list[np.ndarray]:
        if isinstance(inputs, dict):
            inputs = list(inputs.values())
        result: iree_rt.DeviceArray | tuple[iree_rt.DeviceArray] = self._invoker(*inputs)
        if isinstance(result, tuple):
            return [r.to_host() for r in result]
        return [result.to_host()]


if __name__ == "__main__":
    pass
