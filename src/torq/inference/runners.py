import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import time_ns
from typing import Any, Literal

import numpy as np
try:
    import iree.runtime as iree_rt
    IREE_RT_AVAILABLE: bool = True
except ImportError:
    IREE_RT_AVAILABLE: bool = False

try:
    import ai_edge_litert.interpreter as lite_rt
    LITE_RT_AVAILABLE: bool = True
except ImportError:
    try:
        import tensorflow.lite as lite_rt
        LITE_RT_AVAILABLE: bool = True
    except ImportError:
        LITE_RT_AVAILABLE: bool = False

try:
    import onnxruntime as ort
    ONNX_RT_AVAILABLE: bool = True
except ImportError:
    ONNX_RT_AVAILABLE: bool = False

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IOTensorInfo:
    name: str
    shape: list[int | str]
    dtype: Any


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
    def _infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> list[np.ndarray]:
        ...

    def infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> list[np.ndarray]:
        st = time_ns()
        results = self._infer(inputs)
        self._infer_time_ms = (time_ns() - st) / 1e6
        return results


class ORTInferenceRunner(InferenceRunner):

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        n_threads: int | None = None
    ):
        if not ONNX_RT_AVAILABLE:
            raise RuntimeError("ONNX runtime not available in environment")

        super().__init__(model_path)

        self._opts = ort.SessionOptions()
        if isinstance(n_threads, int):
            self._opts.intra_op_num_threads = n_threads
            self._opts.inter_op_num_threads = n_threads
        self._sess = ort.InferenceSession(self._model_path, self._opts, providers=['CPUExecutionProvider'])

    def _infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> list[np.ndarray]:
        return [np.asarray(o) for o in self._sess.run(None, inputs)]


class IREEInferenceRunner(InferenceRunner):

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
            raise RuntimeError("IREE runtime not available in environment")

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


class TFLiteInferenceRunner(InferenceRunner):

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        n_threads: int | None = None,
    ):
        if not LITE_RT_AVAILABLE:
            raise RuntimeError("TFLite runtime not available in environment")

        super().__init__(model_path)

        self._interpreter = lite_rt.Interpreter(
            self._model_path,
            num_threads=n_threads
        )
        self._interpreter.allocate_tensors()
        self._output_details = self._interpreter.get_output_details()
        self._n_outputs = len(self._output_details)

    def _infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> list[np.ndarray]:
        if isinstance(inputs, dict):
            inputs = inputs.values()
        for i, inp in enumerate(inputs):
            self._interpreter.set_tensor(i, inp)
        self._interpreter.invoke()
        return [self._interpreter.get_tensor(self._output_details[i]["index"]) for i in range(self._n_outputs)]


if __name__ == "__main__":
    pass
