import os
import logging

import numpy as np
import onnxruntime as ort
from torq.runtime import InferenceRunner

try:
    import ai_edge_litert.interpreter as lite_rt
except ImportError:
    import tensorflow.lite as lite_rt

logger = logging.getLogger(__name__)


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

    def _infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> list[np.ndarray]:
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

    def _infer(self, inputs: list[np.ndarray] | dict[str, np.ndarray]) -> list[np.ndarray]:
        if isinstance(inputs, dict):
            inputs = inputs.values()
        for i, inp in enumerate(inputs):
            self._interpreter.set_tensor(i, inp)
        self._interpreter.invoke()
        return [self._interpreter.get_tensor(self._output_details[i]["index"]) for i in range(self._n_outputs)]


if __name__ == "__main__":
    pass
