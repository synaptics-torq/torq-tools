import onnx
def maybe_load_onnx_model(model):
    if isinstance(model, onnx.ModelProto):
        return model
    return onnx.load(model)