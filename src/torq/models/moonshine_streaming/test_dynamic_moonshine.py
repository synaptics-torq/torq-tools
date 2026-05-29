from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from datasets import load_dataset, Audio
from transformers import MoonshineStreamingForConditionalGeneration, AutoProcessor


# ============================================================
# Config
# ============================================================

HF_REPO = "UsefulSensors/moonshine-streaming-tiny"
ONNX_DIR = Path("/home/yhtet/projects/moonshine-streaming/torq-tools-dev/models/UsefulSensors/moonshine-streaming-tiny/export/onnx/float/dynamic")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32   # match export

PRINT_IO = True


# ============================================================
# Utilities
# ============================================================

def make_session(model_path: Path):
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def np_dtype_from_ort(ort_type: str):
    ort_type = ort_type.lower()
    if "float16" in ort_type:
        return np.float16
    if "float" in ort_type or "double" in ort_type:
        return np.float32
    if "int64" in ort_type:
        return np.int64
    if "int32" in ort_type:
        return np.int32
    if "bool" in ort_type:
        return np.bool_
    return np.float32


def print_session_io(name, sess):
    print(f"\n{name}")
    print("  Inputs:")
    for x in sess.get_inputs():
        print(f"    {x.name:35s} shape={x.shape} type={x.type}")
    print("  Outputs:")
    for x in sess.get_outputs():
        print(f"    {x.name:35s} shape={x.shape} type={x.type}")


def normalize_name(name: str) -> str:
    return name.lower().strip()


def get_eos_set(config):
    eos = config.eos_token_id
    if eos is None:
        return None
    if isinstance(eos, (list, tuple, set)):
        return set(int(x) for x in eos)
    return {int(eos)}


# ============================================================
# Preprocessor
#   export:
#       inputs:  input_values, attention_mask
#       outputs: input_features, padding_mask
# ============================================================

def run_preprocessor(preprocessor_sess, model_inputs):
    feeds = {
        "input_values": to_numpy(model_inputs["input_values"]).astype(np.float32),
        "attention_mask": to_numpy(model_inputs["attention_mask"]).astype(np.int64),
    }
    outputs = preprocessor_sess.run(None, feeds)
    output_names = [o.name for o in preprocessor_sess.get_outputs()]
    output_map = dict(zip(output_names, outputs))

    input_features = output_map["input_features"]
    padding_mask = output_map["padding_mask"]
    return input_features, padding_mask


# ============================================================
# Encoder
#   export:
#       inputs: input_features, attention_mask
#       outputs: last_hidden_state
# ============================================================

def run_encoder(encoder_sess, input_features, padding_mask):
    feeds = {}

    for inp in encoder_sess.get_inputs():
        name = normalize_name(inp.name)
        dtype = np_dtype_from_ort(inp.type)

        if inp.name == "input_features" or "input_features" in name:
            feeds[inp.name] = input_features.astype(dtype)
        elif inp.name == "attention_mask" or "attention_mask" in name:
            feeds[inp.name] = padding_mask.astype(dtype)
        else:
            raise ValueError(f"Unexpected encoder input: {inp.name}")

    outputs = encoder_sess.run(None, feeds)
    return outputs[0]  # last_hidden_state


# ============================================================
# Decoder helpers
#   export:
#       decoder / decoder_with_past first output is last_hidden_state
#       NOT logits
# ============================================================

def load_token_embeddings(onnx_dir: Path):
    path = onnx_dir / "decoder_token_embeddings.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return np.load(path).astype(np.float32)


def hidden_to_logits(last_hidden_state: np.ndarray, token_embeddings: np.ndarray) -> np.ndarray:
    """
    last_hidden_state: [batch, seq, hidden]
    token_embeddings:  [vocab, hidden]
    returns:           [batch, seq, vocab]
    """
    return np.matmul(last_hidden_state.astype(np.float32), token_embeddings.T.astype(np.float32))


def is_past_input_name(name: str):
    n = normalize_name(name)
    return "past" in n or "key_values" in n or "cache" in n


def build_decoder_feeds(
    decoder_sess,
    decoder_input_ids,
    encoder_hidden_states,
    encoder_attention_mask,
    past_key_values=None,
):
    feeds = {}
    past_iter = iter([] if past_key_values is None else past_key_values)

    for inp in decoder_sess.get_inputs():
        name = normalize_name(inp.name)
        dtype = np_dtype_from_ort(inp.type)

        if inp.name == "decoder_input_ids" or "decoder_input_ids" in name:
            feeds[inp.name] = decoder_input_ids.astype(dtype)

        elif inp.name == "encoder_hidden_states" or "encoder_hidden_states" in name:
            feeds[inp.name] = encoder_hidden_states.astype(dtype)

        elif inp.name == "encoder_attention_mask" or "encoder_attention_mask" in name:
            feeds[inp.name] = encoder_attention_mask.astype(dtype)

        elif is_past_input_name(inp.name):
            try:
                pkv = next(past_iter)
            except StopIteration:
                raise ValueError(f"Missing past tensor for input: {inp.name}")
            feeds[inp.name] = pkv.astype(dtype)

        else:
            raise ValueError(f"Unexpected decoder input: {inp.name}")

    return feeds


def parse_decoder_outputs(ort_outputs, token_embeddings):
    """
    Your export returns:
      [last_hidden_state] + kv outputs

    So convert hidden states -> logits with tied embeddings.
    """
    if len(ort_outputs) == 0:
        raise ValueError("Decoder returned no outputs")

    last_hidden_state = ort_outputs[0]
    logits = hidden_to_logits(last_hidden_state, token_embeddings)
    present = ort_outputs[1:]
    return logits, present


# ============================================================
# ONNX generation
# ============================================================

def onnx_greedy_generate(
    preprocessor_sess,
    encoder_sess,
    decoder_sess,
    decoder_with_past_sess,
    model_inputs,
    config,
    max_length,
    token_embeddings,
):
    # 1) preprocessor
    input_features, padding_mask = run_preprocessor(preprocessor_sess, model_inputs)

    # 2) encoder
    encoder_hidden_states = run_encoder(encoder_sess, input_features, padding_mask)

    # 3) start token
    start_token_id = config.decoder_start_token_id
    if start_token_id is None:
        start_token_id = config.bos_token_id
    if start_token_id is None:
        raise ValueError("No decoder_start_token_id or bos_token_id found")

    eos_set = get_eos_set(config)

    generated = [int(start_token_id)]

    # 4) first step
    decoder_input_ids = np.array([[start_token_id]], dtype=np.int64)
    feeds = build_decoder_feeds(
        decoder_sess=decoder_sess,
        decoder_input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=padding_mask,
        past_key_values=None,
    )

    ort_outputs = decoder_sess.run(None, feeds)
    logits, past_key_values = parse_decoder_outputs(ort_outputs, token_embeddings)

    next_token_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
    generated.append(next_token_id)

    if eos_set is not None and next_token_id in eos_set:
        return np.array([generated], dtype=np.int64)

    # 5) remaining steps
    while len(generated) < max_length:
        decoder_input_ids = np.array([[next_token_id]], dtype=np.int64)

        feeds = build_decoder_feeds(
            decoder_sess=decoder_with_past_sess,
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=padding_mask,
            past_key_values=past_key_values,
        )

        ort_outputs = decoder_with_past_sess.run(None, feeds)
        logits, past_key_values = parse_decoder_outputs(ort_outputs, token_embeddings)

        next_token_id = int(np.argmax(logits[:, -1, :], axis=-1)[0])
        generated.append(next_token_id)

        if eos_set is not None and next_token_id in eos_set:
            break

    return np.array([generated], dtype=np.int64)


# ============================================================
# Optional first-step debug
# ============================================================

def debug_first_step(model, pt_inputs, onnx_hidden, onnx_mask, decoder_sess, token_embeddings):
    start_token_id = model.config.decoder_start_token_id
    if start_token_id is None:
        start_token_id = model.config.bos_token_id

    # PyTorch first-step logits
    with torch.no_grad():
        decoder_input_ids = torch.tensor([[start_token_id]], device=pt_inputs["input_values"].device)
        out = model(
            input_values=pt_inputs["input_values"],
            attention_mask=pt_inputs["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            use_cache=True,
            return_dict=True,
        )
        pt_logits = out.logits[:, -1, :].detach().cpu().numpy()

    # ONNX first-step logits
    feeds = build_decoder_feeds(
        decoder_sess=decoder_sess,
        decoder_input_ids=np.array([[start_token_id]], dtype=np.int64),
        encoder_hidden_states=onnx_hidden,
        encoder_attention_mask=onnx_mask,
        past_key_values=None,
    )
    ort_outputs = decoder_sess.run(None, feeds)
    onnx_logits, _ = parse_decoder_outputs(ort_outputs, token_embeddings)
    onnx_logits = onnx_logits[:, -1, :]

    print("\nFirst-step debug")
    print("  PT   argmax:", int(pt_logits.argmax(-1)[0]))
    print("  ONNX argmax:", int(onnx_logits.argmax(-1)[0]))
    print("  max abs diff:", float(np.max(np.abs(pt_logits - onnx_logits))))
    print("  mean abs diff:", float(np.mean(np.abs(pt_logits - onnx_logits))))


# ============================================================
# Main
# ============================================================

def main():
    # ----- Load baseline model -----
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(HF_REPO).to(device).to(torch_dtype)
    model.eval()

    # For tokenization and other transforms
    processor = AutoProcessor.from_pretrained(HF_REPO)

    # ----- Load dataset -----
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    dataset = dataset.cast_column("audio", Audio(processor.feature_extractor.sampling_rate))
    sample = dataset[0]["audio"]

    # ----- Preprocess -----
    inputs = processor(sample["array"], return_tensors="pt")

    print("Processor keys:", list(inputs.keys()))
    for k, v in inputs.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    pt_inputs = {k: v.to(device) for k, v in inputs.items()}
    cpu_inputs = {k: v.cpu() for k, v in inputs.items()}

    # ----- max_length -----
    token_limit_factor = 6.5 / processor.feature_extractor.sampling_rate
    seq_lens = pt_inputs["attention_mask"].sum(dim=-1)
    max_length = int((seq_lens * token_limit_factor).max().item())
    print(f"\nComputed max_length = {max_length}")

    # ----- PyTorch baseline -----
    with torch.no_grad():
        pt_generated_ids = model.generate(**pt_inputs, max_length=max_length)

    pt_text = processor.decode(pt_generated_ids[0], skip_special_tokens=True)

    # ----- Load ONNX -----
    preprocessor_sess = make_session(ONNX_DIR / "preprocessor.onnx")
    encoder_sess = make_session(ONNX_DIR / "encoder.onnx")
    decoder_sess = make_session(ONNX_DIR / "decoder.onnx")
    decoder_with_past_sess = make_session(ONNX_DIR / "decoder_with_past.onnx")
    token_embeddings = load_token_embeddings(ONNX_DIR)

    if PRINT_IO:
        print_session_io("preprocessor.onnx", preprocessor_sess)
        print_session_io("encoder.onnx", encoder_sess)
        print_session_io("decoder.onnx", decoder_sess)
        print_session_io("decoder_with_past.onnx", decoder_with_past_sess)

    print(f"\nLoaded token embeddings: {token_embeddings.shape}")

    # Optional: sanity debug first decoder step
    onnx_features, onnx_mask = run_preprocessor(preprocessor_sess, cpu_inputs)
    onnx_hidden = run_encoder(encoder_sess, onnx_features, onnx_mask)
    debug_first_step(model, pt_inputs, onnx_hidden, onnx_mask, decoder_sess, token_embeddings)

    # ----- ONNX generation -----
    onnx_generated_ids = onnx_greedy_generate(
        preprocessor_sess=preprocessor_sess,
        encoder_sess=encoder_sess,
        decoder_sess=decoder_sess,
        decoder_with_past_sess=decoder_with_past_sess,
        model_inputs=cpu_inputs,
        config=model.config,
        max_length=max_length,
        token_embeddings=token_embeddings,
    )

    onnx_text = processor.decode(onnx_generated_ids[0], skip_special_tokens=True)

    # ----- Compare -----
    print("\n==============================")
    print("Comparison")
    print("==============================")
    print("PyTorch transcription:", pt_text)
    print("ONNX    transcription:", onnx_text)
    print("Exact text match     :", pt_text == onnx_text)

    pt_ids = pt_generated_ids[0].detach().cpu().tolist()
    onnx_ids = onnx_generated_ids[0].tolist()

    print("\nPyTorch token ids:", pt_ids)
    print("ONNX    token ids:", onnx_ids)
    print("Exact token match:", pt_ids == onnx_ids)

    print("\nPyTorch token count:", len(pt_ids))
    print("ONNX    token count:", len(onnx_ids))
    print("Exact token match:", len(pt_ids) == len(onnx_ids))


if __name__ == "__main__":
    main()
