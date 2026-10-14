import argparse
import os

parser = argparse.ArgumentParser(description="out/in-feature INT4 GPTQ quantization")
parser.add_argument("--grouping", choices=["out", "in"], default="in",
                    help="Quantization grouping axis. 'in' (default) = in_features "
                         "(group_size 32); 'out' = out_features (block_structure '32x1').")
parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                    help="Compute device. 'auto' (default) picks cuda if present, else cpu. "
                         "'cpu' is the CUDA-free path (slower). Apple Silicon / MPS is not "
                         "supported by the GPTQ backend, so a Mac runs on cpu.")
parser.add_argument("--gpu", default="0",
                    help="CUDA device index (sets CUDA_VISIBLE_DEVICES) when the device is cuda.")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from compressed_tensors.offload import dispatch_model


def resolve_device(choice):
    """Resolve the requested device for GPTQ. 'auto' picks cuda if present, else cpu.

    Apple Silicon (MPS) is intentionally not offered: the compressed_tensors offload
    backend has no MPS implementation, so GPTQ (calibration and save) cannot run there.
    On a Mac, use cpu.
    """
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but no CUDA GPU is available.")
    return choice


device = resolve_device(args.device)
# bf16 is only reliable on CUDA here; mps/cpu use fp32 (Gemma overflows in fp16).
dtype = torch.bfloat16 if device == "cuda" else torch.float32
print(f"[device] using {device} (dtype={dtype})")

if device != "cuda":
    # Hide MPS/accelerator from the whole quant + save stack. The compressed_tensors
    # offload backend has no MPS implementation, and otherwise both calibration
    # (activation offload cache, CUDA-only pin_memory) and save_pretrained
    # (from_accelerate re-dispatch) place/offload tensors on mps and crash. Report
    # "no accelerator" AND override current_accelerator (which reads a C-level value
    # that is_available patches do not reach) so everything stays on plain CPU.
    torch.accelerator.is_available = lambda *a, **k: False
    torch.accelerator.current_accelerator = lambda *a, **k: torch.device("cpu")
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False


model_id = "google/gemma-3-270m-it"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id)

input_embeddings = model.get_input_embeddings()
output_embeddings = model.get_output_embeddings()
print(
    "input/output embedding tied before quant:",
    input_embeddings.weight.data_ptr() == output_embeddings.weight.data_ptr(),
)



NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

ds = load_dataset(
    DATASET_ID,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]"
).shuffle(seed=5436)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


ds = ds.map(preprocess)

print("===== calibration text example =====")
print(ds[0]["text"][:1000])
print("====================================")


def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# drop short or abnormal samples
ds = ds.filter(lambda x: len(x["input_ids"]) >= 32)
# ds = ds.filter(lambda x: len(x["input_ids"]) >= 256)

print("num samples:", len(ds))
print("first sample length:", len(ds[0]["input_ids"]))
print("first input_ids:", ds[0]["input_ids"][:20])

if args.grouping == "in":
    # in_feature axis, group size 32 (idiomatic group quant) — scale [N, K//32]
    weights_cfg = {
        "num_bits": 4,
        "type": "int",
        "symmetric": False,
        "strategy": "group",
        "group_size": 32,
    }
else:
    # out_feature axis, group size 32 — based on W[n:n+32, k], scale [N//32, K]
    weights_cfg = {
        "num_bits": 4,
        "type": "int",
        "symmetric": False,
        "strategy": "block",
        "block_structure": "32x1",
    }

recipe = GPTQModifier(
    ignore=[],
    config_groups={
        "group_0": {
            "targets": ["Linear"],
            "weights": weights_cfg,
            }
    },
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=min(NUM_CALIBRATION_SAMPLES, len(ds)),
)

# dispatch only after GPTQ finishes (multi-GPU offload helper; CUDA-only)
if device == "cuda":
    dispatch_model(model)

input_embeddings = model.get_input_embeddings()
output_embeddings = model.get_output_embeddings()
print("input embedding module after quant:", type(input_embeddings))
print("output/lm_head module after quant:", type(output_embeddings))
if hasattr(input_embeddings, "weight") and hasattr(output_embeddings, "weight"):
    print(
        "input/output embedding tied after quant:",
        input_embeddings.weight.data_ptr() == output_embeddings.weight.data_ptr(),
    )

## output


messages = [
    {"role": "user", "content": "Introduce yourself in one short sentence."}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

sample = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=False,
).to(device)

output = model.generate(
    **sample,
    max_new_tokens=80,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(output[0], skip_special_tokens=False))

messages = [
    {"role": "user", "content": "What causes rainbows?"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

sample = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=False,
).to(device)

output = model.generate(
    **sample,
    max_new_tokens=80,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(output[0], skip_special_tokens=False))


messages = [
    {"role": "user", "content": "Who wrote Romeo and Juliet?"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

sample = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=False,
).to(device)

output = model.generate(
    **sample,
    max_new_tokens=80,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(output[0], skip_special_tokens=False))


messages = [
    {"role": "user", "content": "12345+54321="}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

sample = tokenizer(
    prompt,
    return_tensors="pt",
    add_special_tokens=False,
).to(device)

output = model.generate(
    **sample,
    max_new_tokens=80,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(output[0], skip_special_tokens=False))

# reflect the calibration sample count in the name (GPTQ has large run-to-run
# variance, so run several times and pick the best — see select_best_gptq.py)
tag = "infeat" if args.grouping == "in" else "outfeat"
SAVE_DIR = (
    model_id.rstrip("/").split("/")[-1]
    + f"-W4A16-G32-{tag}-gptq-{NUM_CALIBRATION_SAMPLES}"
)
if device != "cuda":
    # save_compressed re-dispatches via from_accelerate, which infers the offload
    # device from each module's params; make sure they are all on cpu (oneshot may
    # have placed some on the accelerator) so compressed_tensors uses the CPU cache.
    model.to("cpu")
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
