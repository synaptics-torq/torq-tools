#!/usr/bin/env python3
"""Run Google's quantized Gemma 4 E2B (text-only) model via llama.cpp.

Quant variants (text-only GGUF weights, mmproj files excluded), both from Unsloth's
QAT conversion of Gemma 4 E2B:
    q2  unsloth/gemma-4-E2B-it-qat-GGUF  gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf (~1.5GB, Dynamic 2-bit)
    q4  unsloth/gemma-4-E2B-it-qat-GGUF  gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf (~2.8GB, Dynamic 4-bit)

NOTE: Google's own GGUF conversion (google/gemma-4-E2B-it-qat-q4_0-gguf) crashes on load
in every llama.cpp build tested (GGML_ASSERT(id_to_token.size() == token_to_id.size())
in vocab loading) — a bug in that specific conversion. Unsloth's re-conversion doesn't
have the issue, so both quants here use their repo instead.

Setup:
    pip install llama-cpp-python huggingface_hub
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

Usage:
    python run_gemma4_e2b.py "What is the capital of France?"
    python run_gemma4_e2b.py --quant q4 "What is the capital of France?"
    python run_gemma4_e2b.py                 # interactive chat
    python run_gemma4_e2b.py --gpu-layers 0   # force CPU-only

    # Load a GGUF already sitting on disk instead of downloading via huggingface_hub
    # (useful on memory-constrained devices where the HF downloader's file
    # reconstruction step can OOM):
    python run_gemma4_e2b.py --model-path ./gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf "..."
"""

import argparse
import sys

from llama_cpp import Llama

QUANTS = {
    "q2": ("unsloth/gemma-4-E2B-it-qat-GGUF", "gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf"),
    "q4": ("unsloth/gemma-4-E2B-it-qat-GGUF", "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"),
}


def load_model(n_ctx: int, n_gpu_layers: int, repo_id: str = None, filename: str = None, model_path: str = None) -> Llama:
    if model_path:
        return Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False,flash_attn=True)
    return Llama.from_pretrained(
        repo_id=repo_id,
        filename=filename,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
        flash_attn=True
    )


def chat(llm: Llama, messages: list[dict], max_tokens: int, temperature: float) -> str:
    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    chunks = []
    for piece in stream:
        delta = piece["choices"][0]["delta"].get("content")
        if delta:
            print(delta, end="", flush=True)
            chunks.append(delta)
    print()
    return "".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt", nargs="*", help="Prompt text; omit for interactive chat")
    parser.add_argument("--quant", choices=sorted(QUANTS), default="q2", help="Which quantization to run (ignored if --model-path is set)")
    parser.add_argument("--model-path", help="Path to a local .gguf file, skips the huggingface_hub downloader entirely")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--gpu-layers", type=int, default=-1, help="-1 offloads all layers to GPU if available")
    args = parser.parse_args()

    if args.model_path:
        print(f"Loading local file {args.model_path}...", file=sys.stderr)
        llm = load_model(n_ctx=args.n_ctx, n_gpu_layers=args.gpu_layers, model_path=args.model_path)
    else:
        repo_id, filename = QUANTS[args.quant]
        print(f"Loading {filename} from {repo_id} (downloads on first run)...", file=sys.stderr)
        llm = load_model(n_ctx=args.n_ctx, n_gpu_layers=args.gpu_layers, repo_id=repo_id, filename=filename)

    if args.prompt:
        prompt = " ".join(args.prompt)
        chat(llm, [{"role": "user", "content": prompt}], args.max_tokens, args.temperature)
        return

    print("Interactive chat with Gemma 4 E2B. Type 'exit' or Ctrl+C to quit.")
    history: list[dict] = []
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        history.append({"role": "user", "content": user_input})
        print("\nGemma: ", end="", flush=True)
        reply = chat(llm, history, args.max_tokens, args.temperature)
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
