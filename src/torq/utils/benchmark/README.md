# Benchmarking quantized models

Run quantized models through a standard question set and compare results.

## Run benchmark on board

```bash
python -m torq.utils.benchmark run \
  -m /path/to/model.vmfb --instruct-model -o results_int8.json

python -m torq.utils.benchmark run \
  -m /path/to/model_hybrid.vmfb --instruct-model -o results_hybrid.json
```

Options:
- `-m` — path to model VMFB (or ONNX)
- `--instruct-model` — use Gemma3 instruct chat template
- `--questions-file` — custom JSON list of questions (default: built-in 24 questions)
- `--temperature` — sampling temperature (default: 0.0 = greedy)
- `--runner-path` — path to directory containing `runner.py` (auto-detected if omitted)
- `-j` — number of inference threads

## Compare two benchmark results

```bash
python -m torq.utils.benchmark compare \
  -a results_int8.json -b results_hybrid.json \
  --name-a "Pure Int8" --name-b "Hybrid Int8/Int4" \
  -o comparison.md
```

Generates a markdown report with:
- Summary table (TPS, TTFT, total tokens)
- Side-by-side answers for each question
- Per-question performance metrics
