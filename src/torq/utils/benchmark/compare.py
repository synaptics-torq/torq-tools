# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

"""Generate markdown comparison between two benchmark results."""

from __future__ import annotations


def generate_comparison(
    results_a: list[dict],
    results_b: list[dict],
    *,
    name_a: str = "Model A",
    name_b: str = "Model B",
) -> str:
    """Generate a markdown comparison of two benchmark result sets.

    Parameters
    ----------
    results_a, results_b : list[dict]
        Benchmark results from ``run_benchmark()``.
    name_a, name_b : str
        Display names for each model.

    Returns
    -------
    str
        Markdown-formatted comparison report.
    """
    lines = []
    lines.append(f"# Benchmark Comparison: {name_a} vs {name_b}")
    lines.append("")

    # Summary table
    def _avg(results, key):
        return sum(r[key] for r in results) / len(results)

    def _total(results, key):
        return sum(r[key] for r in results)

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | " + name_a + " | " + name_b + " |")
    lines.append("|--------|" + "-" * (len(name_a) + 2) + "|" + "-" * (len(name_b) + 2) + "|")
    lines.append(
        f"| Avg TPS | {_avg(results_a, 'tps'):.1f} tok/s | {_avg(results_b, 'tps'):.1f} tok/s |"
    )
    lines.append(
        f"| Avg TTFT | {_avg(results_a, 'ttft_ms'):.0f} ms | {_avg(results_b, 'ttft_ms'):.0f} ms |"
    )
    lines.append(
        f"| Total Tokens | {_total(results_a, 'tokens')} | {_total(results_b, 'tokens')} |"
    )
    lines.append(
        f"| Questions | {len(results_a)} | {len(results_b)} |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-question answers
    lines.append("## All Answers")
    lines.append("")

    for i, (a, b) in enumerate(zip(results_a, results_b)):
        lines.append(f"### Q{i+1}: {a['question']}")
        lines.append("")
        lines.append(f"**{name_a}:** {a['answer']}")
        lines.append("")
        lines.append(f"**{name_b}:** {b['answer']}")
        lines.append("")
        lines.append(
            f"*{name_a}: {a['tokens']} tok, {a['tps']} tok/s, TTFT {a['ttft_ms']}ms | "
            f"{name_b}: {b['tokens']} tok, {b['tps']} tok/s, TTFT {b['ttft_ms']}ms*"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
