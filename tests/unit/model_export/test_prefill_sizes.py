# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import sys

import pytest


@pytest.mark.parametrize("header", [
    "TileAndFuse (torq-tile-and-fuse)",
    "TileAndFuse: torq-tile-and-fuse{slice-count=2}",
])
def test_probe_reads_pass_dump_headers(header):
    from torq.model_export import prefill_sizes as probe

    stderr = f"// -----// IR Dump After {header} //----- //\n" + '''
func.func @main() {
  %w = arith.constant dense<1.0> : tensor<8x16xbf16>
  return
}
'''
    assert probe._parse_tile_and_fuse(stderr, 4, 8, 16) == probe.ProbeResult(4, None, False, False)


def test_probe_sweeps_narrow_nonmonotone_plateaus(monkeypatch, tmp_path):
    from torq.model_export import prefill_sizes as probe

    monkeypatch.setattr(probe, "extract_weight_shapes", lambda _: [(8, 16)])
    calls = []
    monkeypatch.setattr(probe, "run_probe", lambda *args: "")

    def result(_, m, k, n):
        calls.append(m)
        return probe.ProbeResult(m, 4 if m == 7 else 8, False, False)

    monkeypatch.setattr(probe, "_parse_tile_and_fuse", result)
    kwargs = dict(cap=12, tool=sys.executable)
    report = probe.recommend(tmp_path / "decoder.mlir", "SL2610", [], tmp_path, **kwargs)
    assert sorted(calls) == [1, *range(4, 13)]
    assert [(p["b_start"], p["b_end"]) for p in report.shapes[0].plateaus] == [(4, 6), (7, 7), (8, 12)]
    assert report.chosen == [4, 7, 12]
    calls.clear()
    cached = probe.recommend(tmp_path / "decoder.mlir", "SL2610", [], tmp_path, **kwargs)
    assert cached.chosen == report.chosen
    assert calls == []


def test_probe_drops_restream_and_no_fit(monkeypatch, tmp_path):
    from torq.model_export import prefill_sizes as probe

    monkeypatch.setattr(probe, "extract_weight_shapes", lambda _: [(8, 16)])
    monkeypatch.setattr(probe, "run_probe", lambda *args: "")
    monkeypatch.setattr(probe, "_parse_tile_and_fuse", lambda _, m, k, n: probe.ProbeResult(
        m, None if m == 8 else 8 if m < 6 else 4, m == 6, m == 6))
    report = probe.recommend(tmp_path / "decoder.mlir", "SL2610", [], tmp_path, cap=8, tool=sys.executable)
    assert report.chosen == [4, 5, 7]
    dropped = {c["b"]: c["reason"] for c in report.candidates if not c["kept"]}
    assert "re-streamed" in dropped[6] and "no tile fit" in dropped[8] and len(dropped) == 2
    with pytest.raises(probe.ProbeError, match="max batch size"):
        probe.recommend(tmp_path / "decoder.mlir", "SL2610", [], tmp_path, cap=3, tool=sys.executable)


def test_recommendation_uses_exporter_option():
    from torq.model_export.prefill_sizes import ProbeReport

    assert ProbeReport([4, 7, 12], [], 12, 16).sizes_flag() == "--native-batch-prefill 4,7,12"
