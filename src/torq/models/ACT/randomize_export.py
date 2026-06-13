#!/usr/bin/env python3
"""Load the ACT policy, RANDOMIZE every weight, export to fp32 ONNX.

Random weights make the fidelity comparison meaningful: the real checkpoint is
input-insensitive/degenerate, so a torq-vs-ORT match on it could be trivial.
Each parameter is replaced with a normal sample matched to that tensor's original
mean/std (keeps activations in a sane range, model stays input-sensitive).

Usage: python randomize_export.py PRETRAINED_DIR -o model_original_randomized.onnx [--seed 0]
"""
import argparse, torch
from lerobot.policies.act.modeling_act import ACTPolicy


class LeRobotFlatWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.model = policy.model if hasattr(policy, "model") else policy

    def forward(self, image_side, state_input):
        batch = {
            "observation.images": [image_side],
            "observation.state": state_input,
            "observation.environment_state": state_input,
            "action": torch.zeros(state_input.shape[0], 100, state_input.shape[-1]),
            "action_is_pad": torch.zeros(state_input.shape[0], 100, dtype=torch.bool),
        }
        out = self.model(batch)
        return out[0] if isinstance(out, tuple) else out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pretrained_dir")
    ap.add_argument("-o", "--out", default="model_original_randomized.onnx")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply each tensor's std by this. <1 keeps the residual stream inside "
                         "bf16's range (full-std random weights are OOD and overflow the bf16 pipeline).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    policy = ACTPolicy.from_pretrained(args.pretrained_dir).float().eval()

    with torch.no_grad():
        for name, p in policy.named_parameters():
            if not p.is_floating_point():
                continue
            std = (float(p.data.std()) if p.numel() > 1 else 0.0) * args.scale
            mean = float(p.data.mean())
            p.data = torch.randn_like(p) * (std if std > 1e-6 else 0.02 * args.scale) + mean
    print(f"randomized all float parameters (per-tensor mean/std preserved, scale={args.scale})")

    wrapped = LeRobotFlatWrapper(policy).eval()
    img = torch.randn(1, 3, 480, 640, dtype=torch.float32)
    st = torch.randn(1, 6, dtype=torch.float32)
    with torch.no_grad():
        a = wrapped(img, st)
    assert a.shape[1] == 100 and torch.isfinite(a).all(), f"bad output {a.shape}, finite={torch.isfinite(a).all()}"

    torch.onnx.export(
        wrapped, (img, st), args.out,
        input_names=["image_side", "state"], output_names=["action"],
        opset_version=16, do_constant_folding=True,
    )
    print(f"wrote {args.out}  (action {tuple(a.shape)}, range [{a.min():.3f},{a.max():.3f}])")


if __name__ == "__main__":
    main()
