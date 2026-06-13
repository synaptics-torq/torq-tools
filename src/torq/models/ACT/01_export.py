#!/usr/bin/env python3
"""Stage 1 — export the LeRobot ACT policy (.safetensors) to a single fp32 ONNX.

Adapted from work-dev/lerobot/convert_model.py. The ACT policy's select_action()
contains an eval()/control flow that torch.export / Dynamo can't trace, so we wrap
the underlying model and call .forward() directly with a fixed batch dict.

Inputs : pretrained_model/ dir containing model.safetensors (+ config.json)
Output : lerobot_model.onnx  (fp32, opset 16)
         image_side[1,3,480,640] f32, state[1,6] f32  ->  action[1,100,6] f32

Usage:  python 01_export.py /path/to/pretrained_model -o lerobot_model.onnx
"""
import argparse, torch
from lerobot.policies.act.modeling_act import ACTPolicy


class LeRobotFlatWrapper(torch.nn.Module):
    """Call the ACT model.forward() directly, bypassing select_action()'s eval()."""
    def __init__(self, policy):
        super().__init__()
        self.model = policy.model if hasattr(policy, "model") else policy

    def forward(self, image_side, state_input):
        batch = {
            "observation.images": [image_side],
            "observation.state": state_input,
            "observation.environment_state": state_input,
            # ACT's variational objective needs these even at inference:
            "action": torch.zeros(state_input.shape[0], 100, state_input.shape[-1]),
            "action_is_pad": torch.zeros(state_input.shape[0], 100, dtype=torch.bool),
        }
        out = self.model(batch)
        return out[0] if isinstance(out, tuple) else out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pretrained_dir", help="dir with model.safetensors + config.json")
    ap.add_argument("-o", "--out", default="lerobot_model.onnx")
    args = ap.parse_args()

    policy = ACTPolicy.from_pretrained(args.pretrained_dir).float().eval()
    wrapped = LeRobotFlatWrapper(policy).eval()

    img = torch.randn(1, 3, 480, 640, dtype=torch.float32)
    st = torch.randn(1, 6, dtype=torch.float32)
    with torch.no_grad():
        a = wrapped(img, st)
        assert a.shape[1] == 100, f"expected chunk_size 100, got {a.shape}"

    torch.onnx.export(
        wrapped, (img, st), args.out,
        input_names=["image_side", "state"], output_names=["action"],
        opset_version=16, do_constant_folding=True,
    )
    print(f"wrote {args.out}  (action shape {tuple(a.shape)})")


if __name__ == "__main__":
    main()
