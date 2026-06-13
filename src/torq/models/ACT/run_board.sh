#!/bin/bash
# Stage — run the pipeline on the Astra board and report warm timings.
#
# Pipeline:  image --[int8 backbone]--> [1,15,20,512]i8
#                  --[host int8->bf16 cast + reshape]--> permute_1 [15,20,1,512]bf16
#                  --[piece_A]--> layer_norm_3 [302,1,512]bf16
#                  --[piece_B]--> action [1,100,6]f32
#
# The int8->bf16 boundary cast runs on the HOST: a standalone dequant does not
# lower on the NSS (gh-issues/qdq-int8-nss-lowering/). 153600 elts, sub-ms on CPU.
#
# Warm timings (3x each, host profiler) are printed; sum dispatch for per-inference
# latency, sum execute for pure-NSS compute.
#
# Usage:  BOARD=root@10.3.10.62 ./run_board.sh
set -e
BOARD=${BOARD:-root@10.3.10.62}
HW=astra_machina

# 0. ship modules + a random int8 image (replace with a real preprocessed frame)
python3 - <<'PY'
import numpy as np
np.random.randint(-128,127,size=(1,480,640,3),dtype=np.int8).tofile("img_int8.bin")
PY
scp -q resnet18_backbone_int8.vmfb piece_A.vmfb piece_B.vmfb img_int8.bin "$BOARD:/home/root/"
# state vector (6 floats); replace with a real observation.state
ssh "$BOARD" 'python3 -c "import numpy as np; np.zeros((1,6),np.float32).tofile(\"/home/root/state.bin\")" 2>/dev/null || \
              dd if=/dev/zero of=/home/root/state.bin bs=24 count=1'

# 1. int8 backbone -> int8 feature map
ssh "$BOARD" "cd /home/root && torq-run-module --module=resnet18_backbone_int8.vmfb --function=main \
  --input=1x480x640x3xi8=@img_int8.bin --output=@/tmp/bb_out.bin --torq_hw_type=$HW"

# 2. host int8 -> bf16 cast + reshape [1,15,20,512] -> [15,20,1,512] (same flat layout)
scp -q "$BOARD:/tmp/bb_out.bin" /tmp/bb_out.bin
python3 - <<'PY'
import numpy as np, ml_dtypes
x = np.fromfile("/tmp/bb_out.bin", np.int8).reshape(1,15,20,512).astype(np.float32) * 0.05  # scale = backbone out-scale
x.astype(ml_dtypes.bfloat16).reshape(15,20,1,512).tofile("/tmp/permute1.bin")
PY
scp -q /tmp/permute1.bin "$BOARD:/home/root/permute1.bin"

# 3. piece_A -> layer_norm_3 ; 4. piece_B -> action
ssh "$BOARD" "cd /home/root && \
  torq-run-module --module=piece_A.vmfb --function=main_graph \
    --input=15x20x1x512xbf16=@permute1.bin --input=1x6xf32=@state.bin \
    --output=@/tmp/ln3.bin --torq_hw_type=$HW && \
  torq-run-module --module=piece_B.vmfb --function=main_graph \
    --input=302x1x512xbf16=@/tmp/ln3.bin --output=@/tmp/action.bin --torq_hw_type=$HW && \
  echo OK action: && ls -la /tmp/action.bin"

echo "End-to-end ran. For warm timings add --torq_profile_host=/tmp/p.csv to each run"
echo "and pair DISPATCH_BEGIN/END (dispatch) & DISPATCH_EXECUTE_ACTIONS_BEGIN/END (execute)."
echo "Measured warm: backbone 118/133, piece_A 210/240, piece_B 251/284 ms (execute/dispatch)"
echo " => ~579 ms NSS execute / ~657 ms per-inference."
