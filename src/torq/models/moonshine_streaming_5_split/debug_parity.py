import os
import sys
import numpy as np
import torch
import soundfile as sf
import onnxruntime as ort
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.cache_utils import EncoderDecoderCache, DynamicCache
from types import SimpleNamespace

# Add src to python path
sys.path.append("/home/yhtet/projects/moonshine-streaming/core/torq-tools-dev/src")

from torq.models.moonshine_streaming_5_split.export import (
    StatefulPreprocessorWrapper,
    EncoderWrapper,
    AdapterWrapper,
    CrossKVGeneratorWrapper,
    DecoderKVWrapper,
)

def debug_parity():
    model_id = "UsefulSensors/moonshine-streaming-tiny"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    # Load ONNX models
    onnx_dir = "/home/yhtet/projects/moonshine-streaming/core/torq-tools-dev/models/UsefulSensors/moonshine-streaming-tiny/export/onnx/float/dynamic"
    sess_preproc = ort.InferenceSession(os.path.join(onnx_dir, "frontend.onnx"), providers=['CPUExecutionProvider'])
    sess_encoder = ort.InferenceSession(os.path.join(onnx_dir, "encoder.onnx"), providers=['CPUExecutionProvider'])
    sess_adapter = ort.InferenceSession(os.path.join(onnx_dir, "adapter.onnx"), providers=['CPUExecutionProvider'])
    sess_cross_kv = ort.InferenceSession(os.path.join(onnx_dir, "cross_kv.onnx"), providers=['CPUExecutionProvider'])
    sess_decoder = ort.InferenceSession(os.path.join(onnx_dir, "decoder_kv.onnx"), providers=['CPUExecutionProvider'])

    # 1. Load and prepare audio
    wav_path = "/home/yhtet/projects/moonshine-streaming/core/torq-tools-dev/src/torq/models/moonshine_streaming/OSR_us_000_0010_8k.wav"
    data, sr = sf.read(wav_path)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        data = resample_poly(data, up=16000, down=sr).astype(np.float32)
    
    # Run processor
    input_values = processor(data, sampling_rate=16000, return_tensors="pt").input_values

    # 2. Run Eager PyTorch Preprocessor
    with torch.no_grad():
        features_eager, _ = model.model.encoder.embedder(input_values)
    print(f"Eager Preprocessor Features shape: {features_eager.shape}")

    # 3. Run Stateful Preprocessor step-by-step
    stateful_preproc = StatefulPreprocessorWrapper(model).eval()
    
    # Initialize states (PyTorch)
    sample_buffer = torch.zeros(1, 79, dtype=torch.float32)
    sample_len = torch.zeros(1, dtype=torch.int64)
    conv1_buffer = torch.zeros(1, 320, 4, dtype=torch.float32)
    conv2_buffer = torch.zeros(1, 640, 4, dtype=torch.float32)
    frame_count = torch.zeros(1, dtype=torch.int64)

    # Initialize states (ONNX)
    onnx_sample_buffer = np.zeros((1, 79), dtype=np.float32)
    onnx_sample_len = np.zeros(1, dtype=np.int64)
    onnx_conv1_buffer = np.zeros((1, 320, 4), dtype=np.float32)
    onnx_conv2_buffer = np.zeros((1, 640, 4), dtype=np.float32)
    onnx_frame_count = np.zeros(1, dtype=np.int64)
    
    chunk_len = 640
    accum_features_pt = []
    accum_features_onnx = []
    audio_len = input_values.shape[-1]
    
    with torch.no_grad():
        for offset in range(0, audio_len, chunk_len):
            chunk = input_values[:, offset:offset+chunk_len]
            if chunk.shape[-1] < chunk_len:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_len - chunk.shape[-1]))
            
            # PyTorch
            res_pt = stateful_preproc(
                chunk, sample_buffer, sample_len, conv1_buffer, conv2_buffer, frame_count
            )
            features_chunk_pt, sample_buffer, sample_len, conv1_buffer, conv2_buffer, frame_count = res_pt
            accum_features_pt.append(features_chunk_pt)

            # ONNX
            res_onnx = sess_preproc.run(None, {
                "audio_chunk": chunk.numpy(),
                "sample_buffer": onnx_sample_buffer,
                "sample_len": onnx_sample_len,
                "conv1_buffer": onnx_conv1_buffer,
                "conv2_buffer": onnx_conv2_buffer,
                "frame_count": onnx_frame_count,
            })
            features_chunk_onnx, onnx_sample_buffer, onnx_sample_len, onnx_conv1_buffer, onnx_conv2_buffer, onnx_frame_count = res_onnx
            accum_features_onnx.append(features_chunk_onnx)

    features_stateful_pt = torch.cat(accum_features_pt, dim=1)
    features_stateful_onnx = np.concatenate(accum_features_onnx, axis=1)
    print(f"Stateful PT Features shape: {features_stateful_pt.shape}")
    print(f"Stateful ONNX Features shape: {features_stateful_onnx.shape}")

    # Compare preprocessor features
    min_len = min(features_eager.shape[1], features_stateful_pt.shape[1])
    diff_pt = np.abs(features_eager[:, :min_len].numpy() - features_stateful_pt[:, :min_len].numpy())
    diff_onnx = np.abs(features_eager[:, :min_len].numpy() - features_stateful_onnx[:, :min_len])
    print(f"Preprocessor PT vs Eager Max Diff: {diff_pt.max():.6f}")
    print(f"Preprocessor ONNX vs Eager Max Diff: {diff_onnx.max():.6f}")

    # 4. Compare Encoder
    encoder = EncoderWrapper(model).eval()
    with torch.no_grad():
        encoded_eager = model.model.encoder(input_values, attention_mask=torch.ones(input_values.shape[:2], dtype=torch.long))[0]
        encoded_pt = encoder(features_stateful_pt[:, :min_len])
        encoded_onnx = sess_encoder.run(None, {"features": features_stateful_onnx[:, :min_len]})[0]
    
    diff_enc_pt = np.abs(encoded_eager[:, :min_len].numpy() - encoded_pt.numpy())
    diff_enc_onnx = np.abs(encoded_eager[:, :min_len].numpy() - encoded_onnx)
    print(f"Encoder PT vs Eager Max Diff: {diff_enc_pt.max():.6f}")
    print(f"Encoder ONNX vs Eager Max Diff: {diff_enc_onnx.max():.6f}")

    # 5. Compare Adapter (BEFORE decoder monkey patching)
    adapter = AdapterWrapper(model.model.decoder).eval()
    with torch.no_grad():
        pos_embeddings_eager = model.model.decoder.pos_emb(torch.arange(min_len))
        memory_eager = model.model.decoder.proj(encoded_eager[:, :min_len] + pos_embeddings_eager)
        
        memory_pt = adapter(encoded_pt, torch.zeros(1, dtype=torch.int64))
        memory_onnx = sess_adapter.run(None, {
            "encoded": encoded_onnx,
            "pos_offset": np.zeros(1, dtype=np.int64)
        })[0]

    diff_adapt_pt = np.abs(memory_eager.numpy() - memory_pt.numpy())
    diff_adapt_onnx = np.abs(memory_eager.numpy() - memory_onnx)
    print(f"Adapter PT vs Eager Max Diff: {diff_adapt_pt.max():.6f}")
    print(f"Adapter ONNX vs Eager Max Diff: {diff_adapt_onnx.max():.6f}")

    # 6. Compare Cross KV (BEFORE decoder monkey patching)
    cross_kv = CrossKVGeneratorWrapper(model.model.decoder).eval()
    with torch.no_grad():
        k_cross_pt, v_cross_pt = cross_kv(memory_pt)
        k_cross_onnx, v_cross_onnx = sess_cross_kv.run(None, {"memory": memory_onnx})
        
        # Eager cross KV
        k_cross_list = []
        v_cross_list = []
        for layer in model.model.decoder.layers:
            attn = layer.encoder_attn
            k_proj = attn.k_proj(memory_eager).view(1, min_len, 8, 40).transpose(1, 2)
            v_proj = attn.v_proj(memory_eager).view(1, min_len, 8, 40).transpose(1, 2)
            k_cross_list.append(k_proj)
            v_cross_list.append(v_proj)
        k_cross_eager = torch.stack(k_cross_list, dim=0).numpy()
        v_cross_eager = torch.stack(v_cross_list, dim=0).numpy()

    diff_k_cross_pt = np.abs(k_cross_eager - k_cross_pt.numpy())
    diff_k_cross_onnx = np.abs(k_cross_eager - k_cross_onnx)
    print(f"Cross K PT vs Eager Max Diff: {diff_k_cross_pt.max():.6f}")
    print(f"Cross K ONNX vs Eager Max Diff: {diff_k_cross_onnx.max():.6f}")

    # First token is BOS
    tokens = [1]
    
    with torch.no_grad():
        # Eager first step BEFORE decoder monkey patching
        pkv_eager = EncoderDecoderCache(DynamicCache(), DynamicCache())
        dec_out_eager = model.model.decoder(
            input_ids=torch.tensor([[tokens[-1]]]),
            encoder_hidden_states=encoded_eager[:, :min_len],  # original decoder takes raw encoder output
            past_key_values=pkv_eager,
            use_cache=True,
        )
        logits_eager = model.proj_out(dec_out_eager.last_hidden_state).numpy()

    # 7. Compare Decoder step-by-step (MONKEY PATCHES DECODER IN-PLACE NOW)
    decoder_kv = DecoderKVWrapper(model).eval()
    
    # Initialize decoder states
    k_self_pt = torch.zeros(6, 1, 8, 0, 40)
    v_self_pt = torch.zeros(6, 1, 8, 0, 40)

    k_self_onnx = np.zeros((6, 1, 8, 0, 40), dtype=np.float32)
    v_self_onnx = np.zeros((6, 1, 8, 0, 40), dtype=np.float32)
    
    with torch.no_grad():
        # PT first step
        res_dec_pt = decoder_kv(
            torch.tensor([[tokens[-1]]]),
            k_self_pt,
            v_self_pt,
            torch.from_numpy(k_cross_eager),
            torch.from_numpy(v_cross_eager)
        )
        logits_pt, _, _, _, _ = res_dec_pt
        logits_pt = logits_pt.numpy()

        # ONNX first step
        res_dec_onnx = sess_decoder.run(None, {
            "token": np.array([[tokens[-1]]], dtype=np.int64),
            "k_self": k_self_onnx,
            "v_self": v_self_onnx,
            "out_k_cross": k_cross_eager,
            "out_v_cross": v_cross_eager,
        })
        logits_onnx, _, _, _, _ = res_dec_onnx
        
    diff_logits_pt = np.abs(logits_eager - logits_pt)
    diff_logits_onnx = np.abs(logits_eager - logits_onnx)
    print(f"Decoder Logits PT vs Eager Max Diff: {diff_logits_pt.max():.6f}")
    print(f"Decoder Logits ONNX vs Eager Max Diff: {diff_logits_onnx.max():.6f}")
    
    print(f"Eager Logits: {logits_eager[0, -1, :5]}")
    print(f"ONNX Logits: {logits_onnx[0, -1, :5]}")
    
    next_tok_eager = logits_eager[0, -1].argmax().item()
    next_tok_onnx = logits_onnx[0, -1].argmax().item()
    print(f"Eager Next Token: {next_tok_eager}, ONNX Next Token: {next_tok_onnx}")

if __name__ == "__main__":
    debug_parity()

