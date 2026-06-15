import os
import sys
import numpy as np
import soundfile as sf
import onnxruntime as ort

def compare():
    # Paths to the ONNX models
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[4]
    
    base_dir = project_root / "models_base" / "UsefulSensors" / "moonshine-streaming-tiny" / "export" / "onnx" / "float" / "dynamic"
    split_dir = project_root / "models" / "UsefulSensors" / "moonshine-streaming-tiny" / "export" / "onnx" / "float" / "dynamic"

    print("Loading baseline model from:", os.path.join(base_dir, "preprocessor.onnx"))
    sess_base = ort.InferenceSession(os.path.join(base_dir, "preprocessor.onnx"), providers=['CPUExecutionProvider'])
    print("Loading 5-split model from:", os.path.join(split_dir, "frontend.onnx"))
    sess_split = ort.InferenceSession(os.path.join(split_dir, "frontend.onnx"), providers=['CPUExecutionProvider'])

    # 1. Load test audio
    wav_path = project_root / "src" / "torq" / "models" / "moonshine_streaming" / "OSR_us_000_0010_8k.wav"
    data, sr = sf.read(wav_path)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        data = resample_poly(data, up=16000, down=sr).astype(np.float32)

    # Truncate to 80,000 samples to match baseline validation shape
    max_samples = 80000
    if len(data) > max_samples:
        data = data[:max_samples]
    
    # Base model expects shape [1, seq_len]
    speech = data[np.newaxis, :]
    attention_mask = np.ones_like(speech, dtype=np.int64)

    # 2. Run baseline preprocessor (stateless, all at once)
    print("Running baseline preprocessor...")
    features_base, padding_mask_base = sess_base.run(None, {
        "input_values": speech,
        "attention_mask": attention_mask
    })
    print(f"Baseline features shape: {features_base.shape}")

    # 3. Run 5-split preprocessor (stateful, chunk-by-chunk)
    print("Running 5-split stateful preprocessor...")
    sample_buffer = np.zeros((1, 79), dtype=np.float32)
    sample_len = np.zeros(1, dtype=np.int64)
    conv1_buffer = np.zeros((1, 320, 4), dtype=np.float32)
    conv2_buffer = np.zeros((1, 640, 4), dtype=np.float32)
    frame_count = np.zeros(1, dtype=np.int64)

    chunk_len = 640
    accum_features = []
    
    audio_len = speech.shape[-1]
    step = 0
    for offset in range(0, audio_len, chunk_len):
        chunk = speech[:, offset : offset + chunk_len]
        if chunk.shape[-1] < chunk_len:
            chunk = np.pad(chunk, ((0, 0), (0, chunk_len - chunk.shape[-1])), mode="constant")
        
        outputs = sess_split.run(None, {
            "audio_chunk": chunk,
            "sample_buffer": sample_buffer,
            "sample_len": sample_len,
            "conv1_buffer": conv1_buffer,
            "conv2_buffer": conv2_buffer,
            "frame_count": frame_count,
        })
        features_chunk, sample_buffer, sample_len, conv1_buffer, conv2_buffer, frame_count = outputs
        accum_features.append(features_chunk)
        
        # Compare current chunk outputs against the corresponding slice of baseline
        start_feat_idx = step * 2
        end_feat_idx = start_feat_idx + features_chunk.shape[1]
        base_slice = features_base[:, start_feat_idx:end_feat_idx]
        diff_chunk = np.abs(base_slice - features_chunk)
        print(f"Step {step:3d} (audio offset {offset:5d}): Max Diff = {diff_chunk.max():.6f}, Mean Diff = {diff_chunk.mean():.6f}")
        step += 1

    features_split = np.concatenate(accum_features, axis=1)
    print(f"5-split accumulated features shape: {features_split.shape}")

    # Compare features overall
    min_len = min(features_base.shape[1], features_split.shape[1])
    diff = np.abs(features_base[:, :min_len] - features_split[:, :min_len])
    print(f"\nOverall Comparison (length {min_len}):")
    print(f"Max Absolute Difference: {diff.max():.6f}")
    print(f"Mean Absolute Difference: {diff.mean():.6f}")

if __name__ == "__main__":
    compare()
