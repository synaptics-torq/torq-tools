# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import argparse
import time
from typing import List, Tuple
from pathlib import Path
import os

import numpy as np
from datasets import load_dataset as load_ds
from jiwer import wer
from whisper.normalizers import EnglishTextNormalizer
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
import pandas as pd

# Import existing moonshine inference classes
try:
    from ._inference import MoonshineDynamic, MoonshineStatic, load_moonshine
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from _inference import MoonshineDynamic, MoonshineStatic


def _find_models(model_dir: str | Path) -> tuple[list[Path], str]:
    """Find ONNX or VMFB models in the given directory."""
    p = Path(model_dir)
    onnx = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".onnx")
    vmfb = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".vmfb")
    if onnx and vmfb:
        raise ValueError(
            "Model directory contains both ONNX and VMFB models; choose one format."
        )
    models = onnx or vmfb
    if not models:
        raise FileNotFoundError(f"No .onnx or .vmfb models found in {p}")
    kind = "onnx" if onnx else "vmfb"
    return models, kind


def load_tokenizer(model_size: str):
    """Load tokenizer for the given model size."""
    tokenizer_file = hf_hub_download(f"UsefulSensors/moonshine-{model_size}", "tokenizer.json")
    return Tokenizer.from_file(tokenizer_file)


def check_dataset_audio_sizes(dataset, max_samples=None):
    """Check the audio sizes in the dataset to determine appropriate max_inp_len."""
    print("Checking audio sizes in ENTIRE dataset...")
    audio_lengths = []
    
    for i, example in enumerate(tqdm(dataset)):
        audio = example["audio"]["array"]
        audio_lengths.append(len(audio))
        if max_samples is not None and i >= max_samples - 1:
            break
    
    audio_lengths = np.array(audio_lengths)
    total_samples = len(audio_lengths)
    print(f"\nAudio length statistics (ALL {total_samples} samples):")
    print(f"  Min length: {audio_lengths.min():,} samples")
    print(f"  Max length: {audio_lengths.max():,} samples") 
    print(f"  Mean length: {audio_lengths.mean():.0f} samples")
    print(f"  95th percentile: {np.percentile(audio_lengths, 95):.0f} samples")
    print(f"  99th percentile: {np.percentile(audio_lengths, 99):.0f} samples")
    
    return audio_lengths.max()


def validate(onnx_models_dir: str, vmfb_models_dir: str, model_size: str = "tiny", max_inp_len: int = None, max_dec_len: int = None, max_samples: int = None, bf16_vmfb_models_dir: str = None, task_topology_group_count: int = 128):
    """
    Validate and compare WER between ONNX Runtime fp32, IREE Runtime fp32, and IREE Runtime bf16 for static models on suitable samples.
    """
    # Configure IREE runtime task topology if not already set
    if task_topology_group_count and "IREE_RUN_MODULE_FLAGS" not in os.environ:
        os.environ["IREE_RUN_MODULE_FLAGS"] = f"--task_topology_group_count={task_topology_group_count}"
        print(f"Set IREE_RUN_MODULE_FLAGS={os.environ['IREE_RUN_MODULE_FLAGS']}")
    elif task_topology_group_count:
        # Append/override existing flag if different
        flags = os.environ.get("IREE_RUN_MODULE_FLAGS", "")
        if f"--task_topology_group_count=" not in flags:
            os.environ["IREE_RUN_MODULE_FLAGS"] = (flags + f" --task_topology_group_count={task_topology_group_count}").strip()
            print(f"Updated IREE_RUN_MODULE_FLAGS={os.environ['IREE_RUN_MODULE_FLAGS']}")

    # Load dataset (remove deprecated trust_remote_code usage). If this fails, surface clear guidance.
    try:
        dataset = load_ds(
            path="hf-audio/esb-datasets-test-only-sorted",
            name="librispeech",
            split="test.clean",
        )
        print("Loaded ESB test.clean dataset (no train/dev splits).")
    except Exception as e:
        print(f"Primary dataset load failed: {e}")
        print("Falling back to torchaudio test-clean loader (will download if missing)...")
        try:
            import torchaudio
            from torchaudio.datasets import LIBRISPEECH
            ds_t = LIBRISPEECH(root=os.path.join(os.getcwd(), 'librispeech_data'), url='test-clean', download=True)
            dataset = []
            for j in range(len(ds_t)):
                waveform, sr, *_rest = ds_t[j]
                if waveform.ndim > 1 and waveform.shape[0] > 1:
                    waveform = waveform.mean(0, keepdim=True)
                dataset.append({
                    'audio': {'array': waveform.squeeze(0).numpy(), 'sampling_rate': sr},
                    'text': _rest[-1].strip()
                })
            print(f"Fallback torchaudio test-clean loaded: {len(dataset)} samples")
        except Exception as e2:
            print(f"Fallback torchaudio loader also failed: {e2}")
            return
    
    # Skip audio size checking - we already know the parameters
    if max_inp_len is None or max_dec_len is None:
        print("Error: Both --max-inp-len and --max-dec-len must be provided")
        return
    
    normalizer = EnglishTextNormalizer()
    tokenizer = load_tokenizer(model_size)

    print(f"Starting validation with max_inp_len={max_inp_len:,} samples ({max_inp_len/16000:.1f}s)")
    if max_samples:
        print(f"Processing up to {max_samples} audio samples from the dataset...")
    else:
        print("Processing all suitable samples in dataset that fit within input length limit...")
    print()

    # First pass: determine which samples are valid for all runtimes
    print("=== Pre-filtering samples to ensure all runtimes process the same data ===")
    valid_samples, skip_stats = filter_valid_samples(dataset, max_inp_len, max_samples)
    
    if len(valid_samples) == 0:
        print("No valid samples found that fit within the model constraints!")
        return
    
    print(f"Found {len(valid_samples)} valid samples to process with all runtimes\n")

    # ONNX Runtime (fp32)
    print("=== ONNX Runtime FP32 Processing ===")
    model_dirs = {'onnx': onnx_models_dir, 'iree': vmfb_models_dir}
    expected_texts, predicted_texts_onnx_fp32, onnx_fp32_time, onnx_fp32_details = process_valid_samples(valid_samples, model_dirs, model_size, max_inp_len, max_dec_len, tokenizer, backend="onnx")
    wer_onnx_fp32 = wer(
        normalizer(" ".join(expected_texts)),
        normalizer(" ".join(predicted_texts_onnx_fp32)),
    )

    # IREE Runtime (fp32)
    print("\n=== IREE Runtime FP32 Processing ===")
    _, predicted_texts_iree_fp32, iree_fp32_time, iree_fp32_details = process_valid_samples(valid_samples, model_dirs, model_size, max_inp_len, max_dec_len, tokenizer, backend="iree")
    wer_iree_fp32 = wer(
        normalizer(" ".join(expected_texts)),
        normalizer(" ".join(predicted_texts_iree_fp32)),
    )

    # IREE Runtime (bf16) - only if bf16 models directory is provided
    if bf16_vmfb_models_dir:
        print("\n=== IREE Runtime BF16 Processing ===")
        bf16_model_dirs = {'onnx': onnx_models_dir, 'iree': bf16_vmfb_models_dir}
        _, predicted_texts_iree_bf16, iree_bf16_time, iree_bf16_details = process_valid_samples(valid_samples, bf16_model_dirs, model_size, max_inp_len, max_dec_len, tokenizer, backend="iree")
        wer_iree_bf16 = wer(
            normalizer(" ".join(expected_texts)),
            normalizer(" ".join(predicted_texts_iree_bf16)),
        )
    else:
        predicted_texts_iree_bf16, iree_bf16_time, iree_bf16_details, wer_iree_bf16 = [], 0, [], float('inf')
        print("\n=== IREE Runtime BF16 Processing ===")
        print("BF16 models directory not provided, skipping BF16 validation")

    # Export comparison to Excel
    print("\n=== Exporting Results to Excel ===")
    export_to_excel(onnx_fp32_details, iree_fp32_details, wer_onnx_fp32, wer_iree_fp32, onnx_fp32_time, iree_fp32_time, iree_bf16_details, wer_iree_bf16, iree_bf16_time, skip_stats=skip_stats)

    # Summarize results
    num_samples = len(expected_texts)
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Samples processed: {num_samples}")
    print("Skipped samples (pre-filter):")
    print(f"  Decode errors: {skip_stats.get('decode_errors', 0)}")
    print(f"  Audio access errors: {skip_stats.get('audio_access_errors', 0)}")
    print(f"  Too long (> conservative limit): {skip_stats.get('too_long', 0)}")
    print(f"  Total skipped: {skip_stats.get('total', 0)}")
    print(f"Model input limit: {max_inp_len:,} samples ({max_inp_len/16000:.1f}s)")
    print()
    print(f"ONNX Runtime FP32 WER: {100. * wer_onnx_fp32:.2f}%")
    print(f"IREE Runtime FP32 WER: {100. * wer_iree_fp32:.2f}%")
    if bf16_vmfb_models_dir:
        print(f"IREE Runtime BF16 WER: {100. * wer_iree_bf16:.2f}%")
    print()
    print(f"ONNX Runtime FP32 time: {onnx_fp32_time:.2f}s")
    print(f"IREE Runtime FP32 time: {iree_fp32_time:.2f}s")
    if bf16_vmfb_models_dir:
        print(f"IREE Runtime BF16 time: {iree_bf16_time:.2f}s")
    print()
    if abs(wer_onnx_fp32 - wer_iree_fp32) < 0.001:
        print("✅ FP32 WER MATCH: ONNX and IREE FP32 produce identical results!")
    else:
        print(f"⚠️  FP32 WER DIFFERENCE: {100. * abs(wer_onnx_fp32-wer_iree_fp32):.2f}%")
    
    if bf16_vmfb_models_dir:
        if abs(wer_iree_fp32 - wer_iree_bf16) < 0.001:
            print("✅ PRECISION WER MATCH: IREE FP32 and BF16 produce identical results!")
        else:
            print(f"⚠️  PRECISION WER DIFFERENCE: {100. * abs(wer_iree_fp32-wer_iree_bf16):.2f}%")
    print(f"{'='*80}")


def filter_valid_samples(dataset, max_inp_len, max_samples):
    """Filter dataset to find samples that fit within model constraints and gather skip stats.

    Returns (valid_samples_list, skip_stats_dict)
    skip_stats keys: decode_errors, audio_access_errors, too_long, total
    """
    valid_samples = []
    decode_error_count = 0
    audio_access_error_count = 0
    too_long_count = 0
    message_budget = 10  # total printed skip messages before suppression notice
    printed = 0

    conservative_max_inp_len = 500000  # ~31 seconds * 16kHz
    print(f"Pre-filtering samples with conservative limit: {conservative_max_inp_len:,} samples (~{conservative_max_inp_len/16000:.1f}s)")

    total_len = None
    try:
        total_len = len(dataset)
    except Exception:
        pass
    i = 0
    while True:
        if total_len is not None and i >= total_len:
            break
        if max_samples is not None and len(valid_samples) >= max_samples:
            print(f"Reached maximum sample limit of {max_samples}, stopping pre-filter...")
            break
        try:
            example = dataset[i]
        except IndexError:
            break
        except Exception as e:
            if printed < message_budget:
                print(f"Decoding error at sample {i}: {e}")
                printed += 1
                if printed == message_budget:
                    print("... (suppressing further skip messages)")
            decode_error_count += 1
            i += 1
            continue
        try:
            audio = example["audio"]["array"]
        except Exception as e:
            if printed < message_budget:
                print(f"Audio access error at sample {i}: {e}")
                printed += 1
                if printed == message_budget:
                    print("... (suppressing further skip messages)")
            audio_access_error_count += 1
            i += 1
            continue

        if len(audio) > conservative_max_inp_len:
            if printed < message_budget:
                print(f"Skipping sample {i}: audio length = {len(audio):,} samples (exceeds {conservative_max_inp_len:,} limit)")
                printed += 1
                if printed == message_budget:
                    print("... (suppressing further skip messages)")
            too_long_count += 1
            i += 1
            continue

        valid_samples.append((i, example))
        i += 1

    total_skipped = decode_error_count + audio_access_error_count + too_long_count
    print(
        f"Pre-filtering complete: {len(valid_samples)} valid samples, "
        f"{total_skipped} skipped (decode={decode_error_count}, audio_access={audio_access_error_count}, too_long={too_long_count})"
    )
    skip_stats = {
        'decode_errors': decode_error_count,
        'audio_access_errors': audio_access_error_count,
        'too_long': too_long_count,
        'total': total_skipped,
    }
    return valid_samples, skip_stats


def process_valid_samples(valid_samples, model_dirs, model_size, max_inp_len, max_dec_len, tokenizer, backend="onnx"):
    """Process a pre-filtered list of valid samples."""
    expected_texts, predicted_texts = [], []
    sample_details = []  # For Excel export
    start = time.time()
    
    print(f"Processing {len(valid_samples)} pre-validated samples for {backend}...")
    
    # Load the model once for this backend
    model_dir = model_dirs[backend]
    model = load_moonshine(model_dir, model_size, max_inp_len, max_dec_len)
    
    # Process samples sequentially
    for i, (sample_idx, example) in enumerate(valid_samples):
        try:
            audio = example["audio"]["array"]
            audio_input = audio[np.newaxis, :].astype(np.float32)
            
            # Use the same transcription logic as infer.py
            predicted_text = _transcribe_audio(audio_input, model, tokenizer)
            
            expected_texts.append(" " + example["text"])
            predicted_texts.append(" " + predicted_text)
            sample_details.append({
                'sample_idx': sample_idx,
                'audio_length': len(audio),
                'expected_text': example["text"],
                'predicted_text': predicted_text
            })
            
            if (i + 1) % 5 == 0 or (i + 1) == len(valid_samples):
                print(f"  Completed {i + 1}/{len(valid_samples)} samples...")
                
        except Exception as e:
            print(f"Failed to process sample {sample_idx}: {str(e)}")
            continue
    
    elapsed = time.time() - start
    print(f"{backend} processing complete: {len(expected_texts)} successful samples in {elapsed:.2f}s")
    
    return expected_texts, predicted_texts, elapsed, sample_details


def _transcribe_audio(audio_array: np.ndarray, runner, tokenizer) -> str:
    """Transcribe audio using the same logic as infer.py"""
    # audio_array is already in the right format: [1, num_samples] float32
    tokens = runner.run(audio_array)
    text = tokenizer.decode_batch(tokens, skip_special_tokens=True)[0]
    return text


def process_dataset_with_timing(dataset, model, tokenizer, backend="onnx", max_inp_len=None, max_samples=None) -> Tuple[List[str], List[str], float]:
    expected_texts, predicted_texts = [], []
    start = time.time()
    
    processed_count = 0
    skipped_count = 0
    
    # Use a more conservative input length limit to account for decoder token limits
    # Estimate roughly 25 tokens per second, so with 1175 token limit, use ~35 seconds max
    conservative_max_inp_len = min(max_inp_len, 560000) if max_inp_len else 560000  # 35 seconds * 16kHz
    
    print(f"Processing up to {max_samples if max_samples else 'all'} samples from dataset...")
    
    # Process samples that fit within our model's input length limit
    for i, example in enumerate(dataset):
        # Stop if we've reached the maximum number of samples to process
        if max_samples is not None and processed_count >= max_samples:
            print(f"Reached maximum sample limit of {max_samples}, stopping...")
            break
        audio = example["audio"]["array"]
        
        # Skip samples that exceed our conservative input length capacity
        if len(audio) > conservative_max_inp_len:
            skipped_count += 1
            if skipped_count <= 10:  # Only print first 10 skips to avoid spam
                print(f"Skipping sample {i}: audio length = {len(audio):,} samples (exceeds conservative limit={conservative_max_inp_len:,})")
            elif skipped_count == 11:
                print("... (suppressing further skip messages)")
            continue
            
        audio_input = audio[np.newaxis, :].astype(np.float32)
        print(f"Processing sample {i}: audio length = {len(audio):,} samples")
        
        try:
            # Use the same transcription logic as infer.py
            predicted_text = _transcribe_audio(audio_input, model, tokenizer)
            
            expected_texts.append(" " + example["text"])
            predicted_texts.append(" " + predicted_text)
            processed_count += 1
            
            if processed_count % 50 == 0:
                print(f"  Processed {processed_count} samples so far...")
                
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            print(f"  Audio length was {len(audio):,} samples, skipping...")
            skipped_count += 1
            continue
        
    print(f"\nDataset processing complete for {backend}:")
    print(f"  Processed: {processed_count} samples")
    print(f"  Skipped: {skipped_count} samples (too long or errors)")
    print(f"  Total dataset size: {len(dataset)} samples")
        
    elapsed = time.time() - start
    return expected_texts, predicted_texts, elapsed


def export_to_excel(onnx_fp32_details, iree_fp32_details, wer_onnx_fp32, wer_iree_fp32, onnx_fp32_time=0, iree_fp32_time=0, iree_bf16_details=None, wer_iree_bf16=None, iree_bf16_time=0, skip_stats=None):
    """Export comparison results to Excel file. Supports 2 or 3 runtime comparisons."""
    if len(onnx_fp32_details) != len(iree_fp32_details):
        print(f"Warning: Mismatched sample counts - ONNX FP32: {len(onnx_fp32_details)}, IREE FP32: {len(iree_fp32_details)}")
        return
    
    has_bf16 = iree_bf16_details is not None and len(iree_bf16_details) > 0
    if has_bf16 and len(iree_fp32_details) != len(iree_bf16_details):
        print(f"Warning: Mismatched sample counts - IREE FP32: {len(iree_fp32_details)}, IREE BF16: {len(iree_bf16_details)}")
        return
    
    # Combine the data for Excel export
    comparison_data = []
    
    for i in range(len(onnx_fp32_details)):
        onnx_sample = onnx_fp32_details[i]
        iree_fp32_sample = iree_fp32_details[i]
        
        # Check if samples match
        if onnx_sample['sample_idx'] != iree_fp32_sample['sample_idx']:
            print(f"Warning: Sample index mismatch at position {i}")
            continue
        
        row_data = {
            'Sample_Index': onnx_sample['sample_idx'],
            'Audio_Length_Samples': onnx_sample['audio_length'],
            'Audio_Duration_Seconds': onnx_sample['audio_length'] / 16000,
            'Expected_Text': onnx_sample['expected_text'],
            'ONNX_FP32_Output': onnx_sample['predicted_text'],
            'IREE_FP32_Output': iree_fp32_sample['predicted_text'],
            'ONNX_FP32_vs_IREE_FP32_Match': onnx_sample['predicted_text'] == iree_fp32_sample['predicted_text']
        }
        
        # Add BF16 columns if available
        if has_bf16 and i < len(iree_bf16_details):
            iree_bf16_sample = iree_bf16_details[i]
            if iree_fp32_sample['sample_idx'] == iree_bf16_sample['sample_idx']:
                row_data.update({
                    'IREE_BF16_Output': iree_bf16_sample['predicted_text'],
                    'IREE_FP32_vs_BF16_Match': iree_fp32_sample['predicted_text'] == iree_bf16_sample['predicted_text'],
                    'All_Outputs_Match': (onnx_sample['predicted_text'] == iree_fp32_sample['predicted_text'] == iree_bf16_sample['predicted_text'])
                })
            else:
                row_data.update({
                    'IREE_BF16_Output': 'Sample index mismatch',
                    'IREE_FP32_vs_BF16_Match': False,
                    'All_Outputs_Match': False
                })
        else:
            # Legacy column for backward compatibility (two-runtime comparison)
            if not has_bf16:
                row_data['Outputs_Match'] = onnx_sample['predicted_text'] == iree_fp32_sample['predicted_text']
        
        comparison_data.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Add summary statistics
    summary_data = {
        'Metric': ['Total Samples', 'ONNX FP32 WER (%)', 'IREE FP32 WER (%)'],
        'Value': [
            len(comparison_data),
            f"{100.0 * wer_onnx_fp32:.2f}%",
            f"{100.0 * wer_iree_fp32:.2f}%"
        ]
    }

    if skip_stats:
        summary_data['Metric'].extend([
            'Skipped (decode errors)',
            'Skipped (audio access errors)',
            'Skipped (too long)',
            'Skipped (total)'
        ])
        summary_data['Value'].extend([
            skip_stats.get('decode_errors', 0),
            skip_stats.get('audio_access_errors', 0),
            skip_stats.get('too_long', 0),
            skip_stats.get('total', 0)
        ])
    
    # Add timing if provided
    if onnx_fp32_time > 0:
        summary_data['Metric'].append('ONNX FP32 Time (s)')
        summary_data['Value'].append(f"{onnx_fp32_time:.2f}s")
    if iree_fp32_time > 0:
        summary_data['Metric'].append('IREE FP32 Time (s)')
        summary_data['Value'].append(f"{iree_fp32_time:.2f}s")
    
    if has_bf16 and wer_iree_bf16 is not None:
        summary_data['Metric'].extend(['IREE BF16 WER (%)', 'FP32 vs BF16 Identical Outputs'])
        summary_data['Value'].extend([
            f"{100.0 * wer_iree_bf16:.2f}%",
            f"{df['IREE_FP32_vs_BF16_Match'].sum()} / {len(df)} ({100.0 * df['IREE_FP32_vs_BF16_Match'].mean():.1f}%)" if 'IREE_FP32_vs_BF16_Match' in df.columns else "N/A"
        ])
        if iree_bf16_time > 0:
            summary_data['Metric'].append('IREE BF16 Time (s)')
            summary_data['Value'].append(f"{iree_bf16_time:.2f}s")
    else:
        # Legacy summary for two-runtime comparison
        summary_data['Metric'].extend(['WER Difference (%)', 'Identical Outputs'])
        summary_data['Value'].extend([
            f"{100.0 * abs(wer_onnx_fp32 - wer_iree_fp32):.2f}%",
            f"{df['Outputs_Match'].sum()} / {len(df)} ({100.0 * df['Outputs_Match'].mean():.1f}%)" if 'Outputs_Match' in df.columns else "N/A"
        ])
    
    summary_df = pd.DataFrame(summary_data)
    
    # Export to Excel with multiple sheets
    output_file = "outputs_compare.xlsx"
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main comparison data
        df.to_excel(writer, sheet_name='Sample_Comparison', index=False)
        
        # Summary statistics
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Adjust column widths for readability
        worksheet = writer.sheets['Sample_Comparison']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"Results exported to: {output_file}")
    print(f"  - Sample_Comparison sheet: {len(df)} samples with detailed comparisons")
    print(f"  - Summary sheet: Overall WER statistics and match percentages")
    if has_bf16:
        print(f"  - Includes BF16 vs FP32 precision comparison")


def process_dataset_with_timing(dataset, model, tokenizer, backend="onnx", max_inp_len=None, max_samples=None) -> Tuple[List[str], List[str], float]:
    expected_texts, predicted_texts = [], []
    start = time.time()
    
    processed_count = 0
    skipped_count = 0
    
    # Use a more conservative input length limit to account for decoder token limits
    # Estimate roughly 25 tokens per second, so with 1175 token limit, use ~35 seconds max
    conservative_max_inp_len = min(max_inp_len, 560000) if max_inp_len else 560000  # 35 seconds * 16kHz
    
    print(f"Processing up to {max_samples if max_samples else 'all'} samples from dataset...")
    
    # Process samples that fit within our model's input length limit
    for i, example in enumerate(dataset):
        # Stop if we've reached the maximum number of samples to process
        if max_samples is not None and processed_count >= max_samples:
            print(f"Reached maximum sample limit of {max_samples}, stopping...")
            break
        audio = example["audio"]["array"]
        
        # Skip samples that exceed our conservative input length capacity
        if len(audio) > conservative_max_inp_len:
            skipped_count += 1
            if skipped_count <= 10:  # Only print first 10 skips to avoid spam
                print(f"Skipping sample {i}: audio length = {len(audio):,} samples (exceeds conservative limit={conservative_max_inp_len:,})")
            elif skipped_count == 11:
                print("... (suppressing further skip messages)")
            continue
            
        audio_input = audio[np.newaxis, :].astype(np.float32)
        print(f"Processing sample {i}: audio length = {len(audio):,} samples")
        
        try:
            # Use the same transcription logic as infer.py
            predicted_text = _transcribe_audio(audio_input, model, tokenizer)
            
            expected_texts.append(" " + example["text"])
            predicted_texts.append(" " + predicted_text)
            processed_count += 1
            
            if processed_count % 50 == 0:
                print(f"  Processed {processed_count} samples so far...")
                
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            print(f"  Audio length was {len(audio):,} samples, skipping...")
            skipped_count += 1
            continue
        
    print(f"\nDataset processing complete for {backend}:")
    print(f"  Processed: {processed_count} samples")
    print(f"  Skipped: {skipped_count} samples (too long or errors)")
    print(f"  Total dataset size: {len(dataset)} samples")
        
    elapsed = time.time() - start
    return expected_texts, predicted_texts, elapsed


def parse_arguments():
    parser = argparse.ArgumentParser(description="Validate and compare ONNX FP32, IREE FP32, and IREE BF16 WER.")
    parser.add_argument("--onnx-models-dir", required=True, help="Directory containing ONNX FP32 models")
    parser.add_argument("--vmfb-models-dir", required=True, help="Directory containing VMFB FP32 models")
    parser.add_argument("--bf16-vmfb-models-dir", help="Directory containing VMFB BF16 models (optional)")
    parser.add_argument("--model-size", default="tiny", choices=["base", "tiny"], help="Model size")
    parser.add_argument("--max-inp-len", type=int, help="Maximum input length (required for static models)")
    parser.add_argument("--max-dec-len", type=int, help="Maximum decoder length (required for static models)")
    parser.add_argument("--max-samples", type=int, default=250, help="Maximum number of audio samples to process (omit for all valid)")
    parser.add_argument("--validate", action="store_true", help="Run validation and comparison")
    parser.add_argument("--task-topology-group-count", type=int, default=128, help="IREE --task_topology_group_count value (default: 128)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    validate(
            args.onnx_models_dir,
            args.vmfb_models_dir,
            args.model_size,
            args.max_inp_len,
            args.max_dec_len,
            args.max_samples,
            args.bf16_vmfb_models_dir,
            args.task_topology_group_count,
        )

if __name__ == "__main__":
    main()
