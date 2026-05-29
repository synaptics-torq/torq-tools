

import torch
from datasets import load_dataset, Audio
from transformers import MoonshineStreamingForConditionalGeneration, AutoProcessor

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the streaming model and processor
model = MoonshineStreamingForConditionalGeneration.from_pretrained(
    "UsefulSensors/moonshine-streaming-small"
).to(device).to(torch_dtype)

processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-streaming-tiny")

# Load Hugging Face's internal dummy ASR test dataset
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(processor.feature_extractor.sampling_rate))
sample = dataset[0]["audio"]

# Process the raw audio array
inputs = processor(sample["array"], return_tensors="pt").to(device)

# Dynamic length calculation for Moonshine's flexible input windows
token_limit_factor = 6.5 / processor.feature_extractor.sampling_rate
seq_lens = inputs.attention_mask.sum(dim=-1)
max_length = int((seq_lens * token_limit_factor).max().item())

# Generate and print the transcription
generated_ids = model.generate(**inputs, max_length=max_length)
print("Transcription:", processor.decode(generated_ids[0], skip_special_tokens=True))