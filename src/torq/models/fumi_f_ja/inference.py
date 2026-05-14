import json
import numpy as np
import onnxruntime as ort
import soundfile as sf

MODEL_PATH = "tsuki_static_float32.onnx"
VOCAB_PATH = "vocab.json"
SAMPLE_RATE = 24000

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

def make_symbols(vocab):
    symbols = []
    symbols.append(vocab["pad"])
    symbols.extend(list(vocab["punctuation"]))
    symbols.extend(list(vocab["letters"]))
    symbols.extend(list(vocab["letters_ipa"]))
    return list(dict.fromkeys(symbols))

symbols = make_symbols(vocab)

def build_mapping(offset=0):
    return {s: i + offset for i, s in enumerate(symbols)}

def normalize_audio(audio):
    audio = np.asarray(audio).squeeze().astype(np.float32)
    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs * 0.95
    return audio

def encode_text(text, symbol_to_id, add_bos_eos=False, bos_id=1, eos_id=2):
    ids = []

    for ch in text:
        if ch not in symbol_to_id:
            raise ValueError(f"Unknown character: {repr(ch)}")
        ids.append(symbol_to_id[ch])

    if add_bos_eos:
        ids = [bos_id] + ids + [eos_id]

    texts = np.array([ids], dtype=np.int64)
    text_lengths = np.array([len(ids)], dtype=np.int64)

    return texts, text_lengths

def encode_static(text, symbol_to_id, s26=100, add_bos_eos=True):
    PAD_ID = symbol_to_id["$"]
    BOS_ID = 1
    EOS_ID = 2

    ids = [symbol_to_id[ch] for ch in text]

    if add_bos_eos:
        ids = [BOS_ID] + ids + [EOS_ID]

    text_length = len(ids)  # real length, before padding

    if len(ids) > s26:
        ids = ids[:s26]
        text_length = s26

    ids += [PAD_ID] * (s26 - len(ids))

    texts = np.array([ids], dtype=np.int64)
    text_lengths = np.array([text_length], dtype=np.int64)

    return texts, text_lengths

sess = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
)

print("Vocab size:", len(symbols))
print("Symbols:", "".join(symbols))

print("\nInputs:")
for i in sess.get_inputs():
    print(i.name, i.shape, i.type)

'''print("\nOutputs:")
for o in sess.get_outputs():
    print(o.name, o.shape, o.type)'''



# --------------------------------------------------
# Main phrase test
# --------------------------------------------------
best_offset = 0
best_add_bos_eos = True

symbol_to_id = build_mapping(best_offset)

text = "sajoːnara"


# Using the dynamic (original) model
'''texts, text_lengths = encode_text(
    text,
    symbol_to_id,
    add_bos_eos=True,
)'''


# Using the static model
texts, text_lengths = encode_static(
    text,
    symbol_to_id,
    s26=11,
    add_bos_eos=True,
)

outputs = sess.run(
    None,
    {
        "texts": texts,
        "text_lengths": text_lengths,
    },
)

audio = normalize_audio(outputs[0])

sf.write(f"static_{text}_{SAMPLE_RATE}.wav", audio, SAMPLE_RATE)

print("ids:", texts.tolist())
print("length:", text_lengths.tolist())

raw_audio = np.asarray(outputs[0]).squeeze()

total_samples = raw_audio.shape[-1]
audio_frames = total_samples // 300

print("raw audio shape:", raw_audio.shape)
print("total_samples:", total_samples)
print("audio_frames:", audio_frames)
print("audio_frames exact:", total_samples / 300)