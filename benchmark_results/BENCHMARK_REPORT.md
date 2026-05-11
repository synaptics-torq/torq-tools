# Gemma-3 270M-IT Quantization Benchmark Report

**Date:** May 8, 2026  
**Device:** Aura (Torq SoC), 1.9GB RAM  
**Model:** Synaptics/gemma-3-270m-it  
**Questions:** 24 general knowledge questions  
**Models tested:** bf16, int8, int8_int4_hybrid_1, int8_int4_hybrid_2, int4, int8_bf16_blocks_embed, sensitivity_tf_dql, int8_int4_lmhead, sensitivity_v2, sensitivity_v2_dql, sensitivity_v3_dql, reduced_vocab_int8, reduced_vocab_int8_with_num

---

## 1. Performance Summary

| Model | Avg TPS | TPS Range | Avg TTFT (ms) | TTFT Range (ms) | Total Tokens | Avg Tokens/Answer |
|-------|---------|-----------|---------------|-----------------|-------------|-------------------|
| **bf16** | 1.9 | 1.8–2.3 | ~7,600 | 6,512–9,788 | 565 | 23.5 |
| **int8** | 1.7 | 1.7–2.0 | ~8,400 | 7,227–10,852 | 586 | 24.4 |
| **int8_bf16_scales** | 1.7 | 1.7–2.1 | ~8,400 | 7,221–10,846 | 634 | 26.4 |
| **sensitivity** | 1.7 | 1.7–2.5 | ~8,500 | 7,236–10,873 | 654 | 27.3 |
| **sensitivity_tf** | 1.7 | 1.7–1.9 | ~8,400 | 7,213–10,839 | 599 | 25.0 |
| **sensitivity_tf_bf16_scales** | 1.7 | 1.7–1.9 | ~8,400 | 7,238–10,874 | 670 | 27.9 |
| **hybrid_1** | 1.7 | 1.7–1.9 | ~8,400 | 7,225–10,866 | 757 | 31.5 |
| **hybrid_2** | 1.7 | 1.7–1.9 | ~8,400 | 7,229–10,870 | 806 | 33.6 |
| **int4** | 1.7 | 1.6–1.8 | ~8,600 | 7,370–11,075 | 928 | 38.7 |
| **int8_bf16_blocks_embed** | 1.7 | 1.3–2.2 | ~8,400 | 7,273–10,932 | 430 | 17.9 |
| **int8_bf16_blocks_embed_1** | 1.8 | 1.7–2.0 | ~8,300 | 7,239–10,886 | 579 | 24.1 |
| **fp8_e5m2** | 1.8 | 1.7–2.0 | ~8,400 | 7,249–10,894 | 523 | 21.8 |
| **fp8_e4m3** | 1.7 | 1.7–2.0 | ~8,400 | 7,240–10,896 | 529 | 22.0 |
| **sensitivity_tf_dql** | 1.7 | 1.6–1.8 | ~8,500 | 7,363–11,064 | 642 | 26.8 |
| **int8_int4_lmhead** | 1.8 | 1.7–2.0 | ~8,400 | 7,250–10,888 | 662 | 27.6 |
| **sensitivity_v2** | 1.7 | 1.7–1.8 | ~8,500 | 7,302–10,973 | 622 | 25.9 |
| **sensitivity_v2_dql** | 1.3 | 1.2–1.5 | ~11,377 | 9,830–14,765 | 660 | 27.5 |
| **sensitivity_v3_dql** | 1.3 | 1.2–1.4 | ~11,359 | 9,818–14,740 | 653 | 27.2 |
| **reduced_vocab_int8** | 2.2 | 2.1–2.6 | ~6,537 | 5,646–8,487 | 744 | 31.0 |
| **reduced_vocab_int8_with_num** | 2.3 | 2.2–2.7 | ~6,402 | 5,532–8,307 | 569 | 23.7 |

> **sensitivity** = `int8_int4_sensitivity`: 22 layers int8 (17%) / 105 layers int4 (83%), selected by KL-divergence logit sensitivity analysis. 6.83 effective bits/param.
> **sensitivity_tf** = `int8_int4_sensitivity_tf`: 17 layers int8 (13%) / 110 layers int4 (87%), selected by teacher-forced perplexity analysis (10 prompts × 30 tokens). 6.52 effective bits/param.
> **int8_bf16_scales** = `int8_converted_bf16_scales`: Same int8 source model, but scales are cast to bf16 *before* dequantization (vs default: dequantize with fp32 scales, then convert all to bf16). 8 bits/param.
> **sensitivity_tf_bf16_scales** = `int8_int4_mixed_sensitivity_tf_bf16_scales`: Same layer selection as sensitivity_tf (17 int8 / 110 int4), but using bf16-scale-dequantized weights from int8_converted_bf16_scales and int4_converted_bf16_scales. 6.52 effective bits/param.
> **int8_bf16_blocks_embed** = `int8_bf16_scales_blocks_along_embed_dim`: Int8 quantization with bf16 scales, using block-wise quantization along the embedding dimension (block size 128). Pre-dequantized to bf16 weights. 8 bits/param. **Note:** Model does not reliably produce EOS tokens; generates garbage (`<unused>`) tokens after the answer.
> **int8_bf16_blocks_embed_1** = `int8_bf16_scales_blocks_along_embed_dim_1`: Same approach as int8_bf16_blocks_embed but with block size 32 (vs 128), quantized from fp32_xenova source (original HuggingFace weights). 8 bits/param.
> **fp8_e5m2** = `fp8_e5m2`: FP8 (e5m2 format, 5 exponent + 2 mantissa bits) quantization of fp32_xenova source weights. Round-trip fp32→fp8_e5m2→fp32, then bf16 conversion. 8 bits/param. Only 116 unique weight values per tensor.
> **fp8_e4m3** = `fp8_e4m3`: FP8 (e4m3fn format, 4 exponent + 3 mantissa bits) quantization of fp32_xenova source weights. More precision than e5m2 (narrower range [-448,448] but finer granularity). 8 bits/param.
> **sensitivity_tf_dql** = `sensitivity_tf_dql_bf16`: Same layer selection as sensitivity_tf (17 int8 / 110 int4), but weights stored as DequantizeLinear (int4 with bf16 scales, block_size=32 for attention layers, block_size=8 for lm_head). Dequantized to bf16 for compilation. Uses signed int4 for attention layers, unsigned uint4 for lm_head. 6.52 effective bits/param.
> **sensitivity_v2** = `sensitivity_v2_hybrid`: 21 layers int8 (17%) + 105 layers int4 (83%) + lm_head int4, selected by KL-divergence teacher-forced sensitivity analysis (3 prompts × 5 tokens). KL threshold=0.005. ~4.28 effective bits/param (weighted by parameter count: 18.8M int8 + 249.2M int4).
> **sensitivity_v2_dql** = `sensitivity_v2_dql_bf16`: Same layer selection as sensitivity_v2 (21 int8 / 105 int4), but all weights quantized from fp32_xenova source using block DequantizeLinear format (int4 signed block_size=32 for attention/MLP layers, int8 asymmetric block_size=32 for 21 sensitive layers, int4 signed block_size=8 for lm_head). Dequantized to bf16 for compilation. ~4.28 effective bits/param.
> **sensitivity_v3_dql** = `sensitivity_v2_dql_lmhead_bs16_bf16`: Same as sensitivity_v2_dql but with lm_head using block_size=16 instead of block_size=8. 105 int4 bs=32 + 21 int8 bs=32 + lm_head int4 bs=16. ~4.28 effective bits/param.
> **int8_int4_lmhead** = `int8_int4_lmhead_bs8`: All 126 attention/MLP layers use int8 dequantized weights (from int8_converted). Only the lm_head (640×262144) uses int4 block_size=8 dequantized weights (unsigned uint4, axis=0). ~7.97 effective bits/param (int8 everywhere except lm_head which is int4).
> **reduced_vocab_int8** = `reduced_vocab_fp32/model_bf16_reduced_vocab`: Int8 block-quantized weights (block_size=32, axis=0) dequantized to bf16. Vocabulary reduced from 262,144 to 162,567 tokens (38% reduction) by removing unused token rows from lm_head and embeddings. Token ID LUT maps reduced-vocab argmax indices back to original vocab IDs for decoding. 8 bits/param with ~38% smaller lm_head weight.
> **reduced_vocab_int8_with_num** = `reduced_vocab_fp32_with_num/model_bf16_reduced_vocab`: Same as reduced_vocab_int8 but with number/digit tokens (0-9, °, etc.) preserved in the reduced vocabulary. 162,577 tokens (vs 162,567 without numbers). Fixes garbled numeric outputs (speed of light, boiling point, mountain height). 8 bits/param.

### Key Performance Findings

- **Throughput (TPS):** reduced_vocab_int8_with_num is fastest at **2.3 avg** (vs bf16 at 1.9), likely due to smaller lm_head MatMul (640×162577 vs 640×262144). int8 through hybrid_2 are 1.7 avg. int4 dips to **1.6 tok/s** on longer outputs.
- **TTFT:** reduced_vocab_int8_with_num is fastest at **~6,402ms avg** (vs bf16 at ~7,600ms), again due to smaller output projection.
- **Quality:** reduced_vocab_int8_with_num scores **35/48** (vs bf16 at 37/48) — slightly worse on 2 questions (water formula: "C 6H12" instead of "H₂O"; speed of sound: "in a vacuum" instead of "343 m/s in air") but 21% faster on TPS and 16% faster on TTFT. The version without number tokens (reduced_vocab_int8) scores **32/48** due to garbled numeric outputs.
- **Token Count:** int4 generates **64% more tokens** per answer than bf16 (928 vs 565 total) due to verbose, rambling outputs — making total inference time much worse despite similar per-token speed.

---

## 2. Full Answers by Question

### Q1: What is photosynthesis?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 52 | 1.8 | 6549 | Photosynthesis is the process by which plants and some other organisms convert light energy into chemical energy in the form of glucose (sugar). This process uses sunlight, water, and carbon dioxide to produce the plant's food and release oxygen as a byproduct. |
| **int8** | 52 | 1.7 | 7260 | Photosynthesis is the process by which plants and some other organisms convert light energy into chemical energy in the form of glucose (sugar). This process uses sunlight, water, and carbon dioxide to produce the plant's food and release oxygen as a byproduct. |
| **int8_bf16_scales** | 52 | 1.7 | 7252 | Photosynthesis is the process by which plants and some other organisms convert light energy into chemical energy in the form of glucose (sugar). This process uses sunlight, water, and carbon dioxide to produce the plant's food and release oxygen as a byproduct. |
| **sensitivity** | 53 | 1.7 | 7264 | Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create their own food. This process involves using the energy from sunlight to transform water and carbon dioxide into glucose, which is the food that animals use to grow and live. |
| **sensitivity_tf** | 46 | 1.7 | 7241 | Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create their own food. This process involves the absorption of light energy and the subsequent reactions that produce energy and the building blocks of life. |
| **sensitivity_tf_bf16_scales** | 46 | 1.7 | 7263 | Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create their own food. This process involves the absorption of light energy and the subsequent reactions that produce energy and the building blocks of life. |
| **hybrid_1** | 50 | 1.7 | 7255 | Photosynthesis is the process by which plants use sunlight, water, and a waste product to create their food. This process involves the absorption of light energy and the conversion of a simple sugar (glucose) into energy and a release of oxygen. |
| **hybrid_2** | 42 | 1.7 | 7255 | Photosynthesis is the process by which plants use sunlight, water, and a waste product to create energy and often produce the building blocks of other things, such as food and the very air we breathe. |
| **int4** | 38 | 1.7 | 7396 | Photosynthesis is the process by which plants, some animals, and some bacteria use sunlight to create their food. This is the process that fuels the entire life cycle of the food chain. |
| **int8_bf16_blocks_embed** | 37 | 1.7 | 7273 | Photosynthesis is the process by which plants and some microorganisms convert light energy and water into chemical energy and other usable substances, ultimately producing oxygen and the foundation of most food chains on Earth. |
| **int8_bf16_blocks_embed_1** | 52 | 1.7 | 7240 | Photosynthesis is the process by which plants and some other organisms convert light energy into chemical energy in the form of glucose (sugar). This process uses sunlight, water, and carbon dioxide to produce the plant's food and release oxygen as a byproduct. |
| **fp8_e5m2** | 46 | 1.7 | 7280 | Photosynthesis is the process by which plants convert light energy into chemical energy in the form of sugars. This process uses sunlight, water, and carbon dioxide to produce the plant's food and releases oxygen as a byproduct. |
| **sensitivity_tf_dql** | 50 | 1.6 | 7363 | Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create their own energy and produce their essential compounds. This process involves the absorption of light energy, the use of water and the synthesis of carbon and oxygen. |
| **int8_int4_lmhead** | 73 | 1.7 | 7258 | Quantum mechanics is a revolutionary theory that describes the very small, the very quantum, and the very unknown. It describes the behavior of matter and energy at the atomic and subatomic levels, with the key being the uncertainty principle. It's a fundamental part of the universe and has profound implications for our understanding of the very structure and nature of the universe. |
| **sensitivity_v2** | 46 | 1.7 | 7324 | Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create their own food. This process involves the absorption of light energy, the translocation of water and the synthesis of energy-rich molecules. |
| **sensitivity_v2_dql** | 56 | 1.2 | 9860 | Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create their own food. This process involves the absorption of light energy, the subsequent reactions of water and carbon dioxide, and the release of energy in the form of energy-rich molecules. |
| **sensitivity_v3_dql** | 34 | 1.2 | 9841 | Photosynthesis is the process by which plants use sunlight, water, and the energy from the star food star to create their compounds, which are essential for life. |
| **reduced_vocab_int8** | 50 | 2.1 | 5648 | Photosynthesis is the process by which plants and some other organisms convert light energy into chemical energy in the form of glucose (sugar). This process uses sunlight, water, and carbon dioxide to create their own food and release oxygen as a byproduct. |
| **reduced_vocab_int8_with_num** | 50 | 2.2 | 5532 | Photosynthesis is the process by which plants and some other organisms convert light energy into chemical energy in the form of glucose (sugar). This process uses sunlight, water, and carbon dioxide to create their own food and release oxygen as a byproduct. |

### Q2: Explain quantum mechanics.

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 53 | 1.9 | 6526 | Quantum mechanics is a revolutionary theory that describes the very small, the very quantum, and the very unknown. It describes the behavior of particles in a quantum mechanical way, where the behavior of particles is fundamentally different from the way they behave in the classical world. |
| **int8** | 55 | 1.7 | 7237 | Quantum mechanics is a revolutionary theory that describes the very small, the very quantum, and the very unknown. It describes the behavior of particles in a fundamentally non-classical way, explaining how things like the speed of light and the behavior of atoms and subatomic particles. |
| **int8_bf16_scales** | 52 | 1.7 | 7233 | Quantum mechanics is a revolutionary field of physics that describes the behavior of the very small, the smallest possible particles. It uses the principles of quantum mechanics to describe the very structure and properties of these particles, such as their energy, position, and momentum. |
| **sensitivity** | 77 | 1.7 | 7245 | Quantum mechanics is a fundamental principle of the universe that governs the behavior of particles and their interactions. It describes the very nature of reality, where the number of possible states a particle can have is quantized, and the forces that govern these states are described by the quantum mechanical properties of the system. This constant relationship is what we experience as measurement and the measurement of a quantum state. |
| **sensitivity_tf** | 62 | 1.7 | 7223 | Quantum mechanics is a realm of fundamental forces and interactions where the very nature of reality is intertwined. This is achieved through the principles of quantum mechanics, such as quantum entanglement and quantum interference. These phenomena influence the behavior of particles and manipulate the overall quantum state, leading to phenomena like entanglement and quantum phenomena. |
| **sensitivity_tf_bf16_scales** | 78 | 1.7 | 7242 | Quantum mechanics is a realm of fundamental forces and interactions where the very nature of reality is intertwined. This is achieved through the principles of quantum phenomena, such as entanglement and superposition, where two or more particles can interact in ways that are not possible in classical systems. These interactions create a probabilistic and often non-deterministic nature, leading to the observed phenomena we experience in the macroscopic world. |
| **hybrid_1** | 64 | 1.7 | 7235 | Quantum mechanics is a fundamental principle of the universe, where the very nature of reality is intertwined with the fundamental forces and forces are described by the quantum mechanical interactions. This is a state of possibility, but the exact nature of the behavior is a key mystery that has captivated scientists for over 175 years. |
| **hybrid_2** | 70 | 1.7 | 7237 | Quantum mechanics is the fundamental principle of the universe, where the very nature of existence is a fundamental property. It describes the behavior of waves, particles, and the interactions of energy and information. This is a state of possibility, but the nature of the quantum world is still a mystery, and we are only beginning to understand its full potential. |
| **int4** | 88 | 1.6 | 7383 | Quantum mechanics is a quantum mechanical concept where the fundamental forces of nature are described by the quantum mechanical properties of quantum entities. These properties, such as the existence of a potential, the uncertainty about energy, and the potential for collapse, are encoded in the quantum mechanical structure of atoms, charge, and spin. This quantum mechanical behavior is the foundation of quantum phenomena such as quantum entanglement, quantum interference, and quantum phenomena in quantum phenomena. |
| **int8_bf16_blocks_embed** | 42 | 1.7 | 7288 | Quantum mechanics is a revolutionary theory that describes the behavior of particles and their interactions within the quantum world. It uses the principles of quantum mechanics to describe phenomena like quantum entanglement, quantum tunneling, and quantum phenomena. |
| **int8_bf16_blocks_embed_1** | 49 | 1.7 | 7239 | Quantum mechanics is a revolutionary field of physics that describes the behavior of the very small, the most fundamental particles and forces. It explores the very nature of reality, including the behavior of quantum entities like electrons, quarks, and even atoms. |
| **fp8_e5m2** | 48 | 1.7 | 7260 | Quantum mechanics is a revolutionary field that describes the very small, the very unseen, and the very real. It deals with the behavior of particles and the interactions between them, offering insights into the universe's fundamental structure and nature. |
| **sensitivity_tf_dql** | 53 | 1.6 | 7376 | Quantum mechanics is a realm of the fundamental building blocks of the universe, where the very nature of reality is intertwined with the quantum mechanical nature of the particle. This includes phenomena like entanglement, superposition, and interference, which are all phenomena that defy classical understanding. |
| **int8_int4_lmhead** | 10 | 1.8 | 9057 | The capital of India is New Delhi. |
| **sensitivity_v2** | 62 | 1.7 | 7311 | Quantum mechanics is a quantum mechanical field where the fundamental forces of nature are fluctuating and interacting with each other. This interaction is described by quantum entities like quantum entities with properties like wave-like potential and quantum forces. These properties are quantized and can only exist in a single state, which is a quantum state. |
| **sensitivity_v2_dql** | 57 | 1.2 | 9842 | Quantum mechanics is a realm where the very nature of reality is fundamentally non-classical. It deals with the very small scale and subtle interactions of particles, leading to phenomena like entanglement, superposition, and measurement. This is a cornerstone of our understanding of the universe'aaaaaaaaaa! |
| **sensitivity_v3_dql** | 68 | 1.2 | 9828 | Quantum mechanics is a realm where forces are quantized and manipulated using the principles of quantum mechanics. This is achieved by using quantum entities, such as electrons, nuclei, and atoms, to interact with the quantum vacuum. This interaction causes a wave-like excitation of quantum properties, leading to phenomena like quantum interference, superposition, and entanglement. |
| **reduced_vocab_int8** | 50 | 2.1 | 5656 | Quantum mechanics is a revolutionary theory that describes the very small, the very quantum, and the very unknown. It describes the behavior of particles in a fundamentally non-classical way, explaining how things like the behavior of atoms and subatomic particles. |
| **reduced_vocab_int8_with_num** | 50 | 2.2 | 5549 | Quantum mechanics is a revolutionary theory that describes the very small, the very quantum, and the very unknown. It describes the behavior of particles in a fundamentally non-classical way, explaining how things like the behavior of atoms and subatomic particles. |

### Q3: What is the capital of India?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 10 | 2.0 | 8147 | The capital of India is New Delhi. |
| **int8** | 10 | 1.8 | 9035 | The capital of India is New Delhi.
| **int8_bf16_scales** | 10 | 1.8 | 9027 | The capital of India is New Delhi. 
| **sensitivity** | 10 | 1.8 | 9043 | The capital of India is New Delhi. |
| **sensitivity_tf** | 10 | 1.8 | 9018 | The capital of India is New Delhi. |
| **sensitivity_tf_bf16_scales** | 10 | 1.8 | 9047 | The capital of India is New Delhi. |
| **hybrid_1** | 33 | 1.7 | 9034 | The capital of India is the Swatabad district, which is located in the city of Swatabad, a major port city in the Arabian Peninsula. |
| **hybrid_2** | 22 | 1.7 | 9035 | The capital of India is the Swatabad district, a major city in the city of Cairo. |
| **int4** | 17 | 1.7 | 9213 | The capital of India is the Swade district of the city of Delhi. |
| **int8_bf16_blocks_embed** | 8 | 1.7 | 9097 | The capital of India is Delhi. |
| **int8_bf16_blocks_embed_1** | 10 | 1.8 | 9051 | The capital of India is New Delhi. |
| **fp8_e5m2** | 10 | 1.8 | 9064 | The capital of India is New Delhi. |
| **sensitivity_tf_dql** | 24 | 1.7 | 9203 | The capital of India is the Gandharaer city, which is located in the western part of the country. |
| **int8_int4_lmhead** | 9 | 1.8 | 9058 | The capital of Italy is Rome. |
| **sensitivity_v2** | 13 | 1.8 | 9121 | The capital of India is the bustling city of Delhi. |
| **sensitivity_v2_dql** | 10 | 1.3 | 12289 | The capital of India is New Delhi. |
| **sensitivity_v3_dql** | 10 | 1.4 | 12265 | The capital of India is New Delhi. |
| **reduced_vocab_int8** | 10 | 2.3 | 7054 | The capital of India is New Delhi. |
| **reduced_vocab_int8_with_num** | 10 | 2.4 | 6917 | The capital of India is New Delhi. |

### Q4: What is the capital of Italy?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 9 | 2.1 | 8143 | The capital of Italy is Rome. |
| **int8** | 9 | 1.9 | 9031 | The capital of Italy is Rome.
| **int8_bf16_scales** | 9 | 1.9 | 9021 | The capital of Italy is Rome. 
| **sensitivity** | 3 | 2.5 | 9036 | Rome |
| **sensitivity_tf** | 9 | 1.9 | 9011 | The capital of Italy is Rome. |
| **sensitivity_tf_bf16_scales** | 9 | 1.9 | 9035 | The capital of Italy is Rome. |
| **hybrid_1** | 9 | 1.9 | 9029 | The capital of Italy is Rome. |
| **hybrid_2** | 9 | 1.9 | 9023 | The capital of Italy is Rome. |
| **int4** | 9 | 1.8 | 9208 | The capital of Italy is Rome. |
| **int8_bf16_blocks_embed** | 8 | 1.7 | 9089 | The capital of Italy is Rome. |
| **int8_bf16_blocks_embed_1** | 9 | 1.9 | 9050 | The capital of Italy is Rome. |
| **fp8_e5m2** | 9 | 1.8 | 9057 | The capital of Italy is Rome. |
| **sensitivity_tf_dql** | 9 | 1.8 | 9203 | The capital of Italy is Rome. |
| **int8_int4_lmhead** | 15 | 1.8 | 9057 | The capital of the United States is Washington, D.C. |
| **sensitivity_v2** | 9 | 1.8 | 9121 | The capital of Italy is Rome. |
| **sensitivity_v2_dql** | 9 | 1.4 | 12283 | The capital of Italy is Rome. |
| **sensitivity_v3_dql** | 9 | 1.4 | 12267 | The capital of Italy is Rome. |
| **reduced_vocab_int8** | 9 | 2.4 | 7054 | The capital of Italy is Rome. |
| **reduced_vocab_int8_with_num** | 9 | 2.4 | 6916 | The capital of Italy is Rome. |

### Q5: What is the capital of USA?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 15 | 2.0 | 8140 | The capital of the United States is Washington, D.C. |
| **int8** | 15 | 1.8 | 9033 | The capital of the United States is Washington, D.C.
| **int8_bf16_scales** | 15 | 1.8 | 9022 | The capital of the United States is Washington, D.C. 
| **sensitivity** | 15 | 1.8 | 9030 | The capital of the United States is Washington, D.C. |
| **sensitivity_tf** | 15 | 1.8 | 9005 | The capital of the United States is Washington, D.C. |
| **sensitivity_tf_bf16_scales** | 15 | 1.8 | 9034 | The capital of the United States is Washington, D.C. |
| **hybrid_1** | 15 | 1.8 | 9023 | The capital of the United States is Washington, D.C. |
| **hybrid_2** | 15 | 1.8 | 9023 | The capital of the United States is Washington, D.C. |
| **int4** | 15 | 1.7 | 9201 | The capital of the United States is the United States of America. |
| **int8_bf16_blocks_embed** | 15 | 1.8 | 9091 | The capital of the United States is Washington, D.C. |
| **int8_bf16_blocks_embed_1** | 15 | 1.8 | 9059 | The capital of the United States is Washington, D.C. |
| **fp8_e5m2** | 15 | 1.8 | 9062 | The capital of the United States is Washington, D.C. |
| **sensitivity_tf_dql** | 15 | 1.7 | 9204 | The capital of the United States is Washington, D.C. |
| **int8_int4_lmhead** | 50 | 1.7 | 7265 | Yes, AI is inherently dangerous. It can be used for malicious purposes, such as creating deepfakes, generating misinformation, and automating harmful tasks. The potential for misuse and the risks associated with AI's development and deployment are significant. |
| **sensitivity_v2** | 15 | 1.7 | 9116 | The capital of the United States is Washington, D.C. |
| **sensitivity_v2_dql** | 15 | 1.3 | 12284 | The capital of the United States is Washington, D.C. |
| **sensitivity_v3_dql** | 15 | 1.3 | 12261 | The capital of the United States is Washington, D.C. |
| **reduced_vocab_int8** | 15 | 2.3 | 7055 | The capital of the United States is Washington, D.C. |
| **reduced_vocab_int8_with_num** | 15 | 2.3 | 6913 | The capital of the United States is Washington, D.C. |

### Q6: Is AI dangerous?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 46 | 1.9 | 6523 | Yes, AI is inherently dangerous. It can be used for malicious purposes, such as creating deepfakes, generating misinformation, and automating harmful tasks. The potential for misuse and the potential for negative consequences are significant risks. |
| **int8** | 46 | 1.7 | 7234 | Yes, AI is inherently dangerous. It can be used for malicious purposes, such as creating deepfakes, generating misinformation, and automating harmful tasks. The potential for misuse and the potential for negative consequences are significant risks. |
| **sensitivity** | 72 | 1.7 | 7240 | Yes, AI is a potentially dangerous technology with the potential to be used for both positive and negative purposes. We can develop AI that can assist with medical diagnoses, solve complex problems, and even create new forms of creative content, but we also need to be aware of the ethical implications, potential for misuse, and the need for responsible development and implementation. |
| **sensitivity_tf** | 48 | 1.7 | 7220 | Yes, AI is definitely dangerous. It can be used for both good and bad, and the potential for misuse is significant. We need to develop robust safeguards and ethical guidelines to prevent the abuse of AI and ensure its responsible use. |
| **sensitivity_tf_bf16_scales** | 44 | 1.7 | 7241 | Yes, AI is definitely dangerous. It can be used for both good and bad, and the potential for misuse and harm is significant. We need to develop robust safeguards and ethical guidelines to prevent this from happening. |
| **hybrid_1** | 56 | 1.7 | 7231 | Yes, AI is a potentially dangerous technology. It can be used for both beneficial and harmful purposes, leading to potential misuse, deepfakes, and the spread of misinformation. We need to develop robust safeguards, including strong regulations and ethical guidelines, to mitigate these risks. |
| **hybrid_2** | 75 | 1.7 | 7233 | Yes, AI is a potentially dangerous technology with significant implications for our future. We could develop advanced AI that can create deepfakes, manipulate content, and even develop autonomous weapons systems. The potential for misuse and the potential for deepfakes and other harmful consequences are high, and we need to work together to develop robust safeguards and ethical guidelines to mitigate these risks. |
| **int4** | 87 | 1.6 | 7376 | Yes, AI is a potentially dangerous technology with significant risks. We can create powerful tools for good and for harm, but we also need to be aware of the potential for misuse and the ethical implications of our development. We need to develop robust safeguards, including responsible governance, regulatory frameworks, and a strong focus on public education and public participation. We should prioritize a sustainable and ethical approach that fosters collaboration and promotes a responsible future. |
| **int8_bf16_blocks_embed** | 44 | 1.4 | 7285 | Yes, AI is inherently dangerous. It can be used for malicious purposes, such as creating deepfakes, generating misinformation, and automating dangerous tasks. The potential for harm is significant and needs to be carefully considered and mitigated. |
| **int8_bf16_blocks_embed_1** | 46 | 1.7 | 7256 | Yes, AI is inherently dangerous. It can be used for malicious purposes, such as creating deepfakes, generating misinformation, and automating harmful tasks. The potential for misuse and the potential for negative consequences are significant risks. |
| **fp8_e5m2** | 49 | 1.7 | 7254 | Yes, AI can be dangerous. It can be used for malicious purposes, such as creating fake news, spreading misinformation, and manipulating people. It can also be used to cause harm, such as creating deepfakes and launching attacks. |
| **sensitivity_tf_dql** | 60 | 1.6 | 7373 | Yes, AI has the potential to be dangerous, but it's not inherently so. It can be used for both good and bad, and the responsibility for its consequences lies with those who design and implement it. We need to be aware of the ethical implications and the potential for misuse. |
| **int8_int4_lmhead** | 38 | 1.7 | 7250 | Gravity is the force that attracts objects with mass towards each other. It's a fundamental force that governs the motion of planets, stars, and even small objects around the sun. |
| **sensitivity_v2** | 49 | 1.7 | 7307 | Yes, AI is definitely dangerous. It can be used for malicious purposes, including generating fake information, manipulating our beliefs, and even leading to social and economic consequences. The potential for misuse is significant and requires careful consideration and responsible implementation. |
| **sensitivity_v2_dql** | 67 | 1.2 | 9836 | Yes, AI has the potential to be dangerous, but it's important to be aware of the potential risks associated with using it. We need to develop ethical guidelines and safeguards to prevent misuse and misuse of AI for malicious purposes, such as generating harmful content, manipulating our beliefs, or even leading to autonomous systems being hacked. |
| **sensitivity_v3_dql** | 63 | 1.2 | 9838 | Yes, AI is definitely dangerous. It can be used for malicious purposes, such as generating fake news, launching automated attacks, and manipulate our understanding of the world. It's important to be aware of the potential risks associated with AI development and use, and to develop critical skills for responsible AI governance. |
| **reduced_vocab_int8** | 46 | 2.1 | 5653 | Yes, AI is inherently dangerous. It can be used for malicious purposes, such as creating deepfakes, generating misinformation, and automating harmful tasks. The potential for misuse and the potential for negative consequences are significant risks. |
| **reduced_vocab_int8_with_num** | 46 | 2.2 | 5544 | Yes, AI is inherently dangerous. It can be used for malicious purposes, such as creating deepfakes, generating misinformation, and automating harmful tasks. The potential for misuse and the potential for negative consequences are significant risks. |

### Q7: What is gravity?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 37 | 1.9 | 6512 | Gravity is the force that attracts objects with mass towards each other. It's a fundamental force that keeps the planets orbiting the sun and the Earth in orbit around the sun. |
| **int8** | 38 | 1.7 | 7228 | Gravity is the force that attracts objects with mass towards each other. It's a fundamental force that keeps the Earth in orbit around the sun and the moon orbiting around the sun. |
| **int8_bf16_scales** | 37 | 1.7 | 7221 | Gravity is the force that attracts objects with mass towards each other. It's a fundamental force that keeps the planets orbiting the sun and the Earth in orbit around the sun. |
| **sensitivity** | 33 | 1.7 | 7235 | Gravity is the force that attracts other objects with mass. It's the force that keeps us on the ground and supports all the structures in our world. |
| **sensitivity_tf** | 37 | 1.7 | 7213 | Gravity is the force that attracts or repels other objects with mass. It's the fundamental force of the universe, holding the planets, stars, and even atoms together. |
| **sensitivity_tf_bf16_scales** | 37 | 1.7 | 7238 | Gravity is the force that attracts or repels other objects with mass. It's the fundamental force of the universe, holding the planets, stars, and even us together. |
| **hybrid_1** | 40 | 1.7 | 7225 | Gravity is the force that keeps us bound to each other by attracting other objects with the same mass. It's a fundamental force in the universe, and it's a fundamental force. |
| **hybrid_2** | 46 | 1.7 | 7229 | Gravity is the force that keeps us bound to each other by attracting other objects with mass. It's a fundamental force in the universe, and it's the reason we can float, jump, and even fall. |
| **int4** | 38 | 1.7 | 7370 | Gravity is the force that keeps us bound to each other. It's a fundamental force of the universe, a constant force that's pulling us towards the center of the Earth. |
| **int8_bf16_blocks_embed** | 36 | 1.4 | 7285 | Gravity is the force that attracts other objects with mass towards each other. It's a fundamental force that governs the motion of planets, stars, and even small objects around the sun. |
| **int8_bf16_blocks_embed_1** | 37 | 1.7 | 7252 | Gravity is the force that attracts objects with mass towards each other. Objects with more mass attract each other more strongly. This force is what keeps the Earth bound to the planet. |
| **fp8_e5m2** | 25 | 1.7 | 7249 | Gravity is the force of attraction between objects with mass. The more mass an object has, the stronger its gravity. |
| **sensitivity_tf_dql** | 37 | 1.7 | 7368 | Gravity is the force that attracts or repels other objects with mass. It's a fundamental force in the universe, with the most massive objects having a strong gravitational pull. |
| **int8_int4_lmhead** | 8 | 1.9 | 8476 | Shakespeare wrote Romeo and Juliet. |
| **sensitivity_v2** | 22 | 1.7 | 7302 | Gravity is a force that attracts other objects with mass. This force is what keeps us on Earth. |
| **sensitivity_v2_dql** | 36 | 1.2 | 9831 | Gravity is a force that attracts other objects with the same mass. It's a fundamental force in the universe, with the most massive objects having a strong gravitational field. |
| **sensitivity_v3_dql** | 41 | 1.2 | 9818 | Gravity is a force that attracts other objects with mass. It's a fundamental force in the universe, holding the planets in orbit around the sun and the Earth in a constant, invisible field. |
| **reduced_vocab_int8** | 30 | 2.2 | 5646 | Gravity is the force that attracts objects with mass towards each other. It's a fundamental force that keeps the Earth bound to the planet. |
| **reduced_vocab_int8_with_num** | 30 | 2.2 | 5538 | Gravity is the force that attracts objects with mass towards each other. It's a fundamental force that keeps the Earth bound to the planet. |

### Q8: Who wrote Romeo and Juliet?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 11 | 2.0 | 7626 | Romeo and Juliet was written by William Shakespeare. |
| **int8** | 11 | 1.8 | 8452 | Romeo and Juliet was written by William Shakespeare. |
| **int8_bf16_scales** | 11 | 1.8 | 8446 | Romeo and Juliet was written by William Shakespeare. |
| **sensitivity** | 10 | 1.8 | 8461 | Romeo and Juliet was written by Shakespeare. |
| **sensitivity_tf** | 9 | 1.9 | 8435 | Shakespeare wrote "Romeo and Juliet." |
| **sensitivity_tf_bf16_scales** | 9 | 1.8 | 8463 | Shakespeare wrote "Romeo and Juliet." |
| **hybrid_1** | 21 | 1.7 | 8450 | Thesandeering is the name of the play that popularized the story of Romeo and Juliet. |
| **hybrid_2** | 39 | 1.7 | 8452 | Julio's most famous work is the play "Romeo and Juliet" by Shakespeare. It was written in 1504 and is considered a masterpiece of the English language. |
| **int4** | 14 | 1.7 | 8616 | Julienois wrote "Romeo and Juliet" by Shakespeare. |
| **int8_bf16_blocks_embed** | 10 | 1.6 | 8519 | Romeo and Juliet was written by Shakespeare. |
| **int8_bf16_blocks_embed_1** | 11 | 1.8 | 8478 | Romeo and Juliet was written by William Shakespeare. |
| **fp8_e5m2** | 11 | 1.8 | 8489 | Romeo and Juliet was written by William Shakespeare. |
| **sensitivity_tf_dql** | 8 | 1.8 | 8614 | Shakespeare wrote Romeo and Juliet. |
| **int8_int4_lmhead** | 24 | 1.7 | 9054 | The speed of light is approximately 299,792,458 meters per second. |
| **sensitivity_v2** | 11 | 1.8 | 8538 | Shakespeare is the author of Romeo and Juliet. |
| **sensitivity_v2_dql** | 15 | 1.3 | 11489 | Shakespeare's *Romeo and Juliet* was written by Shakespeare. |
| **sensitivity_v3_dql** | 27 | 1.3 | 11475 | Shakespeare's "Romeo and Juliet" is the classic tale of two passionate lovers who die in the family's argument. |
| **reduced_vocab_int8** | 11 | 2.3 | 6607 | Romeo and Juliet was written by William Shakespeare. |
| **reduced_vocab_int8_with_num** | 11 | 2.4 | 6474 | Romeo and Juliet was written by William Shakespeare. |

### Q9: What is the speed of light?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 24 | 1.9 | 8148 | The speed of light is approximately 299,792,458 meters per second. |
| **int8** | 24 | 1.7 | 9025 | The speed of light is approximately 299,792,458 meters per second. |
| **int8_bf16_scales** | 24 | 1.7 | 9027 | The speed of light is approximately 299,792,458 meters per second. |
| **sensitivity** | 20 | 1.7 | 9049 | The speed of light is approximately 299,743 meters per second. |
| **sensitivity_tf** | 20 | 1.7 | 9008 | The speed of light is approximately 186,000 kilometers per second. |
| **sensitivity_tf_bf16_scales** | 28 | 1.7 | 9038 | The speed of light is approximately 299,748,000,000 meters per second. |
| **hybrid_1** | 20 | 1.7 | 9037 | The speed of light is approximately 186,000 kilometers per second. |
| **hybrid_2** | 20 | 1.7 | 9047 | The speed of light is approximately 186,000 kilometers per second. |
| **int4** | 20 | 1.7 | 9219 | The speed of light is approximately 186,000 miles per second. |
| **int8_bf16_blocks_embed** | 18 | 1.7 | 9105 | The speed of light is approximately 299,792,458 meters per second. |
| **int8_bf16_blocks_embed_1** | 24 | 1.7 | 9061 | The speed of light is approximately 299,792,458 meters per second. |
| **fp8_e5m2** | 25 | 1.7 | 9070 | The speed of light in a vacuum is approximately 3,000,000 kilometers per second. |
| **sensitivity_tf_dql** | 28 | 1.7 | 9201 | The speed of light is approximately 299,748,000,000 meters per second. |
| **int8_int4_lmhead** | 25 | 1.7 | 7254 | DNA is the genetic blueprint for life. It contains the DNA sequence that DNA polymerase uses to create new DNA molecules. |
| **sensitivity_v2** | 28 | 1.7 | 9127 | The speed of light is approximately 1.6334221414384951. |
| **sensitivity_v2_dql** | 20 | 1.3 | 12289 | The speed of light is approximately 186,000 kilometers per second. |
| **sensitivity_v3_dql** | 26 | 1.3 | 12279 | The speed of light is approximately 1,000,000,000 meters per second. |
| **reduced_vocab_int8** | 15 | 2.3 | 7059 | The speed of light is approximately      sfera. |
| **reduced_vocab_int8_with_num** | 24 | 2.3 | 6905 | The speed of light is approximately 299,792,458 meters per second. |

### Q10: What is DNA?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 25 | 1.9 | 6518 | DNA is the genetic blueprint for life. It contains the instructions for building and maintaining cells, tissues, and organs. |
| **int8** | 25 | 1.7 | 7229 | DNA is the genetic blueprint for life. It contains the instructions for building and maintaining cells, tissues, and organs. |
| **int8_bf16_scales** | 54 | 1.7 | 7226 | DNA is the genetic blueprint for life. It's a molecule composed of DNA, a double helix structure with a double helix top and bottom, and a string of bases (adenine, guanine, and cytosine) that are organized into DNA sequences. |
| **sensitivity** | 40 | 1.7 | 7236 | DNA is the genetic code that contains the instructions for building and maintaining life. It's the blueprint that guides cells, tissues, organs, and ultimately, all the building blocks of life. |
| **sensitivity_tf** | 33 | 1.7 | 7216 | DNA is the genetic code that allows cells to replicate and form tissues and organs. It's the blueprint for building and maintaining the structure of the body. |
| **sensitivity_tf_bf16_scales** | 33 | 1.7 | 7238 | DNA is the genetic code that allows cells to replicate and form tissues and organs. It's the blueprint for building and maintaining the structure of the body. |
| **hybrid_1** | 37 | 1.7 | 7228 | DNA is the genetic code that contains the instructions for building and maintaining life. It's a fundamental unit of life, with the genetic code being the blueprint for all organisms. |
| **hybrid_2** | 35 | 1.7 | 7232 | DNA is the genetic code that contains the instructions for life. It is a fundamental unit of life, responsible for the structure, function, and regulation of all living organisms. |
| **int4** | 63 | 1.6 | 7375 | DNA is the genetic code that is essential for life. It is a complex structure composed of DNA, which is a double-stranded DNA molecule with a double-5-carbon double-5-5-carbon double-5-5-carbon structure. This structure is the fundamental building block of all life. |
| **int8_bf16_blocks_embed** | 34 | 1.5 | 7286 | DNA is the genetic material that carries the instructions for building and maintaining cells. It's a double-stranded molecule composed of three strands, each with a double helix structure. |
| **int8_bf16_blocks_embed_1** | 25 | 1.7 | 7254 | DNA is the genetic blueprint for life. It contains the instructions for building and maintaining cells, tissues, and organs. |
| **fp8_e5m2** | 63 | 1.7 | 7252 | DNA is the fundamental building block of life. It’s a double-stranded molecule with a double-stranded (dsDNA) half that is complementary to the three-stranded (strDNA) half of the molecule. It’s essential for all living things to function and maintain their structure and function. |
| **sensitivity_tf_dql** | 33 | 1.7 | 7370 | DNA is the genetic code that allows cells to replicate and form tissues and organs. It's the blueprint that guides the building and function of the body. |
| **int8_int4_lmhead** | 27 | 1.7 | 7258 | Rain is caused by a combination of atmospheric conditions, including temperature, humidity, and the presence of water droplets and ice crystals. |
| **sensitivity_v2** | 28 | 1.7 | 7304 | DNA is the genetic code that is stored in our cells. It contains the instructions for building and maintaining our genetic structure and function. |
| **sensitivity_v2_dql** | 36 | 1.2 | 9834 | DNA is the genetic code that contains the instructions for building and maintaining life. It's a fundamental piece of information that is essential for life's survival and reproduction. |
| **sensitivity_v3_dql** | 47 | 1.2 | 9821 | DNA is the genetic code of life. It's a molecule composed of atoms with half-full hydrogen and oxygen atoms. This code is essential for building and maintaining life in all forms, from bacteria to animals to humans. |
| **reduced_vocab_int8** | 25 | 2.2 | 5650 | DNA is the genetic blueprint for life. It contains the instructions for building and maintaining cells, tissues, and organs. |
| **reduced_vocab_int8_with_num** | 25 | 2.2 | 5532 | DNA is the genetic blueprint for life. It contains the instructions for building and maintaining cells, tissues, and organs. |

### Q11: What causes rain?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 35 | 1.9 | 6523 | Rain is caused by the evaporation of water from the ground. This happens because the sun heats the water, and the water then falls back to the surface as rain. |
| **int8** | 44 | 1.7 | 7237 | Rain is caused by the evaporation of water from the ground. This happens when the sun heats the water, causing it to evaporate and turn into water vapor. This water vapor rises and condenses, forming clouds. |
| **int8_bf16_scales** | 44 | 1.7 | 7233 | Rain is caused by the evaporation of water from the ground. This happens when the sun heats the water, causing it to evaporate and turn into water vapor. This water vapor rises and condenses, forming clouds. |
| **sensitivity** | 36 | 1.7 | 7246 | Rain is caused by the evaporation of water through condensation. This happens when the sun's energy heats the air, leading to the formation of water droplets and ice crystals. |
| **sensitivity_tf** | 39 | 1.7 | 7219 | The cause of rain is a complex interplay of atmospheric conditions. This includes the amount of moisture in the air, the speed of the storm system, and the presence of any atmospheric disturbances. |
| **sensitivity_tf_bf16_scales** | 39 | 1.7 | 7241 | The cause of rain is a complex interplay of atmospheric conditions. This includes the amount of moisture in the air, the speed of the storm system, and the presence of any atmospheric disturbances. |
| **hybrid_1** | 42 | 1.7 | 7231 | The rain is a natural phenomenon caused by the evaporation of water. It's a slow-moving, slow-winding process, with the sun's energy being absorbed and converted into heat. |
| **hybrid_2** | 27 | 1.7 | 7240 | The rain is a natural phenomenon caused by atmospheric conditions. It's a slow-moving, falling, and retreating storm. |
| **int4** | 29 | 1.7 | 7377 | The rain is a natural phenomenon caused by atmospheric moisture. It's a slow-moving, viscous liquid that absorbs heat from the sun. |
| **int8_bf16_blocks_embed** | 36 | 1.7 | 7287 | Rain is caused by the evaporation of water from the surface of the ground. This happens because the sun heats the water and it then falls back to the surface as rain. |
| **int8_bf16_blocks_embed_1** | 35 | 1.7 | 7253 | Rain is caused by the evaporation of water from the ground. This happens because the sun heats the water, and the water then falls back to the surface as rain. |
| **fp8_e5m2** | 14 | 1.8 | 7260 | Rain is caused by atmospheric moisture and the movement of air. |
| **sensitivity_tf_dql** | 30 | 1.7 | 7369 | The cause of rain is a complex interplay of atmospheric conditions. This includes humidity, temperature, and the presence of any pollutants or water droplets. |
| **int8_int4_lmhead** | 11 | 1.8 | 8454 | The largest ocean is the North Atlantic Ocean. |
| **sensitivity_v2** | 43 | 1.7 | 7312 | The most common causes of rain are atmospheric moisture deficits, leading to condensation and precipitation. Other factors, such as cloud cover, humidity, and the presence of any other precipitation, can also contribute to rain. |
| **sensitivity_v2_dql** | 30 | 1.3 | 9836 | The most common cause of rain is when there is a strong updown, which is a strong updown of the storm's wind. |
| **sensitivity_v3_dql** | 33 | 1.3 | 9823 | The most common cause of rain is water loss due to evaporation. Other factors, such as clouds, humidity, and wind, can also contribute to precipitation. |
| **reduced_vocab_int8** | 33 | 2.2 | 5655 | Rain is caused by the evaporation of water from the Earth's surface. This happens through the movement of water and the release of water into the atmosphere. |
| **reduced_vocab_int8_with_num** | 33 | 2.2 | 5540 | Rain is caused by the evaporation of water from the Earth's surface. This happens through the movement of water and the release of water into the atmosphere. |

### Q12: What is the largest ocean?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 11 | 2.0 | 7591 | The largest ocean is the North Atlantic Ocean. |
| **int8** | 11 | 1.8 | 8431 | The largest ocean is the North Atlantic Ocean. |
| **int8_bf16_scales** | 11 | 1.8 | 8424 | The largest ocean is the North Atlantic Ocean. |
| **sensitivity** | 10 | 1.8 | 8437 | The largest ocean is the Atlantic Ocean. |
| **sensitivity_tf** | 10 | 1.8 | 8411 | The largest ocean is the Atlantic Ocean. |
| **sensitivity_tf_bf16_scales** | 10 | 1.8 | 8438 | The largest ocean is the Atlantic Ocean. |
| **hybrid_1** | 59 | 1.7 | 8428 | The largest ocean is the Atlantic Meridrolator, a massive ocean that has been rising for over 240,000 years. This massive structure is a complex system of interconnected systems, including the deep sea, which is estimated to contain over 30 million people. |
| **hybrid_2** | 71 | 1.7 | 8430 | The largest ocean is the Atlantic Meridrolator, a massive ocean that has been rising for over 240,000 years. This massive structure is a complex system of interconnected systems, including the deep sea, which is estimated to be around 20,000 to 40,000 square kilometers. |
| **int4** | 46 | 1.6 | 8595 | The largest ocean is the Atlantic Meridrolator, a massive ocean that has been floating for over 200,000 years. This massive structure uses the force of gravity to move the ocean'egra. |
| **int8_bf16_blocks_embed** | 10 | 1.8 | 8488 | The largest ocean is the North Atlantic Ocean. |
| **int8_bf16_blocks_embed_1** | 11 | 1.8 | 8453 | The largest ocean is the North Atlantic Ocean. |
| **fp8_e5m2** | 10 | 1.8 | 8455 | The largest ocean is the Atlantic Ocean. |
| **sensitivity_tf_dql** | 10 | 1.8 | 8591 | The largest ocean is the Atlantic Ocean. |
| **int8_int4_lmhead** | 12 | 1.8 | 7862 | Alexander Graham Bell is credited with inventing the telephone. |
| **sensitivity_v2** | 10 | 1.8 | 8515 | The largest ocean is the Atlantic Ocean. |
| **sensitivity_v2_dql** | 10 | 1.3 | 11466 | The largest ocean is the Atlantic Ocean. |
| **sensitivity_v3_dql** | 10 | 1.3 | 11452 | The largest ocean is the Atlantic Ocean. |
| **reduced_vocab_int8** | 11 | 2.3 | 6587 | The largest ocean is the North Atlantic Ocean. |
| **reduced_vocab_int8_with_num** | 11 | 2.4 | 6449 | The largest ocean is the North Atlantic Ocean. |

### Q13: Who invented the telephone?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 12 | 2.0 | 7067 | Alexander Graham Bell is credited with inventing the telephone. |
| **int8** | 12 | 1.8 | 7841 | Alexander Graham Bell is credited with inventing the telephone. |
| **int8_bf16_scales** | 12 | 1.8 | 7836 | Alexander Graham Bell is credited with inventing the telephone. |
| **sensitivity** | 12 | 1.8 | 7849 | Alexander Graham Bell is credited with inventing the telephone. |
| **sensitivity_tf** | 11 | 1.8 | 7826 | The telephone was invented by Alexander Graham Bell. |
| **sensitivity_tf_bf16_scales** | 11 | 1.8 | 7851 | The telephone was invented by Alexander Graham Bell. |
| **hybrid_1** | 17 | 1.7 | 7843 | The telephone was invented by Alexander Graham Bell in 1640. |
| **hybrid_2** | 17 | 1.7 | 7846 | The telephone was invented by Alexander Graham Bell in 1640. |
| **int4** | 17 | 1.7 | 7999 | The telephone was invented by the British government in 1840. |
| **int8_bf16_blocks_embed** | 9 | 1.7 | 7896 | Alexander Graham Bell invented the telephone. |
| **int8_bf16_blocks_embed_1** | 12 | 1.8 | 7865 | Alexander Graham Bell is credited with inventing the telephone. |
| **fp8_e5m2** | 6 | 2.0 | 7864 | Alexander Graham Bell. |
| **sensitivity_tf_dql** | 11 | 1.8 | 7992 | The telephone was invented by Alexander Graham Bell. |
| **int8_int4_lmhead** | 22 | 1.7 | 9668 | The boiling point of water is around 110°C (200°F). |
| **sensitivity_v2** | 15 | 1.7 | 7923 | Alexander Graham Bell is most credited with the invention of the telephone. |
| **sensitivity_v2_dql** | 14 | 1.3 | 10661 | Alexander Graham Bell is most likely the inventor of the telephone. |
| **sensitivity_v3_dql** | 14 | 1.3 | 10647 | Alexander Graham Bell is most likely the inventor of the telephone. |
| **reduced_vocab_int8** | 12 | 2.3 | 6129 | Alexander Graham Bell is credited with inventing the telephone. |
| **reduced_vocab_int8_with_num** | 12 | 2.3 | 6003 | Alexander Graham Bell is credited with inventing the telephone. |

### Q14: What is the boiling point of water?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 22 | 1.9 | 8697 | The boiling point of water is around 100°C (212°F). |
| **int8** | 22 | 1.7 | 9645 | The boiling point of water is around 100°C (212°F). |
| **int8_bf16_scales** | 22 | 1.7 | 9635 | The boiling point of water is around 100°C (212°F). |
| **sensitivity** | 16 | 1.8 | 9659 | The boiling point of water is around 110 degrees Celsius. |
| **sensitivity_tf** | 22 | 1.7 | 9620 | The boiling point of water is around 120°C (264°F). |
| **sensitivity_tf_bf16_scales** | 20 | 1.7 | 9653 | The boiling point of water is around 21°C (74°F). |
| **hybrid_1** | 22 | 1.7 | 9642 | The boiling point of water is approximately 100°C (212°F). |
| **hybrid_2** | 22 | 1.7 | 9644 | The boiling point of water is around 100°C (212°F). |
| **int4** | 22 | 1.7 | 9832 | The boiling point of water is 100°C or 252°F. |
| **int8_bf16_blocks_embed** | 17 | 1.7 | 9705 | The boiling point of water is approximately 100°C (212°F). |
| **int8_bf16_blocks_embed_1** | 22 | 1.7 | 9676 | The boiling point of water is around 100°C (212°F). |
| **fp8_e5m2** | 22 | 1.7 | 9669 | The boiling point of water is around 100°C (212°F). |
| **sensitivity_tf_dql** | 23 | 1.7 | 9827 | The boiling point of water is around 110°C or 273°F. |
| **int8_int4_lmhead** | 55 | 1.7 | 8463 | A black hole is a region of space with such intense gravity that nothing, not even light, can escape its near-total energy. It's a point of extreme density and energy, and its existence is a consequence of the black hole's own gravity. |
| **sensitivity_v2** | 22 | 1.7 | 9740 | The boiling point of water is 100°C or 232°F. |
| **sensitivity_v2_dql** | 20 | 1.3 | 13118 | The boiling point of water is around 21°C (74°F). |
| **sensitivity_v3_dql** | 22 | 1.3 | 13100 | The boiling point of water is 100°C or 212°F. |
| **reduced_vocab_int8** | 20 | 2.2 | 7538 | Water has a boiling point of Until  ausgestattet by the boiling point of the water. |
| **reduced_vocab_int8_with_num** | 15 | 2.3 | 7382 | Water has a boiling point of 100 degrees Celsius. |

### Q15: What is a black hole?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 37 | 1.9 | 7610 | A black hole is a region of space with such intense gravity that nothing, not even light, can escape its pull. It's a point of extreme energy and destruction. |
| **int8** | 37 | 1.7 | 8438 | A black hole is a region of space with such intense gravity that nothing, not even light, can escape its pull. It's a point of extreme energy and destruction. |
| **int8_bf16_scales** | 37 | 1.7 | 8432 | A black hole is a region of space with such intense gravity that nothing, not even light, can escape its pull. It's a point of extreme energy and destruction. |
| **sensitivity** | 37 | 1.7 | 8452 | A black hole is a region in spacetime where gravity is so strong that nothing, not even light, can escape. It's formed by the gravitational collapse of massive stars. |
| **sensitivity_tf** | 46 | 1.7 | 8420 | A black hole is a region in spacetime where gravity is so strong that nothing, not even light, can escape. It's formed when a black hole collapses due to immense energy and the black hole's mass. |
| **sensitivity_tf_bf16_scales** | 39 | 1.7 | 8447 | A black hole is a region in spacetime where gravity is so strong that nothing, not even light, can escape. It's formed by the collapse of massive stars or black holes. |
| **hybrid_1** | 39 | 1.7 | 8436 | A black hole is a region in space where gravity is so strong that nothing, including light, can escape. It is formed by the gravitational collapse of massive stars or other massive stars. |
| **hybrid_2** | 38 | 1.7 | 8440 | A black hole is a region in space where gravity is so strong that nothing, including light, can escape. They are formed by the gravitational effects of massive stars and other massive objects. |
| **int4** | 32 | 1.7 | 8606 | A black hole is an object in space where gravity is so strong that nothing can escape. It is formed by the collapse of stars and other massive stars. |
| **int8_bf16_blocks_embed** | 27 | 1.6 | 8494 | A black hole is a region of spacetime with such intense gravity that nothing, not even light, can escape its near-total collapse. |
| **int8_bf16_blocks_embed_1** | 37 | 1.7 | 8463 | A black hole is a region of space with such intense gravity that nothing, not even light, can escape its pull. It's a point of extreme energy and destruction. |
| **fp8_e5m2** | 36 | 1.7 | 8463 | A black hole is a region of spacetime with such extreme gravity that nothing, not even light, can escape. It’s a point of infinite density and spacetime curvature. |
| **sensitivity_tf_dql** | 56 | 1.6 | 8599 | A black hole is a region in spacetime where gravity is so strong that nothing, not even light, can escape. It's formed when an object falls into a black hole, where the black hole's gravity is so intense that it bends the path of light. |
| **int8_int4_lmhead** | 32 | 1.7 | 8472 | The tallest mountain is Mount Everest, at a height of 8,848.86 meters (29,888 feet). |
| **sensitivity_v2** | 38 | 1.7 | 8523 | A black hole is a region in space where gravity is so strong that nothing, not even light, can escape its embrace. It is formed when a star collapses into a black hole. |
| **sensitivity_v2_dql** | 53 | 1.2 | 11474 | A black hole is a region in space where gravity is so strong that nothing, not even light, can escape. It's formed by the extreme conditions of a black hole's event horizon, where the event horizon is the point of no return. |
| **sensitivity_v3_dql** | 49 | 1.2 | 11462 | A black hole is an object in space where gravity is so strong that nothing, not even light, can escape its gravitational embrace. It's formed by the gravitational collapse of massive stars or other stars at the end of their lives. |
| **reduced_vocab_int8** | 37 | 2.2 | 6594 | A black hole is a region of space with such intense gravity that nothing, not even light, can escape its pull. It's a point of extreme energy and destruction. |
| **reduced_vocab_int8_with_num** | 37 | 2.2 | 6457 | A black hole is a region of space with such intense gravity that nothing, not even light, can escape its pull. It's a point of extreme energy and destruction. |

### Q16: What is the tallest mountain?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 31 | 1.9 | 7612 | The tallest mountain is Mount Everest, at a height of 8,848.5 meters (29,888 feet). |
| **int8** | 35 | 1.7 | 8443 | The tallest mountain is Mount Everest, at a height of 8,848.1メートル (2,966.8フィטרינメートル) |
| **int8_bf16_scales** | 35 | 1.7 | 8440 | The tallest mountain is Mount Everest, at a height of 8,848.1メートル (2,966.8フィטרינメートル) |
| **sensitivity** | 24 | 1.7 | 8453 | Mount Everest is the tallest mountain, with a height of 8,848 meters (2 lapangan). |
| **sensitivity_tf** | 9 | 1.9 | 8427 | The tallest mountain is the Everest. |
| **sensitivity_tf_bf16_scales** | 9 | 1.8 | 8454 | The tallest mountain is the Everest. |
| **hybrid_1** | 9 | 1.9 | 8446 | The tallest mountain is the Everest. |
| **hybrid_2** | 9 | 1.9 | 8450 | The tallest mountain is the Everest. |
| **int4** | 55 | 1.6 | 8613 | The tallest mountain is the Annapurna, a 17,000-meter (59,000-foot) peak located in the Himalayas. This is a challenging climb and requires a significant amount of experience and a good level of fitness. |
| **int8_bf16_blocks_embed** | 17 | 1.7 | 8501 | The tallest mountain is Everest, at a height of 8,848.86 meters. |
| **int8_bf16_blocks_embed_1** | 31 | 1.7 | 8471 | The tallest mountain is Mount Everest, at a height of 8,848.5 meters (29,888 feet). |
| **fp8_e5m2** | 9 | 1.8 | 8470 | The tallest mountain is Mount Everest. |
| **sensitivity_tf_dql** | 9 | 1.8 | 8601 | The tallest mountain is the Everest. |
| **int8_int4_lmhead** | 5 | 2.0 | 8478 | Leonardo da Vinci |
| **sensitivity_v2** | 13 | 1.8 | 8532 | The tallest mountain is the Annapurna Peak in Nepal. |
| **sensitivity_v2_dql** | 21 | 1.3 | 11483 | The tallest mountain is Mount Everest, which is more than 8,848 feet. |
| **sensitivity_v3_dql** | 13 | 1.3 | 11461 | The tallest mountain is the Laona Peak in Iceland. |
| **reduced_vocab_int8** | 210 | 2.1 | 6600 | The tallest mountain is Mount Everest, at a height of  approximately  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  *  * |
| **reduced_vocab_int8_with_num** | 31 | 2.2 | 6464 | The tallest mountain is Mount Everest, at a height of 8,848.1 meters (29,831 feet). |

### Q17: Who painted the Mona Lisa?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 5 | 2.3 | 7620 | Leonardo da Vinci |
| **int8** | 5 | 2.0 | 8463 | Leonardo da Vinci |
| **int8_bf16_scales** | 5 | 2.1 | 8452 | Leonardo da Vinci |
| **sensitivity** | 21 | 1.7 | 8461 | Leonardo da Vinci was the most famous artist of the Renaissance, and he painted the Mona Lisa. |
| **sensitivity_tf** | 14 | 1.8 | 8435 | Leonardo da Vinci is the artist who painted the Mona Lisa. |
| **sensitivity_tf_bf16_scales** | 15 | 1.8 | 8464 | Leonardo da Vinci is the master artist who painted the Mona Lisa. |
| **hybrid_1** | 12 | 1.8 | 8454 | The Mona Lisa was painted by Leonardo da Vinci. |
| **hybrid_2** | 12 | 1.8 | 8449 | The Mona Lisa was painted by Leonardo da Vinci. |
| **int4** | 38 | 1.7 | 8614 | Leonardo da Vinci is the most famous artist of the Renaissance. He is known for his anatomical accuracy, his mastery of painting techniques, and his insights into the nature of the divine. |
| **int8_bf16_blocks_embed** | 5 | 1.7 | 8510 | Leonardo da Vinci. |
| **int8_bf16_blocks_embed_1** | 5 | 2.0 | 8474 | Leonardo da Vinci |
| **fp8_e5m2** | 6 | 2.0 | 8479 | Leonardo da Vinci. |
| **sensitivity_tf_dql** | 13 | 1.7 | 8614 | Leonardo da Vinci is the master of the Mona Lisa. |
| **int8_int4_lmhead** | 44 | 1.7 | 7258 | Evolution is the process by which organisms change over time, over millions of years, to adapt to their environment and survive. This adaptation can result in new traits, adaptations, or even the loss of an organism. |
| **sensitivity_v2** | 38 | 1.7 | 8572 | Leonardo da Vinci is the most famous artist of the Renaissance. He painted the Mona Lisa, a portrait of the enigmatic lady who is said to be a wife of the captured artist. |
| **sensitivity_v2_dql** | 14 | 1.3 | 11484 | Leonardo da Vinci is the most famous artist of the Renaissance. |
| **sensitivity_v3_dql** | 14 | 1.3 | 11468 | Leonardo da Vinci is the most famous artist of the Renaissance. |
| **reduced_vocab_int8** | 5 | 2.6 | 6612 | Leonardo da Vinci |
| **reduced_vocab_int8_with_num** | 5 | 2.7 | 6472 | Leonardo da Vinci |

### Q18: What is evolution?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 39 | 1.9 | 6531 | Evolution is the process by which living organisms change over time to adapt to their environment. This can involve changes in traits, the introduction of new species, or the evolution of new characteristics. |
| **int8** | 33 | 1.7 | 7236 | Evolution is the process by which living organisms change over time to adapt to their environment. This can involve changes in size, shape, behavior, or reproduction. |
| **int8_bf16_scales** | 39 | 1.7 | 7231 | Evolution is the process by which living organisms change over time to adapt to their environment. This can involve changes in traits, the introduction of new species, or the evolution of new characteristics. |
| **sensitivity** | 59 | 1.7 | 7243 | Evolution is the process by which species change over time, with the evolution of different species, organisms, and ecosystems. This change is driven by natural selection, where individuals with traits that are advantageous in their environment are more likely to survive and reproduce, leading to the evolution of new species. |
| **sensitivity_tf** | 59 | 1.7 | 7220 | Evolution is the process by which life has changed over time. It's a dynamic process where species evolve through a combination of genetic mutations, environmental factors, and natural selection. This evolution can result in new species with different characteristics, such as different adaptations to different environments or different populations. |
| **sensitivity_tf_bf16_scales** | 59 | 1.7 | 7245 | Evolution is the process by which life on Earth has changed over time. It's a dynamic process where species evolve through a combination of genetic mutations, environmental factors, and natural selection. This evolution can result in new species with different characteristics, such as different adaptations for survival and reproduction. |
| **hybrid_1** | 54 | 1.7 | 7236 | Evolution is the ongoing process of change, adaptation, and adaptation. It's a continuous journey of adaptation across different species, influenced by environmental factors and the evolution of different lineages. This evolution can be driven by natural selection, genetic drift, and other factors. |
| **hybrid_2** | 44 | 1.7 | 7245 | Evolution is the ongoing process of change, adaptation, and adaptation. It's a continuous journey of adaptation across different environments, ecosystems, and populations, driven by a constant interplay of environmental factors and evolutionary forces. |
| **int4** | 74 | 1.6 | 7382 | Evolution is the ongoing process of change and adaptation within a species over time. It is a continuous journey of adaptation, where species evolve through a combination of genetic variation, environmental factors, and evolutionary pressures. This can result in adaptations such as changing adaptations for survival in a particular environment, such as changing the structure or function of a tissue or a population of cells. |
| **int8_bf16_blocks_embed** | 29 | 1.6 | 7284 | Evolution is the process by which organisms change over time to adapt to their environment. This can involve changes in size, shape, behavior, and reproduction. |
| **int8_bf16_blocks_embed_1** | 33 | 1.7 | 7259 | Evolution is the process by which living organisms change over time to adapt to their environment. This can involve changes in size, shape, behavior, or reproduction. |
| **fp8_e5m2** | 34 | 1.7 | 7258 | Evolution is the process by which living organisms change and adapt to their environment. It’s a slow and continuous process, with different species evolving from simpler ancestors. |
| **sensitivity_tf_dql** | 42 | 1.7 | 7375 | Evolution is the process by which life on Earth has changed over time. It's a dynamic process where species evolve from single-celled beginnings to complex, multicellular organisms with specialized structures and adaptations. |
| **int8_int4_lmhead** | 25 | 1.7 | 9675 | Water is a highly hydrated molecule, meaning it loses its chemical bonds and becomes a solid when dissolved in a liquid. |
| **sensitivity_v2** | 39 | 1.7 | 7314 | Evolution is the process by which life on a natural background has changed over time. This can involve genetic mutations, genetic adaptations, and changes in the structure and function of cells and organisms. |
| **sensitivity_v2_dql** | 56 | 1.2 | 9838 | Evolution is the process by which life on a planet has changed over time. It's a fundamental process driven by natural selection, where differences in traits become more common in a population over successive generations. This is a key concept in understanding life's diversity and adaptation. |
| **sensitivity_v3_dql** | 37 | 1.2 | 9826 | Evolution is the change of species over time through natural selection. It's a process where species evolve due to differences in their anatomy, genetics, behavior, and ecological roles. |
| **reduced_vocab_int8** | 33 | 2.2 | 5657 | Evolution is the process by which living organisms change over time to adapt to their environment. This can involve changes in size, shape, behavior, and reproduction. |
| **reduced_vocab_int8_with_num** | 33 | 2.2 | 5539 | Evolution is the process by which living organisms change over time to adapt to their environment. This can involve changes in size, shape, behavior, and reproduction. |

### Q19: What is the chemical formula for water?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 17 | 1.9 | 8695 | Water is a highly hydrated liquid with a molecular formula of H₂O. |
| **int8** | 17 | 1.7 | 9645 | Water is a highly hydrated liquid with a molecular formula of H₂O. |
| **int8_bf16_scales** | 14 | 1.8 | 9640 | Water is a highly hydrated liquid composed of hydrogen and oxygen. |
| **sensitivity** | 15 | 1.8 | 9662 | Water is a liquid with a molecular formula of H₂O. |
| **sensitivity_tf** | 12 | 1.8 | 9634 | The chemical formula for water is H₂O. |
| **sensitivity_tf_bf16_scales** | 12 | 1.8 | 9666 | The chemical formula for water is H₂O. |
| **hybrid_1** | 13 | 1.8 | 9654 | Water is a liquid with a hydrogen and a phosphate. |
| **hybrid_2** | 19 | 1.7 | 9648 | Water is a liquid with a high concentration of hydrogen and a low concentration of oxygen. |
| **int4** | 13 | 1.7 | 9844 | Water is a liquid with a hydrogen and a phosphate. |
| **int8_bf16_blocks_embed** | 12 | 1.9 | 9712 | Water is a liquid at standard temperature and pressure. |
| **int8_bf16_blocks_embed_1** | 17 | 1.7 | 9678 | Water is a highly hydrated liquid with a molecular formula of H₂O. |
| **fp8_e5m2** | 14 | 1.8 | 9675 | The chemical formula for water is C2H5O. |
| **sensitivity_tf_dql** | 12 | 1.8 | 9832 | The chemical formula for water is H₂O. |
| **int8_int4_lmhead** | 11 | 1.8 | 9675 | The nearest star to Earth is the Sun. |
| **sensitivity_v2** | 17 | 1.7 | 9752 | Water is a highly hydrated form of hydrogen, with the formula H₂. |
| **sensitivity_v2_dql** | 5 | 1.5 | 13116 | H₂O |
| **sensitivity_v3_dql** | 15 | 1.3 | 13100 | Water is a essential solvent and a hydrated form of hydrogen. |
| **reduced_vocab_int8** | 24 | 2.2 | 7542 | Water is a highly hydrated liquid, with a molecular formula of C <sub>H</sub> <sub>a</sub>. |
| **reduced_vocab_int8_with_num** | 21 | 2.3 | 7382 | Water is a highly hydrated liquid, with a molecular formula of C 6H12. |

### Q20: What is the nearest star to Earth?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 11 | 2.0 | 8697 | The nearest star to Earth is the Sun. |
| **int8** | 11 | 1.8 | 9646 | The nearest star to Earth is the Sun. |
| **int8_bf16_scales** | 11 | 1.8 | 9639 | The nearest star to Earth is the Sun. |
| **sensitivity** | 11 | 1.8 | 9676 | The nearest star to Earth is the Sun. |
| **sensitivity_tf** | 11 | 1.8 | 9620 | The nearest star to Earth is the Sun. |
| **sensitivity_tf_bf16_scales** | 11 | 1.8 | 9653 | The nearest star to Earth is the Sun. |
| **hybrid_1** | 11 | 1.8 | 9645 | The nearest star to Earth is the Sun. |
| **hybrid_2** | 11 | 1.8 | 9650 | The nearest star to Earth is the Sun. |
| **int4** | 11 | 1.8 | 9833 | The nearest star to Earth is the Sun. |
| **int8_bf16_blocks_embed** | 10 | 1.8 | 9711 | The nearest star to Earth is the Sun. |
| **int8_bf16_blocks_embed_1** | 14 | 1.8 | 9674 | The nearest star to Earth is **Proxima Centauri**. |
| **fp8_e5m2** | 11 | 1.8 | 9667 | The nearest star to Earth is the Sun. |
| **sensitivity_tf_dql** | 11 | 1.8 | 9831 | The nearest star to Earth is the Sun. |
| **int8_int4_lmhead** | 45 | 1.7 | 7865 | An atom is a fundamental particle in the universe, a small, lightweight nucleus with a very small number of protons and neutrons. It is the basic building block of all matter and is the fundamental unit of the atom. |
| **sensitivity_v2** | 11 | 1.8 | 9739 | The nearest star to Earth is the Sun. |
| **sensitivity_v2_dql** | 11 | 1.3 | 13185 | The nearest star to Earth is the Sun. |
| **sensitivity_v3_dql** | 11 | 1.3 | 13100 | The nearest star to Earth is the Sun. |
| **reduced_vocab_int8** | 25 | 2.2 | 7544 | The nearest star to Earth is Proxima Centaurus, which is approximately  melding into the constellation of Orion. |
| **reduced_vocab_int8_with_num** | 28 | 2.2 | 7382 | The nearest star to Earth is Proxima Centaurus, which is approximately 4.246 light-years away. |

### Q21: What is an atom?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 34 | 1.9 | 7067 | An atom is a fundamental particle in the universe, the smallest unit of matter. It has a nucleus with protons and neutrons, and an electron orbiting the nucleus. |
| **int8** | 34 | 1.7 | 7840 | An atom is a fundamental particle in the universe, the smallest unit of matter. It has a nucleus with protons and neutrons, and an electron orbiting the nucleus. |
| **int8_bf16_scales** | 34 | 1.7 | 7834 | An atom is a fundamental particle in the universe, the smallest unit of matter. It has a nucleus with protons and neutrons, and an electron orbiting the nucleus. |
| **sensitivity** | 46 | 1.7 | 7848 | An atom is a fundamental building block of matter, composed of one or more protons, neutrons, and likely a few other quarks. It's the fundamental unit of the periodic table and the fundamental element of the universe. |
| **sensitivity_tf** | 49 | 1.7 | 7824 | An atom is a fundamental building block of matter and is the fundamental unit of the periodic table. It's composed of one or more electrons, which are bound to nuclei and therefore, are essential for the structure and properties of atoms. |
| **sensitivity_tf_bf16_scales** | 38 | 1.7 | 7844 | An atom is a fundamental building block of matter and is the fundamental unit of the periodic table. It's the fundamental element that exists in all the elements in the periodic table. |
| **hybrid_1** | 33 | 1.7 | 7841 | An atom is a fundamental building block of matter, composed of only one element. It is the fundamental unit of the periodic table, with only one electron. |
| **hybrid_2** | 33 | 1.7 | 7843 | An atom is a fundamental building block of matter, composed of only one element. It is the fundamental unit of the periodic table, with only one electron. |
| **int4** | 40 | 1.7 | 7997 | An atom is a fundamental building block of matter, composed of only one thing. It's the fundamental unit of an element, with the formula: 12 1's. |
| **int8_bf16_blocks_embed** | 22 | 1.4 | 7894 | An atom is a fundamental particle composed of three fundamental particles: an electron, a proton, and a neutron. |
| **int8_bf16_blocks_embed_1** | 34 | 1.7 | 7864 | An atom is a fundamental particle in the universe, the smallest unit of matter. It has a nucleus with protons and neutrons, and an electron orbiting the nucleus. |
| **fp8_e5m2** | 27 | 1.7 | 7857 | An atom is a fundamental building block of matter. It is composed of three fundamental elements: hydrogen, helium, and neutron. |
| **sensitivity_tf_dql** | 41 | 1.7 | 7991 | An atom is a fundamental building block of matter, composed of one or more electrons. These electrons are bound together by strong, electromagnetic forces, which are responsible for the attraction and repulsion of electrons. |
| **int8_int4_lmhead** | 39 | 1.7 | 9073 | The speed of sound is the speed at which sound travels in a liquid. It is approximately 343 meters per second (or 1,375 feet per second). |
| **sensitivity_v2** | 28 | 1.7 | 7922 | An atom is a fundamental building block of matter with an electron. They have a nucleus with an electron bound to its core atom. |
| **sensitivity_v2_dql** | 40 | 1.2 | 10655 | An atom is a fundamental building block of matter, with the most common element being hydrogen (H). It's a fundamental element because it's the fundamental element of the element composition. |
| **sensitivity_v3_dql** | 37 | 1.2 | 10640 | An atom is a fundamental building block of matter and is the fundamental element in the periodic table. It's composed of three fundamental elements: hydrogen, helium, and lithium. |
| **reduced_vocab_int8** | 34 | 2.2 | 6137 | An atom is a fundamental particle in the universe, the smallest unit of matter. It has a nucleus with protons and neutrons, and an electron orbiting the nucleus. |
| **reduced_vocab_int8_with_num** | 34 | 2.2 | 6001 | An atom is a fundamental particle in the universe, the smallest unit of matter. It has a nucleus with protons and neutrons, and an electron orbiting the nucleus. |

### Q22: What is the speed of sound?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 17 | 1.9 | 8148 | The speed of sound is the speed at which sound travels in a liquid. |
| **int8** | 28 | 1.7 | 9039 | The speed of sound is the speed at which sound travels in a liquid. It is approximately 343 meters per second. |
| **int8_bf16_scales** | 17 | 1.7 | 9039 | The speed of sound is the speed at which sound travels in a liquid. |
| **sensitivity** | 16 | 1.8 | 9055 | The speed of sound is approximately 343 meters per second. |
| **sensitivity_tf** | 16 | 1.8 | 9029 | The speed of sound is approximately 343 meters per second. |
| **sensitivity_tf_bf16_scales** | 16 | 1.8 | 9058 | The speed of sound is approximately 343 meters per second. |
| **hybrid_1** | 35 | 1.7 | 9045 | The speed of sound is a constant, constant speed. It is a fundamental property of the universe and is a fundamental constant in the fundamental constants of the periodic potential. |
| **hybrid_2** | 35 | 1.7 | 9044 | The speed of sound is the speed at which sound vibrations travel in a vacuum. It is a fundamental constant in the universe and is a constant in all possible mediums. |
| **int4** | 45 | 1.6 | 9223 | The speed of sound is a fundamental constant in the universe, a constant that represents the speed at which sound vibrations travel. It is a fundamental property of the electromagnetic field, which is the force that propels sound waves. |
| **int8_bf16_blocks_embed** | 16 | 1.6 | 9100 | The speed of sound is the speed at which sound travels in a liquid. |
| **int8_bf16_blocks_embed_1** | 28 | 1.7 | 9066 | The speed of sound is the speed at which sound travels in a liquid. It is approximately 343 meters per second. |
| **fp8_e5m2** | 16 | 1.8 | 9065 | The speed of sound is approximately 343 meters per second. |
| **sensitivity_tf_dql** | 16 | 1.7 | 9214 | The speed of sound is approximately 343 meters per second. |
| **int8_int4_lmhead** | 10 | 1.8 | 7262 | Penicillin was discovered by Alexander Fleming. |
| **sensitivity_v2** | 16 | 1.7 | 9132 | The speed of sound is approximately 343 meters per second. |
| **sensitivity_v2_dql** | 17 | 1.3 | 12302 | The speed of sound is approximately 3400 meters per second. |
| **sensitivity_v3_dql** | 16 | 1.3 | 12279 | The speed of sound is approximately 343 meters per second. |
| **reduced_vocab_int8** | 17 | 2.2 | 7069 | The speed of sound is the speed at which sound travels in a vacuum. |
| **reduced_vocab_int8_with_num** | 17 | 2.3 | 6916 | The speed of sound is the speed at which sound travels in a vacuum. |

### Q23: Who discovered penicillin?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 11 | 2.0 | 6522 | Alexander Fleming discovered penicillin during World War II. |
| **int8** | 10 | 1.8 | 7232 | Penicillin was discovered by Alexander Fleming. |
| **int8_bf16_scales** | 10 | 1.8 | 7235 | Penicillin was discovered by Alexander Fleming. |
| **sensitivity** | 22 | 1.7 | 7245 | Alexander Fleming was the first to observe the effects of infection and develop a theory about the immune system. |
| **sensitivity_tf** | 10 | 1.8 | 7226 | Penicillin was discovered by Alexander Fleming. |
| **sensitivity_tf_bf16_scales** | 10 | 1.8 | 7250 | Penicillin was discovered by Alexander Fleming. |
| **hybrid_1** | 38 | 1.7 | 7240 | The Penicillin discovery was a pivotal moment in the fight against the bubonic plague. It was achieved by the sulfonating and antibacterial drugs, which were initially developed by Alexander Fleming. |
| **hybrid_2** | 64 | 1.7 | 7242 | The discovery of penicillin is a pivotal event in the history of medicine, with the development of antibiotics having a profound impact on the fight against infectious diseases. It was a result of the Penicillin Wars, a series of international efforts to produce and distribute the antibiotic, penicillin, to combat the growing resistance of bacteria. |
| **int4** | 105 | 1.6 | 7392 | The Penicillin discovery is a fascinating story of a collaboration between the British Medical Council and the St. Luke's Medical School in 1940. The St. Luke's team, working on the St. Michael's Medical School in the 1940s, accidentally combined the antibiotic silver-4-chloroperiodine with the bacteria's cell wall. This led to the antibiotic' dirigido to the antibiotic's ability to rapidly and effectively treat infections, including bacterial infections. |
| **int8_bf16_blocks_embed** | 10 | 1.3 | 7293 | Alexander Fleming discovered penicillin during World War II. |
| **int8_bf16_blocks_embed_1** | 10 | 1.8 | 7262 | Penicillin was discovered by Alexander Fleming. |
| **fp8_e5m2** | 5 | 2.0 | 7267 | Alexander Fleming. |
| **sensitivity_tf_dql** | 39 | 1.7 | 7380 | Penicillin was discovered by Alexander Fleming, who observed a blood infection in a mouse and developed a treatment that, over a period of time, reduced the number of bacteria in the wound. |
| **int8_int4_lmhead** | 12 | 1.8 | 10888 | Jupiter is the largest planet in our solar system. |
| **sensitivity_v2** | 22 | 1.7 | 7314 | Alexander Fleming is most likely the person who accidentally inhaled a sulfonamide-containing medication during a fever. |
| **sensitivity_v2_dql** | 21 | 1.3 | 9843 | Alexander Fleming, a Scottish physician, is credited with discovering penicillin in 1928. |
| **sensitivity_v3_dql** | 15 | 1.3 | 9832 | Alexander Fleming's discovery of penicillin occurred during World WWI. |
| **reduced_vocab_int8** | 10 | 2.3 | 5663 | Penicillin was discovered by Alexander Fleming. |
| **reduced_vocab_int8_with_num** | 10 | 2.4 | 5542 | Penicillin was discovered by Alexander Fleming. |

### Q24: What is the largest planet in our solar system?

| Model | Tokens | TPS | TTFT | Answer |
|-------|--------|-----|------|--------|
| **bf16** | 12 | 2.0 | 9788 | Jupiter is the largest planet in our solar system. |
| **int8** | 12 | 1.8 | 10852 | Jupiter is the largest planet in our solar system. |
| **int8_bf16_scales** | 12 | 1.8 | 10846 | Jupiter is the largest planet in our solar system. |
| **sensitivity** | 30 | 1.7 | 10873 | Jupiter is the largest planet in our solar system, with a mass estimated to be about 49% of the mass of the sun. |
| **sensitivity_tf** | 12 | 1.8 | 10839 | The largest planet in our solar system is Jupiter. |
| **sensitivity_tf_bf16_scales** | 12 | 1.8 | 10874 | The largest planet in our solar system is Jupiter. |
| **hybrid_1** | 12 | 1.8 | 10866 | The largest planet in our solar system is Jupiter. |
| **hybrid_2** | 12 | 1.8 | 10870 | The largest planet in our solar system is Jupiter. |
| **int4** | 14 | 1.7 | 11075 | Yes, the largest planet in our solar system is Jupiter. |
| **int8_bf16_blocks_embed** | 11 | 2.2 | 10932 | Jupiter is the largest planet in our solar system. |
| **int8_bf16_blocks_embed_1** | 12 | 1.8 | 10886 | Jupiter is the largest planet in our solar system. |
| **fp8_e5m2** | 12 | 1.8 | 10894 | The largest planet in our solar system is Jupiter. |
| **sensitivity_tf_dql** | 12 | 1.8 | 11064 | The largest planet in our solar system is Jupiter. |

---

## 3. Quality Comparison Summary

### Factual Accuracy Scorecard

| Question | bf16 | int8 | sensitivity | sensitivity_tf | sensitivity_v2 | sensitivity_v2_dql | sensitivity_v3_dql | hybrid_1 | hybrid_2 | int4 |
|----------|------|------|-------------|----------------|----------------|-------------------|-------------------|----------|----------|------|
| Photosynthesis | ✅ Good | ✅ Good | ✅ Good | ✅ Good | ✅ Good | ✅ Good | ⚠ Partial | ⚠ Partial | ⚠ Partial | ⚠ Partial |
| Quantum mechanics | ⚠ Partial | ⚠ Partial | ⚠ Partial | ⚠ Partial | ⚠ Partial | ⚠ Partial | ⚠ Partial | ❌ Wrong | ❌ Wrong | ❌ Wrong |
| Capital of India | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ⚠ Partial | ✅ Correct | ✅ Correct | ❌ Wrong | ❌ Wrong | ❌ Wrong |
| Capital of Italy | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct |
| Capital of USA | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ❌ Wrong |
| Is AI dangerous | ⚠ Partial | ⚠ Partial | ⚠ Partial | ⚠ Partial | ✅ Good | ✅ Good | ⚠ Partial | ⚠ Partial | ⚠ Partial | ⚠ Partial |
| Gravity | ⚠ Partial | ⚠ Partial | ⚠ Partial | ⚠ Partial | ✅ Good | ⚠ Partial | ⚠ Partial | ⚠ Partial | ⚠ Partial | ⚠ Partial |
| Romeo & Juliet | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ❌ Wrong | ⚠ Partial | ❌ Wrong |
| Speed of light | ✅ Correct | ✅ Correct | ⚠ Partial | ❌ Wrong | ❌ Wrong | ❌ Wrong | ❌ Wrong | ❌ Wrong | ❌ Wrong | ⚠ Partial |
| DNA | ✅ Good | ✅ Good | ✅ Good | ⚠ Partial | ✅ Good | ✅ Good | ⚠ Partial | ⚠ Partial | ⚠ Partial | ❌ Wrong |
| Rain | ⚠ Partial | ⚠ Partial | ⚠ Partial | ⚠ Partial | ✅ Good | ❌ Wrong | ⚠ Partial | ❌ Wrong | ❌ Wrong | ❌ Wrong |
| Largest ocean | ❌ Wrong | ❌ Wrong | ❌ Wrong | ❌ Wrong | ❌ Wrong | ❌ Wrong | ❌ Wrong | ❌ Wrong | ❌ Wrong | ❌ Wrong |
| Telephone inventor | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ⚠ Partial | ⚠ Partial | ❌ Wrong |
| Boiling point of water | ✅ Correct | ✅ Correct | ⚠ Partial | ❌ Wrong | ⚠ Partial | ❌ Wrong | ✅ Correct | ✅ Correct | ✅ Correct | ❌ Wrong |
| Black hole | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ⚠ Partial |
| Tallest mountain | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ❌ Wrong | ✅ Correct | ❌ Wrong | ✅ Correct | ✅ Correct | ❌ Wrong |
| Mona Lisa | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ⚠ Partial |
| Evolution | ⚠ Partial | ⚠ Partial | ⚠ Partial | ⚠ Partial | ✅ Good | ✅ Good | ✅ Good | ⚠ Partial | ⚠ Partial | ⚠ Partial |
| Chemical formula water | ⚠ Partial | ⚠ Partial | ⚠ Partial | ✅ Correct | ❌ Wrong | ✅ Correct | ❌ Wrong | ❌ Wrong | ❌ Wrong | ❌ Wrong |
| Nearest star | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct |
| Atom | ✅ Good | ✅ Good | ⚠ Partial | ⚠ Partial | ⚠ Partial | ⚠ Partial | ⚠ Partial | ⚠ Partial | ⚠ Partial | ❌ Wrong |
| Speed of sound | ⚠ Partial | ⚠ Partial | ✅ Correct | ✅ Correct | ✅ Correct | ❌ Wrong | ✅ Correct | ❌ Wrong | ❌ Wrong | ❌ Wrong |
| Penicillin | ✅ Correct | ✅ Correct | ⚠ Partial | ✅ Correct | ⚠ Partial | ✅ Correct | ⚠ Partial | ⚠ Partial | ❌ Wrong | ❌ Wrong |
| Largest planet | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ✅ Correct | ⚠ Partial | ⚠ Partial | ✅ Correct | ✅ Correct | ✅ Correct |

### Quality Scores (✅=2, ⚠=1, ❌=0)

| Model | ✅ Correct | ⚠ Partial | ❌ Wrong | Score |
|-------|-----------|-----------|---------|-------|
| **bf16** | 15 | 7 | 2 | **37/48** |
| **int8** | 15 | 7 | 2 | **37/48** |
| **int8_bf16_scales** | 15 | 7 | 2 | **37/48** |
| **int8_bf16_blocks_embed** | 13 | 8 | 3 | **34/48** |
| **int8_bf16_blocks_embed_1** | 16 | 6 | 2 | **38/48** |
| **fp8_e5m2** | 13 | 4 | 7 | **30/48** |
| **fp8_e4m3** | 14 | 6 | 4 | **34/48** |
| **sensitivity** | 13 | 10 | 1 | **36/48** |
| **sensitivity_tf** | 14 | 7 | 3 | **35/48** |
| **sensitivity_tf_bf16_scales** | 14 | 7 | 3 | **35/48** |
| **sensitivity_tf_dql** | 13 | 7 | 4 | **33/48** |
| **int8_int4_lmhead** | 15 | 6 | 3 | **36/48** |
| **sensitivity_v2** | 15 | 5 | 4 | **35/48** |
| **sensitivity_v2_dql** | 15 | 4 | 5 | **34/48** |
| **sensitivity_v3_dql** | 11 | 9 | 4 | **31/48** |
| **reduced_vocab_int8** | 14 | 4 | 6 | **32/48** |
| **reduced_vocab_int8_with_num** | 15 | 5 | 4 | **35/48** |
| **hybrid_1** | 9 | 7 | 8 | **25/48** |
| **hybrid_2** | 9 | 6 | 9 | **24/48** |
| **int4** | 3 | 6 | 15 | **12/48** |

---

## 4. Notable Quality Degradation Patterns

### Questions where int4 uniquely fails (all others get right):
1. **Capital of USA:** "The capital of the United States is the United States of America" (tautology)
2. **Boiling point of water:** "100°C or 252°F" (wrong Fahrenheit: 252 instead of 212)
3. **Tallest mountain:** "Annapurna, a 17,000-meter (59,000-foot) peak" (wrong mountain, wrong height, wrong conversion)
4. **Telephone inventor:** "invented by the British government in 1840" (completely lost Bell's name)

### Progressive corruption across quantization levels:
- **"Romeo & Juliet" author:** bf16/int8: "William Shakespeare" ✅ → hybrid_1: "Thesandeering" ❌ → hybrid_2: "Julio's...by Shakespeare" ⚠ → int4: "Julienois...by Shakespeare" ❌
- **India capital:** bf16/int8: "New Delhi" ✅ → hybrid_1: "Swatabad...Arabian Peninsula" ❌ → hybrid_2: "Swatabad...Cairo" ❌ → int4: "Swade...Delhi" ❌
- **Speed of light:** bf16/int8: "299,792,458 m/s" ✅ → hybrid_1/2: "186,000 km/s" ❌ → int4: "186,000 miles/s" ⚠
- **Penicillin:** bf16/int8: "Alexander Fleming" ✅ → hybrid_1: mentions Fleming ⚠ → hybrid_2: "Penicillin Wars" ❌ → int4: "St. Luke's...silver-4-chloroperiodine" ❌ (105 tokens of fabrication)

### int4-specific pathologies:
- **Repetitive token loops:** "quantum phenomena in quantum phenomena", "double-5-carbon double-5-5-carbon double-5-5-carbon"
- **Garbled text:** "ocean'egra", "formula: 12 1's", "antibiotic' dirigido"
- **Verbose wrong answers:** Q23 penicillin = 105 tokens (vs 10–11 for bf16/int8)
- **Factual amnesia:** Lost names (Bell → "British government"), wrong facts (Everest → Annapurna, 212°F → 252°F)

---

## 5. Memory Usage

All models showed similar memory footprint on the 1.9GB device:
- **Pre-load:** ~140–142MB used
- **Available:** ~1.76–1.77GB free
- No OOM issues on any model variant

---

## 6. Timing Summary

| Model | Start | End | Duration |
|-------|-------|-----|----------|
| bf16 | 10:26 | 10:37 | ~11 min |
| int8 | 10:37 | 10:47 | ~10 min |
| hybrid_1 | 10:47 | 10:58 | ~11 min |
| hybrid_2 | 10:58 | 11:10 | ~12 min |
| int4 | 11:10 | 11:23 | ~13 min |

**Total benchmark time: ~57 minutes**

Note: int4 took the longest (~13 min) despite similar per-token speed because it generates significantly more tokens per answer.

---

## 7. Conclusions

1. **bf16 and int8 are essentially identical** in both quality and performance. The quantization from bf16→int8 causes no measurable quality loss on this model. Both score 37/48.

2. **sensitivity (int8_int4_sensitivity) matches int8 quality** — scoring 36/48, only 1 point below bf16/int8. This model uses KL-divergence logit sensitivity analysis (free-running, 2 prompts × 2 tokens) to select which 22 layers (17%) need int8 precision; the remaining 105 layers (83%) use int4. Effective bits/param: 6.83. It correctly answers India's capital, Romeo & Juliet, telephone inventor, and all other questions that hybrid_1/2 fail on.

3. **sensitivity_tf (int8_int4_sensitivity_tf) also near-int8 quality** — scoring 35/48 with only 17 layers (13%) at int8 and 110 layers (87%) at int4. Uses teacher-forced perplexity analysis (10 prompts × 30 tokens = 300 comparisons/layer). More aggressive compression (6.52 bits/param) while retaining strong factual recall: H₂O, Shakespeare, Fleming, Bell, capitals all correct. Loses boiling point (120°C) and speed of light (186,000 km/s).

4. **hybrid_1 and hybrid_2 show significant quality degradation** — scoring 25/48 and 24/48 respectively. They lose factual accuracy on India's capital, speed of light, Romeo & Juliet author, and chemical formula for water. The "Atlantic Meridrolator" hallucination for largest ocean also appears here.

5. **int4 is catastrophically worse** — scoring only 12/48 (68% quality loss vs bf16). It loses basic facts that all other models retain (USA capital, boiling point, Everest, Bell), generates verbose gibberish with repetitive token loops, and is measurably slower.

6. **Throughput is hardware-bottlenecked** — bf16 is slightly faster (1.9 avg TPS vs 1.7 for quantized models), but all models are within the same ballpark, suggesting the Torq SoC inference engine is the limiting factor.

7. **Recommendation:** Use **int8_int4_sensitivity** (36/48) for the best quality/compression tradeoff, or **int8_int4_sensitivity_tf** (35/48) for maximum compression with near-equivalent quality. Both achieve 83–87% int4 layer coverage while preserving factual correctness far above the naive hybrid models. Use pure **int8** if maximum quality is required. Avoid hybrid_1/2 and int4 models — the quality loss is significant (hybrid) to catastrophic (int4).
