# isiZulu Ensemble ASR

**A multi-agent ensemble Automatic Speech Recognition system for isiZulu — a critically under-resourced South African language.**

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-isizulu--asr-blue)](https://huggingface.co/spaces/lsgmgbl/isizulu-asr-demo)

> Research project completed in 2025 under the supervision of Prof. Duncan Coulter, University of Johannesburg. Part of a BSc Honours in Computer Science (Artificial Intelligence).

---

## Overview

Automatic Speech Recognition (ASR) technology has largely failed to serve low-resource languages like isiZulu due to data scarcity and a historical focus on high-resource languages such as English and Mandarin. This project addresses that gap by designing, training, and evaluating a **multi-agent ensemble ASR system** for isiZulu.

The system combines three architecturally distinct deep learning agents whose outputs are fused by a trainable GRU-based ensemble module, producing a single, more accurate transcription.

---

## Architecture

Three independent ASR agents process input audio in parallel. Their logits are concatenated and passed through a GRU-based fusion module which learns to weigh and combine their predictions.

```
Input Audio
    │
    ├──► Wav2Vec2 Agent (Fine-Tuned XLS-R)       ──► Logits ─┐
    ├──► Conformer Agent (Subset Fine-Tuned)      ──► Logits ─┼──► GRU Fusion Module ──► CTC Decoder ──► Transcription
    └──► Custom CNN-RNN-CTC Agent (From Scratch)  ──► Logits ─┘
```

### Agents

**Wav2Vec2 (XLS-R)** — Fine-tuned from `facebook/wav2vec2-xls-r-300m`. Leverages self-supervised pre-training across 128+ languages, then adapted for isiZulu. Best performing individual agent.

**Wav2Vec2 Conformer** — Initialized from `facebook/wav2vec2-conformer-large-960h` and fine-tuned on a subset of the isiZulu data. Combines convolutional layers for local feature extraction with self-attention for global context.

**Custom CNN-RNN-CTC** — Built entirely from scratch on the NCHLT isiZulu corpus. Architecture: 2-layer CNN block → 3-layer bidirectional GRU (hidden size 512) → linear CTC classifier. Establishes a baseline for what is achievable with limited data and no pre-training.

**GRU Fusion Module** — A 2-layer bidirectional GRU (hidden size 512) that learns to integrate logits from all three agents. Logits are padded to uniform length, concatenated along the feature dimension, and decoded via CTC.

---

## Results

Evaluated on the held-out NCHLT isiZulu test set:

| Model | WER | CER |
|---|---|---|
| Wav2Vec2 XLS-R (Fine-Tuned) | 19.49% | 3.62% |
| Ensemble (GRU Fusion) | 19.53% | 3.58% |
| CNN-RNN-CTC (From Scratch) | 42.08% | 7.89% |
| Conformer (Subset Fine-Tuned) | 699.53% | 77.44% |

**Key findings:**
- Fine-tuning large multilingual pre-trained models is highly effective in low-resource settings
- The ensemble matched but did not surpass the best individual agent, likely due to the under-trained Conformer introducing noise
- A custom model trained from scratch on 56 hours of data achieves a functional but lower-accuracy baseline

---

## Additional Experiments

Three alternative fusion strategies were explored beyond the primary GRU fusion model:

| Strategy | WER | CER | Outcome |
|---|---|---|---|
| T5-based post-correction | 20.76% | 3.51% | Marginal CER gain, WER degraded |
| Dictionary-based spelling correction | 21.58% | 3.42% | Degraded — over-corrected valid words |
| Character-level voting fusion | — | — | Abandoned — poor alignment across agents |

---

## Dataset

**NCHLT isiZulu Speech Corpus** — approximately 56 hours of orthographically transcribed speech from 210 unique speakers, collected in non-studio environments at 16kHz.

Source: Barnard et al. (2014). The NCHLT speech corpus of the South African languages. SLTU Workshop.

---

## Setup

```bash
git clone https://github.com/lesego-mogable/isizulu-ensemble-asr.git
cd isizulu-ensemble-asr
pip install -r requirements.txt
```

See the `notebooks/` directory for training and evaluation walkthroughs, and `src/` for model architecture definitions.

---

## Project Structure

```
├── app/              # Inference application
├── notebooks/        # Training and evaluation notebooks
├── src/              # Model architecture definitions
├── training/         # Training scripts and configs
└── requirements.txt
```

---

## Citation

If you use this work, please cite:

```
Mogable, L. (2025). A Multi-Agent Ensemble Approach for Automatic Speech Recognition
in a Low Data Resource Environment. University of Johannesburg.
Supervisor: Prof. Duncan Coulter.
```

---

## Author

**Lesego Mogable**
[linkedin.com/in/lesego-mogable](https://www.linkedin.com/in/lesego-mogable)
