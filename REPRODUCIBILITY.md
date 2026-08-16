# Reproducibility Guide — Nascenia Bengali Medical Dialogue Generation

This guide provides comprehensive instructions to reproduce the leaderboard submission, evaluate candidate model architectures head-to-head, and verify parameter-count compliance of the final Phase 2 submission.

---

## Dual-Architecture Strategy & Approach Summary

To maximize generation fidelity against the official composite metric ($0.5 \times \text{BERTScore\_F1} + 0.3 \times \text{Token\_F1} + 0.2 \times \text{ROUGE-L\_F1}$), we developed and compared two complementary architectures evaluated under identical conditions:

1. **Decoder-Only Pipeline (`titulm-llama-3.2-3b-v2.0` — 3.2B parameters)**:
   - **Stage 1 (Broad Mix SFT)**: Fine-tuned with QLoRA ($r=32$) on competition data augmented with normalized `shetumohanto/doctor_qa_bangla` to expand clinical vocabulary.
   - **Stage 2 (Official Anchor SFT)**: Fine-tuned on cleaned official `sft_train.csv` at a lower learning rate ($5 \times 10^{-5}$) to align response length and stylistic register.
   - **MBR Decoding**: Generates $k=4$ candidate responses and selects the most canonical answer via Minimum Bayes Risk over pairwise ROUGE-L with length filtering [65, 145 words].

2. **Encoder-Decoder Pipeline (`csebuetnlp/banglat5` — 247M parameters)**:
   - **Full Fine-Tuning**: End-to-end full parameter fine-tuning on `sft_train.csv` (no adapter approximation) following the published BanglaT5 sequence-to-sequence recipe (6 epochs, $\text{lr}=3\times 10^{-4}$, label smoothing=0.1).
   - **Bangla Normalization**: Integration of `csebuetnlp/normalizer` for Bengali Unicode standardisation.
   - **Deterministic Beam Search**: Decoded with `num_beams=5`, `no_repeat_ngram_size=3`, `length_penalty=0.6`.
   - **Safety Margin**: At 247M parameters, it is 12x smaller than the 3B cap, ensuring substantial compute efficiency and compliance safety.

3. **Data-Driven Champion Selection**:
   - `07_model_comparison.ipynb` evaluates both models against the 1,000-row held-out `sft_val.csv` ground truth.
   - Generates the official `submission.csv` using exclusively the winning champion architecture, adhering to Phase 2 single-model verification requirements.

---

## Environment Setup

The pipeline is designed to execute in a standard Kaggle Notebook GPU environment (T4 x1 or P100/A100). The required dependencies are installed within the notebooks:

- **Python**: `3.10`+
- **Core Dependencies**:
  - `transformers`, `datasets`, `accelerate`, `sentencepiece`
  - `git+https://github.com/csebuetnlp/normalizer`
  - `unsloth`, `peft`, `trl`, `bitsandbytes` (for decoder-only QLoRA)
  - `bert-score`, `rouge-score`

---

## Step-by-Step Reproduction Workflow

Execute the notebooks in the following sequence:

### 1. Data Ingestion & Splitting (`01_data_pipeline.ipynb`)
- **Action**: Loads and cleans official data, normalizes external Bengali medical QA data, and creates stratified 1,000-row validation and SFT splits.
- **Outputs**:
  - `/kaggle/working/sft_train.csv`
  - `/kaggle/working/sft_val.csv`
  - `/kaggle/working/train_plus_external_clean.csv`

### 2. Model Training (Run either or both in parallel)

#### A. Decoder-Only QLoRA Training (`02_train_qlora.ipynb`)
- **Action**: Runs 2-stage QLoRA fine-tuning on `titulm-llama-3.2-3b-v2.0` and exports merged weights.
- **Outputs**: `/kaggle/working/final_model/`

#### B. BanglaT5 Full Fine-Tuning (`02b_train_banglat5.ipynb`)
- **Action**: Runs 6-epoch full fine-tune of `csebuetnlp/banglat5` (247M parameters) with Bengali normalizer and dynamic padding.
- **Outputs**: `/kaggle/working/banglat5_final/`

### 3. Validation Inference & Scoring

#### A. Decoder-Only Validation (`03_inference_and_submit.ipynb`)
- **Action**: Generates MBR-reranked validation predictions for TituLLM-3B.
- **Outputs**: `/kaggle/working/val_predictions.csv`

#### B. BanglaT5 Validation (`03b_inference_banglat5.ipynb`)
- **Action**: Generates batched beam search validation predictions for BanglaT5 and prints comparative delta against decoder-only baseline.
- **Outputs**: `/kaggle/working/val_predictions_banglat5.csv`

### 4. Local Metric Validation (`04_local_validation.ipynb`)
- **Action**: Runs metric harness unit tests and analyzes error modes on lowest-scoring validation rows.

### 5. Head-to-Head Comparison & Final Submission (`07_model_comparison.ipynb`)
- **Action**: Computes composite, BERTScore, Token F1, and ROUGE-L metrics across all validation rows, evaluates per-row win rates and oracle upper bounds, designates the champion architecture, outputs `/kaggle/working/champion_model.json`, and generates the verified `/kaggle/working/submission.csv`.

### 6. Phase 2 Packaging (`05_phase2_packaging.ipynb`)
- **Action**: Reads `champion_model.json`, verifies parameter count compliance ($< 3.0\text{B}$), generates model card (`model_card.md`), and archives the model and notebooks into `/kaggle/working/phase2_submission.zip`.

### 7. Retrieval Augmentation (Exploratory) (`06_retrieval_addon.ipynb`)
- **Action**: Explores query back-translation and semantic retrieval over HealthCareMagic.

---

## Open-Source License

This submission is distributed under the **Apache-2.0 License** in compliance with competition requirements. See the [`LICENSE`](file:///y:/4-1/lab%20slides/al%20mahmud%20sir/nascenia_1/LICENSE) file at the project root for details.

---

## Limitations and Safety Notice

- **Competition Optimization**: Decoding parameters and length calibrations are optimized for the reference distributions of Nascenia's automated evaluation harness.
- **Non-Clinical Disclaimer**: This model is an artificial intelligence research submission trained on translated and synthetically curated dialogues. It does not provide medical diagnoses or treatment plans. Users must consult licensed healthcare professionals for medical decisions.
