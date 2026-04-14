# DoH Topic Modeling Pipeline

A reproducible NLP pipeline for topic modeling of veterinary science literature, built for SLURM-based HPC clusters. The pipeline combines semantic embeddings, BERTopic clustering, and LLM-based topic labeling to extract and characterize research themes from large abstract bodies of texts.

## Overview

The pipeline processes a deduplicated corpus of scientific abstracts through the following stages:

```
00_01 Pairwise deduplication
00_02 Threshold selection
  ↓
01 Filter abstracts
02 Truncate to model max length
03 Embed with BioBERT (S-BioBERT)
04 Grid search (UMAP + HDBSCAN hyperparameters)
05 BERTopic full run
06 LLM topic labeling (Llama 3.1) ← manual review point
07 Cluster filtering (keep/remove topics)
  ↓
08 Re-embed filtered corpus
09 Retrain BERTopic on curated corpus
10 Final LLM topic labeling & summarization
11 Evaluate and compare runs
```

## Models Used

| Stage | Model | Description |
|---|---|---|
| Embeddings (steps 03, 08) | [`pritamdeka/S-BioBert-snli-multinli-stsb`](https://huggingface.co/pritamdeka/S-BioBert-snli-multinli-stsb) | BioBERT fine-tuned for semantic similarity via sentence-transformers |
| Topic modeling (steps 05, 09) | [BERTopic](https://maartengr.github.io/BERTopic/) | Transformer-based topic modeling with HDBSCAN clustering |
| Topic labeling (steps 06, 10) | [`meta-llama/Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) | Llama 3.1 8B for automatic topic label generation and summarization |

## Requirements

### Hardware
- GPU with CUDA support recommended (steps 03, 06, 08, 10 are GPU-accelerated)
- At least 48 GB RAM for grid search and BERTopic steps
- SLURM cluster with GPU partition

### Software

Install all dependencies:

```bash
pip install -r requirements.txt
```

> **GPU note:** `requirements.txt` includes `faiss-gpu`. If running on CPU only, replace it with `faiss-cpu`.

### HuggingFace Token

Steps 06 and 10 use Llama 3.1, which requires a HuggingFace account with access to Meta Llama models.

1. Request access at [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
2. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Export before running:

```bash
export HF_TOKEN="your_token_here"
```

## Directory Structure

```
.
├── pipeline/
│   ├── 00_01_pairwise_comparison.py / .sh   # Pairwise duplicate detection
│   ├── 00_02_threshold_selection.py / .sh   # Deduplication threshold
│   ├── 00_run_deduplication_pipeline.sh     # Orchestrator: steps 00_01 + 00_02
(pre-deduplication)
│   ├── 01_filter.py / .sh                   # Abstract filtering
│   ├── 02_truncate.py / .sh                 # Truncation to 512 tokens
│   ├── 03_embed.py / .sh                    # BioBERT embeddings
│   ├── 04_grid_search.py / .sh              # UMAP + HDBSCAN grid search
│   ├── 05_bertopic_full.py / .sh            # BERTopic on full corpus
│   ├── 06_llm_labeling.py / .sh             # Llama topic labeling (pre-filtering)
│   ├── 07_cluster_filtering.py / .sh        # Manual topic curation
│   ├── 08_re_embed.py / .sh                 # Re-embed filtered corpus
│   ├── 09_train_rerun_model.py / .sh        # Retrain BERTopic on curated corpus
│   ├── 10_label_topics.py / .sh             # Final Llama topic labeling
│   ├── 11_evaluate_runs.py / .sh            # Compare and evaluate runs
│   ├── run_phase1_pipeline.sh               # Orchestrator: steps 01–06
│   ├── run_phase2_pipeline.sh               # Orchestrator: steps 08–10
│   └── logs/                                # SLURM job logs (gitignored)
├── requirements.txt
├── LICENSE
└── README.md
```

## Usage

### Input

Place your deduplicated abstracts CSV in `input/` before running. The expected file is:

```
input/merged_cleaned_auto_dedup.csv
```

Required columns: `Abstract`, `Title`, `Year`, `Source` (configurable via script arguments).

### Running the pipeline

Submit individual steps or use the orchestrators:

```bash
cd pipeline/
# Deduplication preprocessing step
sbatch run_phase0_dedup_pipeline.sh

# Full phase 1 (deduplication → LLM labeling)
sbatch run_phase1_pipeline.sh

# After manual review of step 06 output → update TOPIC_CLASSIFICATION in 07_cluster_filtering.py
# Then run phase 2:
sbatch run_phase2_pipeline.sh
```

Or submit steps individually:

```bash
sbatch 01_filter.sh
sbatch --dependency=afterok:<JOB_ID> 02_truncate.sh
# ...
```

### Manual intervention (after step 06)

After step 06 completes, review `output/06_llm_labeling/topic_info_with_llm.csv` and update the `TOPIC_CLASSIFICATION` dictionary in `07_cluster_filtering.py` to mark topics as `Keep` or `Remove` based on:

- `LLM_DairySpecificityScore` (1–5)
- `LLM_PotentialNonDairyFocus`
- `LLM_LinkToDairyCattleHealth`

### Estimated runtime

| Phase | Steps | Estimated time |
|---|---|---|
| Phase 1 | 01–06 | ~28–38 hours |
| Manual review | — | variable |
| Phase 2 | 08–10 | ~13–21 hours |
| **Total** | | **~40–60 hours** |

## Configuration

Each shell script contains a `PATHS` section at the top. Paths are derived automatically relative to the script location — no editing required if you follow the standard directory layout. The only mandatory configuration is:

- **`HF_TOKEN`** environment variable (for steps 06 and 10)
- **`CONDA_ENV`** — defaults to `bertopic_gpu`, change if your environment has a different name
- **SBATCH headers** — update `--account`, `--mail-user`, and resource requests for your cluster

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
