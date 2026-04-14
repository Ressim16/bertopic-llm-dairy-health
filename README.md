# BERTopic-LLM Pipeline for Dairy Cattle Health Literature

> A reproducible, large-scale topic modeling pipeline for thematic mapping of veterinary research literature.  
> Combines S-BioBERT embeddings, BERTopic (UMAP + HDBSCAN), and LLaMA 3.1 for automated topic labeling.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![BERTopic](https://img.shields.io/badge/BERTopic-0.16%2B-orange)
![LLaMA](https://img.shields.io/badge/LLaMA-3.1--8B--Instruct-purple)
![SLURM](https://img.shields.io/badge/HPC-SLURM-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

---

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Requirements](#requirements)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Estimated Runtime](#estimated-runtime)
- [Data Availability](#data-availability)
- [Citation](#citation)
- [License](#license)

---

## Overview

This pipeline was developed to support a large-scale bibliometric and thematic analysis of the dairy cattle health research literature (~74,000 abstracts, 2000–2025). It is designed to run on SLURM-based HPC clusters with GPU support, and is fully parameterized for reproducibility.

The pipeline covers the full workflow:
- Semantic deduplication of raw abstract corpora
- S-BioBERT embedding generation
- Hyperparameter optimization (UMAP + HDBSCAN)
- BERTopic topic modeling
- LLM-assisted topic labeling via LLaMA 3.1 8B Instruct
- Manual curation interface
- Model retraining on curated corpus
- Quantitative evaluation of topic quality

---

## Models

| Stage | Model | Source |
|---|---|---|
| Embeddings (steps 03, 08) | `pritamdeka/S-BioBert-snli-multinli-stsb` | [HuggingFace](https://huggingface.co/pritamdeka/S-BioBert-snli-multinli-stsb) |
| Topic modeling (steps 05, 09) | BERTopic | [GitHub](https://maartengr.github.io/BERTopic/) |
| Topic labeling (steps 06, 10) | `meta-llama/Meta-Llama-3.1-8B-Instruct` | [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |

---

## Requirements

### Hardware

- GPU with CUDA support (required for steps 03, 06, 08, 10)
- ≥ 48 GB RAM (grid search and BERTopic steps)
- SLURM cluster with GPU partition

### Software

```bash
pip install -r requirements.txt
```

> **Note:** `requirements.txt` includes `faiss-gpu`. Replace with `faiss-cpu` if running without GPU.

### HuggingFace Access Token

Steps 06 and 10 use LLaMA 3.1, which requires approved access to Meta's model.

1. Request access at [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
2. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Export before running:

```bash
export HF_TOKEN="your_token_here"
```

---

## Directory Structure

```
.
├── pipeline/
│   ├── 00_01_pairwise_comparison.py / .sh    # Pairwise duplicate detection (FAISS)
│   ├── 00_02_threshold_selection.py / .sh    # Deduplication threshold optimization
│   ├── run_phase0_dedup_pipeline.sh          # Orchestrator: steps 00_01 + 00_02
│   ├── 01_filter.py / .sh                    # Abstract filtering
│   ├── 02_truncate.py / .sh                  # Truncation to 512 tokens
│   ├── 03_embed.py / .sh                     # S-BioBERT embedding generation
│   ├── 04_grid_search.py / .sh               # UMAP + HDBSCAN hyperparameter search
│   ├── 05_bertopic_full.py / .sh             # BERTopic run on full corpus
│   ├── 06_llm_labeling.py / .sh              # LLaMA topic labeling (pre-curation)
│   ├── 07_cluster_filtering.py / .sh         # Manual topic curation
│   ├── 08_re_embed.py / .sh                  # Re-embedding of curated corpus
│   ├── 09_train_rerun_model.py / .sh         # BERTopic retraining on curated corpus
│   ├── 10_label_topics.py / .sh              # Final LLaMA topic labeling
│   ├── 11_evaluate_runs.py / .sh             # Quantitative run evaluation
│   ├── run_phase1_pipeline.sh                # Orchestrator: steps 01–06
│   ├── run_phase2_pipeline.sh                # Orchestrator: steps 08–10
│   └── logs/                                 # SLURM job logs (gitignored)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Usage

### Input

Place your abstract corpus CSV in `input/` before running:

```
input/merged_cleaned_auto_dedup.csv
```

Expected columns: `Abstract`, `Title`, `Year`, `Source` (configurable via script arguments).  
Alternatively, run Phase 0 on the raw, unprocessed corpus to generate this file automatically.

### Running the pipeline

```bash
cd pipeline/

# Phase 0 — Deduplication (if starting from raw corpus)
sbatch run_phase0_dedup_pipeline.sh

# Phase 1 — Topic discovery
sbatch run_phase1_pipeline.sh

# → Manual review: update TOPIC_CLASSIFICATION in 07_cluster_filtering.py

# Phase 2 — Refinement and evaluation
sbatch run_phase2_pipeline.sh
```

Or submit steps individually with SLURM dependencies:

```bash
sbatch 01_filter.sh
sbatch --dependency=afterok:<JOB_ID> 02_truncate.sh
# ... and so on
```

### Manual curation step (after step 06)

After step 06 completes, review:

```
output/06_llm_labeling/topic_info_with_llm.csv
```

Update the `TOPIC_CLASSIFICATION` dictionary in `07_cluster_filtering.py` to mark each topic as `Keep` or `Remove` based on:

- `LLM_DairySpecificityScore` (1–5)
- `LLM_PotentialNonDairyFocus`
- `LLM_LinkToDairyCattleHealth`

---

## Configuration

Each shell script contains a `PATHS` section at the top. Paths are derived automatically relative to the script location — **no editing required** if the standard directory layout is followed.

Mandatory configuration before first run:

| Parameter | Location | Description |
|---|---|---|
| `HF_TOKEN` | Environment variable | HuggingFace access token (steps 06, 10) |
| `CONDA_ENV` | Each `.sh` script | Conda environment name (default: `bertopic_gpu`) |
| `--account` | SBATCH headers | Your SLURM account name |
| `--mail-user` | SBATCH headers | Email for job notifications |

---

## Estimated Runtime

| Phase | Steps | Estimated time |
|---|---|---|
| Phase 0 | 00_01 – 00_02 | ~4–8 hours |
| Phase 1 | 01 – 06 | ~28–38 hours |
| Manual review | — | variable |
| Phase 2 | 07 – 11 | ~13–21 hours |
| **Total** | | **~45–67 hours** |

> Runtimes were benchmarked on an NVIDIA H100 GPU (UBELIX HPC cluster, University of Bern).

---

## Data Availability

The abstract corpus used in this study is **not provided** in this repository. The ~74,000 abstracts were retrieved from bibliographic databases (PubMed and Scopus) and remain subject to the copyright terms of their respective publishers and journals. Redistribution of this content is not permitted under those terms.

The pipeline is fully reproducible on any corpus of scientific abstracts that follows the expected input format (see [Usage](#usage)). Researchers wishing to replicate the study should retrieve the abstracts independently using equivalent search queries, which are described in detail in the associated publication.

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@unpublished{Reda Zahri BERTOpic—LLM},
  title   = {<Paper title>},
  author  = {<Authors>},
  note    = {<Note>},
  year    = {<Year>}
}
```

> This section will be updated upon publication.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
