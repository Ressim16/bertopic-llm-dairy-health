#!/usr/bin/env python3
"""
Step 5b: Evaluate Trained BERTopic Models (Initial Run and Re-run)

Computes evaluation metrics for both trained BERTopic models:
  - Step 05: full corpus, before manual topic curation
  - Step 09: filtered corpus, after manual topic curation (re-run)

Metrics computed:
  - Silhouette (UMAP, Euclidean): computed on the exact UMAP low-dimensional
    projection saved immediately after fit_transform() in Steps 05/09.
    Uses the same space and distance metric as the grid search — direct
    comparison across grid search, initial run, and re-run is valid.

  - Coherence c_npmi: topic semantic coherence via Gensim (identical to
    grid-search implementation).

  - Diversity: unique words / total words across topic top-N word lists
    (identical to grid-search implementation).

  - Coverage: 1 - outlier rate.

Inputs (resolved automatically from --output-base):
  Step 05:
    - 05_bertopic_full/bertopic_model/
    - 02_truncate/abstracts_with_truncation.csv
    - 03_embeddings/embeddings.npy
    - 05_bertopic_full/topic_assignments_frozen.csv
    - 05_bertopic_full/umap_embeddings.npy
  Step 09:
    - 09_train_rerun/bertopic_model_rerun/
    - 07_filtered/abstracts_topic_filtered.csv
    - 08_re_embed/embeddings_rerun.npy
    - 09_train_rerun/topic_assignments_frozen_rerun.csv
    - 09_train_rerun/umap_embeddings_rerun.npy

Outputs:
  - evaluation_results.json: All metrics for both runs
  - evaluation_results.csv:  Tabular summary for easy comparison
  - step_5b_evaluate_runs.log: Execution log

Author: Reda Zahri
Date: 2025
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from bertopic import BERTopic
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.metrics import silhouette_score


# ============================================================
# LOGGING
# ============================================================

def configure_logging(log_file: Path = None) -> logging.Logger:
    """Configure logging with file and console output."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(__name__)


def silence_noisy_loggers():
    """Reduce verbosity from third-party libraries."""
    for name in ["numba", "umap", "pynndescent", "llvmlite", "gensim", "hdbscan"]:
        logging.getLogger(name).setLevel(logging.WARNING)


# ============================================================
# METRIC HELPERS  (identical to 04_grid_search.py)
# ============================================================

def simple_tokenize(text: str):
    """Simple regex tokenizer for coherence calculation."""
    return re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", (text or "").lower())


def compute_topic_words(topic_model: BERTopic, top_n: int = 10) -> list:
    """Extract top-N word lists per topic, excluding the outlier topic (-1)."""
    topic_words = []
    for k, items in topic_model.get_topics().items():
        if k == -1:
            continue
        words = [w for (w, _) in items[:top_n]]
        if words:
            topic_words.append(words)
    return topic_words


def topic_diversity(topic_word_lists: list) -> float:
    """Ratio of unique words to total words across all topic word lists."""
    if not topic_word_lists:
        return 0.0
    total = sum(len(ws) for ws in topic_word_lists)
    if total == 0:
        return 0.0
    unique = len({w for ws in topic_word_lists for w in ws})
    return unique / total


def safe_silhouette(umap_space: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette score in UMAP space (Euclidean), excluding outliers and singletons.
    Identical to the implementation in 04_grid_search.py.

    Args:
        umap_space: 2-D array of shape (n_docs, n_umap_components).
        labels:     1-D integer array of topic assignments; -1 = outlier.
    """
    labels = np.array(labels)
    mask = labels != -1
    if mask.sum() < 3:
        return float("nan")

    sizes = dict(zip(*np.unique(labels[mask], return_counts=True)))
    valid_idx = np.array(
        [i for i, lab in enumerate(labels) if lab != -1 and sizes.get(lab, 0) > 1],
        dtype=int,
    )

    if valid_idx.size < 3 or len(np.unique(labels[valid_idx])) < 2:
        return float("nan")

    try:
        return float(
            silhouette_score(umap_space[valid_idx], labels[valid_idx], metric="euclidean")
        )
    except Exception:
        return float("nan")


def compute_coherence_c_npmi(topic_word_lists: list, tokenized_docs: list) -> float:
    """
    Compute Gensim c_npmi coherence over all non-outlier topics.
    Identical to the implementation in 04_grid_search.py.
    """
    if not topic_word_lists or not tokenized_docs:
        return float("nan")

    logger = logging.getLogger(__name__)
    n_docs = len(tokenized_docs)
    no_below = min(5, max(2, int(0.001 * n_docs)))

    dictionary = Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=no_below, no_above=0.9, keep_n=1_000_000)

    vocab = dictionary.token2id
    valid_topics = [
        [t for t in words if t in vocab]
        for words in topic_word_lists
    ]
    valid_topics = [w for w in valid_topics if len(w) >= 2]

    if len(valid_topics) < 2:
        return float("nan")

    corpus = [dictionary.doc2bow(toks) for toks in tokenized_docs]

    try:
        cm = CoherenceModel(
            topics=valid_topics,
            texts=tokenized_docs,
            corpus=corpus,
            dictionary=dictionary,
            coherence="c_npmi",
        )
        return float(cm.get_coherence())
    except Exception as e:
        logger.debug(f"Coherence computation failed: {e}")
        return float("nan")


# ============================================================
# PER-RUN EVALUATION
# ============================================================

def evaluate_run(
    run_label: str,
    model_dir: Path,
    input_csv: Path,
    embeddings_path: Path,
    frozen_assignments_csv: Path,
    umap_path: Path,
    text_column: str,
    top_n_words: int,
    logger: logging.Logger,
) -> dict:
    """
    Evaluate a single trained BERTopic run.

    Topic assignments are read from the frozen_assignments CSV produced by
    Steps 05 / 09 rather than recomputed, preserving manual curation decisions.

    Silhouette is computed on the exact UMAP projection saved during training
    (umap_model.embedding_), using Euclidean distance — identical to the
    grid-search implementation in 04_grid_search.py.

    Args:
        run_label:              Human-readable identifier.
        model_dir:              Directory containing the saved BERTopic model.
        input_csv:              CSV with the corpus abstracts for this run.
        embeddings_path:        .npy S-BioBERT embeddings aligned with input_csv.
                                Used only to validate alignment; not used for silhouette.
        frozen_assignments_csv: CSV with frozen topic assignments.
        umap_path:              .npy file containing the UMAP projection saved
                                immediately after fit_transform() in Steps 05/09.
        text_column:            Column name for abstract text in input_csv.
        top_n_words:            Number of top words per topic.
        logger:                 Logger instance.

    Returns:
        Dictionary of computed metrics.
    """
    logger.info("=" * 70)
    logger.info(f"Evaluating run: {run_label}")
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    logger.info(f"Loading BERTopic model from: {model_dir}")
    topic_model = BERTopic.load(str(model_dir))
    logger.info("Model loaded.")

    # ------------------------------------------------------------------
    # 2. Load corpus texts
    # ------------------------------------------------------------------
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    logger.info(f"Loading corpus from: {input_csv}")
    df = pd.read_csv(input_csv)

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in {input_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    texts = df[text_column].fillna("").astype(str).tolist()
    logger.info(f"Loaded {len(texts):,} documents.")

    # ------------------------------------------------------------------
    # 3. Load frozen topic assignments
    # ------------------------------------------------------------------
    if not frozen_assignments_csv.exists():
        raise FileNotFoundError(f"Frozen assignments not found: {frozen_assignments_csv}")

    logger.info(f"Loading frozen assignments from: {frozen_assignments_csv}")
    assignments_df = pd.read_csv(frozen_assignments_csv)

    topic_col_candidates = [c for c in assignments_df.columns if c.lower() == "topic"]
    if not topic_col_candidates:
        raise ValueError(
            f"No 'Topic' column found in {frozen_assignments_csv}. "
            f"Available columns: {list(assignments_df.columns)}"
        )
    topic_col = topic_col_candidates[0]

    if len(assignments_df) != len(texts):
        raise ValueError(
            f"Frozen assignments ({len(assignments_df)}) and corpus "
            f"({len(texts)}) must have the same length."
        )

    labels = assignments_df[topic_col].values.astype(int)
    logger.info(
        f"Assignments loaded. Topics: {len(np.unique(labels[labels != -1]))} "
        f"(+ outlier class -1)"
    )

    # ------------------------------------------------------------------
    # 4. Validate S-BioBERT embeddings alignment
    # ------------------------------------------------------------------
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    logger.info(f"Validating embeddings alignment: {embeddings_path}")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.shape[0] != len(texts):
        raise ValueError(
            f"Embeddings ({embeddings.shape[0]}) and texts ({len(texts)}) "
            "must have the same length."
        )
    logger.info(f"Embeddings shape: {embeddings.shape}  OK.")

    # ------------------------------------------------------------------
    # 5. Load UMAP projection (saved during training)
    # ------------------------------------------------------------------
    if not umap_path.exists():
        raise FileNotFoundError(
            f"UMAP embeddings not found: {umap_path}\n"
            "Make sure Steps 05/09 were run with the umap_model.embedding_ save patch."
        )

    logger.info(f"Loading UMAP embeddings from: {umap_path}")
    umap_space = np.load(umap_path)
    logger.info(f"UMAP embeddings shape: {umap_space.shape}")

    if umap_space.shape[0] != len(texts):
        raise ValueError(
            f"UMAP embeddings ({umap_space.shape[0]}) and corpus ({len(texts)}) "
            "must have the same length."
        )

    # ------------------------------------------------------------------
    # 6. Compute metrics
    # ------------------------------------------------------------------
    logger.info("Computing metrics...")

    # Silhouette — UMAP space, Euclidean (identical to grid search)
    logger.info("  Computing silhouette (UMAP space, Euclidean)...")
    sil = safe_silhouette(umap_space, labels)
    logger.info(
        f"  Silhouette: {sil:.4f}"
        if not np.isnan(sil)
        else "  Silhouette: nan"
    )

    # Topic word lists
    topic_word_lists = compute_topic_words(topic_model, top_n_words)
    n_topics = len(topic_word_lists)
    logger.info(f"  Topics (excl. outlier): {n_topics}")

    # Coherence
    logger.info("  Tokenizing documents for coherence...")
    tokenized_docs = [simple_tokenize(t) for t in texts]
    logger.info("  Computing c_npmi coherence...")
    coh = compute_coherence_c_npmi(topic_word_lists, tokenized_docs)
    logger.info(
        f"  Coherence c_npmi: {coh:.4f}"
        if not np.isnan(coh)
        else "  Coherence c_npmi: nan"
    )

    # Diversity
    div = topic_diversity(topic_word_lists)
    logger.info(f"  Diversity: {div:.4f}")

    # Coverage
    n_outliers = int(np.sum(labels == -1))
    coverage = 1.0 - n_outliers / len(labels)
    logger.info(f"  Coverage: {coverage:.4f}  (outliers: {n_outliers:,} / {len(labels):,})")

    # ------------------------------------------------------------------
    # 7. Assemble result
    # ------------------------------------------------------------------
    def _fmt(v):
        return round(float(v), 4) if v is not None and not np.isnan(v) else None

    result = {
        "run_label":        run_label,
        "n_documents":      len(texts),
        "n_topics":         n_topics,
        "n_outliers":       n_outliers,
        "coverage":         _fmt(coverage),
        "silhouette":       _fmt(sil),
        "coherence_c_npmi": _fmt(coh),
        "diversity":        _fmt(div),
        "model_dir":        str(model_dir),
        "input_csv":        str(input_csv),
        "umap_path":        str(umap_path),
        "evaluated_at":     datetime.now().isoformat(),
    }

    logger.info("")
    logger.info(f"Run '{run_label}' evaluation complete.")
    logger.info("")

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step 5b: Evaluate trained BERTopic models (initial run and re-run)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python 5b_evaluate_runs.py \\
        --output-base /path/to/output \\
        --output-dir  /path/to/output/5b_evaluate_runs \\
        --text-column Abstract

Paths inferred automatically from --output-base:
  Step 05 model:       <output-base>/05_bertopic_full/bertopic_model/
  Step 05 corpus:      <output-base>/02_truncate/abstracts_with_truncation.csv
  Step 05 embeddings:  <output-base>/03_embeddings/embeddings.npy
  Step 05 assignments: <output-base>/05_bertopic_full/topic_assignments_frozen.csv
  Step 05 UMAP:        <output-base>/05_bertopic_full/umap_embeddings.npy
  Step 09 model:       <output-base>/09_train_rerun/bertopic_model_rerun/
  Step 09 corpus:      <output-base>/07_filtered/abstracts_topic_filtered.csv
  Step 09 embeddings:  <output-base>/08_re_embed/embeddings_rerun.npy
  Step 09 assignments: <output-base>/09_train_rerun/topic_assignments_frozen_rerun.csv
  Step 09 UMAP:        <output-base>/09_train_rerun/umap_embeddings_rerun.npy

Override any individual path with the corresponding --step05-* / --step09-* flag.
        """,
    )

    parser.add_argument(
        "--output-base", type=Path, required=True,
        help="Base output directory (parent of 02_truncate/, 05_bertopic_full/, etc.)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for this evaluation script",
    )

    # Step 05 path overrides
    parser.add_argument("--step05-model-dir",   type=Path, default=None)
    parser.add_argument("--step05-input-csv",   type=Path, default=None)
    parser.add_argument("--step05-embeddings",  type=Path, default=None)
    parser.add_argument("--step05-assignments", type=Path, default=None)
    parser.add_argument("--step05-umap",        type=Path, default=None,
                        help="UMAP projection .npy saved during Step 05 training")

    # Step 09 path overrides
    parser.add_argument("--step09-model-dir",   type=Path, default=None)
    parser.add_argument("--step09-input-csv",   type=Path, default=None)
    parser.add_argument("--step09-embeddings",  type=Path, default=None)
    parser.add_argument("--step09-assignments", type=Path, default=None)
    parser.add_argument("--step09-umap",        type=Path, default=None,
                        help="UMAP projection .npy saved during Step 09 training")

    parser.add_argument(
        "--text-column", type=str, default="Abstract",
        help="Column containing abstract text (default: Abstract)",
    )
    parser.add_argument(
        "--top-n-words", type=int, default=10,
        help="Number of top words per topic for coherence/diversity (default: 10)",
    )

    args = parser.parse_args()
    base = args.output_base

    # Resolve paths
    step05_model       = args.step05_model_dir   or base / "05_bertopic_full" / "bertopic_model"
    step05_input       = args.step05_input_csv   or base / "02_truncate" / "abstracts_with_truncation.csv"
    step05_embeddings  = args.step05_embeddings  or base / "03_embeddings" / "embeddings.npy"
    step05_assignments = args.step05_assignments or base / "05_bertopic_full" / "topic_assignments_frozen.csv"
    step05_umap        = args.step05_umap        or base / "05_bertopic_full" / "umap_embeddings.npy"

    step09_model       = args.step09_model_dir   or base / "09_train_rerun" / "bertopic_model_rerun"
    step09_input       = args.step09_input_csv   or base / "07_filtered" / "abstracts_topic_filtered.csv"
    step09_embeddings  = args.step09_embeddings  or base / "08_re_embed" / "embeddings_rerun.npy"
    step09_assignments = args.step09_assignments or base / "09_train_rerun" / "topic_assignments_frozen_rerun.csv"
    step09_umap        = args.step09_umap        or base / "09_train_rerun" / "umap_embeddings_rerun.npy"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.output_dir / "step_5b_evaluate_runs.log")
    silence_noisy_loggers()

    logger.info("=" * 70)
    logger.info("STEP 5b: EVALUATE BERTOPIC RUNS")
    logger.info("=" * 70)
    logger.info(f"Output dir:    {args.output_dir}")
    logger.info(f"Text column:   {args.text_column}")
    logger.info(f"Top-N words:   {args.top_n_words}")
    logger.info("")
    logger.info("Step 05 paths:")
    logger.info(f"  Model:       {step05_model}")
    logger.info(f"  Corpus:      {step05_input}")
    logger.info(f"  Embeddings:  {step05_embeddings}")
    logger.info(f"  Assignments: {step05_assignments}")
    logger.info(f"  UMAP:        {step05_umap}")
    logger.info("")
    logger.info("Step 09 paths:")
    logger.info(f"  Model:       {step09_model}")
    logger.info(f"  Corpus:      {step09_input}")
    logger.info(f"  Embeddings:  {step09_embeddings}")
    logger.info(f"  Assignments: {step09_assignments}")
    logger.info(f"  UMAP:        {step09_umap}")
    logger.info("=" * 70)
    logger.info("")

    try:
        results = []

        r05 = evaluate_run(
            run_label="initial_run",
            model_dir=step05_model,
            input_csv=step05_input,
            embeddings_path=step05_embeddings,
            frozen_assignments_csv=step05_assignments,
            umap_path=step05_umap,
            text_column=args.text_column,
            top_n_words=args.top_n_words,
            logger=logger,
        )
        results.append(r05)

        r09 = evaluate_run(
            run_label="rerun_after_curation",
            model_dir=step09_model,
            input_csv=step09_input,
            embeddings_path=step09_embeddings,
            frozen_assignments_csv=step09_assignments,
            umap_path=step09_umap,
            text_column=args.text_column,
            top_n_words=args.top_n_words,
            logger=logger,
        )
        results.append(r09)

        # Save outputs
        json_path = args.output_dir / "evaluation_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved: {json_path}")

        csv_path = args.output_dir / "evaluation_results.csv"
        pd.DataFrame(results).to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")

        # Summary table
        logger.info("")
        logger.info("=" * 70)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"{'Metric':<25} {'Initial run (05)':>18} {'Re-run (09)':>18}")
        logger.info("-" * 63)

        metrics = [
            ("n_documents",      "Documents"),
            ("n_topics",         "Topics"),
            ("n_outliers",       "Outliers"),
            ("coverage",         "Coverage"),
            ("silhouette",       "Silhouette (UMAP)"),
            ("coherence_c_npmi", "Coherence (c_npmi)"),
            ("diversity",        "Diversity"),
        ]

        for key, label in metrics:
            v05 = r05.get(key)
            v09 = r09.get(key)
            fmt05 = f"{v05:>18}" if isinstance(v05, int) else (f"{v05:>18.4f}" if v05 is not None else f"{'N/A':>18}")
            fmt09 = f"{v09:>18}" if isinstance(v09, int) else (f"{v09:>18.4f}" if v09 is not None else f"{'N/A':>18}")
            logger.info(f"{label:<25} {fmt05} {fmt09}")

        logger.info("=" * 70)
        logger.info("")
        logger.info("NOTE: Silhouette computed in UMAP space (Euclidean), same as")
        logger.info("      grid search. Direct comparison across all three is valid.")
        logger.info("=" * 70)
        logger.info("")
        logger.info("STEP 5b COMPLETED SUCCESSFULLY")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        logger.error("Step 5b FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())