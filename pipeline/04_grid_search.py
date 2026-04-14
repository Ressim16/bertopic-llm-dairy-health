#!/usr/bin/env python3
"""
Step 04: BERTopic Hyperparameter Grid Search

Performs grid search over BERTopic hyperparameters to find optimal configuration:
- Vectorizer: min_df, max_df
- UMAP: n_neighbors, n_components  
- HDBSCAN: min_cluster_size, min_samples

Evaluates using:
- Silhouette score (cluster quality in UMAP space)
- Coherence c_npmi (topic semantic coherence)
- Diversity (unique words across topics)
- Coverage (1 - outlier rate)

Inputs:
    - abstracts_with_truncation.csv: Abstracts from Step 02
    - embeddings.npy: Embeddings from Step 03

Outputs:
    - grid_search_results.csv: All configurations with metrics
    - grid_search_results_ranked.csv: Ranked by average rank
    - grid_search_summary.json: Best configuration details
    - step_04_grid_search.log: Execution log

Author: Reda Zahri
Date: 2025
"""

import argparse
import gc
import json
import logging
import os
import re
import sys
from pathlib import Path

# Reduce nondeterminism from thread scheduling / BLAS / numba on HPC
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import hdbscan
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from umap import UMAP


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

def configure_logging(log_file: Path = None) -> logging.Logger:
    """Configure logging with file and console output."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)


def silence_noisy_loggers():
    """Reduce verbosity from third-party libraries."""
    for name in [
        "numba",
        "umap",
        "pynndescent",
        "llvmlite",
        "gensim",
        "hdbscan",
    ]:
        logging.getLogger(name).setLevel(logging.WARNING)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Optional: if torch is installed (won't fail otherwise)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def simple_tokenize(text: str):
    """Simple regex tokenizer for coherence calculation."""
    return re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", (text or "").lower())


def compute_topic_words(topic_model, top_n=10):
    """Extract topic word lists excluding outliers."""
    topics = topic_model.get_topics()
    topic_words = []
    for k, items in topics.items():
        if k == -1:
            continue
        words = [w for (w, _) in items[:top_n]]
        if words:
            topic_words.append(words)
    return topic_words


def topic_diversity(topic_word_lists):
    """Calculate topic diversity: unique words / total words."""
    if not topic_word_lists:
        return 0.0
    total = sum(len(ws) for ws in topic_word_lists)
    if total == 0:
        return 0.0
    unique = len(set(w for ws in topic_word_lists for w in ws))
    return unique / total


def safe_silhouette(umap_embeds, labels):
    """Silhouette score excluding outliers and singletons."""
    labels = np.array(labels)
    mask = labels != -1
    if mask.sum() < 3:
        return np.nan
    
    # Filter singletons
    sizes = dict(zip(*np.unique(labels[mask], return_counts=True)))
    valid_idx = []
    for i, lab in enumerate(labels):
        if lab == -1:
            continue
        if sizes.get(lab, 0) > 1:
            valid_idx.append(i)
    
    valid_idx = np.array(valid_idx, dtype=int)
    if valid_idx.size < 3 or len(np.unique(labels[valid_idx])) < 2:
        return np.nan
    
    try:
        return float(silhouette_score(umap_embeds[valid_idx], labels[valid_idx], metric="euclidean"))
    except Exception:
        return np.nan


def compute_coherence_c_npmi(topic_word_lists, tokenized_docs):
    """Compute Gensim c_npmi coherence."""
    if not topic_word_lists:
        return np.nan
    if not tokenized_docs:
        return np.nan

    logger = logging.getLogger(__name__)

    n_docs = len(tokenized_docs)

    # Make coherence more robust on subsets: keep no_below <= 5 but not too strict on small n
    no_below = min(5, max(2, int(0.001 * n_docs)))
    no_above = 0.9

    dictionary = Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=1000000)

    # Filter topic words to those that survive dictionary filtering
    valid_topics = []
    vocab = dictionary.token2id
    for words in topic_word_lists:
        w = [t for t in words if t in vocab]
        if len(w) >= 2:
            valid_topics.append(w)

    if len(valid_topics) < 2:
        return np.nan

    corpus = [dictionary.doc2bow(toks) for toks in tokenized_docs]

    try:
        cm = CoherenceModel(
            topics=valid_topics,
            texts=tokenized_docs,
            corpus=corpus,
            dictionary=dictionary,
            coherence="c_npmi"
        )
        return float(cm.get_coherence())
    except Exception as e:
        logger.debug(f"Coherence computation failed: {e}")
        return np.nan


def rank_and_choose_best(results_df):
    """Rank configurations and return best by average rank."""
    df = results_df.copy()
    
    # Rank each metric (higher is better)
    for col in ["silhouette", "coherence_c_npmi", "diversity", "coverage"]:
        df[col + "_rank"] = (-df[col]).rank(method="min", na_option="bottom")
    
    # Penalize configurations with >40% outliers
    df["penalty"] = np.where(df["coverage"] < 0.6, 10000, 0)
    
    # Calculate average rank with penalty
    df["avg_rank"] = df[["silhouette_rank", "coherence_c_npmi_rank", 
                         "diversity_rank", "coverage_rank"]].mean(axis=1) + df["penalty"]
    
    # Sort by average rank, then by individual metrics for tie-breaking
    df = df.sort_values(["avg_rank", "coverage_rank", "silhouette_rank", 
                         "coherence_c_npmi_rank", "diversity_rank"])
    
    return df


# ============================================================
# MAIN GRID SEARCH
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step 04: BERTopic hyperparameter grid search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python 04_grid_search.py \\
        --input-file /path/to/02_truncate/abstracts_with_truncation.csv \\
        --embeddings /path/to/03_embeddings/embeddings.npy \\
        --output-dir /path/to/output/04_grid_search \\
        --subset 5000 \\
        --seed 42
        """
    )
    
    parser.add_argument('--input-file', type=Path, required=True, 
                       help='Input CSV from Step 02')
    parser.add_argument('--embeddings', type=Path, required=True, 
                       help='Embeddings from Step 03')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory')
    parser.add_argument('--text-column', type=str, default='Abstract',
                       help='Column containing text (default: Abstract)')
    parser.add_argument('--subset', type=int, default=5000, 
                       help='Use subset for grid search (0=full dataset, default: 5000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--top-n-words', type=int, default=10,
                       help='Number of words per topic (default: 10)')
    parser.add_argument('--max-features', type=int, default=30000,
                       help='Max vocabulary size (default: 30000)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logger = configure_logging(args.output_dir / "step_04_grid_search.log")
    silence_noisy_loggers()
    
    # Log header
    logger.info("=" * 70)
    logger.info("STEP 04: BERTOPIC HYPERPARAMETER GRID SEARCH")
    logger.info("=" * 70)
    logger.info(f"Input file:    {args.input_file}")
    logger.info(f"Embeddings:    {args.embeddings}")
    logger.info(f"Output dir:    {args.output_dir}")
    logger.info(f"Text column:   {args.text_column}")
    logger.info(f"Subset:        {args.subset if args.subset > 0 else 'Full dataset'}")
    logger.info(f"Seed:          {args.seed}")
    logger.info("=" * 70)
    logger.info("")
    
    try:
        # Set seed
        set_seed(args.seed)
        logger.info(f"Random seed set to: {args.seed}")
        
        # Validate input files
        if not args.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        if not args.embeddings.exists():
            raise FileNotFoundError(f"Embeddings not found: {args.embeddings}")
        
        # Load data
        logger.info("Loading data...")
        df = pd.read_csv(args.input_file)
        
        if args.text_column not in df.columns:
            raise ValueError(
                f"Column '{args.text_column}' not found. "
                f"Available: {list(df.columns)}"
            )
        
        texts_full = df[args.text_column].fillna("").astype(str).tolist()
        
        logger.info("Loading embeddings...")
        emb_full = np.load(args.embeddings, mmap_mode="r")
        
        if emb_full.shape[0] != len(texts_full):
            raise ValueError(
                f"Embeddings ({emb_full.shape[0]}) and texts ({len(texts_full)}) "
                f"must have same length"
            )
        
                # Subset if requested
        if args.subset and args.subset > 0 and args.subset < len(texts_full):
            subset_idx_file = args.output_dir / "subset_idx.npy"

            if subset_idx_file.exists():
                idx = np.load(subset_idx_file)
                idx = np.array(idx, dtype=int)

                # Validate loaded indices: check size AND that indices are within bounds
                max_valid_idx = len(texts_full) - 1
                indices_valid = (idx.size == args.subset) and (idx.max() <= max_valid_idx)
                
                if not indices_valid:
                    if idx.size != args.subset:
                        logger.warning(
                            f"Existing subset_idx.npy has size {idx.size}, expected {args.subset}. Regenerating."
                        )
                    if idx.max() > max_valid_idx:
                        logger.warning(
                            f"Existing subset_idx.npy has max index {idx.max()}, but dataset has only {len(texts_full)} documents. Regenerating."
                        )
                    rng = np.random.RandomState(args.seed)
                    idx = rng.choice(len(texts_full), size=args.subset, replace=False)
                    idx = np.sort(idx)
                    np.save(subset_idx_file, idx)
                    logger.info(f"Regenerated subset indices and saved to {subset_idx_file}")
            else:
                rng = np.random.RandomState(args.seed)
                idx = rng.choice(len(texts_full), size=args.subset, replace=False)
                idx = np.sort(idx)
                np.save(subset_idx_file, idx)

            texts = [texts_full[i] for i in idx]
            embeddings = emb_full[idx]
            logger.info(f"Using subset: {len(texts):,} documents (idx saved to {subset_idx_file})")
        else:
            texts = texts_full
            embeddings = emb_full[:len(texts)]
            logger.info(f"Using full dataset: {len(texts):,} documents")

        
        # Tokenize once for coherence
        logger.info("Tokenizing for coherence calculation...")
        tokenized_docs = [simple_tokenize(t) for t in texts]
        logger.info("")
        
        # Define grid
        grid = {
            "min_df": [0.003, 0.005],
            "max_df": [0.5, 0.8],
            "n_neighbors": [5, 10, 15],
            "n_components": [5, 10, 15],
            "min_cluster_size": [5, 10, 15],
            "min_samples": [1, 5, 10],
        }
        
        total_runs = int(np.prod([len(v) for v in grid.values()]))
        logger.info(f"Grid search configuration:")
        for param, values in grid.items():
            logger.info(f"  {param}: {values}")
        logger.info(f"Total configurations: {total_runs}")
        logger.info("")
        
        # Run grid search
        rows = []
        run_idx = 0
        results_csv = args.output_dir / "grid_search_results.csv"
        
        logger.info("Starting grid search...")
        logger.info("")
        
        for min_df in grid["min_df"]:
            for max_df in grid["max_df"]:
                if isinstance(min_df, float) and isinstance(max_df, float) and max_df <= min_df:
                    continue
                    
                vectorizer_model = CountVectorizer(
                    stop_words="english",
                    ngram_range=(1, 3),
                    min_df=min_df,
                    max_df=max_df,
                    max_features=args.max_features,
                    strip_accents="unicode",
                    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b"
                )
                
                for n_neighbors in grid["n_neighbors"]:
                    for n_components in grid["n_components"]:
                        umap_model = UMAP(
                            n_neighbors=n_neighbors,
                            n_components=n_components,
                            min_dist=0.0,
                            metric="cosine",
                            random_state=args.seed,
                            low_memory=True
                        )
                        
                        for min_cluster_size in grid["min_cluster_size"]:
                            for min_samples in grid["min_samples"]:
                                run_idx += 1
                                tag = (f"min_df={min_df}|max_df={max_df}|"
                                       f"n_neighbors={n_neighbors}|n_components={n_components}|"
                                       f"min_cluster_size={min_cluster_size}|min_samples={min_samples}")
                                logger.info(f"[{run_idx}/{total_runs}] {tag}")
                                
                                hdbscan_model = hdbscan.HDBSCAN(
                                    min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    metric="euclidean",
                                    cluster_selection_method="eom",
                                    prediction_data=False
                                )
                                
                                representation_model = MaximalMarginalRelevance(
                                    diversity=0.7,
                                    top_n_words=args.top_n_words
                                )
                                
                                topic_model = BERTopic(
                                    language="english",
                                    embedding_model=None,
                                    vectorizer_model=vectorizer_model,
                                    representation_model=representation_model,
                                    umap_model=umap_model,
                                    hdbscan_model=hdbscan_model,
                                    calculate_probabilities=False,
                                    verbose=False
                                )
                                
                                topics, _ = topic_model.fit_transform(texts, embeddings)
                                
                                # Get UMAP embedding
                                if hasattr(topic_model.umap_model, "embedding_"):
                                    umap_space = topic_model.umap_model.embedding_
                                else:
                                    umap_space = topic_model.umap_model.transform(embeddings)
                                
                                labels = np.array(topics)
                                
                                # Calculate metrics
                                sil = safe_silhouette(umap_space, labels)
                                topic_word_lists = compute_topic_words(topic_model, args.top_n_words)
                                coh = compute_coherence_c_npmi(topic_word_lists, tokenized_docs)
                                div = topic_diversity(topic_word_lists)
                                
                                n_topics = len(topic_word_lists)
                                outliers = int(np.sum(labels == -1))
                                coverage = 1.0 - outliers / len(labels)
                                
                                row = {
                                    "min_df": min_df,
                                    "max_df": max_df,
                                    "n_neighbors": n_neighbors,
                                    "n_components": n_components,
                                    "min_cluster_size": min_cluster_size,
                                    "min_samples": min_samples,
                                    "n_topics": n_topics,
                                    "outliers": outliers,
                                    "coverage": coverage,
                                    "silhouette": sil,
                                    "coherence_c_npmi": coh,
                                    "diversity": div,
                                    "tag": tag,
                                }
                                rows.append(row)
                                
                                logger.info(f"  → topics={n_topics}, outliers={outliers}, "
                                          f"coverage={coverage:.3f}, silhouette={sil:.3f}")
                                
                                # Save incrementally
                                pd.DataFrame([row]).to_csv(
                                    results_csv,
                                    mode="a",
                                    header=not results_csv.exists(),
                                    index=False
                                )
                                
                                # Clean up
                                del topic_model
                                gc.collect()
        
        logger.info("")
        logger.info("Grid search complete. Ranking results...")
        
        # Rank results
        results = pd.DataFrame(rows)
        results_ranked = rank_and_choose_best(results)
        
        ranked_csv = args.output_dir / "grid_search_results_ranked.csv"
        results_ranked.to_csv(ranked_csv, index=False)
        logger.info(f"Saved ranked results: {ranked_csv}")
        
        # Get best configuration
        best = results_ranked.iloc[0].to_dict()
        
        summary_json = args.output_dir / "grid_search_summary.json"
        with open(summary_json, 'w') as f:
            json.dump(best, f, indent=2)
        logger.info(f"Saved best configuration: {summary_json}")
        
        # Log best configuration
        logger.info("")
        logger.info("=" * 70)
        logger.info("BEST CONFIGURATION")
        logger.info("=" * 70)
        logger.info(f"Rank: 1 / {len(results_ranked)}")
        logger.info(f"Average rank: {best['avg_rank']:.2f}")
        logger.info("")
        logger.info("Hyperparameters:")
        logger.info(f"  min_df:            {best['min_df']}")
        logger.info(f"  max_df:            {best['max_df']}")
        logger.info(f"  n_neighbors:       {best['n_neighbors']}")
        logger.info(f"  n_components:      {best['n_components']}")
        logger.info(f"  min_cluster_size:  {best['min_cluster_size']}")
        logger.info(f"  min_samples:       {best['min_samples']}")
        logger.info("")
        logger.info("Performance:")
        logger.info(f"  Topics:            {best['n_topics']}")
        logger.info(f"  Outliers:          {best['outliers']} ({(1-best['coverage'])*100:.1f}%)")
        logger.info(f"  Coverage:          {best['coverage']:.4f}")
        logger.info(f"  Silhouette:        {best['silhouette']:.4f}")
        logger.info(f"  Coherence (c_npmi):{best['coherence_c_npmi']:.4f}")
        logger.info(f"  Diversity:         {best['diversity']:.4f}")
        logger.info("=" * 70)
        
        # Log completion
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 04 COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info("Output files:")
        logger.info(f"  - {results_csv}")
        logger.info(f"  - {ranked_csv}")
        logger.info(f"  - {summary_json}")
        logger.info(f"  - {args.output_dir}/step_04_grid_search.log")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        logger.error("Step 04 FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())