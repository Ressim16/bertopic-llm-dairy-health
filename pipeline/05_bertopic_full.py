#!/usr/bin/env python3
"""
Step 05: Train BERTopic Model with Optimal Hyperparameters

This script trains a BERTopic model using the best hyperparameters
identified in Step 04 grid search. It reads the optimal configuration
from grid_search_summary.json and trains on the full dataset.

Inputs:
    - abstracts_with_truncation.csv: Abstracts from Step 02
    - embeddings.npy: Embeddings from Step 03
    - grid_search_summary.json: Best hyperparameters from Step 04

Outputs:
    - bertopic_model/: Saved BERTopic model directory
    - topic_info_full.csv: Topic information
    - document_info_full.csv: Document-topic assignments
    - topic_assignments_frozen.csv: Frozen assignments for reproducibility
    - step_05_summary.json: Training summary
    - step_05_bertopic_full.log: Execution log

Author: Reda Zahri
Date: 2025
"""

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Set thread limits before importing numpy/sklearn
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import hdbscan
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer
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


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ============================================================
# DATA LOADING
# ============================================================

def load_hyperparameters(grid_search_summary: Path, logger: logging.Logger) -> dict:
    """
    Load best hyperparameters from grid search results.
    
    Args:
        grid_search_summary: Path to grid_search_summary.json
        logger: Logger instance
        
    Returns:
        Dictionary with hyperparameters
    """
    logger.info(f"Loading hyperparameters from: {grid_search_summary}")
    
    with open(grid_search_summary, 'r') as f:
        params = json.load(f)
    
    # Extract required hyperparameters
    required_keys = ['min_df', 'max_df', 'n_neighbors', 'n_components', 
                     'min_cluster_size', 'min_samples']
    
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required hyperparameter '{key}' in {grid_search_summary}")
    
    logger.info("Best hyperparameters from grid search:")
    for key in required_keys:
        logger.info(f"  {key}: {params[key]}")
    
    if 'n_topics' in params:
        logger.info(f"  Expected topics: ~{params['n_topics']}")
    if 'coverage' in params:
        logger.info(f"  Expected coverage: ~{params['coverage']:.4f}")
    
    return params


def load_data(input_file: Path, text_column: str, logger: logging.Logger) -> tuple:
    """
    Load input data.
    
    Args:
        input_file: Path to CSV file
        text_column: Column name containing text
        logger: Logger instance
        
    Returns:
        Tuple of (dataframe, texts_list)
    """
    logger.info(f"Loading data from: {input_file}")
    
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df):,} rows")
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available: {list(df.columns)}")
    
    texts = df[text_column].fillna("").astype(str).tolist()
    logger.info(f"Extracted {len(texts):,} texts from column '{text_column}'")
    
    return df, texts


def load_embeddings(embeddings_path: Path, n_texts: int, logger: logging.Logger) -> np.ndarray:
    """
    Load embeddings.
    
    Args:
        embeddings_path: Path to .npy file
        n_texts: Expected number of texts
        logger: Logger instance
        
    Returns:
        Embeddings array
    """
    logger.info(f"Loading embeddings from: {embeddings_path}")
    
    embeddings = np.load(embeddings_path)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    
    if embeddings.shape[0] != n_texts:
        if embeddings.shape[0] > n_texts:
            logger.warning(f"Embeddings ({embeddings.shape[0]}) > texts ({n_texts}). Truncating embeddings.")
            embeddings = embeddings[:n_texts]
        else:
            raise ValueError(
                f"Embeddings ({embeddings.shape[0]}) < texts ({n_texts}). "
                "Cannot proceed - embeddings and texts must be aligned."
            )
    
    return embeddings


# ============================================================
# MODEL TRAINING
# ============================================================

def train_bertopic(
    texts: list,
    embeddings: np.ndarray,
    params: dict,
    seed: int,
    top_n_words: int,
    max_features: int,
    logger: logging.Logger
) -> tuple:
    """
    Train BERTopic model with specified hyperparameters.
    
    Args:
        texts: List of documents
        embeddings: Document embeddings
        params: Hyperparameters dictionary
        seed: Random seed
        top_n_words: Number of words per topic
        max_features: Max vocabulary size
        logger: Logger instance
        
    Returns:
        Tuple of (topic_model, topics)
    """
    logger.info("Building BERTopic model components...")
    
    # Vectorizer
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=params['min_df'],
        max_df=params['max_df'],
        max_features=max_features,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b"
    )
    logger.info(f"  Vectorizer: min_df={params['min_df']}, max_df={params['max_df']}, max_features={max_features}")
    
    # UMAP
    umap_model = UMAP(
        n_neighbors=params['n_neighbors'],
        n_components=params['n_components'],
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
        low_memory=True
    )
    logger.info(f"  UMAP: n_neighbors={params['n_neighbors']}, n_components={params['n_components']}")
    
    # HDBSCAN
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=params['min_cluster_size'],
        min_samples=params['min_samples'],
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True  # Enable for transform() on new data
    )
    logger.info(f"  HDBSCAN: min_cluster_size={params['min_cluster_size']}, min_samples={params['min_samples']}")
    
    # Representation model
    representation_model = MaximalMarginalRelevance(
        diversity=0.7,
        top_n_words=top_n_words
    )
    logger.info(f"  Representation: MMR with diversity=0.7, top_n_words={top_n_words}")
    
    # BERTopic
    topic_model = BERTopic(
        language="english",
        embedding_model=None,  # We provide pre-computed embeddings
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=True
    )
    
    logger.info("")
    logger.info("Training BERTopic model on full dataset...")
    logger.info(f"  Documents: {len(texts):,}")
    logger.info(f"  Embedding dimensions: {embeddings.shape[1]}")
    logger.info("")
    
    topics, _ = topic_model.fit_transform(texts, embeddings)
    
    logger.info("Training complete!")
    
    return topic_model, topics


# ============================================================
# OUTPUT GENERATION
# ============================================================

def save_outputs(
    topic_model: BERTopic,
    texts: list,
    topics: list,
    output_dir: Path,
    logger: logging.Logger
) -> tuple:
    """
    Save model and generate output files.
    
    Args:
        topic_model: Trained BERTopic model
        texts: List of documents
        topics: Topic assignments
        output_dir: Output directory
        logger: Logger instance
        
    Returns:
        Tuple of (topic_info_df, doc_info_df)
    """
    # Save model using pickle format (more reliable than safetensors for some versions)
    model_path = output_dir / "bertopic_model"
    logger.info(f"Saving model to: {model_path}")
    
    try:
        # Try safetensors first (preferred format)
        topic_model.save(model_path, serialization="safetensors", save_ctfidf=True, save_embedding_model=False)
        logger.info("Model saved successfully (safetensors format)")
    except TypeError as e:
        # Fall back to pickle format if safetensors fails (JSON serialization issues)
        logger.warning(f"Safetensors save failed ({e}), falling back to pickle format...")
        model_path_pkl = output_dir / "bertopic_model.pkl"
        topic_model.save(model_path_pkl, serialization="pickle", save_ctfidf=True, save_embedding_model=False)
        logger.info(f"Model saved successfully (pickle format): {model_path_pkl}")
    except Exception as e:
        # Last resort: save without ctfidf
        logger.warning(f"Standard save failed ({e}), trying minimal save...")
        model_path_minimal = output_dir / "bertopic_model_minimal"
        try:
            topic_model.save(model_path_minimal, serialization="pickle", save_ctfidf=False, save_embedding_model=False)
            logger.info(f"Model saved (minimal pickle format): {model_path_minimal}")
        except Exception as e2:
            logger.error(f"All save methods failed: {e2}")
            logger.warning("Continuing without saved model - outputs will still be generated")
    
    # Get topic info
    topic_info = topic_model.get_topic_info()
    topic_info_path = output_dir / "topic_info_full.csv"
    topic_info.to_csv(topic_info_path, index=False)
    logger.info(f"Saved topic info: {topic_info_path}")
    
    # Get document info
    doc_info = topic_model.get_document_info(texts)
    doc_info_path = output_dir / "document_info_full.csv"
    doc_info.to_csv(doc_info_path, index=False)
    logger.info(f"Saved document info: {doc_info_path}")
    
    # Save frozen topic assignments
    assignments_df = pd.DataFrame({
        'doc_id': range(len(topics)),
        'topic': [int(t) for t in topics]  # Convert to native int
    })
    assignments_path = output_dir / "topic_assignments_frozen.csv"
    assignments_df.to_csv(assignments_path, index=False)
    logger.info(f"Saved frozen assignments: {assignments_path}")
    
    # Save UMAP projection immediately after fit_transform() while embedding_ is in memory
    np.save(output_dir / "umap_embeddings.npy", topic_model.umap_model.embedding_)
    logger.info(f"Saved UMAP embeddings: shape={topic_model.umap_model.embedding_.shape}")
    
    return topic_info, doc_info


def save_summary(
    params: dict,
    topics: list,
    topic_info: pd.DataFrame,
    output_dir: Path,
    grid_search_path: Path,
    logger: logging.Logger
) -> dict:
    """
    Save training summary.
    
    Args:
        params: Hyperparameters used
        topics: Topic assignments
        topic_info: Topic info DataFrame
        output_dir: Output directory
        grid_search_path: Path to grid search results
        logger: Logger instance
        
    Returns:
        Summary dictionary
    """
    topics_array = np.array(topics)
    n_topics = len(topic_info[topic_info['Topic'] != -1])
    outliers = int((topics_array == -1).sum())
    coverage = 1.0 - outliers / len(topics)
    
    summary = {
        "generated_at": datetime.now().isoformat(),
        "n_documents": len(topics),
        "n_topics_excluding_outlier": n_topics,
        "outliers_count": outliers,
        "outliers_pct": round(float(outliers / len(topics)), 4),
        "coverage": round(float(coverage), 4),
        "hyperparameters": {
            "min_df": params['min_df'],
            "max_df": params['max_df'],
            "n_neighbors": params['n_neighbors'],
            "n_components": params['n_components'],
            "min_cluster_size": params['min_cluster_size'],
            "min_samples": params['min_samples']
        },
        "hyperparameters_source": str(grid_search_path),
        "model_path": str(output_dir / "bertopic_model"),
        "outputs": {
            "topic_info": str(output_dir / "topic_info_full.csv"),
            "document_info": str(output_dir / "document_info_full.csv"),
            "frozen_assignments": str(output_dir / "topic_assignments_frozen.csv")
        }
    }
    
    # Add expected metrics from grid search if available
    for key in ['silhouette', 'coherence_c_npmi', 'diversity']:
        if key in params:
            summary[f"expected_{key}"] = params[key]
    
    summary_path = output_dir / "step_05_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {summary_path}")
    
    return summary


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Step 05: Train BERTopic with optimal hyperparameters from grid search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python 05_bertopic_full.py \\
        --input-file /path/to/abstracts_with_truncation.csv \\
        --embeddings /path/to/embeddings.npy \\
        --grid-search-summary /path/to/grid_search_summary.json \\
        --output-dir /path/to/output/05_bertopic_full
        """
    )
    
    parser.add_argument(
        '--input-file',
        type=Path,
        required=True,
        help='Input CSV file with abstracts (from Step 02)'
    )
    parser.add_argument(
        '--embeddings',
        type=Path,
        required=True,
        help='Embeddings .npy file (from Step 03)'
    )
    parser.add_argument(
        '--grid-search-summary',
        type=Path,
        required=True,
        help='Grid search summary JSON (from Step 04)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='Abstract',
        help='Column name containing text (default: Abstract)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--top-n-words',
        type=int,
        default=10,
        help='Number of words per topic (default: 10)'
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=30000,
        help='Max vocabulary size (default: 30000)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logger = configure_logging(args.output_dir / "step_05_bertopic_full.log")
    
    # Set seed
    set_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    
    # Log header
    logger.info("=" * 70)
    logger.info("STEP 05: TRAIN BERTOPIC WITH OPTIMAL HYPERPARAMETERS")
    logger.info("=" * 70)
    logger.info(f"Input file:          {args.input_file}")
    logger.info(f"Embeddings:          {args.embeddings}")
    logger.info(f"Grid search summary: {args.grid_search_summary}")
    logger.info(f"Output dir:          {args.output_dir}")
    logger.info(f"Text column:         {args.text_column}")
    logger.info(f"Seed:                {args.seed}")
    logger.info("=" * 70)
    logger.info("")
    
    try:
        # Validate input files
        if not args.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        if not args.embeddings.exists():
            raise FileNotFoundError(f"Embeddings not found: {args.embeddings}")
        if not args.grid_search_summary.exists():
            raise FileNotFoundError(f"Grid search summary not found: {args.grid_search_summary}")
        
        # Load hyperparameters from grid search
        params = load_hyperparameters(args.grid_search_summary, logger)
        logger.info("")
        
        # Load data
        df, texts = load_data(args.input_file, args.text_column, logger)
        logger.info("")
        
        # Load embeddings
        embeddings = load_embeddings(args.embeddings, len(texts), logger)
        logger.info("")
        
        # Train model
        topic_model, topics = train_bertopic(
            texts=texts,
            embeddings=embeddings,
            params=params,
            seed=args.seed,
            top_n_words=args.top_n_words,
            max_features=args.max_features,
            logger=logger
        )
        logger.info("")
        
        # Save outputs
        topic_info, doc_info = save_outputs(
            topic_model=topic_model,
            texts=texts,
            topics=topics,
            output_dir=args.output_dir,
            logger=logger
        )
        logger.info("")
        
        # Save summary
        summary = save_summary(
            params=params,
            topics=topics,
            topic_info=topic_info,
            output_dir=args.output_dir,
            grid_search_path=args.grid_search_summary,
            logger=logger
        )
        
        # Log results
        logger.info("")
        logger.info("=" * 70)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 70)
        logger.info(f"Documents:           {summary['n_documents']:,}")
        logger.info(f"Topics (excl. -1):   {summary['n_topics_excluding_outlier']}")
        logger.info(f"Outliers:            {summary['outliers_count']:,} ({summary['outliers_pct']:.2%})")
        logger.info(f"Coverage:            {summary['coverage']:.4f}")
        logger.info("")
        logger.info("Hyperparameters used:")
        for key, val in summary['hyperparameters'].items():
            logger.info(f"  {key}: {val}")
        logger.info("=" * 70)
        
        # Log completion
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 05 COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info("Output files:")
        logger.info(f"  - {args.output_dir}/bertopic_model/")
        logger.info(f"  - {args.output_dir}/topic_info_full.csv")
        logger.info(f"  - {args.output_dir}/document_info_full.csv")
        logger.info(f"  - {args.output_dir}/topic_assignments_frozen.csv")
        logger.info(f"  - {args.output_dir}/step_05_summary.json")
        logger.info(f"  - {args.output_dir}/step_05_bertopic_full.log")
        logger.info("")
        logger.info("Next step: Run 06_llm_labeling.sh")
        logger.info("=" * 70)
        
        # Cleanup
        del topic_model
        gc.collect()
        
        return 0
        
    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        logger.error("Step 05 FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())