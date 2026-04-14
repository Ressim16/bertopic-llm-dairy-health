#!/usr/bin/env python3
"""
Step 001: Pairwise Comparison for Duplicate Detection

This script merges PubMed and Scopus datasets and identifies exact + potential duplicate
abstracts using TF-IDF vectorization and FAISS similarity search. Candidate pairs
above a specified similarity threshold are exported for manual labeling.

Inputs:
    - pubmed_abstracts_luis_string_2000_2025_cleaned.csv: PubMed abstracts
    - scopus_abstracts_articles_luis_2000_2025_cleaned.csv: Scopus abstracts

Outputs:
    - candidate_pairs_for_labeling.csv: Similar pairs for manual review
    - pairwise_statistics.json: Processing statistics
    - pairwise_summary.txt: Human-readable summary
    - step_001_pairwise.log: Execution log

Author: Reda Zahri
Date: 2025
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


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


# ============================================================
# DATA LOADING AND HARMONIZATION
# ============================================================

def load_and_harmonize_data(
    pubmed_path: Path,
    scopus_path: Path,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Load PubMed and Scopus datasets and harmonize column names.
    
    Args:
        pubmed_path: Path to PubMed CSV file
        scopus_path: Path to Scopus CSV file
        logger: Logger instance
        
    Returns:
        Merged DataFrame with harmonized columns
    """
    logger.info("Loading PubMed data...")
    df_pubmed = pd.read_csv(pubmed_path)
    logger.info(f"  PubMed records: {len(df_pubmed):,}")
    
    logger.info("Loading Scopus data...")
    df_scopus = pd.read_csv(scopus_path)
    logger.info(f"  Scopus records: {len(df_scopus):,}")
    
    # Harmonize Scopus column names to match PubMed
    logger.info("Harmonizing column names...")
    df_scopus = df_scopus.rename(columns={
        'doi': 'DOI',
        'title': 'Title',
        'abstract': 'Abstract',
        'creator': 'Authors',
        'publicationName': 'Journal',
        'coverDate': 'Year'
    })
    
    # Add source identifier
    df_pubmed["source"] = "pubmed"
    df_scopus["source"] = "scopus"
    
    # Merge datasets
    logger.info("Merging datasets...")
    df_merged = pd.concat([df_pubmed, df_scopus], ignore_index=True)
    
    # Keep only required columns
    required_cols = ['DOI', 'Title', 'Abstract', 'Authors', 'Journal', 'Year', 'source']
    df_merged = df_merged[required_cols]
    
    logger.info(f"  Total merged records: {len(df_merged):,}")
    
    return df_merged


def remove_exact_duplicates(
    df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Remove exact duplicates based on Title and Abstract.
    
    Args:
        df: Input DataFrame
        logger: Logger instance
        
    Returns:
        DataFrame with exact duplicates removed
    """
    logger.info("Removing exact duplicates (Title + Abstract)...")
    n_before = len(df)
    df_dedup = df.drop_duplicates(subset=["Title", "Abstract"], keep="first").reset_index(drop=True)
    n_after = len(df_dedup)
    n_removed = n_before - n_after
    
    logger.info(f"  Exact duplicates removed: {n_removed:,}")
    logger.info(f"  Records after exact dedup: {n_after:,}")
    
    return df_dedup


# ============================================================
# SIMILARITY SEARCH
# ============================================================

def compute_similarity_pairs(
    df: pd.DataFrame,
    threshold: float,
    top_k: int,
    tfidf_max_df: float,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Compute pairwise similarities using TF-IDF and FAISS.
    
    Args:
        df: DataFrame with 'Abstract' column
        threshold: Minimum similarity threshold for candidate pairs
        top_k: Number of nearest neighbors to search
        tfidf_max_df: Maximum document frequency for TF-IDF
        logger: Logger instance
        
    Returns:
        DataFrame of candidate duplicate pairs
    """
    # Prepare abstracts
    abstracts = df["Abstract"].fillna("").tolist()
    
    # TF-IDF vectorization
    logger.info("Computing TF-IDF vectors...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=tfidf_max_df)
    tfidf_matrix = vectorizer.fit_transform(abstracts)
    logger.info(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Convert to dense and normalize for cosine similarity
    logger.info("Converting to dense matrix and L2 normalizing...")
    dense_matrix = tfidf_matrix.toarray().astype('float32')
    dense_matrix = normalize(dense_matrix, norm='l2', axis=1)
    
    # Build FAISS index
    logger.info("Building FAISS index (Inner Product)...")
    index = faiss.IndexFlatIP(dense_matrix.shape[1])
    index.add(dense_matrix)
    
    # Search for similar items
    logger.info(f"Searching for top-{top_k} neighbors...")
    similarities, indices = index.search(dense_matrix, top_k)
    
    # Collect candidate pairs above threshold
    logger.info(f"Collecting candidate pairs (threshold >= {threshold})...")
    seen = set()
    duplicate_pairs = []
    
    for i in range(len(indices)):
        for rank in range(1, top_k):  # Skip self (rank 0)
            j = int(indices[i][rank])
            score = float(similarities[i][rank])
            
            if score >= threshold and j != i:
                # Ensure consistent ordering to avoid duplicates
                pair = (min(i, j), max(i, j))
                if pair not in seen:
                    seen.add(pair)
                    duplicate_pairs.append({
                        "Index_1": pair[0],
                        "Index_2": pair[1],
                        "Similarity": score,
                        "Title_1": df.iloc[pair[0]]["Title"],
                        "Title_2": df.iloc[pair[1]]["Title"],
                        "Authors_1": df.iloc[pair[0]]["Authors"],
                        "Authors_2": df.iloc[pair[1]]["Authors"],
                        "Source_1": df.iloc[pair[0]]["source"],
                        "Source_2": df.iloc[pair[1]]["source"],
                        "is_duplicate": ""  # Empty column for manual review
                    })
    
    logger.info(f"  Found {len(duplicate_pairs):,} candidate pairs")
    
    # Convert to DataFrame and sort by similarity
    pairs_df = pd.DataFrame(duplicate_pairs)
    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values(by="Similarity", ascending=False).reset_index(drop=True)
    
    return pairs_df


# ============================================================
# OUTPUT FUNCTIONS
# ============================================================

def save_outputs(
    pairs_df: pd.DataFrame,
    stats: dict,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Save all output files.
    
    Args:
        pairs_df: DataFrame of candidate pairs
        merged_df: (Not saved in this step) Merged and exact-deduplicated DataFrame
        stats: Statistics dictionary
        output_dir: Output directory
        logger: Logger instance
    """
    # Save candidate pairs for labeling
    pairs_path = output_dir / "candidate_pairs_for_labeling.csv"
    pairs_df.to_csv(pairs_path, index=False)
    logger.info(f"Saved candidate pairs: {pairs_path}")
    
    # Save statistics as JSON
    json_path = output_dir / "pairwise_statistics.json"
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics: {json_path}")
    
    # Save human-readable summary
    summary_path = output_dir / "pairwise_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("STEP 001: PAIRWISE COMPARISON SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Data Sources:\n")
        f.write(f"  PubMed records:               {stats['pubmed_records']:>10,}\n")
        f.write(f"  Scopus records:               {stats['scopus_records']:>10,}\n")
        f.write(f"  Total merged:                 {stats['total_merged']:>10,}\n\n")
        
        f.write("Exact Deduplication:\n")
        f.write(f"  Exact duplicates removed:     {stats['exact_duplicates_removed']:>10,}\n")
        f.write(f"  Records after exact dedup:    {stats['after_exact_dedup']:>10,}\n\n")
        
        f.write("Similarity Search:\n")
        f.write(f"  Similarity threshold:         {stats['similarity_threshold']:>10.2f}\n")
        f.write(f"  Top-k neighbors searched:     {stats['top_k']:>10}\n")
        f.write(f"  Candidate pairs found:        {stats['candidate_pairs']:>10,}\n\n")
        
        f.write("Next Step:\n")
        f.write("  Manually label 'is_duplicate' column (1=duplicate, 0=not)\n")
        f.write("  Then run 00_02_threshold_selection.sh\n\n")
        
        f.write("=" * 70 + "\n")
    
    logger.info(f"Saved summary: {summary_path}")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Step 001: Pairwise comparison for duplicate detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python 001_pairwise_comparison.py \\
        --pubmed-file /path/to/pubmed.csv \\
        --scopus-file /path/to/scopus.csv \\
        --output-dir /path/to/output/001_pairwise \\
        --threshold 0.70 \\
        --top-k 10
        """
    )
    
    parser.add_argument(
        '--pubmed-file',
        type=Path,
        required=True,
        help='Path to PubMed CSV file'
    )
    parser.add_argument(
        '--scopus-file',
        type=Path,
        required=True,
        help='Path to Scopus CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.70,
        help='Minimum similarity threshold for candidate pairs (default: 0.70)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of nearest neighbors to search (default: 10)'
    )
    parser.add_argument(
        '--tfidf-max-df',
        type=float,
        default=0.95,
        help='Maximum document frequency for TF-IDF (default: 0.95)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logger = configure_logging(args.output_dir / "step_001_pairwise.log")
    
    # Log header
    logger.info("=" * 70)
    logger.info("STEP 001: PAIRWISE COMPARISON")
    logger.info("=" * 70)
    logger.info(f"PubMed file:   {args.pubmed_file}")
    logger.info(f"Scopus file:   {args.scopus_file}")
    logger.info(f"Output dir:    {args.output_dir}")
    logger.info(f"Threshold:     {args.threshold}")
    logger.info(f"Top-k:         {args.top_k}")
    logger.info(f"TF-IDF max_df: {args.tfidf_max_df}")
    logger.info("=" * 70)
    logger.info("")
    
    try:
        # Validate input files
        if not args.pubmed_file.exists():
            raise FileNotFoundError(f"PubMed file not found: {args.pubmed_file}")
        if not args.scopus_file.exists():
            raise FileNotFoundError(f"Scopus file not found: {args.scopus_file}")
        
        # Load and harmonize data
        df_merged = load_and_harmonize_data(
            args.pubmed_file,
            args.scopus_file,
            logger
        )
        
        # Track statistics
        stats = {
            'pubmed_records': int(len(pd.read_csv(args.pubmed_file))),
            'scopus_records': int(len(pd.read_csv(args.scopus_file))),
            'total_merged': int(len(df_merged))
        }
        
        # Remove exact duplicates
        df_dedup = remove_exact_duplicates(df_merged, logger)
        stats['exact_duplicates_removed'] = stats['total_merged'] - len(df_dedup)
        stats['after_exact_dedup'] = int(len(df_dedup))
        
        # Compute similarity pairs
        logger.info("")
        pairs_df = compute_similarity_pairs(
            df_dedup,
            threshold=args.threshold,
            top_k=args.top_k,
            tfidf_max_df=args.tfidf_max_df,
            logger=logger
        )
        
        stats['similarity_threshold'] = args.threshold
        stats['top_k'] = args.top_k
        stats['candidate_pairs'] = int(len(pairs_df))
        
        # Save outputs
        logger.info("")
        # NOTE: We intentionally do NOT persist the merged_exact_dedup.csv here
        # to reduce I/O and runtime. Step 00_02 can rebuild the merged/exact-dedup
        # dataset from the raw inputs.
        save_outputs(pairs_df, stats, args.output_dir, logger)
        
        # Log completion
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 001 COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info("Output files:")
        logger.info(f"  - {args.output_dir}/candidate_pairs_for_labeling.csv")
        logger.info(f"  - {args.output_dir}/pairwise_statistics.json")
        logger.info(f"  - {args.output_dir}/pairwise_summary.txt")
        logger.info(f"  - {args.output_dir}/step_001_pairwise.log")
        logger.info("=" * 70)
        logger.info("")
        logger.info("NEXT: Manually label 'is_duplicate' column, then run 002_threshold_selection.sh")
        
        return 0
        
    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        logger.error("Step 001 FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())