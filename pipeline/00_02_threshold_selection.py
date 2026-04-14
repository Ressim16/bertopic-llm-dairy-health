#!/usr/bin/env python3
"""
Step 002: Threshold Selection and Auto-Deduplication

This script uses manually labeled duplicate pairs to:
1. Evaluate precision/recall/F1 across similarity thresholds
2. Automatically select optimal threshold (targeting high precision)
3. Apply the selected threshold to deduplicate the full dataset

Inputs:
    - candidate_pairs_labeled.csv: Manually labeled candidate pairs
    - merged_exact_dedup.csv: Dataset after exact deduplication (from Step 001)

Outputs:
    - threshold_metrics.csv: Metrics for each evaluated threshold
    - merged_cleaned_auto_dedup.csv: Final deduplicated dataset
    - auto_drop_log.csv: Log of dropped records with their representatives
    - similarity_edges.csv: Audit trail of similarity edges used
    - threshold_statistics.json: Processing statistics
    - threshold_summary.txt: Human-readable summary
    - step_002_threshold.log: Execution log

Author: Reda Zahri
Date: 2025
"""

import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
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
# HELPER FUNCTIONS
# ============================================================

def load_and_harmonize_data(pubmed_path: Path, scopus_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load PubMed and Scopus datasets and harmonize column names (same logic as Step 001)."""
    logger.info("Loading PubMed data...")
    df_pubmed = pd.read_csv(pubmed_path)
    logger.info(f"  PubMed records: {len(df_pubmed):,}")

    logger.info("Loading Scopus data...")
    df_scopus = pd.read_csv(scopus_path)
    logger.info(f"  Scopus records: {len(df_scopus):,}")

    logger.info("Harmonizing column names...")
    df_scopus = df_scopus.rename(columns={
        'doi': 'DOI',
        'title': 'Title',
        'abstract': 'Abstract',
        'creator': 'Authors',
        'publicationName': 'Journal',
        'coverDate': 'Year'
    })

    df_pubmed["source"] = "pubmed"
    df_scopus["source"] = "scopus"

    logger.info("Merging datasets...")
    df_merged = pd.concat([df_pubmed, df_scopus], ignore_index=True)

    required_cols = ['DOI', 'Title', 'Abstract', 'Authors', 'Journal', 'Year', 'source']
    missing = [c for c in required_cols if c not in df_merged.columns]
    if missing:
        raise ValueError(f"Missing required columns after harmonization: {missing}")

    df_merged = df_merged[required_cols]
    logger.info(f"  Total merged records: {len(df_merged):,}")
    return df_merged


def remove_exact_duplicates(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Remove exact duplicates based on Title + Abstract (same logic as Step 001)."""
    logger.info("Removing exact duplicates (Title + Abstract)...")
    n_before = len(df)
    df_dedup = df.drop_duplicates(subset=["Title", "Abstract"], keep="first").reset_index(drop=True)
    n_after = len(df_dedup)
    logger.info(f"  Exact duplicates removed: {n_before - n_after:,}")
    logger.info(f"  Records after exact dedup: {n_after:,}")
    return df_dedup


def fbeta(precision: float, recall: float, beta: float) -> float:
    """
    Compute F-beta score safely.
    
    Args:
        precision: Precision value
        recall: Recall value
        beta: Beta parameter (< 1 favors precision, > 1 favors recall)
        
    Returns:
        F-beta score
    """
    if precision == 0 and recall == 0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision + recall)
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def clean_for_tfidf(text: str) -> str:
    """
    Light normalization for TF-IDF deduplication.
    
    Note: This is only for dedup matching, not for downstream BERTopic.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class UnionFind:
    """
    Disjoint Set Union (Union-Find) data structure.
    
    Used to enforce transitive deduplication groups:
    if A~B and B~C, then {A, B, C} form one group.
    """
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        """Find root with path compression."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, a: int, b: int) -> None:
        """Union by rank."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_thresholds(
    labeled_df: pd.DataFrame,
    sim_col: str,
    label_col: str,
    t_min: float,
    t_max: float,
    t_step: float,
    beta: float,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Evaluate precision/recall/F1 across a range of thresholds.
    
    Args:
        labeled_df: DataFrame with similarity scores and labels
        sim_col: Column name for similarity scores
        label_col: Column name for labels (0/1)
        t_min: Minimum threshold to evaluate
        t_max: Maximum threshold to evaluate
        t_step: Step size for threshold sweep
        beta: Beta parameter for F-beta score
        logger: Logger instance
        
    Returns:
        DataFrame with metrics for each threshold
    """
    # Validate and clean data
    tmp = labeled_df[[sim_col, label_col]].copy()
    tmp[label_col] = pd.to_numeric(tmp[label_col], errors="coerce")
    tmp = tmp[tmp[label_col].isin([0, 1])].copy()
    
    if len(tmp) == 0:
        raise ValueError("No labeled rows found. Ensure is_duplicate contains 0/1 values.")
    
    y_true = tmp[label_col].astype(int).values
    sims = pd.to_numeric(tmp[sim_col], errors="coerce").values
    
    if np.isnan(sims).any():
        raise ValueError("Some similarity values could not be parsed as numbers.")
    
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    logger.info(f"Labeled pairs: {len(tmp):,} (duplicates: {n_pos:,}, non-duplicates: {n_neg:,})")
    
    # Evaluate thresholds
    thresholds = np.arange(t_min, t_max + 1e-12, t_step)
    rows = []
    
    for t in thresholds:
        y_pred = (sims >= t).astype(int)
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f_b = fbeta(prec, rec, beta)
        
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        
        rows.append({
            "threshold": float(t),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            f"f{beta:g}": float(f_b),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "predicted_duplicates": int(y_pred.sum())
        })
    
    return pd.DataFrame(rows)


def select_optimal_threshold(
    metrics_df: pd.DataFrame,
    target_precision: float,
    beta: float,
    logger: logging.Logger
) -> tuple:
    """
    Select optimal threshold based on target precision.
    
    Args:
        metrics_df: DataFrame with threshold metrics
        target_precision: Minimum precision target
        beta: Beta parameter for F-beta (used as fallback)
        logger: Logger instance
        
    Returns:
        Tuple of (selected_threshold, selection_info_dict)
    """
    # Best F1 threshold
    best_f1 = metrics_df.loc[metrics_df["f1"].idxmax()].to_dict()
    logger.info(f"Best F1 threshold: {best_f1['threshold']:.3f} "
                f"(P={best_f1['precision']:.3f}, R={best_f1['recall']:.3f}, "
                f"F1={best_f1['f1']:.3f})")
    
    # Best F-beta threshold
    fb_col = f"f{beta:g}"
    best_fb = metrics_df.loc[metrics_df[fb_col].idxmax()].to_dict()
    logger.info(f"Best F{beta:g} threshold: {best_fb['threshold']:.3f} "
                f"(P={best_fb['precision']:.3f}, R={best_fb['recall']:.3f}, "
                f"F{beta:g}={best_fb[fb_col]:.3f})")
    
    # Find minimum threshold meeting precision target
    eligible = metrics_df[
        (metrics_df["precision"] >= target_precision) & 
        (metrics_df["predicted_duplicates"] > 0)
    ]
    
    if len(eligible) > 0:
        best_prec = eligible.sort_values("threshold").iloc[0].to_dict()
        auto_threshold = float(best_prec["threshold"])
        selection_method = f"precision >= {target_precision}"
        logger.info(f"Threshold with precision >= {target_precision}: {auto_threshold:.3f} "
                    f"(P={best_prec['precision']:.3f}, R={best_prec['recall']:.3f})")
    else:
        # Fallback to best F-beta
        auto_threshold = float(best_fb["threshold"])
        selection_method = f"fallback to best F{beta:g}"
        logger.warning(f"No threshold achieved precision >= {target_precision}")
        logger.warning(f"Using fallback: best F{beta:g} threshold = {auto_threshold:.3f}")
    
    logger.info(f"AUTO-DROP threshold selected: {auto_threshold:.3f}")
    
    selection_info = {
        "selected_threshold": auto_threshold,
        "selection_method": selection_method,
        "best_f1_threshold": best_f1["threshold"],
        "best_fb_threshold": best_fb["threshold"],
        "target_precision": target_precision,
        "beta": beta
    }
    
    return auto_threshold, selection_info


# ============================================================
# AUTO-DEDUPLICATION
# ============================================================

def filter_by_year(
    df: pd.DataFrame,
    min_year: int,
    max_year: int,
    year_col: str,
    logger: logging.Logger
) -> tuple:
    """
    Filter documents by publication year.
    
    Args:
        df: Input DataFrame
        min_year: Minimum year (inclusive)
        max_year: Maximum year (inclusive)
        year_col: Column name containing year information
        logger: Logger instance
        
    Returns:
        Tuple of (filtered_df, n_removed)
    """
    n_before = len(df)
    
    # Extract year from the column (handle various formats)
    def extract_year(val):
        if pd.isna(val):
            return None
        val_str = str(val)
        # Try to extract 4-digit year
        match = re.search(r'(19|20)\d{2}', val_str)
        if match:
            return int(match.group())
        # Try direct conversion for numeric values
        try:
            year = int(float(val_str))
            if 1900 <= year <= 2100:
                return year
        except (ValueError, TypeError):
            pass
        return None
    
    df = df.copy()
    df['_extracted_year'] = df[year_col].apply(extract_year)
    
    # Log year distribution before filtering
    valid_years = df['_extracted_year'].dropna()
    if len(valid_years) > 0:
        logger.info(f"  Year range in data: {int(valid_years.min())} - {int(valid_years.max())}")
    
    # Count documents outside range
    n_before_min = (df['_extracted_year'] < min_year).sum()
    n_after_max = (df['_extracted_year'] > max_year).sum()
    n_missing_year = df['_extracted_year'].isna().sum()
    
    logger.info(f"  Documents before {min_year}: {n_before_min:,}")
    logger.info(f"  Documents after {max_year}: {n_after_max:,}")
    logger.info(f"  Documents with missing/invalid year: {n_missing_year:,}")
    
    # Filter: keep only documents within year range (exclude missing years)
    mask = (df['_extracted_year'] >= min_year) & (df['_extracted_year'] <= max_year)
    df_filtered = df[mask].drop(columns=['_extracted_year']).reset_index(drop=True)
    
    n_removed = n_before - len(df_filtered)
    logger.info(f"  Removed {n_removed:,} documents outside {min_year}-{max_year}")
    logger.info(f"  Remaining: {len(df_filtered):,} documents")
    
    return df_filtered, n_removed


def apply_auto_deduplication(
    df: pd.DataFrame,
    auto_threshold: float,
    top_k: int,
    tfidf_max_df: float,
    tfidf_min_df: int,
    logger: logging.Logger
) -> tuple:
    """
    Apply automatic deduplication using the selected threshold.
    
    Args:
        df: DataFrame to deduplicate
        auto_threshold: Similarity threshold for duplicate detection
        top_k: Number of nearest neighbors to search
        tfidf_max_df: TF-IDF max document frequency
        tfidf_min_df: TF-IDF min document frequency
        logger: Logger instance
        
    Returns:
        Tuple of (dedup_df, drop_log_df, edges_df)
    """
    # Prepare texts
    texts = df["Abstract"].fillna("").astype(str).apply(clean_for_tfidf).tolist()
    
    # TF-IDF vectorization
    logger.info("Computing TF-IDF vectors for auto-deduplication...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=tfidf_max_df,
        min_df=tfidf_min_df
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Convert to dense and normalize
    logger.info("Converting to dense matrix and L2 normalizing...")
    dense_matrix = tfidf_matrix.toarray().astype("float32")
    dense_matrix = normalize(dense_matrix, norm="l2", axis=1)
    
    # Build FAISS index
    logger.info("Building FAISS index...")
    index = faiss.IndexFlatIP(dense_matrix.shape[1])
    index.add(dense_matrix)
    
    # Search for neighbors
    logger.info(f"Searching for top-{top_k} neighbors...")
    sim_mat, nbr_mat = index.search(dense_matrix, top_k)
    
    # Build Union-Find structure
    uf = UnionFind(len(df))
    edges = []
    
    for i in range(len(df)):
        for rank in range(1, top_k):  # Skip self
            j = int(nbr_mat[i][rank])
            score = float(sim_mat[i][rank])
            
            if j == -1 or i == j:
                continue
            
            # Record edge for audit
            if score >= auto_threshold:
                a, b = (i, j) if i < j else (j, i)
                edges.append((a, b, score))
                uf.union(i, j)
    
    # Deduplicate edge list
    if edges:
        edges_df = (pd.DataFrame(edges, columns=["Index_1", "Index_2", "Similarity"])
                    .drop_duplicates(subset=["Index_1", "Index_2"])
                    .sort_values("Similarity", ascending=False)
                    .reset_index(drop=True))
    else:
        edges_df = pd.DataFrame(columns=["Index_1", "Index_2", "Similarity"])
    
    logger.info(f"Found {len(edges_df):,} unique similarity edges >= {auto_threshold:.3f}")
    
    # Build groups and select representatives
    groups = {}
    for i in range(len(df)):
        root = uf.find(i)
        groups.setdefault(root, []).append(i)
    
    keep_indices = []
    drop_rows = []
    
    for root, members in groups.items():
        # Select representative (lowest index for determinism)
        rep = min(members)
        keep_indices.append(rep)
        
        for m in members:
            if m == rep:
                continue
            drop_rows.append({
                "kept_index": rep,
                "dropped_index": m,
                "kept_title": df.iloc[rep]["Title"],
                "dropped_title": df.iloc[m]["Title"],
                "kept_source": df.iloc[rep]["source"],
                "dropped_source": df.iloc[m]["source"],
                "kept_doi": df.iloc[rep]["DOI"],
                "dropped_doi": df.iloc[m]["DOI"],
            })
    
    # Create output DataFrames
    keep_indices = sorted(set(keep_indices))
    dedup_df = df.iloc[keep_indices].reset_index(drop=True)
    drop_log_df = pd.DataFrame(drop_rows)
    
    logger.info(f"Deduplication complete: kept {len(dedup_df):,}, dropped {len(drop_log_df):,}")
    
    return dedup_df, drop_log_df, edges_df


# ============================================================
# OUTPUT FUNCTIONS
# ============================================================

def save_outputs(
    metrics_df: pd.DataFrame,
    dedup_df: pd.DataFrame,
    drop_log_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    stats: dict,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Save all output files.
    
    Args:
        metrics_df: Threshold metrics DataFrame
        dedup_df: Deduplicated DataFrame
        drop_log_df: Drop log DataFrame
        edges_df: Similarity edges DataFrame
        stats: Statistics dictionary
        output_dir: Output directory
        logger: Logger instance
    """
    # Save threshold metrics
    metrics_path = output_dir / "threshold_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved threshold metrics: {metrics_path}")
    
    # Save deduplicated dataset
    dedup_path = output_dir / "merged_cleaned_auto_dedup.csv"
    dedup_df.to_csv(dedup_path, index=False)
    logger.info(f"Saved deduplicated dataset: {dedup_path}")
    
    # Save drop log
    drop_path = output_dir / "auto_drop_log.csv"
    drop_log_df.to_csv(drop_path, index=False)
    logger.info(f"Saved drop log: {drop_path}")
    
    # Save similarity edges
    edges_path = output_dir / "similarity_edges.csv"
    edges_df.to_csv(edges_path, index=False)
    logger.info(f"Saved similarity edges: {edges_path}")
    
    # Save statistics as JSON
    json_path = output_dir / "threshold_statistics.json"
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics: {json_path}")
    
    # Save human-readable summary
    summary_path = output_dir / "threshold_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("STEP 002: THRESHOLD SELECTION AND AUTO-DEDUPLICATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Threshold Evaluation:\n")
        f.write(f"  Labeled pairs evaluated:      {stats['labeled_pairs']:>10,}\n")
        f.write(f"  Positive labels (duplicates): {stats['positive_labels']:>10,}\n")
        f.write(f"  Negative labels:              {stats['negative_labels']:>10,}\n")
        f.write(f"  Threshold range:              {stats['t_min']:.3f} - {stats['t_max']:.3f}\n")
        f.write(f"  Step size:                    {stats['t_step']:.3f}\n\n")
        
        f.write("Selected Threshold:\n")
        f.write(f"  Auto-drop threshold:          {stats['auto_threshold']:>10.3f}\n")
        f.write(f"  Selection method:             {stats['selection_method']}\n")
        f.write(f"  Target precision:             {stats['target_precision']:>10.2f}\n")
        f.write(f"  Beta (F-beta):                {stats['beta']:>10.2f}\n\n")
        
        f.write("Year Filtering:\n")
        f.write(f"  Year range:                   {stats['min_year']} - {stats['max_year']}\n")
        f.write(f"  Records before year filter:   {stats['input_records_after_year_filter']:>10,}\n")
        f.write(f"  Records removed (out of range):{stats['records_removed_by_year']:>10,}\n")
        f.write(f"  Records after year filter:    {stats['records_after_year_filter']:>10,}\n\n")
        
        f.write("Auto-Deduplication Results:\n")
        f.write(f"  Input records (after year):   {stats['records_after_year_filter']:>10,}\n")
        f.write(f"  Records kept:                 {stats['records_kept']:>10,}\n")
        f.write(f"  Records dropped:              {stats['records_dropped']:>10,}\n")
        f.write(f"  Similarity edges used:        {stats['similarity_edges']:>10,}\n")
        f.write(f"  Deduplication groups:         {stats['dedup_groups']:>10,}\n\n")
        
        f.write("Output Files:\n")
        f.write(f"  - threshold_metrics.csv: Metrics for all evaluated thresholds\n")
        f.write(f"  - merged_cleaned_auto_dedup.csv: Final deduplicated dataset\n")
        f.write(f"  - auto_drop_log.csv: Log of dropped records\n")
        f.write(f"  - similarity_edges.csv: Audit trail of similarity edges\n\n")
        
        f.write("=" * 70 + "\n")
    
    logger.info(f"Saved summary: {summary_path}")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Step 002: Threshold selection and auto-deduplication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python 002_threshold_selection.py \\
        --labeled-file /path/to/candidate_pairs_labeled.csv \\
        --input-file /path/to/merged_exact_dedup.csv \\
        --output-dir /path/to/output/002_threshold \\
        --target-precision 0.99 \\
        --beta 0.5
        """
    )
    
    # Input files
    parser.add_argument(
        '--labeled-file',
        type=Path,
        required=True,
        help='Path to manually labeled candidate pairs CSV'
    )
    parser.add_argument(
        '--input-file',
        type=Path,
        required=False,
        default=None,
        help=(
            'Optional: Path to merged dataset from Step 001 (merged_exact_dedup.csv). '
            'If not provided, you must provide --pubmed-file and --scopus-file and the script will rebuild '
            'the merged + exact-dedup dataset in-memory.'
        )
    )

    parser.add_argument(
        '--pubmed-file',
        type=Path,
        required=False,
        default=None,
        help='Optional: PubMed CSV (used when --input-file is not provided)'
    )
    parser.add_argument(
        '--scopus-file',
        type=Path,
        required=False,
        default=None,
        help='Optional: Scopus CSV (used when --input-file is not provided)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for results'
    )
    
    # Threshold evaluation parameters
    parser.add_argument(
        '--sim-col',
        type=str,
        default='Similarity',
        help='Column name for similarity scores (default: Similarity)'
    )
    parser.add_argument(
        '--label-col',
        type=str,
        default='is_duplicate',
        help='Column name for labels (default: is_duplicate)'
    )
    parser.add_argument(
        '--t-min',
        type=float,
        default=0.70,
        help='Minimum threshold to evaluate (default: 0.70)'
    )
    parser.add_argument(
        '--t-max',
        type=float,
        default=0.995,
        help='Maximum threshold to evaluate (default: 0.995)'
    )
    parser.add_argument(
        '--t-step',
        type=float,
        default=0.005,
        help='Threshold step size (default: 0.005)'
    )
    parser.add_argument(
        '--target-precision',
        type=float,
        default=0.99,
        help='Target precision for auto-drop threshold (default: 0.99)'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0.5,
        help='Beta for F-beta score (default: 0.5, favors precision)'
    )
    
    # Deduplication parameters
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
        help='TF-IDF max document frequency (default: 0.95)'
    )
    parser.add_argument(
        '--tfidf-min-df',
        type=int,
        default=1,
        help='TF-IDF min document frequency (default: 1)'
    )
    
    # Year filtering parameters
    parser.add_argument(
        '--min-year',
        type=int,
        default=2000,
        help='Minimum publication year to include (default: 2000)'
    )
    parser.add_argument(
        '--max-year',
        type=int,
        default=2025,
        help='Maximum publication year to include (default: 2025)'
    )
    parser.add_argument(
        '--year-col',
        type=str,
        default='Year',
        help='Column name containing year information (default: Year)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logger = configure_logging(args.output_dir / "step_002_threshold.log")
    
    # Log header
    logger.info("=" * 70)
    logger.info("STEP 002: THRESHOLD SELECTION AND AUTO-DEDUPLICATION")
    logger.info("=" * 70)
    logger.info(f"Labeled file:      {args.labeled_file}")
    logger.info(f"Input file:        {args.input_file}")
    logger.info(f"Output dir:        {args.output_dir}")
    logger.info(f"Threshold range:   {args.t_min} - {args.t_max} (step: {args.t_step})")
    logger.info(f"Target precision:  {args.target_precision}")
    logger.info(f"Beta:              {args.beta}")
    logger.info(f"Year range:        {args.min_year} - {args.max_year}")
    logger.info("=" * 70)
    logger.info("")
    
    try:
        # Validate input files
        if not args.labeled_file.exists():
            raise FileNotFoundError(f"Labeled file not found: {args.labeled_file}")

        # Dataset source: either an explicit input file (from Step 001) OR raw PubMed/Scopus
        if args.input_file is not None:
            if not args.input_file.exists():
                raise FileNotFoundError(f"Input file not found: {args.input_file}")
        else:
            if args.pubmed_file is None or args.scopus_file is None:
                raise ValueError(
                    "You must provide --input-file OR both --pubmed-file and --scopus-file."
                )
            if not args.pubmed_file.exists():
                raise FileNotFoundError(f"PubMed file not found: {args.pubmed_file}")
            if not args.scopus_file.exists():
                raise FileNotFoundError(f"Scopus file not found: {args.scopus_file}")
        
        # Load labeled pairs
        logger.info("Loading labeled pairs...")
        labeled_df = pd.read_csv(args.labeled_file)
        
        # Validate columns
        for col in [args.sim_col, args.label_col]:
            if col not in labeled_df.columns:
                raise ValueError(f"Column '{col}' not found. Available: {list(labeled_df.columns)}")
        
        # Evaluate thresholds
        logger.info("")
        logger.info("Evaluating thresholds...")
        metrics_df = evaluate_thresholds(
            labeled_df,
            args.sim_col,
            args.label_col,
            args.t_min,
            args.t_max,
            args.t_step,
            args.beta,
            logger
        )
        
        # Select optimal threshold
        logger.info("")
        auto_threshold, selection_info = select_optimal_threshold(
            metrics_df,
            args.target_precision,
            args.beta,
            logger
        )
        
        # Load input dataset
        logger.info("")
        if args.input_file is not None:
            logger.info("Loading dataset for deduplication from --input-file...")
            df = pd.read_csv(args.input_file)
            logger.info(f"Loaded {len(df):,} records")
        else:
            logger.info("Rebuilding merged + exact-deduplicated dataset in-memory (no merged CSV saved in Step 001)...")
            df_merged = load_and_harmonize_data(args.pubmed_file, args.scopus_file, logger)
            df = remove_exact_duplicates(df_merged, logger)
            logger.info(f"Records after exact dedup: {len(df):,}")
        
        # Filter by year
        logger.info("")
        logger.info(f"Filtering documents by year ({args.min_year}-{args.max_year})...")
        df, n_year_removed = filter_by_year(
            df,
            args.min_year,
            args.max_year,
            args.year_col,
            logger
        )
        
        # Apply auto-deduplication
        logger.info("")
        dedup_df, drop_log_df, edges_df = apply_auto_deduplication(
            df,
            auto_threshold,
            args.top_k,
            args.tfidf_max_df,
            args.tfidf_min_df,
            logger
        )
        
        # Count labeled pairs statistics
        tmp = labeled_df[[args.label_col]].copy()
        tmp[args.label_col] = pd.to_numeric(tmp[args.label_col], errors="coerce")
        tmp = tmp[tmp[args.label_col].isin([0, 1])]
        n_pos = int((tmp[args.label_col] == 1).sum())
        n_neg = int((tmp[args.label_col] == 0).sum())
        
        # Compute statistics
        stats = {
            'labeled_pairs': int(len(tmp)),
            'positive_labels': n_pos,
            'negative_labels': n_neg,
            't_min': float(args.t_min),
            't_max': float(args.t_max),
            't_step': float(args.t_step),
            'auto_threshold': float(auto_threshold),
            'selection_method': selection_info['selection_method'],
            'target_precision': float(args.target_precision),
            'beta': float(args.beta),
            'min_year': int(args.min_year),
            'max_year': int(args.max_year),
            'records_removed_by_year': int(n_year_removed),
            'input_records_after_year_filter': int(len(df) + n_year_removed),
            'records_after_year_filter': int(len(df)),
            'records_kept': int(len(dedup_df)),
            'records_dropped': int(len(drop_log_df)),
            'similarity_edges': int(len(edges_df)),
            'dedup_groups': int(len(dedup_df))
        }
        
        # Save outputs
        logger.info("")
        save_outputs(
            metrics_df,
            dedup_df,
            drop_log_df,
            edges_df,
            stats,
            args.output_dir,
            logger
        )
        
        # Log completion
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 002 COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info("Output files:")
        logger.info(f"  - {args.output_dir}/threshold_metrics.csv")
        logger.info(f"  - {args.output_dir}/merged_cleaned_auto_dedup.csv")
        logger.info(f"  - {args.output_dir}/auto_drop_log.csv")
        logger.info(f"  - {args.output_dir}/similarity_edges.csv")
        logger.info(f"  - {args.output_dir}/threshold_statistics.json")
        logger.info(f"  - {args.output_dir}/threshold_summary.txt")
        logger.info(f"  - {args.output_dir}/step_002_threshold.log")
        logger.info("")
        logger.info("Next: Copy merged_cleaned_auto_dedup.csv to input directory for Step 01")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        logger.error("Step 002 FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())