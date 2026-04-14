#!/usr/bin/env python3
"""
Step 02: Truncate Abstracts to Model Token Limit

This script truncates abstracts to fit within the embedding model's token limit
(512 tokens for PubMedBERT). Creates a new column with truncated text while preserving
the original abstract.

Inputs:
    - filtered_abstracts.csv: Filtered abstract dataset from Step 01

Outputs:
    - abstracts_with_truncation.csv: Original + truncated abstracts
    - truncation_statistics.json: Statistics on token lengths
    - truncation_summary.txt: Human-readable summary
    - step_02_truncate.log: Execution log

Author: Reda Zahri
Date: 2025
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


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
# TOKENIZATION FUNCTIONS
# ============================================================

def get_token_statistics(
    texts: pd.Series,
    tokenizer,
    logger: logging.Logger
) -> tuple:
    """
    Calculate token length statistics for a collection of texts.
    
    Args:
        texts: Series of text strings
        tokenizer: HuggingFace tokenizer
        logger: Logger instance
        
    Returns:
        Tuple of (statistics_dict, lengths_array)
    """
    logger.info("Calculating token statistics...")
    
    lengths = texts.apply(
        lambda t: len(tokenizer(
            t,
            add_special_tokens=True,
            truncation=False
        )["input_ids"])
    )
    
    stats = {
        'n_documents': int(len(texts)),
        'min': int(lengths.min()),
        'max': int(lengths.max()),
        'mean': float(lengths.mean()),
        'median': float(lengths.median()),
        'std': float(lengths.std()),
        'p25': float(np.percentile(lengths, 25)),
        'p50': float(np.percentile(lengths, 50)),
        'p75': float(np.percentile(lengths, 75)),
        'p90': float(np.percentile(lengths, 90)),
        'p95': float(np.percentile(lengths, 95)),
        'p99': float(np.percentile(lengths, 99)),
    }
    
    return stats, lengths


def truncate_texts(
    texts: pd.Series,
    tokenizer,
    max_length: int,
    logger: logging.Logger
) -> pd.Series:
    """
    Truncate texts to specified token length.
    
    Args:
        texts: Series of text strings
        tokenizer: HuggingFace tokenizer
        max_length: Maximum token length
        logger: Logger instance
        
    Returns:
        Series of truncated texts
    """
    logger.info(f"Truncating texts to {max_length} tokens...")
    
    def truncate_single(text: str) -> str:
        """Truncate a single text to max_length tokens."""
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length
        )
        return tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
    
    truncated = texts.apply(truncate_single)
    
    return truncated


# ============================================================
# OUTPUT FUNCTIONS
# ============================================================

def save_outputs(
    df: pd.DataFrame,
    stats_before: dict,
    stats_after: dict,
    max_length: int,
    n_exceeding: int,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Save truncated data and statistics.
    
    Args:
        df: DataFrame with original and truncated columns
        stats_before: Statistics before truncation
        stats_after: Statistics after truncation
        max_length: Token limit
        n_exceeding: Number of texts exceeding limit
        output_dir: Output directory
        logger: Logger instance
    """
    # Save CSV with both original and truncated abstracts
    output_csv = output_dir / "abstracts_with_truncation.csv"
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved data with truncation: {output_csv}")
    
    # Calculate percentage exceeding
    pct_exceeding = (n_exceeding / stats_before['n_documents']) * 100
    
    # Save statistics as JSON
    combined_stats = {
        'max_token_limit': max_length,
        'n_documents': stats_before['n_documents'],
        'n_exceeding_limit': int(n_exceeding),
        'pct_exceeding_limit': float(pct_exceeding),
        'before_truncation': stats_before,
        'after_truncation': stats_after
    }
    
    json_path = output_dir / "truncation_statistics.json"
    with open(json_path, 'w') as f:
        json.dump(combined_stats, f, indent=2)
    logger.info(f"Saved statistics: {json_path}")
    
    # Save human-readable summary
    summary_path = output_dir / "truncation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("STEP 02: ABSTRACT TRUNCATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Token Limit: {max_length}\n")
        f.write(f"Total Documents: {stats_before['n_documents']:,}\n")
        f.write(f"Exceeding Limit: {n_exceeding:,} ({pct_exceeding:.2f}%)\n\n")
        
        f.write("Token Length Statistics (Before Truncation):\n")
        f.write(f"  Min:       {stats_before['min']:>8,} tokens\n")
        f.write(f"  Max:       {stats_before['max']:>8,} tokens\n")
        f.write(f"  Mean:      {stats_before['mean']:>8,.1f} tokens\n")
        f.write(f"  Median:    {stats_before['median']:>8,.1f} tokens\n")
        f.write(f"  Std Dev:   {stats_before['std']:>8,.1f} tokens\n")
        f.write(f"  P90:       {stats_before['p90']:>8,.1f} tokens\n")
        f.write(f"  P95:       {stats_before['p95']:>8,.1f} tokens\n")
        f.write(f"  P99:       {stats_before['p99']:>8,.1f} tokens\n\n")
        
        f.write("Token Length Statistics (After Truncation):\n")
        f.write(f"  Min:       {stats_after['min']:>8,} tokens\n")
        f.write(f"  Max:       {stats_after['max']:>8,} tokens\n")
        f.write(f"  Mean:      {stats_after['mean']:>8,.1f} tokens\n")
        f.write(f"  Median:    {stats_after['median']:>8,.1f} tokens\n")
        f.write(f"  Std Dev:   {stats_after['std']:>8,.1f} tokens\n\n")
        
        f.write("=" * 70 + "\n")
    
    logger.info(f"Saved summary: {summary_path}")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Step 02: Truncate abstracts to model token limit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Example usage:
    python 02_truncate.py \\
        --input-file /path/to/01_filter/filtered_abstracts.csv \\
        --output-dir /path/to/output/02_truncate \\
        --model-name pritamdeka/S-BioBert-snli-multinli-stsb \\
        --max-tokens 512
        """
    )
    
    parser.add_argument(
        '--input-file',
        type=Path,
        required=True,
        help='Input CSV file from Step 01 (filtered_abstracts.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='pritamdeka/S-BioBert-snli-multinli-stsb',
        help='Model name for tokenizer (default: BioBERT)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum token length (default: 512)'
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='Abstract',
        help='Column containing abstract text (default: Abstract)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logger = configure_logging(args.output_dir / "step_02_truncate.log")
    
    # Log header
    logger.info("=" * 70)
    logger.info("STEP 02: TRUNCATE ABSTRACTS")
    logger.info("=" * 70)
    logger.info(f"Input file:    {args.input_file}")
    logger.info(f"Output dir:    {args.output_dir}")
    logger.info(f"Model:         {args.model_name}")
    logger.info(f"Max tokens:    {args.max_tokens}")
    logger.info(f"Text column:   {args.text_column}")
    logger.info("=" * 70)
    logger.info("")
    
    try:
        # Validate input file
        if not args.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        
        # Load data
        logger.info("Loading data...")
        df = pd.read_csv(args.input_file)
        logger.info(f"Loaded {len(df):,} abstracts")
        
        # Check required column
        if args.text_column not in df.columns:
            raise ValueError(
                f"Column '{args.text_column}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Remove any existing rows with missing abstracts
        initial_count = len(df)
        df = df.dropna(subset=[args.text_column]).copy()
        df[args.text_column] = df[args.text_column].astype(str).str.strip()
        
        if len(df) < initial_count:
            logger.warning(f"Removed {initial_count - len(df)} rows with missing abstracts")
        
        logger.info(f"Processing {len(df):,} abstracts")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {args.model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Handle extremely large model_max_length values
        true_max = args.max_tokens
        if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length < 1_000_000:
            true_max = min(args.max_tokens, tokenizer.model_max_length)
        
        logger.info(f"Effective max length: {true_max} tokens")
        
        # Get statistics before truncation
        stats_before, lengths_before = get_token_statistics(
            df[args.text_column],
            tokenizer,
            logger
        )
        
        n_exceeding = (lengths_before > true_max).sum()
        pct_exceeding = n_exceeding / len(df) * 100
        logger.info(f"Abstracts exceeding {true_max} tokens: {n_exceeding:,} ({pct_exceeding:.2f}%)")
        
        # Truncate
        truncated_column_name = f"{args.text_column}_trunc{true_max}"
        logger.info(f"Creating truncated column: {truncated_column_name}")
        
        df[truncated_column_name] = truncate_texts(
            df[args.text_column],
            tokenizer,
            true_max,
            logger
        )
        
        # Get statistics after truncation
        stats_after, lengths_after = get_token_statistics(
            df[truncated_column_name],
            tokenizer,
            logger
        )
        
        # Save outputs
        save_outputs(
            df,
            stats_before,
            stats_after,
            true_max,
            n_exceeding,
            args.output_dir,
            logger
        )
        
        # Log summary
        logger.info("")
        logger.info("Truncation Summary:")
        logger.info(f"  Before: mean={stats_before['mean']:.1f}, max={stats_before['max']}")
        logger.info(f"  After:  mean={stats_after['mean']:.1f}, max={stats_after['max']}")
        logger.info(f"  Texts truncated: {n_exceeding:,} ({pct_exceeding:.2f}%)")
        
        # Log completion
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 02 COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info("Output files:")
        logger.info(f"  - {args.output_dir}/abstracts_with_truncation.csv")
        logger.info(f"  - {args.output_dir}/truncation_statistics.json")
        logger.info(f"  - {args.output_dir}/truncation_summary.txt")
        logger.info(f"  - {args.output_dir}/step_02_truncate.log")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        logger.error("Step 02 FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())