#!/usr/bin/env python3
"""
Step 01: Filter Abstracts for Dairy Cattle Epidemiology

This script applies inclusion and exclusion filters to abstract data:
1. Inclusion: Keep abstracts with epidemiological risk/association terms
2. Exclusion: Remove abstracts about non-target species and human-only topics

Inputs:
    - merged_cleaned_auto_dedup.csv: Cleaned abstract dataset

Outputs:
    - filtered_abstracts.csv: Abstracts passing both filters
    - filter_statistics.json: Filtering statistics
    - filter_summary.txt: Human-readable summary
    - step_01_filter.log: Execution log

Author: Reda Zahri
Date: 2025
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd


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
# INCLUSION PATTERN: Epidemiological risk/association terms
# ============================================================

RISK_PATTERN = re.compile(
    r"""
    (risk\ factor[s]?)|
    (relative\ risk)|
    (risk\ ratio)|
    (rate\ ratio)|
    (\bRR\b\s*[=]?\s*\d+(\.\d+)?)|
    (odds\ ratio)|
    (\bOR\b\s*[=]?\s*\d+(\.\d+)?)|
    (p[-\s]?value)|
    (associat\w*)|
    (confidence\ interval)|
    (95\%?\s*CI)|
    (CI\s*95\%?)|
    (case[-\s]?control)|
    (cohort\ study)|
    (prevalence\ ratio)|
    (\bPR\b\s*[=]?\s*\d+(\.\d+)?)|
    (cross[-\s]?sectional)
    """,
    re.IGNORECASE | re.VERBOSE
)


# ============================================================
# EXCLUSION PATTERN: Non-target species & human-only terms
# ============================================================

EXCLUSION_PATTERN = re.compile(
    r"""
    # Non-target animals (non-cattle)
    \b(donkey|mule|horse|equine|camel|camelid|llama|alpaca)s?\b|
    \b(whale|dolphin|porpoise)s?\b|
    \bseal(s)?\b|
    \bsea\s+lions?\b|
    \bwalrus(es)?\b|
    \b(poultry|chicken|broiler|hen|turkey|duck)s?\b|
    \b(goose|geese)\b|
    \b(badger|deer|elk|moose|caribou|antelope|boar)\b|
    \b(murine|fox|bat|primate|monkey)s?\b|
    
    # Human population / medical terms
    \b(infant|toddler|adolescent)s?\b|
    \belderly\b|
    \bICU\b|
    \bbreastfeeding\b|
    \bdiabetes\b|
    \bDVT\b|
    \bmenopaus\w*\b|
    \bmalaria\b|
    \bwoman\b|\bwomen\b
    """,
    re.IGNORECASE | re.VERBOSE
)


# ============================================================
# FILTER FUNCTIONS
# ============================================================

def has_risk_marker(text: str) -> bool:
    """
    Check if text contains epidemiological risk/association markers.
    
    Args:
        text: Text to check
        
    Returns:
        True if risk markers found, False otherwise
    """
    if not isinstance(text, str):
        return False
    return bool(RISK_PATTERN.search(text))


def contains_excluded_term(text: str) -> bool:
    """
    Check if text contains excluded species or human-only terms.
    
    Args:
        text: Text to check
        
    Returns:
        True if excluded terms found, False otherwise
    """
    if not isinstance(text, str):
        return False
    return bool(EXCLUSION_PATTERN.search(text))


def apply_filters(df: pd.DataFrame, logger: logging.Logger) -> tuple:
    """
    Apply inclusion and exclusion filters to abstracts.
    
    Args:
        df: DataFrame with 'Abstract' column
        logger: Logger instance
        
    Returns:
        Tuple of (filtered_df, statistics_dict)
    """
    logger.info(f"Initial dataset: {len(df):,} abstracts")
    
    # Apply inclusion filter
    logger.info("Applying inclusion filter (epidemiological terms)...")
    df['has_risk_marker'] = df['Abstract'].apply(has_risk_marker)
    n_with_risk = df['has_risk_marker'].sum()
    logger.info(f"  Abstracts with risk markers: {n_with_risk:,} ({n_with_risk/len(df)*100:.2f}%)")
    
    # Apply exclusion filter
    logger.info("Applying exclusion filter (non-target species/humans)...")
    df['contains_excluded'] = df['Abstract'].apply(contains_excluded_term)
    n_excluded = df['contains_excluded'].sum()
    logger.info(f"  Abstracts with excluded terms: {n_excluded:,} ({n_excluded/len(df)*100:.2f}%)")
    
    # Final filtering: must have risk markers AND not have excluded terms
    filtered = df[(df['has_risk_marker']) & (~df['contains_excluded'])].copy()
    
    logger.info(f"Final filtered dataset: {len(filtered):,} abstracts ({len(filtered)/len(df)*100:.2f}% retained)")
    
    # Calculate statistics
    stats = {
        'total_input': int(len(df)),
        'with_risk_markers': int(n_with_risk),
        'with_excluded_terms': int(n_excluded),
        'final_kept': int(len(filtered)),
        'final_removed': int(len(df) - len(filtered)),
        'retention_rate': float(len(filtered) / len(df))
    }
    
    return filtered, stats


def save_outputs(
    filtered_df: pd.DataFrame,
    stats: dict,
    output_dir: Path,
    logger: logging.Logger
) -> None:
    """
    Save filtered data and statistics.
    
    Args:
        filtered_df: Filtered DataFrame
        stats: Statistics dictionary
        output_dir: Output directory
        logger: Logger instance
    """
    # Save filtered CSV (without temporary filter columns)
    output_csv = output_dir / "filtered_abstracts.csv"
    filtered_df_clean = filtered_df.drop(columns=['has_risk_marker', 'contains_excluded'])
    filtered_df_clean.to_csv(output_csv, index=False)
    logger.info(f"Saved filtered abstracts: {output_csv}")
    
    # Save statistics as JSON
    json_path = output_dir / "filter_statistics.json"
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics: {json_path}")
    
    # Save human-readable summary
    summary_path = output_dir / "filter_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("STEP 01: ABSTRACT FILTERING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Filtering Criteria:\n")
        f.write("  Inclusion: Epidemiological risk/association terms\n")
        f.write("  Exclusion: Non-cattle species and human-only medical terms\n\n")
        
        f.write("Results:\n")
        f.write(f"  Total input abstracts:        {stats['total_input']:>10,}\n")
        f.write(f"  With risk markers:            {stats['with_risk_markers']:>10,} "
                f"({stats['with_risk_markers']/stats['total_input']*100:>6.2f}%)\n")
        f.write(f"  With excluded terms:          {stats['with_excluded_terms']:>10,} "
                f"({stats['with_excluded_terms']/stats['total_input']*100:>6.2f}%)\n")
        f.write(f"  Final kept (both filters):    {stats['final_kept']:>10,} "
                f"({stats['retention_rate']*100:>6.2f}%)\n")
        f.write(f"  Final removed:                {stats['final_removed']:>10,} "
                f"({stats['final_removed']/stats['total_input']*100:>6.2f}%)\n\n")
        
        f.write("Filter Logic:\n")
        f.write("  kept = (has_risk_marker == True) AND (contains_excluded == False)\n\n")
        
        f.write("=" * 70 + "\n")
    
    logger.info(f"Saved summary: {summary_path}")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Step 01: Filter abstracts for dairy cattle epidemiology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python 01_filter.py \\
        --input-file /path/to/merged_cleaned_auto_dedup.csv \\
        --output-dir /path/to/output/01_filter \\
        --text-column Abstract
        """
    )
    
    parser.add_argument(
        '--input-file',
        type=Path,
        required=True,
        help='Input CSV file with abstracts'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory for filtered data'
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='Abstract',
        help='Column name containing abstract text (default: Abstract)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logger = configure_logging(args.output_dir / "step_01_filter.log")
    
    # Log header
    logger.info("=" * 70)
    logger.info("STEP 01: FILTER ABSTRACTS")
    logger.info("=" * 70)
    logger.info(f"Input file:    {args.input_file}")
    logger.info(f"Output dir:    {args.output_dir}")
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
        logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
        
        # Check required column
        if args.text_column not in df.columns:
            raise ValueError(
                f"Column '{args.text_column}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Apply filters
        filtered_df, stats = apply_filters(df, logger)
        
        # Save outputs
        save_outputs(filtered_df, stats, args.output_dir, logger)
        
        # Log completion
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 01 COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info("Output files:")
        logger.info(f"  - {args.output_dir}/filtered_abstracts.csv")
        logger.info(f"  - {args.output_dir}/filter_statistics.json")
        logger.info(f"  - {args.output_dir}/filter_summary.txt")
        logger.info(f"  - {args.output_dir}/step_01_filter.log")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        logger.error("Step 01 FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())