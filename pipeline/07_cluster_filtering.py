#!/usr/bin/env python3
"""
Step 07: Filter abstracts based on manual topic review

This script filters the dataset to keep only abstracts from topics classified as "Keep"
during manual review. This comes AFTER Step 06 (LLM labeling), which provides
automated assessments of dairy cattle health relevance to help with your review.

Typical workflow:
1. Step 05: BERTopic clustering
2. Step 06: LLM labeling (automated dairy health relevance assessment)
3. Manual review: Review LLM labels and decide Keep/Remove for each topic
4. Step 07: This script - filter based on your decisions

The script is used to:
1. Remove off-topic abstracts before downstream analysis
2. Create a refined dataset for further processing  
3. Prepare data for potential pipeline re-runs

The script preserves the original Abstract column for downstream use.

Usage:
    python 07_cluster_filtering.py \
        --input-file /path/to/abstracts_with_truncation.csv \
        --document-info /path/to/document_info_full.csv \
        --output-dir /path/to/output \
        --text-column Abstract_trunc512

IMPORTANT: The --text-column parameter must match the column used in step 05 clustering.
If you used truncated abstracts (Abstract_trunc512) for clustering, you MUST use that same
column here. This ensures that the filtered output maintains consistency with the clustering.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import pandas as pd
import json


# =============================================================================
# TOPIC CLASSIFICATION FROM MANUAL REVIEW
# =============================================================================
# This dictionary should be updated based on your manual review of topics
# Each topic ID is classified as either "Keep" or "Remove"

TOPIC_CLASSIFICATION = {
    -1: "Keep",
    0: "Keep",
    1: "Keep",
    2: "Remove",
    3: "Keep",
    4: "Keep",
    5: "Keep",
    6: "Remove",
    7: "Keep",
    8: "Remove",
    9: "Keep",
    10: "Keep",
    11: "Keep",
    12: "Remove",
    13: "Remove",
    14: "Remove",
    15: "Keep",
    16: "Keep",
    17: "Keep",
    18: "Keep",
    19: "Keep",
    20: "Keep",
    21: "Keep",
    22: "Keep",
    23: "Remove",
    24: "Remove",
    25: "Remove",
    26: "Keep",
    27: "Keep",
    28: "Keep",
    29: "Remove",
    30: "Remove",
    31: "Keep",
    32: "Keep",
    33: "Keep",
    34: "Keep",
    35: "Remove",
    36: "Remove",
    37: "Keep",
    38: "Remove",
    39: "Keep",
    40: "Keep",
    41: "Keep",
    42: "Keep",
    43: "Keep",
    44: "Remove",
    45: "Keep",
    46: "Keep",
    47: "Keep",
    48: "Keep",
    49: "Keep",
    50: "Keep",
    51: "Keep",
    52: "Remove",
    53: "Keep",
    54: "Keep",
    55: "Remove",
    56: "Remove",
    57: "Remove",
    58: "Keep",
    59: "Remove",
    60: "Remove",
    61: "Remove",
    62: "Remove",
    63: "Keep",
    64: "Remove",
    65: "Keep",
    66: "Keep",
    67: "Keep",
    68: "Keep",
    69: "Keep",
    70: "Keep",
    71: "Remove",
    72: "Remove",
    73: "Remove",
    74: "Remove",
    75: "Keep",
    76: "Keep",
    77: "Remove",
    78: "Keep",
    79: "Remove",
    80: "Remove",
    81: "Remove",
    82: "Keep",
    83: "Remove",
    84: "Remove",
    85: "Keep",
    86: "Keep",
    87: "Remove",
    88: "Remove",
    89: "Keep",
    90: "Keep",
    91: "Remove",
    92: "Keep",
    93: "Keep",
    94: "Keep",
    95: "Remove",
    96: "Keep",
    97: "Keep",
    98: "Keep",
    99: "Keep",
    100: "Keep",
    101: "Keep",
    102: "Remove",
    103: "Remove",
    104: "Remove",
    105: "Remove",
    106: "Keep",
    107: "Remove",
    108: "Keep",
    109: "Remove",
    110: "Remove",
    111: "Keep",
    112: "Keep",
    113: "Remove",
    114: "Remove",
    115: "Keep",
    116: "Remove",
    117: "Keep",
    118: "Keep",
    119: "Remove",
    120: "Remove",
    121: "Remove",
    122: "Keep",
    123: "Keep",
    124: "Keep",
    125: "Keep",
    126: "Keep",
    127: "Remove",
    128: "Keep",
    129: "Keep",
    130: "Keep",
    131: "Keep",
    132: "Remove",
    133: "Keep",
    134: "Keep",
    135: "Remove",
    136: "Keep",
    137: "Remove",
    138: "Remove",
    139: "Remove",
    140: "Keep",
    141: "Keep",
    142: "Keep",
    143: "Keep",
    144: "Remove",
    145: "Keep",
    146: "Keep",
    147: "Keep",
    148: "Keep",
    149: "Keep",
    150: "Keep",
    151: "Keep",
    152: "Keep",
    153: "Remove",
    154: "Keep",
    155: "Remove",
    156: "Keep",
    157: "Remove",
    158: "Keep",
    159: "Remove",
    160: "Remove",
    161: "Keep",
    162: "Remove",
    163: "Remove",
    164: "Remove",
    165: "Remove",
    166: "Remove",
    167: "Keep",
    168: "Keep",
    169: "Remove",
    170: "Remove",
    171: "Keep",
    172: "Keep",
    173: "Remove",
    174: "Remove",
}

# Precompute topics to remove
DROP_TOPICS = {topic_id for topic_id, status in TOPIC_CLASSIFICATION.items() if status == "Remove"}


def setup_logging(output_dir):
    """Setup logging to both file and console."""
    log_file = os.path.join(output_dir, "step_07_cluster_filtering.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_document_info(doc_info_path, logger):
    """Load document_info_full.csv with topic assignments."""
    logger.info(f"Loading document info: {doc_info_path}")
    
    if not os.path.exists(doc_info_path):
        raise FileNotFoundError(f"Document info not found: {doc_info_path}")
    
    doc_info = pd.read_csv(doc_info_path)
    logger.info(f"Loaded {len(doc_info):,} document-topic assignments")
    
    # Validate required columns
    required_cols = ['Topic', 'Document']
    missing = [col for col in required_cols if col not in doc_info.columns]
    if missing:
        raise ValueError(f"document_info_full.csv missing required columns: {missing}")
    
    return doc_info


def load_input_data(input_file, text_column, logger):
    """Load the input CSV with abstracts."""
    logger.info(f"Loading input data: {input_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df):,} rows")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Validate text column exists
    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found in input CSV. "
            f"Available columns: {list(df.columns)}"
        )
    
    return df


def filter_by_topics(doc_info, original_df, text_column, logger):
    """Filter documents based on topic classification."""
    logger.info("Filtering documents by topic classification...")

    # Filter document_info to keep only "Keep" topics
    kept_docs = doc_info[~doc_info["Topic"].isin(DROP_TOPICS)].copy()

    logger.info(f"Topic filtering:")
    logger.info(f"  Total topics:    {len(TOPIC_CLASSIFICATION)}")
    logger.info(f"  Keep topics:     {len(TOPIC_CLASSIFICATION) - len(DROP_TOPICS)}")
    logger.info(f"  Remove topics:   {len(DROP_TOPICS)}")
    logger.info(f"  Documents kept:  {len(kept_docs):,} / {len(doc_info):,}")
    logger.info(f"  Retention rate:  {len(kept_docs)/len(doc_info)*100:.2f}%")

    # Validate text column length consistency (early warning for mismatches)
    sample_doc = kept_docs['Document'].iloc[0] if len(kept_docs) > 0 else ""
    sample_input = original_df[text_column].iloc[0] if len(original_df) > 0 else ""

    avg_doc_len = kept_docs['Document'].astype(str).str.len().mean()
    avg_input_len = original_df[text_column].astype(str).str.len().mean()

    if abs(avg_doc_len - avg_input_len) > 100:
        logger.warning(
            f"Text length mismatch detected! "
            f"Avg document_info length: {avg_doc_len:.0f}, "
            f"Avg input file length: {avg_input_len:.0f}. "
            f"Are you using the correct --text-column? "
            f"(Current: '{text_column}')"
        )

    # Create set of kept abstracts for matching
    kept_abstracts = set(kept_docs['Document'].astype(str).str.strip())
    logger.info(f"Created set of {len(kept_abstracts):,} unique abstracts to keep")
    
    # Match with original CSV
    logger.info(f"Matching with original CSV using column: '{text_column}'")
    original_df[text_column] = original_df[text_column].astype(str).str.strip()
    filtered_df = original_df[original_df[text_column].isin(kept_abstracts)].copy()
    
    logger.info(f"Matched {len(filtered_df):,} rows from original CSV")
    
    if len(filtered_df) == 0:
        raise ValueError(
            "No abstracts matched! This might mean:\n"
            "  1. Text was modified during preprocessing\n"
            "  2. Wrong column selected for matching\n"
            "  3. Document column in document_info doesn't match input CSV"
        )
    
    # Check match rate
    match_rate = len(filtered_df) / len(kept_docs) * 100
    if match_rate < 95:
        logger.warning(
            f"Low match rate: {match_rate:.2f}%. "
            f"Expected {len(kept_docs):,} matches but got {len(filtered_df):,}"
        )
    
    return filtered_df


def save_outputs(filtered_df, output_dir, logger):
    """Save filtered data and summary files."""
    logger.info("Saving outputs...")
    
    # Save filtered CSV with descriptive name indicating topic-based filtering
    output_csv = os.path.join(output_dir, "abstracts_topic_filtered.csv")
    filtered_df.to_csv(output_csv, index=False)
    logger.info(f"Saved topic-filtered CSV: {output_csv}")
    logger.info(f"  Rows: {len(filtered_df):,}")
    logger.info(f"  Columns: {len(filtered_df.columns)}")
    
    return output_csv


def generate_summary(doc_info, original_df, filtered_df, output_dir, logger):
    """Generate and save summary statistics."""
    logger.info("Generating summary...")
    
    # Calculate statistics
    n_topics_total = len(TOPIC_CLASSIFICATION)
    n_topics_keep = n_topics_total - len(DROP_TOPICS)
    n_topics_remove = len(DROP_TOPICS)
    
    n_docs_original = len(original_df)
    n_docs_filtered = len(filtered_df)
    retention_rate = n_docs_filtered / n_docs_original
    
    n_docs_from_topic_info = len(doc_info[~doc_info["Topic"].isin(DROP_TOPICS)])
    
    summary = {
        "generated_at": datetime.now().isoformat(),
        "topic_classification": {
            "total_topics": n_topics_total,
            "keep_topics": n_topics_keep,
            "remove_topics": n_topics_remove,
            "removed_topic_ids": sorted(list(DROP_TOPICS))
        },
        "filtering_results": {
            "original_documents": n_docs_original,
            "filtered_documents": n_docs_filtered,
            "documents_removed": n_docs_original - n_docs_filtered,
            "retention_rate": round(retention_rate, 4),
            "expected_from_topic_info": n_docs_from_topic_info,
            "match_rate": round(n_docs_filtered / n_docs_from_topic_info, 4) if n_docs_from_topic_info > 0 else 0
        },
        "preserved_columns": list(filtered_df.columns),
        "paths": {
            "filtered_csv": os.path.join(output_dir, "abstracts_topic_filtered.csv"),
            "summary_json": os.path.join(output_dir, "step_07_summary.json"),
            "log_file": os.path.join(output_dir, "step_07_cluster_filtering.log")
        }
    }
    
    # Save JSON summary
    summary_json = os.path.join(output_dir, "step_07_summary.json")
    with open(summary_json, 'w') as f:
        json.dump(summary, indent=2, fp=f)
    logger.info(f"Saved summary: {summary_json}")
    
    # Save detailed text summary
    summary_txt = os.path.join(output_dir, "step_07_filtering_details.txt")
    with open(summary_txt, 'w') as f:
        f.write("="*70 + "\n")
        f.write("STEP 07: TOPIC-BASED FILTERING SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Generated: {summary['generated_at']}\n\n")
        
        f.write("TOPIC CLASSIFICATION:\n")
        f.write(f"  Total topics:  {n_topics_total}\n")
        f.write(f"  Keep topics:   {n_topics_keep}\n")
        f.write(f"  Remove topics: {n_topics_remove}\n\n")
        
        f.write(f"REMOVED TOPICS ({n_topics_remove}):\n")
        sorted_dropped = sorted(DROP_TOPICS)
        for i in range(0, len(sorted_dropped), 10):
            chunk = sorted_dropped[i:i+10]
            f.write("  " + ", ".join(str(t) for t in chunk) + "\n")
        f.write("\n")
        
        f.write("FILTERING RESULTS:\n")
        f.write(f"  Original documents:  {n_docs_original:,}\n")
        f.write(f"  Filtered documents:  {n_docs_filtered:,}\n")
        f.write(f"  Documents removed:   {n_docs_original - n_docs_filtered:,}\n")
        f.write(f"  Retention rate:      {retention_rate*100:.2f}%\n\n")
        
        f.write("PRESERVED COLUMNS:\n")
        for col in filtered_df.columns:
            f.write(f"  - {col}\n")
        f.write("\n")
        
        f.write("OUTPUT FILES:\n")
        for key, path in summary['paths'].items():
            f.write(f"  {key}: {path}\n")
        f.write("\n")
        
        f.write("NEXT STEPS:\n")
        f.write("  This filtered dataset can be used for:\n")
        f.write("  1. Downstream analyses on relevant topics only\n")
        f.write("  2. Re-running the pipeline with refined data\n")
        f.write("  3. Further manual review or validation\n")
        f.write("="*70 + "\n")
    
    logger.info(f"Saved detailed summary: {summary_txt}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Filter abstracts based on manual topic review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to input CSV with abstracts (from step 02)"
    )
    parser.add_argument(
        "--document-info",
        required=True,
        help="Path to document_info_full.csv (from step 05)"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save filtered outputs"
    )
    
    # Optional arguments
    parser.add_argument(
        "--text-column",
        default="Abstract_trunc512",
        help="Name of the column containing text data (default: Abstract_trunc512). "
             "IMPORTANT: Must match the column used in step 05 clustering. "
             "If you used truncated abstracts for clustering (recommended), use Abstract_trunc512."
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    try:
        logger.info("="*60)
        logger.info("STEP 07: TOPIC-BASED FILTERING")
        logger.info("="*60)
        logger.info(f"Input file:     {args.input_file}")
        logger.info(f"Document info:  {args.document_info}")
        logger.info(f"Output dir:     {args.output_dir}")
        logger.info(f"Text column:    {args.text_column}")
        logger.info("="*60)
        
        # Load data
        doc_info = load_document_info(args.document_info, logger)
        original_df = load_input_data(args.input_file, args.text_column, logger)
        
        # Filter by topics
        filtered_df = filter_by_topics(doc_info, original_df, args.text_column, logger)
        
        # Save outputs
        output_csv = save_outputs(filtered_df, args.output_dir, logger)
        
        # Generate summary
        summary = generate_summary(doc_info, original_df, filtered_df, args.output_dir, logger)
        
        # Print summary
        logger.info("")
        logger.info("="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        logger.info(f"Original documents:  {summary['filtering_results']['original_documents']:,}")
        logger.info(f"Filtered documents:  {summary['filtering_results']['filtered_documents']:,}")
        logger.info(f"Retention rate:      {summary['filtering_results']['retention_rate']*100:.2f}%")
        logger.info("")
        logger.info("Output files:")
        logger.info(f"  - {os.path.basename(summary['paths']['filtered_csv'])}")
        logger.info(f"  - {os.path.basename(summary['paths']['summary_json'])}")
        logger.info(f"  - step_07_filtering_details.txt")
        logger.info("="*60)
        logger.info("SUCCESS: Step 07 completed")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()