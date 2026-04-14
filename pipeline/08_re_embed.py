#!/usr/bin/env python3
"""
Step 08: Re-embed Filtered Abstracts (After Topic Filtering)

After manually filtering topics in Step 06, this script re-generates embeddings
for only the abstracts assigned to kept topics. Uses the same embedding approach
as Step 03 (chunk-mean pooling).

Inputs:
    - filtered_abstracts.csv: Abstracts with only kept topics (from Step 06)

Outputs:
    - embeddings_rerun.npy: Dense embeddings for filtered abstracts
    - embedding_metadata_rerun.json: Metadata about embeddings
    - embedding_summary_rerun.txt: Human-readable summary
    - step_08_re_embed.log: Execution log

Author: Reda Zahri
Date: 2025
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, models
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


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# CHUNKING FUNCTIONS (Same as Step 03)
# ============================================================

def chunk_token_ids(
    input_ids: List[int],
    chunk_length: int,
    stride: int
) -> List[List[int]]:
    """
    Split token IDs into overlapping chunks.
    
    Args:
        input_ids: List of token IDs
        chunk_length: Maximum tokens per chunk
        stride: Step size between chunks (e.g., 409 for ~80% overlap)
        
    Yields:
        Chunks of token IDs
    """
    n = len(input_ids)
    
    # If text fits in one chunk, return as-is
    if n <= chunk_length:
        yield input_ids
        return
    
    # Generate overlapping chunks
    start = 0
    while start < n:
        end = min(start + chunk_length, n)
        chunk = input_ids[start:end]
        
        if chunk:
            yield chunk
        
        if end == n:
            break
            
        start += stride


def embed_with_chunking(
    texts: List[str],
    model: SentenceTransformer,
    tokenizer,
    chunk_length: int,
    stride: int,
    batch_size: int,
    logger: logging.Logger
) -> np.ndarray:
    """
    Generate embeddings using chunk-mean pooling for long texts.
    
    Args:
        texts: List of text strings
        model: Sentence transformer model
        tokenizer: HuggingFace tokenizer
        chunk_length: Maximum tokens per chunk
        stride: Stride between chunks
        batch_size: Batch size for encoding
        logger: Logger instance
        
    Returns:
        Array of embeddings
    """
    logger.info("Encoding texts with chunk-mean pooling...")
    
    # First pass: identify short vs long texts
    logger.info("Analyzing text lengths...")
    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False
    )
    
    lengths = [len(ids) for ids in encoded["input_ids"]]
    short_indices = [i for i, length in enumerate(lengths) if length <= chunk_length]
    long_indices = [i for i, length in enumerate(lengths) if length > chunk_length]
    
    logger.info(f"  Short texts (≤{chunk_length} tokens): {len(short_indices):,}")
    logger.info(f"  Long texts (>{chunk_length} tokens): {len(long_indices):,}")
    
    # Initialize output array
    embeddings = [None] * len(texts)
    
    # Encode short texts in batch
    if short_indices:
        logger.info("Encoding short texts in batch...")
        short_texts = [texts[i] for i in short_indices]
        short_embeddings = model.encode(
            short_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        for idx, embedding in zip(short_indices, short_embeddings):
            embeddings[idx] = embedding
    
    # Encode long texts with chunking
    if long_indices:
        logger.info(f"Encoding long texts with chunking (stride={stride})...")
        
        for i, text_idx in enumerate(long_indices):
            if i % 100 == 0 and i > 0:
                logger.info(f"  Processing long text {i}/{len(long_indices)}")
            
            input_ids = encoded["input_ids"][text_idx]
            
            # Generate chunks
            chunks = list(chunk_token_ids(input_ids, chunk_length, stride))
            
            # Decode chunks back to text
            chunk_texts = [
                tokenizer.decode(chunk, skip_special_tokens=True)
                for chunk in chunks
            ]
            
            # Encode chunks
            chunk_embeddings = model.encode(
                chunk_texts,
                batch_size=min(batch_size, 64),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Average chunk embeddings
            mean_embedding = np.mean(chunk_embeddings, axis=0)
            
            embeddings[text_idx] = mean_embedding
        
        logger.info(f"  Completed all {len(long_indices):,} long texts")
    
    # Stack into array
    embeddings_array = np.stack(embeddings)
    
    logger.info(f"Generated embeddings shape: {embeddings_array.shape}")
    
    return embeddings_array


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def format_bytes(bytes_value: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def get_gpu_info() -> dict:
    """Get information about available GPUs."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'devices': []
    }
    
    if torch.cuda.is_available():
        info['device_count'] = torch.cuda.device_count()
        info['cuda_version'] = torch.version.cuda
        
        for i in range(info['device_count']):
            device_info = {
                'id': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3)
            }
            info['devices'].append(device_info)
    
    return info


# ============================================================
# OUTPUT FUNCTIONS
# ============================================================

def save_outputs(
    embeddings: np.ndarray,
    output_dir: Path,
    args,
    gpu_info: dict,
    logger: logging.Logger
) -> None:
    """
    Save embeddings and metadata.
    
    Args:
        embeddings: Embedding array
        output_dir: Output directory
        args: Command-line arguments
        gpu_info: GPU information dictionary
        logger: Logger instance
    """
    # Save embeddings with _rerun suffix
    embeddings_path = output_dir / "embeddings_rerun.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved embeddings: {embeddings_path}")
    logger.info(f"  Shape: {embeddings.shape}")
    logger.info(f"  Dtype: {embeddings.dtype}")
    logger.info(f"  Size: {format_bytes(embeddings.nbytes)}")
    
    # Save metadata
    metadata = {
        'model_name': args.model_name,
        'n_texts': int(embeddings.shape[0]),
        'embedding_dim': int(embeddings.shape[1]),
        'dtype': str(embeddings.dtype),
        'size_bytes': int(embeddings.nbytes),
        'chunk_length': args.chunk_length,
        'stride': args.stride,
        'batch_size': args.batch_size,
        'fp16': args.fp16,
        'seed': args.seed,
        'text_column': args.text_column,
        'source_step': 'step_08_re_embed',
        'note': 'Re-embedded after topic filtering (Step 06)',
        'gpu_used': gpu_info['cuda_available'],
        'device_count': gpu_info['device_count']
    }
    
    if gpu_info['cuda_available'] and gpu_info['devices']:
        metadata['gpu_device'] = gpu_info['devices'][0]['name']
        metadata['gpu_memory_gb'] = gpu_info['devices'][0]['memory_total_gb']
    
    metadata_path = output_dir / "embedding_metadata_rerun.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {metadata_path}")
    
    # Save summary
    summary_path = output_dir / "embedding_summary_rerun.txt"
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("STEP 08: RE-EMBEDDING SUMMARY (After Topic Filtering)\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Text Column: {args.text_column}\n")
        f.write(f"Random Seed: {args.seed}\n\n")
        
        f.write("Embedding Configuration:\n")
        f.write(f"  Chunk length: {args.chunk_length} tokens\n")
        f.write(f"  Stride:       {args.stride} tokens (~{(1 - args.stride/args.chunk_length)*100:.0f}% overlap)\n")
        f.write(f"  Batch size:   {args.batch_size}\n")
        f.write(f"  Precision:    {'float16' if args.fp16 else 'float32'}\n\n")
        
        f.write("Output:\n")
        f.write(f"  Documents:       {embeddings.shape[0]:,}\n")
        f.write(f"  Embedding dim:   {embeddings.shape[1]}\n")
        f.write(f"  Total size:      {format_bytes(embeddings.nbytes)}\n\n")
        
        f.write("Hardware:\n")
        f.write(f"  GPU available:   {gpu_info['cuda_available']}\n")
        if gpu_info['cuda_available'] and gpu_info['devices']:
            f.write(f"  GPU device:      {gpu_info['devices'][0]['name']}\n")
            f.write(f"  GPU memory:      {gpu_info['devices'][0]['memory_total_gb']:.1f} GB\n")
        f.write("\n")
        
        f.write("Note:\n")
        f.write("  These embeddings are for FILTERED abstracts only.\n")
        f.write("  Only abstracts assigned to 'kept' topics (from Step 06) are included.\n")
        f.write("  This enables re-clustering with better topic quality.\n\n")
        
        f.write("=" * 70 + "\n")
    
    logger.info(f"Saved summary: {summary_path}")


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Step 08: Re-embed filtered abstracts with chunk-mean pooling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python 08_re_embed.py \\
        --input-file /path/to/06_filtered/filtered_abstracts.csv \\
        --output-dir /path/to/output/08_re_embed \\
        --text-column Abstract \\
        --seed 42 \\
        --fp16
        """
    )
    
    parser.add_argument(
        '--input-file',
        type=Path,
        required=True,
        help='Input CSV from Step 06 (filtered_abstracts.csv)'
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
        help='Sentence transformer model name'
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='Abstract',
        help='Column to embed (Abstract or Abstract_trunc512)'
    )
    parser.add_argument(
        '--chunk-length',
        type=int,
        default=512,
        help='Maximum tokens per chunk (default: 512)'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=409,
        help='Stride between chunks (default: 409, ~80%% overlap)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help='Batch size for encoding (default: 128)'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use float16 precision (saves memory)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logger = configure_logging(args.output_dir / "step_08_re_embed.log")
    
    # Log header
    logger.info("=" * 70)
    logger.info("STEP 08: RE-EMBED FILTERED ABSTRACTS")
    logger.info("=" * 70)
    logger.info(f"Input file:    {args.input_file}")
    logger.info(f"Output dir:    {args.output_dir}")
    logger.info(f"Model:         {args.model_name}")
    logger.info(f"Text column:   {args.text_column}")
    logger.info(f"Chunk length:  {args.chunk_length}")
    logger.info(f"Stride:        {args.stride}")
    logger.info(f"Batch size:    {args.batch_size}")
    logger.info(f"FP16:          {args.fp16}")
    logger.info(f"Seed:          {args.seed}")
    logger.info("=" * 70)
    logger.info("")
    
    try:
        # Set seed
        set_seed(args.seed)
        logger.info(f"Random seed set to: {args.seed}")
        
        # Get GPU info
        gpu_info = get_gpu_info()
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {gpu_info['cuda_available']}")
        if gpu_info['cuda_available']:
            logger.info(f"CUDA version: {gpu_info['cuda_version']}")
            logger.info(f"GPU devices: {gpu_info['device_count']}")
            for dev in gpu_info['devices']:
                logger.info(f"  - {dev['name']} ({dev['memory_total_gb']:.1f} GB)")
        logger.info("")
        
        # Validate input file
        if not args.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        
        # Load data
        logger.info("Loading filtered data...")
        df = pd.read_csv(args.input_file)
        logger.info(f"Loaded {len(df):,} filtered abstracts")
        
        # Check column exists
        if args.text_column not in df.columns:
            raise ValueError(
                f"Column '{args.text_column}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Remove missing values
        df = df.dropna(subset=[args.text_column]).copy()
        texts = df[args.text_column].astype(str).tolist()
        logger.info(f"Processing {len(texts):,} abstracts")
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        logger.info("")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Handle model_max_length
        if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length < 1_000_000:
            effective_chunk_length = min(args.chunk_length, tokenizer.model_max_length)
        else:
            effective_chunk_length = args.chunk_length
        
        logger.info(f"Effective chunk length: {effective_chunk_length} tokens")
        
        # Load embedding model
        logger.info("Loading embedding model...")
        word_embedding = models.Transformer(
            args.model_name,
            max_seq_length=effective_chunk_length
        )
        pooling = models.Pooling(
            word_embedding.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        model = SentenceTransformer(
            modules=[word_embedding, pooling],
            device=device
        )
        logger.info("Model loaded successfully")
        logger.info("")
        
        # Generate embeddings
        embeddings = embed_with_chunking(
            texts=texts,
            model=model,
            tokenizer=tokenizer,
            chunk_length=effective_chunk_length,
            stride=args.stride,
            batch_size=args.batch_size,
            logger=logger
        )
        
        # Convert to desired dtype
        if args.fp16:
            embeddings = embeddings.astype(np.float16)
            logger.info("Converted embeddings to float16")
        
        # Save outputs
        save_outputs(embeddings, args.output_dir, args, gpu_info, logger)
        
        # Log completion
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 08 COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info("Output files:")
        logger.info(f"  - {args.output_dir}/embeddings_rerun.npy")
        logger.info(f"  - {args.output_dir}/embedding_metadata_rerun.json")
        logger.info(f"  - {args.output_dir}/embedding_summary_rerun.txt")
        logger.info(f"  - {args.output_dir}/step_08_re_embed.log")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        logger.error("Step 08 FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())