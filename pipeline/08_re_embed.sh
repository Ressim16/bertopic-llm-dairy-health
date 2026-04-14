#!/bin/bash

#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=08_re_embed
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-user=your.email@institution.ch
#SBATCH --mail-type=fail,end
#SBATCH --output=logs/08_re_embed_%j.out
#SBATCH --error=logs/08_re_embed_%j.err

# ============================================================
# Step 08: Re-embed Filtered Abstracts
# ============================================================

# ============================================================
# PATHS - EDIT THESE FOR YOUR ENVIRONMENT
# ============================================================

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base output directory
OUTPUT_BASE="$(dirname "${SCRIPT_DIR}")/output"

# Input from Step 07 (filtered abstracts after topic filtering)
INPUT_FILE="${OUTPUT_BASE}/07_cluster_filtering/abstracts_topic_filtered.csv"

# Output directory for this step
OUTPUT_DIR="${OUTPUT_BASE}/08_re_embed"

# Model and parameters (same as Step 03)
EMBEDDING_MODEL="pritamdeka/S-BioBert-snli-multinli-stsb"
TEXT_COLUMN="Abstract"
CHUNK_LENGTH=512
STRIDE=409
BATCH_SIZE=128
SEED=42

# Conda environment
CONDA_ENV="bertopic_gpu"

# ============================================================
# EXECUTION
# ============================================================

# Load environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
module load GCC/11.2.0

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Log environment
echo "=========================================="
echo "STEP 08: RE-EMBED FILTERED ABSTRACTS"
echo "=========================================="
echo "Date:         $(date)"
echo "Host:         $(hostname)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Input:        ${INPUT_FILE}"
echo "Output:       ${OUTPUT_DIR}"
echo "Model:        ${EMBEDDING_MODEL}"
echo "Text column:  ${TEXT_COLUMN}"
echo "Chunk length: ${CHUNK_LENGTH}"
echo "Stride:       ${STRIDE}"
echo "Batch size:   ${BATCH_SIZE}"
echo "Seed:         ${SEED}"
echo "=========================================="
echo ""
echo "NOTE: This re-embeds only the FILTERED abstracts"
echo "      from Step 06 (kept topics only)."
echo ""

# Check GPU
echo "GPU Information:"
nvidia-smi || echo "WARNING: nvidia-smi not available"
echo ""

# Verify PyTorch GPU access
echo "Verifying PyTorch GPU access:"
python - <<'PYCHECK'
import torch
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version:    {torch.version.cuda}")
    print(f"  GPU device:      {torch.cuda.get_device_name(0)}")
    print(f"  GPU count:       {torch.cuda.device_count()}")
PYCHECK
echo ""

# Check if input file exists
if [ ! -f "${INPUT_FILE}" ]; then
    echo "ERROR: Input file not found: ${INPUT_FILE}"
    echo "Make sure Step 07 completed successfully."
    exit 1
fi

# Show input file stats
echo "Input file statistics:"
python - <<PYSTATS
import pandas as pd
df = pd.read_csv("${INPUT_FILE}")
print(f"  Total abstracts: {len(df):,}")
if 'Topic' in df.columns:
    kept = (df['Topic'] != -1).sum()
    print(f"  Kept topics:     {kept:,}")
    print(f"  Outliers:        {(df['Topic'] == -1).sum():,}")
PYSTATS
echo ""

# Execute Python script
python "${SCRIPT_DIR}/08_re_embed.py" \
    --input-file "${INPUT_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-name "${EMBEDDING_MODEL}" \
    --text-column "${TEXT_COLUMN}" \
    --chunk-length ${CHUNK_LENGTH} \
    --stride ${STRIDE} \
    --batch-size ${BATCH_SIZE} \
    --seed ${SEED} \
    --fp16

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS: Step 08 completed"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  - ${OUTPUT_DIR}/embeddings_rerun.npy"
    echo "  - ${OUTPUT_DIR}/embedding_metadata_rerun.json"
    echo "  - ${OUTPUT_DIR}/embedding_summary_rerun.txt"
    echo "  - ${OUTPUT_DIR}/step_08_re_embed.log"
    echo ""
    echo "Embeddings info:"
    python - <<PYINFO
import numpy as np
emb = np.load("${OUTPUT_DIR}/embeddings_rerun.npy")
print(f"  Shape: {emb.shape}")
print(f"  Dtype: {emb.dtype}")
print(f"  Size:  {emb.nbytes / (1024**2):.2f} MB")
PYINFO
    echo ""
    echo "Next step: Run 09_re_cluster.sh"
else
    echo "FAILED: Step 08 failed with exit code ${EXIT_CODE}"
    echo "=========================================="
    echo ""
    echo "Check logs:"
    echo "  - ${OUTPUT_DIR}/step_08_re_embed.log"
    echo "  - ${SCRIPT_DIR}/logs/08_re_embed_${SLURM_JOB_ID}.err"
fi
echo ""

conda deactivate
exit ${EXIT_CODE}