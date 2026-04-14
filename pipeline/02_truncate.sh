#!/bin/bash

#SBATCH --account=gratis
#SBATCH --job-name=02_truncate
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --mail-user=your.email@institution.ch
#SBATCH --mail-type=fail,end
#SBATCH --output=logs/02_truncate_%j.out
#SBATCH --error=logs/02_truncate_%j.err

# ============================================================
# Step 02: Truncate Abstracts
# ============================================================

# ============================================================
# PATHS - EDIT THESE FOR YOUR ENVIRONMENT
# ============================================================

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base output directory
OUTPUT_BASE="$(dirname "${SCRIPT_DIR}")/output"

# Input from Step 01
INPUT_FILE="${OUTPUT_BASE}/01_filter/filtered_abstracts.csv"

# Output directory for this step
OUTPUT_DIR="${OUTPUT_BASE}/02_truncate"

# Model and parameters
EMBEDDING_MODEL="pritamdeka/S-BioBert-snli-multinli-stsb"
MAX_TOKENS=512

# Conda environment
CONDA_ENV="bertopic_gpu"

# ============================================================
# EXECUTION
# ============================================================

# Load environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Log environment
echo "=========================================="
echo "STEP 02: TRUNCATE ABSTRACTS"
echo "=========================================="
echo "Date:         $(date)"
echo "Host:         $(hostname)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "Input:        ${INPUT_FILE}"
echo "Output:       ${OUTPUT_DIR}"
echo "Model:        ${EMBEDDING_MODEL}"
echo "Max tokens:   ${MAX_TOKENS}"
echo "=========================================="
echo ""

# Check if input file exists
if [ ! -f "${INPUT_FILE}" ]; then
    echo "ERROR: Input file not found: ${INPUT_FILE}"
    echo "Make sure Step 01 completed successfully."
    exit 1
fi

# Execute Python script
python "${SCRIPT_DIR}/02_truncate.py" \
    --input-file "${INPUT_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-name "${EMBEDDING_MODEL}" \
    --max-tokens ${MAX_TOKENS} \
    --text-column "Abstract"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS: Step 02 completed"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  - ${OUTPUT_DIR}/abstracts_with_truncation.csv"
    echo "  - ${OUTPUT_DIR}/truncation_statistics.json"
    echo "  - ${OUTPUT_DIR}/truncation_summary.txt"
    echo "  - ${OUTPUT_DIR}/step_02_truncate.log"
    echo ""
    echo "Next step: Run 03_embed.sh"
else
    echo "FAILED: Step 02 failed with exit code ${EXIT_CODE}"
    echo "=========================================="
    echo ""
    echo "Check logs:"
    echo "  - ${OUTPUT_DIR}/step_02_truncate.log"
    echo "  - ${SCRIPT_DIR}/logs/02_truncate_${SLURM_JOB_ID}.err"
fi
echo ""

conda deactivate
exit ${EXIT_CODE}