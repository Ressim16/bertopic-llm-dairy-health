#!/bin/bash

#SBATCH --account=gratis
#SBATCH --job-name=01_filter
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mail-user=your.email@institution.ch
#SBATCH --mail-type=fail,end
#SBATCH --output=logs/01_filter_%j.out
#SBATCH --error=logs/01_filter_%j.err

# ============================================================
# Step 01: Filter Abstracts
# ============================================================

# ============================================================
# PATHS - EDIT THESE FOR YOUR ENVIRONMENT
# ============================================================

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Input directory containing merged_cleaned_auto_dedup.csv
INPUT_DIR="$(dirname "${SCRIPT_DIR}")/input"

# Base output directory
OUTPUT_BASE="$(dirname "${SCRIPT_DIR}")/output"

# Output directory for this step
OUTPUT_DIR="${OUTPUT_BASE}/01_filter"

# Input file name
INPUT_FILE="${INPUT_DIR}/merged_cleaned_auto_dedup.csv"

# Conda environment
CONDA_ENV="bertopic_gpu"

# ============================================================
# EXECUTION
# ============================================================

# Load environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# Create log directory
mkdir -p "${SCRIPT_DIR}/logs"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Log environment
echo "=========================================="
echo "STEP 01: FILTER ABSTRACTS"
echo "=========================================="
echo "Date:       $(date)"
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-N/A}"
echo "Input:      ${INPUT_FILE}"
echo "Output:     ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Check if input file exists
if [ ! -f "${INPUT_FILE}" ]; then
    echo "ERROR: Input file not found: ${INPUT_FILE}"
    exit 1
fi

# Execute Python script
python "${SCRIPT_DIR}/01_filter.py" \
    --input-file "${INPUT_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --text-column "Abstract"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS: Step 01 completed"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  - ${OUTPUT_DIR}/filtered_abstracts.csv"
    echo "  - ${OUTPUT_DIR}/filter_statistics.json"
    echo "  - ${OUTPUT_DIR}/filter_summary.txt"
    echo "  - ${OUTPUT_DIR}/step_01_filter.log"
    echo ""
    echo "Next step: Run 02_truncate.sh"
else
    echo "FAILED: Step 01 failed with exit code ${EXIT_CODE}"
    echo "=========================================="
    echo ""
    echo "Check logs:"
    echo "  - ${OUTPUT_DIR}/step_01_filter.log"
    echo "  - ${SCRIPT_DIR}/logs/01_filter_${SLURM_JOB_ID}.err"
fi
echo ""

conda deactivate
exit ${EXIT_CODE}