#!/bin/bash

#SBATCH --account=gratis
#SBATCH --job-name=07_filtering
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --mail-user=your.email@institution.ch
#SBATCH --mail-type=fail,end
#SBATCH --output=logs/07_cluster_filtering_%j.out
#SBATCH --error=logs/07_cluster_filtering_%j.err

# ============================================================
# Step 07: Filter Abstracts Based on Manual Topic Review
# ============================================================
# This script filters the dataset to keep only abstracts from
# topics classified as "Keep" during manual review.
# ============================================================

# ============================================================
# PATHS - EDIT THESE FOR YOUR ENVIRONMENT
# ============================================================

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base output directory
OUTPUT_BASE="$(dirname "${SCRIPT_DIR}")/output"

# Input from Step 02 (original abstracts)
INPUT_FILE="${OUTPUT_BASE}/02_truncate/abstracts_with_truncation.csv"

# Input from Step 05 (document-topic assignments)
DOCUMENT_INFO="${OUTPUT_BASE}/05_bertopic_full/document_info_full.csv"

# Output directory for this step
OUTPUT_DIR="${OUTPUT_BASE}/07_cluster_filtering"

# Parameters
TEXT_COLUMN="Abstract"

# Conda environment
CONDA_ENV="bertopic_gpu"

# ============================================================
# EXECUTION
# ============================================================

# Load environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# Set thread limits
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Log environment
echo "=========================================="
echo "STEP 07: TOPIC-BASED FILTERING"
echo "=========================================="
echo "Date:         $(date)"
echo "Host:         $(hostname)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "CPUs:         ${SLURM_CPUS_PER_TASK:-4}"
echo "Input file:   ${INPUT_FILE}"
echo "Doc info:     ${DOCUMENT_INFO}"
echo "Output:       ${OUTPUT_DIR}"
echo "Text column:  ${TEXT_COLUMN}"
echo "=========================================="
echo ""

# Check if input files exist
if [ ! -f "${INPUT_FILE}" ]; then
    echo "ERROR: Input file not found: ${INPUT_FILE}"
    echo "Make sure Step 02 completed successfully."
    exit 1
fi

if [ ! -f "${DOCUMENT_INFO}" ]; then
    echo "ERROR: Document info not found: ${DOCUMENT_INFO}"
    echo "Make sure Step 05 completed successfully."
    exit 1
fi

echo "Topic filtering configuration:"
echo "  - Filters abstracts based on manual topic review"
echo "  - Keeps only topics classified as 'Keep'"
echo "  - Preserves all original columns"
echo "  - Creates filtered dataset for downstream analysis"
echo ""

# Execute Python script
python "${SCRIPT_DIR}/07_cluster_filtering.py" \
    --input-file "${INPUT_FILE}" \
    --document-info "${DOCUMENT_INFO}" \
    --output-dir "${OUTPUT_DIR}" \
    --text-column "${TEXT_COLUMN}"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS: Step 07 completed"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  - ${OUTPUT_DIR}/abstracts_topic_filtered.csv"
    echo "  - ${OUTPUT_DIR}/step_07_summary.json"
    echo "  - ${OUTPUT_DIR}/step_07_filtering_details.txt"
    echo "  - ${OUTPUT_DIR}/step_07_cluster_filtering.log"
    echo ""
    echo "Summary:"
    python - <<PYSUMMARY
import json
summary_file = "${OUTPUT_DIR}/step_07_summary.json"
try:
    with open(summary_file) as f:
        summary = json.load(f)
    
    fr = summary['filtering_results']
    tc = summary['topic_classification']
    
    print(f"  Original documents:  {fr['original_documents']:,}")
    print(f"  Filtered documents:  {fr['filtered_documents']:,}")
    print(f"  Documents removed:   {fr['documents_removed']:,}")
    print(f"  Retention rate:      {fr['retention_rate']*100:.2f}%")
    print(f"")
    print(f"  Topics kept:         {tc['keep_topics']} / {tc['total_topics']}")
    print(f"  Topics removed:      {tc['remove_topics']}")
except Exception as e:
    print(f"  Could not read summary: {e}")
PYSUMMARY
    echo ""
    echo "Next steps:"
    echo "  Option A: Use filtered dataset for downstream analyses"
    echo "  Option B: Re-run pipeline (Steps 03-05) with filtered data"
    echo ""
else
    echo "FAILED: Step 07 failed with exit code ${EXIT_CODE}"
    echo "=========================================="
    echo ""
    echo "Check logs:"
    echo "  - ${OUTPUT_DIR}/step_07_cluster_filtering.log"
    echo "  - ${SCRIPT_DIR}/logs/07_cluster_filtering_${SLURM_JOB_ID}.err"
fi
echo ""

conda deactivate
exit ${EXIT_CODE}