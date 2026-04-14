#!/bin/bash

#SBATCH --account=gratis
#SBATCH --job-name=00_dedup_pipeline
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mail-user=your.email@institution.ch
#SBATCH --mail-type=fail,end
#SBATCH --output=logs/00_dedup_pipeline_%j.out
#SBATCH --error=logs/00_dedup_pipeline_%j.err

# ============================================================
# Deduplication Pipeline: Run 00_01 and 00_02 sequentially
# ============================================================
#
# This script runs both deduplication steps in a single SLURM job.
# Use this when you already have the labeled pairs file ready.
#
# ============================================================

# ============================================================
# PATHS
# ============================================================

INPUT_DIR="$(dirname "${SCRIPT_DIR}")/input"
OUTPUT_BASE="$(dirname "${SCRIPT_DIR}")/output/test_reproducibility"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Labeled file (must exist before running this pipeline)
LABELED_FILE="${INPUT_DIR}/duplicates_review_70_for_labeling.csv"

# ============================================================
# ENVIRONMENT
# ============================================================

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "bertopic_gpu"

mkdir -p "${SCRIPT_DIR}/logs"

echo "=========================================="
echo "DEDUPLICATION PIPELINE"
echo "=========================================="
echo "Date:       $(date)"
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-N/A}"
echo "=========================================="
echo ""

# ============================================================
# PRE-FLIGHT CHECK
# ============================================================

# Check if labeled file exists (required for Step 00_02)
if [ ! -f "${LABELED_FILE}" ]; then
    echo "ERROR: Labeled file not found: ${LABELED_FILE}"
    echo ""
    echo "This pipeline requires the labeled pairs file to exist."
    echo "If you haven't labeled yet, run 00_01 separately first."
    exit 1
fi

echo "Pre-flight check passed: labeled file exists"
echo ""

# ============================================================
# STEP 00_01: PAIRWISE COMPARISON
# ============================================================

echo "=========================================="
echo "STARTING STEP 00_01: PAIRWISE COMPARISON"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Source the script content directly (to stay in same job)
OUTPUT_DIR_01="${OUTPUT_BASE}/00_01_pairwise"
mkdir -p "${OUTPUT_DIR_01}"

PUBMED_FILE="${INPUT_DIR}/pubmed_abstracts.csv"
SCOPUS_FILE="${INPUT_DIR}/scopus_abstracts.csv"

python "${SCRIPT_DIR}/00_01_pairwise_comparison.py" \
    --pubmed-file "${PUBMED_FILE}" \
    --scopus-file "${SCOPUS_FILE}" \
    --output-dir "${OUTPUT_DIR_01}" \
    --threshold 0.70 \
    --top-k 10 \
    --tfidf-max-df 0.95

EXIT_CODE_01=$?

echo ""
echo "Step 00_01 finished at: $(date)"

if [ ${EXIT_CODE_01} -ne 0 ]; then
    echo "FAILED: Step 00_01 failed with exit code ${EXIT_CODE_01}"
    conda deactivate
    exit ${EXIT_CODE_01}
fi

echo "SUCCESS: Step 00_01 completed"
echo ""

# ============================================================
# STEP 00_02: THRESHOLD SELECTION
# ============================================================

echo "=========================================="
echo "STARTING STEP 00_02: THRESHOLD SELECTION"
echo "=========================================="
echo "Start time: $(date)"
echo ""

OUTPUT_DIR_02="${OUTPUT_BASE}/00_02_threshold"
mkdir -p "${OUTPUT_DIR_02}"

python "${SCRIPT_DIR}/00_02_threshold_selection.py" \
    --labeled-file "${LABELED_FILE}" \
    --input-file "${OUTPUT_DIR_01}/merged_exact_dedup.csv" \
    --output-dir "${OUTPUT_DIR_02}" \
    --t-min 0.70 \
    --t-max 0.995 \
    --t-step 0.005 \
    --target-precision 0.99 \
    --beta 0.5 \
    --top-k 10 \
    --tfidf-max-df 0.95 \
    --tfidf-min-df 1 \
    --min-year 2000 \
    --max-year 2025

EXIT_CODE_02=$?

echo ""
echo "Step 00_02 finished at: $(date)"

if [ ${EXIT_CODE_02} -ne 0 ]; then
    echo "FAILED: Step 00_02 failed with exit code ${EXIT_CODE_02}"
    conda deactivate
    exit ${EXIT_CODE_02}
fi

echo "SUCCESS: Step 00_02 completed"
echo ""

# ============================================================
# PIPELINE COMPLETE
# ============================================================

echo "=========================================="
echo "PIPELINE COMPLETED SUCCESSFULLY"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Output files:"
echo ""
echo "Step 00_01 (Pairwise Comparison):"
echo "  - ${OUTPUT_DIR_01}/merged_exact_dedup.csv"
echo "  - ${OUTPUT_DIR_01}/candidate_pairs_for_labeling.csv"
echo "  - ${OUTPUT_DIR_01}/pairwise_statistics.json"
echo ""
echo "Step 00_02 (Threshold Selection):"
echo "  - ${OUTPUT_DIR_02}/merged_cleaned_auto_dedup.csv"
echo "  - ${OUTPUT_DIR_02}/threshold_metrics.csv"
echo "  - ${OUTPUT_DIR_02}/auto_drop_log.csv"
echo "  - ${OUTPUT_DIR_02}/threshold_statistics.json"
echo ""
echo "Next step:"
echo "  cp ${OUTPUT_DIR_02}/merged_cleaned_auto_dedup.csv ${INPUT_DIR}/"
echo "  sbatch ${SCRIPT_DIR}/01_filter.sh"
echo "=========================================="

conda deactivate
exit 0