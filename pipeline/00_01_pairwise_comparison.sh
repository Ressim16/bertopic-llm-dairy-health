#!/bin/bash

#SBATCH --account=gratis
#SBATCH --job-name=00_01_pairwise
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mail-user=your.email@institution.ch
#SBATCH --mail-type=fail,end
#SBATCH --output=logs/00_01_pairwise_%j.out
#SBATCH --error=logs/00_01_pairwise_%j.err

# ============================================================
# Step 001: Pairwise Comparison for Duplicate Detection
# ============================================================

# ============================================================
# PATHS - EDIT THESE FOR YOUR ENVIRONMENT
# ============================================================

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Input directory containing raw CSV files
INPUT_DIR="$(dirname "${SCRIPT_DIR}")/input"

# Base output directory
OUTPUT_BASE="$(dirname "${SCRIPT_DIR}")/output"

# Output directory for this step
OUTPUT_DIR="${OUTPUT_BASE}/00_01_pairwise"

# Input files
PUBMED_FILE="${INPUT_DIR}/pubmed_abstracts.csv"
SCOPUS_FILE="${INPUT_DIR}/scopus_abstracts.csv"

# Conda environment
CONDA_ENV="bertopic_gpu"

# ============================================================
# PARAMETERS
# ============================================================

# Similarity threshold for candidate pairs (lower = more pairs for labeling)
THRESHOLD=0.70

# Number of nearest neighbors to search
TOP_K=10

# TF-IDF maximum document frequency
TFIDF_MAX_DF=0.95

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
echo "STEP 001: PAIRWISE COMPARISON"
echo "=========================================="
echo "Date:       $(date)"
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID:-N/A}"
echo "PubMed:     ${PUBMED_FILE}"
echo "Scopus:     ${SCOPUS_FILE}"
echo "Output:     ${OUTPUT_DIR}"
echo "Threshold:  ${THRESHOLD}"
echo "Top-k:      ${TOP_K}"
echo "=========================================="
echo ""

# ------------------------------------------------------------
# Lightweight runtime estimate + wall-clock timer
# ------------------------------------------------------------
PUBMED_N=$(( $(wc -l < "${PUBMED_FILE}") - 1 ))
SCOPUS_N=$(( $(wc -l < "${SCOPUS_FILE}") - 1 ))
TOTAL_N=$(( PUBMED_N + SCOPUS_N ))

# Empirical baseline (adjust if you have better numbers)
BASE_DOCS=74000
BASE_MIN=40

# Linear scaling estimate (good enough for logging)
EST_MIN=$(( (TOTAL_N * BASE_MIN) / BASE_DOCS ))
EST_MAX=$(( (TOTAL_N * (BASE_MIN + 30)) / BASE_DOCS ))

echo "Doc counts (raw): PubMed=${PUBMED_N}, Scopus=${SCOPUS_N}, Total=${TOTAL_N}"
echo "Estimated wall time: ~${EST_MIN}-${EST_MAX} minutes (baseline: ${BASE_MIN} min @ ${BASE_DOCS} docs)"
echo ""

START_TS=$(date +%s)

# Check if input files exist
if [ ! -f "${PUBMED_FILE}" ]; then
    echo "ERROR: PubMed file not found: ${PUBMED_FILE}"
    exit 1
fi

if [ ! -f "${SCOPUS_FILE}" ]; then
    echo "ERROR: Scopus file not found: ${SCOPUS_FILE}"
    exit 1
fi

# Start timer
START_TS=$(date +%s)

# Execute Python script
python "${SCRIPT_DIR}/00_01_pairwise_comparison.py" \
    --pubmed-file "${PUBMED_FILE}" \
    --scopus-file "${SCOPUS_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --threshold "${THRESHOLD}" \
    --top-k "${TOP_K}" \
    --tfidf-max-df "${TFIDF_MAX_DF}"

EXIT_CODE=$?

# Stop timer
END_TS=$(date +%s)
ELAPSED_SEC=$((END_TS - START_TS))
ELAPSED_MIN=$((ELAPSED_SEC / 60))
ELAPSED_REM=$((ELAPSED_SEC % 60))

echo ""
echo "Elapsed time: ${ELAPSED_MIN} min ${ELAPSED_REM} sec (${ELAPSED_SEC} seconds total)"

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS: Step 00_01 completed"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  - ${OUTPUT_DIR}/candidate_pairs_for_labeling.csv"
    echo "  - ${OUTPUT_DIR}/pairwise_statistics.json"
    echo "  - ${OUTPUT_DIR}/pairwise_summary.txt"
    echo "  - ${OUTPUT_DIR}/step_001_pairwise.log"
    echo ""
    echo "NEXT STEP:"
    echo "  1. Manually label 'is_duplicate' column in candidate_pairs_for_labeling.csv"
    echo "     (1 = duplicate, 0 = not duplicate)"
    echo "  2. Run 00_02_threshold_selection.sh"
else
    echo "FAILED: Step 00_01 failed with exit code ${EXIT_CODE}"
    echo "=========================================="
    echo ""
    echo "Check logs:"
    echo "  - ${OUTPUT_DIR}/step_00_01_pairwise.log"
    echo "  - ${SCRIPT_DIR}/logs/00_01_pairwise_${SLURM_JOB_ID}.err"
fi
echo ""

END_TS=$(date +%s)
ELAPSED_SEC=$((END_TS - START_TS))
ELAPSED_MIN=$((ELAPSED_SEC / 60))
ELAPSED_REM=$((ELAPSED_SEC % 60))
echo "Elapsed time: ${ELAPSED_MIN} min ${ELAPSED_REM} sec (${ELAPSED_SEC} seconds total)"

conda deactivate
exit ${EXIT_CODE}