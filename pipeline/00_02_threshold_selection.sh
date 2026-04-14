#!/bin/bash

#SBATCH --account=gratis
#SBATCH --job-name=00_02_threshold
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mail-user=your.email@institution.ch
#SBATCH --mail-type=fail,end
#SBATCH --output=logs/00_02_threshold_%j.out
#SBATCH --error=logs/00_02_threshold_%j.err
# ============================================================
# Step 002: Threshold Selection and Auto-Deduplication
# ============================================================

# ============================================================
# PATHS - EDIT THESE FOR YOUR ENVIRONMENT
# ============================================================

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Input directory
INPUT_DIR="$(dirname "${SCRIPT_DIR}")/input"

# Base output directory
OUTPUT_BASE="$(dirname "${SCRIPT_DIR}")/output"

# Output directory for this step
OUTPUT_DIR="${OUTPUT_BASE}/00_02_threshold"

# Input files
# Labeled pairs (manually labeled from Step 00_01 output)
LABELED_FILE="${INPUT_DIR}/duplicates_review_70_for_labeling.csv"

# Raw inputs (Step 00_02 rebuilds the merged+exact-dedup dataset in-memory)
PUBMED_FILE="${INPUT_DIR}/pubmed_abstracts.csv"
SCOPUS_FILE="${INPUT_DIR}/scopus_abstracts.csv"

# Conda environment
CONDA_ENV="bertopic_gpu"

# ============================================================
# PARAMETERS
# ============================================================

# Threshold evaluation range
T_MIN=0.70
T_MAX=0.995
T_STEP=0.005

# Target precision for auto-drop threshold selection
TARGET_PRECISION=0.99

# Beta for F-beta score (< 1 favors precision)
BETA=0.5

# FAISS search parameters
TOP_K=10

# TF-IDF parameters
TFIDF_MAX_DF=0.95
TFIDF_MIN_DF=1

# Year filtering parameters (discard documents outside this range)
MIN_YEAR=2000
MAX_YEAR=2025

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
echo "STEP 002: THRESHOLD SELECTION"
echo "=========================================="
echo "Date:             $(date)"
echo "Host:             $(hostname)"
echo "Job ID:           ${SLURM_JOB_ID:-N/A}"
echo "Labeled file:     ${LABELED_FILE}"
echo "PubMed input:     ${PUBMED_FILE}"
echo "Scopus input:     ${SCOPUS_FILE}"
echo "Output:           ${OUTPUT_DIR}"
echo "Threshold range:  ${T_MIN} - ${T_MAX} (step: ${T_STEP})"
echo "Target precision: ${TARGET_PRECISION}"
echo "Beta:             ${BETA}"
echo "Year range:       ${MIN_YEAR} - ${MAX_YEAR}"
echo "=========================================="
echo ""

# Check if input files exist
if [ ! -f "${LABELED_FILE}" ]; then
    echo "ERROR: Labeled file not found: ${LABELED_FILE}"
    echo ""
    echo "Please ensure you have:"
    echo "  1. Run 00_01_pairwise_comparison.sh"
    echo "  2. Manually labeled the 'is_duplicate' column in candidate_pairs_for_labeling.csv"
    echo "  3. Saved it as: ${LABELED_FILE}"
    exit 1
fi


if [ ! -f "${PUBMED_FILE}" ]; then
    echo "ERROR: PubMed file not found: ${PUBMED_FILE}"
    exit 1
fi

if [ ! -f "${SCOPUS_FILE}" ]; then
    echo "ERROR: Scopus file not found: ${SCOPUS_FILE}"
    exit 1
fi

# ------------------------------------------------------------
# Lightweight runtime estimate + wall-clock timer
# ------------------------------------------------------------
PUBMED_N=$(( $(wc -l < "${PUBMED_FILE}") - 1 ))
SCOPUS_N=$(( $(wc -l < "${SCOPUS_FILE}") - 1 ))
TOTAL_N=$(( PUBMED_N + SCOPUS_N ))

# Empirical baseline (adjust if you have better numbers)
BASE_DOCS=74000
BASE_MIN=50  # Step 002 includes threshold sweep + one FAISS pass

# Linear scaling estimate (good enough for logging)
EST_MIN=$(( (TOTAL_N * BASE_MIN) / BASE_DOCS ))
EST_MAX=$(( (TOTAL_N * (BASE_MIN + 40)) / BASE_DOCS ))

echo "Doc counts (raw): PubMed=${PUBMED_N}, Scopus=${SCOPUS_N}, Total=${TOTAL_N}"
echo "Estimated wall time: ~${EST_MIN}-${EST_MAX} minutes (baseline: ${BASE_MIN} min @ ${BASE_DOCS} docs)"
echo ""

# Start timer
START_TS=$(date +%s)

# Execute Python script
python "${SCRIPT_DIR}/00_02_threshold_selection.py" \
    --labeled-file "${LABELED_FILE}" \
    --pubmed-file "${PUBMED_FILE}" \
    --scopus-file "${SCOPUS_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --t-min "${T_MIN}" \
    --t-max "${T_MAX}" \
    --t-step "${T_STEP}" \
    --target-precision "${TARGET_PRECISION}" \
    --beta "${BETA}" \
    --top-k "${TOP_K}" \
    --tfidf-max-df "${TFIDF_MAX_DF}" \
    --tfidf-min-df "${TFIDF_MIN_DF}" \
    --min-year "${MIN_YEAR}" \
    --max-year "${MAX_YEAR}"

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
    echo "SUCCESS: Step 00_02 completed"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  - ${OUTPUT_DIR}/threshold_metrics.csv"
    echo "  - ${OUTPUT_DIR}/merged_cleaned_auto_dedup.csv"
    echo "  - ${OUTPUT_DIR}/auto_drop_log.csv"
    echo "  - ${OUTPUT_DIR}/similarity_edges.csv"
    echo "  - ${OUTPUT_DIR}/threshold_statistics.json"
    echo "  - ${OUTPUT_DIR}/threshold_summary.txt"
    echo "  - ${OUTPUT_DIR}/step_00_02_threshold.log"
    echo ""
    echo "Next step:"
    echo "  1. Copy the deduplicated dataset to input directory:"
    echo "     cp ${OUTPUT_DIR}/merged_cleaned_auto_dedup.csv ${INPUT_DIR}/"
    echo "  2. Run 01_filter.sh"
else
    echo "FAILED: Step 00_02 failed with exit code ${EXIT_CODE}"
    echo "=========================================="
    echo ""
    echo "Check logs:"
    echo "  - ${OUTPUT_DIR}/step_00_02_threshold.log"
    echo "  - ${SCRIPT_DIR}/logs/00_02_threshold_${SLURM_JOB_ID}.err"
fi
echo ""

conda deactivate
exit ${EXIT_CODE}