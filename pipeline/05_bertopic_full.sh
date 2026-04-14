#!/bin/bash

#SBATCH --account=gratis
#SBATCH --job-name=05_bertopic_full
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mail-user=your.email@institution.ch
#SBATCH --mail-type=fail,end
#SBATCH --output=logs/05_bertopic_full_%j.out
#SBATCH --error=logs/05_bertopic_full_%j.err

# ============================================================
# Step 05: Train BERTopic with Optimal Hyperparameters
# ============================================================
# This script trains a BERTopic model using the best hyperparameters
# identified in Step 04 grid search.
# ============================================================

# ============================================================
# PATHS - EDIT THESE FOR YOUR ENVIRONMENT
# ============================================================

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base output directory
OUTPUT_BASE="$(dirname "${SCRIPT_DIR}")/output"

# Input from Step 02
INPUT_FILE="${OUTPUT_BASE}/02_truncate/abstracts_with_truncation.csv"

# Input from Step 03
EMBEDDINGS="${OUTPUT_BASE}/03_embeddings/embeddings.npy"

# Input from Step 04 (best hyperparameters)
GRID_SEARCH_SUMMARY="${OUTPUT_BASE}/04_grid_search/grid_search_summary.json"

# Output directory for this step
OUTPUT_DIR="${OUTPUT_BASE}/05_bertopic_full"

# Parameters
TEXT_COLUMN="Abstract"
SEED=42
TOP_N_WORDS=10
MAX_FEATURES=30000

# Conda environment
CONDA_ENV="bertopic_gpu"

# ============================================================
# EXECUTION
# ============================================================

# Load environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# Set thread limits for CPU efficiency
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Log environment
echo "=========================================="
echo "STEP 05: TRAIN BERTOPIC MODEL"
echo "=========================================="
echo "Date:         $(date)"
echo "Host:         $(hostname)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "CPUs:         ${SLURM_CPUS_PER_TASK:-8}"
echo "Input:        ${INPUT_FILE}"
echo "Embeddings:   ${EMBEDDINGS}"
echo "Grid search:  ${GRID_SEARCH_SUMMARY}"
echo "Output:       ${OUTPUT_DIR}"
echo "Text column:  ${TEXT_COLUMN}"
echo "Seed:         ${SEED}"
echo "=========================================="
echo ""

# Check if input files exist
if [ ! -f "${INPUT_FILE}" ]; then
    echo "ERROR: Input file not found: ${INPUT_FILE}"
    echo "Make sure Step 02 completed successfully."
    exit 1
fi

if [ ! -f "${EMBEDDINGS}" ]; then
    echo "ERROR: Embeddings not found: ${EMBEDDINGS}"
    echo "Make sure Step 03 completed successfully."
    exit 1
fi

if [ ! -f "${GRID_SEARCH_SUMMARY}" ]; then
    echo "ERROR: Grid search summary not found: ${GRID_SEARCH_SUMMARY}"
    echo "Make sure Step 04 completed successfully."
    exit 1
fi

# Display best hyperparameters from grid search
echo "Best hyperparameters from Step 04 grid search:"
python - <<PYPARAMS
import json
with open("${GRID_SEARCH_SUMMARY}") as f:
    params = json.load(f)
print(f"  min_df:            {params.get('min_df', 'N/A')}")
print(f"  max_df:            {params.get('max_df', 'N/A')}")
print(f"  n_neighbors:       {params.get('n_neighbors', 'N/A')}")
print(f"  n_components:      {params.get('n_components', 'N/A')}")
print(f"  min_cluster_size:  {params.get('min_cluster_size', 'N/A')}")
print(f"  min_samples:       {params.get('min_samples', 'N/A')}")
print(f"")
print(f"Expected performance (from grid search on subset):")
print(f"  Topics:            {params.get('n_topics', 'N/A')}")
print(f"  Coverage:          {params.get('coverage', 'N/A')}")
print(f"  Silhouette:        {params.get('silhouette', 'N/A')}")
PYPARAMS
echo ""

# Execute Python script
python "${SCRIPT_DIR}/05_bertopic_full.py" \
    --input-file "${INPUT_FILE}" \
    --embeddings "${EMBEDDINGS}" \
    --grid-search-summary "${GRID_SEARCH_SUMMARY}" \
    --output-dir "${OUTPUT_DIR}" \
    --text-column "${TEXT_COLUMN}" \
    --seed ${SEED} \
    --top-n-words ${TOP_N_WORDS} \
    --max-features ${MAX_FEATURES}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS: Step 05 completed"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  - ${OUTPUT_DIR}/bertopic_model/"
    echo "  - ${OUTPUT_DIR}/topic_info_full.csv"
    echo "  - ${OUTPUT_DIR}/document_info_full.csv"
    echo "  - ${OUTPUT_DIR}/topic_assignments_frozen.csv"
    echo "  - ${OUTPUT_DIR}/step_05_summary.json"
    echo "  - ${OUTPUT_DIR}/step_05_bertopic_full.log"
    echo ""
    echo "Summary:"
    python - <<PYSUMMARY
import json
summary_file = "${OUTPUT_DIR}/step_05_summary.json"
try:
    with open(summary_file) as f:
        summary = json.load(f)
    print(f"  Documents:     {summary['n_documents']:,}")
    print(f"  Topics:        {summary['n_topics_excluding_outlier']} (excluding outlier topic)")
    print(f"  Outliers:      {summary['outliers_count']:,} ({summary['outliers_pct']:.2%})")
    print(f"  Coverage:      {summary['coverage']:.4f}")
except Exception as e:
    print(f"  Could not read summary: {e}")
PYSUMMARY
    echo ""
    echo "Next step: Run 06_llm_labeling.sh"
else
    echo "FAILED: Step 05 failed with exit code ${EXIT_CODE}"
    echo "=========================================="
    echo ""
    echo "Check logs:"
    echo "  - ${OUTPUT_DIR}/step_05_bertopic_full.log"
    echo "  - ${SCRIPT_DIR}/logs/05_bertopic_full_${SLURM_JOB_ID}.err"
fi
echo ""

conda deactivate
exit ${EXIT_CODE}