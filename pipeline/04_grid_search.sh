#!/bin/bash

#SBATCH --account=gratis
#SBATCH --job-name=04_grid_search
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mail-user=your.email@institution.ch
#SBATCH --mail-type=fail,end
#SBATCH --output=logs/04_grid_search_%j.out
#SBATCH --error=logs/04_grid_search_%j.err

# ============================================================
# Step 04: BERTopic Hyperparameter Grid Search
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

# Output directory for this step
OUTPUT_DIR="${OUTPUT_BASE}/04_grid_search"

# Parameters
TEXT_COLUMN="Abstract"
SUBSET=5000  # Use subset for faster grid search (0 = full dataset)
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
echo "STEP 04: BERTOPIC HYPERPARAMETER GRID SEARCH"
echo "=========================================="
echo "Date:         $(date)"
echo "Host:         $(hostname)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "CPUs:         ${SLURM_CPUS_PER_TASK:-8}"
echo "Input:        ${INPUT_FILE}"
echo "Embeddings:   ${EMBEDDINGS}"
echo "Output:       ${OUTPUT_DIR}"
echo "Text column:  ${TEXT_COLUMN}"
echo "Subset:       ${SUBSET} (0 = full dataset)"
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

echo "Grid search configuration:"
echo "  min_df:            [0.003, 0.005]"
echo "  max_df:            [0.5, 0.8]"
echo "  n_neighbors:       [5, 10, 15]"
echo "  n_components:      [5, 10, 15]"
echo "  min_cluster_size:  [5, 10, 15]"
echo "  min_samples:       [1, 5, 10]"
echo ""
echo "Total configurations: 324"
echo "Estimated time: 12-24 hours (depending on subset size)"
echo ""

# Execute Python script
python "${SCRIPT_DIR}/04_grid_search.py" \
    --input-file "${INPUT_FILE}" \
    --embeddings "${EMBEDDINGS}" \
    --output-dir "${OUTPUT_DIR}" \
    --text-column "${TEXT_COLUMN}" \
    --subset ${SUBSET} \
    --seed ${SEED} \
    --top-n-words ${TOP_N_WORDS} \
    --max-features ${MAX_FEATURES}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS: Step 04 completed"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  - ${OUTPUT_DIR}/grid_search_results.csv"
    echo "  - ${OUTPUT_DIR}/grid_search_results_ranked.csv"
    echo "  - ${OUTPUT_DIR}/grid_search_summary.json"
    echo "  - ${OUTPUT_DIR}/step_04_grid_search.log"
    echo ""
    echo "Best configuration:"
    python - <<PYBEST
import json
with open("${OUTPUT_DIR}/grid_search_summary.json") as f:
    best = json.load(f)
print(f"  min_df:            {best['min_df']}")
print(f"  max_df:            {best['max_df']}")
print(f"  n_neighbors:       {best['n_neighbors']}")
print(f"  n_components:      {best['n_components']}")
print(f"  min_cluster_size:  {best['min_cluster_size']}")
print(f"  min_samples:       {best['min_samples']}")
print(f"  Topics:            {best['n_topics']}")
print(f"  Coverage:          {best['coverage']:.4f}")
print(f"  Silhouette:        {best['silhouette']:.4f}")
PYBEST
    echo ""
    echo "Next step: Run 05_cluster.sh with these hyperparameters"
else
    echo "FAILED: Step 04 failed with exit code ${EXIT_CODE}"
    echo "=========================================="
    echo ""
    echo "Check logs:"
    echo "  - ${OUTPUT_DIR}/step_04_grid_search.log"
    echo "  - ${SCRIPT_DIR}/logs/04_grid_search_${SLURM_JOB_ID}.err"
fi
echo ""

conda deactivate
exit ${EXIT_CODE}