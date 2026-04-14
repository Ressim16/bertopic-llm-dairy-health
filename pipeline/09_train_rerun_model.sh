#!/bin/bash

#SBATCH --account=gratis
#SBATCH --job-name=09_train_rerun
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --mail-user=your.email@institution.ch
#SBATCH --mail-type=fail,end
#SBATCH --output=logs/09_train_rerun_%j.out
#SBATCH --error=logs/09_train_rerun_%j.err

# ============================================================
# Step 09: Train New BERTopic Model on Filtered Data (Re-run)
# ============================================================
# This script TRAINS a new BERTopic model on filtered abstracts
# using the SAME hyperparameters from Step 05.
# ============================================================

# ============================================================
# PATHS - EDIT THESE FOR YOUR ENVIRONMENT
# ============================================================

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base output directory
OUTPUT_BASE="$(dirname "${SCRIPT_DIR}")/output"

# Input from Step 07 (filtered abstracts)
INPUT_FILE="${OUTPUT_BASE}/07_cluster_filtering/abstracts_topic_filtered.csv"

# Input from Step 08 (re-generated embeddings)
EMBEDDINGS="${OUTPUT_BASE}/08_re_embed/embeddings_rerun.npy"

# Input from Step 04 (best hyperparameters - same as original)
GRID_SEARCH_SUMMARY="${OUTPUT_BASE}/04_grid_search/grid_search_summary.json"

# Output directory for this step
OUTPUT_DIR="${OUTPUT_BASE}/09_train_rerun"

# Parameters (should match Step 05)
TEXT_COLUMN="Abstract"  # Or "Abstract_trunc512" if you prefer truncated
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
echo "STEP 09: TRAIN BERTOPIC MODEL (RE-RUN)"
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
echo "IMPORTANT: This will TRAIN a new model on filtered data,"
echo "           NOT load an existing model."
echo "           Topic IDs will be newly assigned."
echo ""
echo "This creates a fresh clustering on the filtered dataset."
echo ""

# Check if input files exist
if [ ! -f "${INPUT_FILE}" ]; then
    echo "ERROR: Input file not found: ${INPUT_FILE}"
    echo "Make sure Step 07 completed successfully."
    exit 1
fi

if [ ! -f "${EMBEDDINGS}" ]; then
    echo "ERROR: Embeddings not found: ${EMBEDDINGS}"
    echo "Make sure Step 08 completed successfully."
    exit 1
fi

if [ ! -f "${GRID_SEARCH_SUMMARY}" ]; then
    echo "ERROR: Grid search summary not found: ${GRID_SEARCH_SUMMARY}"
    echo "Make sure Step 04 completed successfully."
    echo ""
    echo "Alternatively, you can provide manual hyperparameters:"
    echo "  --min-df 0.003 --max-df 0.8 --n-neighbors 5 --n-components 5 \\"
    echo "  --min-cluster-size 15 --min-samples 10"
    exit 1
fi

# Display hyperparameters from grid search
echo "Hyperparameters from Step 04 grid search (will be reused):"
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
PYPARAMS
echo ""

# Show input file stats
echo "Input data statistics:"
python - <<PYSTATS
import pandas as pd
df = pd.read_csv("${INPUT_FILE}")
print(f"  Filtered abstracts: {len(df):,}")
print(f"  Columns available: {', '.join(df.columns[:5])}...")
PYSTATS
echo ""

# Execute Python script
python "${SCRIPT_DIR}/09_train_rerun_model.py" \
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
    echo "SUCCESS: Step 09 completed"
    echo "=========================================="
    echo ""
    echo "✅ New BERTopic model trained successfully!"
    echo "✅ Fresh clustering on filtered dataset"
    echo "✅ Topic IDs are newly assigned (different from original)"
    echo ""
    echo "Output files:"
    echo "  - ${OUTPUT_DIR}/bertopic_model_rerun/"
    echo "  - ${OUTPUT_DIR}/topic_info_rerun.csv"
    echo "  - ${OUTPUT_DIR}/document_info_rerun.csv"
    echo "  - ${OUTPUT_DIR}/topic_assignments_frozen_rerun.csv"
    echo "  - ${OUTPUT_DIR}/clustering_summary_rerun.json"
    echo "  - ${OUTPUT_DIR}/step_09_train_rerun_model.log"
    echo ""
    echo "Summary:"
    python - <<PYSUM
import json
with open("${OUTPUT_DIR}/clustering_summary_rerun.json") as f:
    s = json.load(f)
print(f"  Mode:       {s['mode']}")
print(f"  Documents:  {s['n_documents']:,}")
print(f"  Topics:     {s['n_topics_excluding_outlier']}")
print(f"  Outliers:   {s['outliers_count']:,} ({s['outliers_pct']*100:.2f}%)")
print(f"  Coverage:   {s['coverage']:.4f}")
PYSUM
    echo ""
    echo "Next step: Run 10_label_topics.sh (LLM labeling of new topics)"
else
    echo "FAILED: Step 09 failed with exit code ${EXIT_CODE}"
    echo "=========================================="
    echo ""
    echo "Check logs:"
    echo "  - ${OUTPUT_DIR}/step_09_train_rerun_model.log"
    echo "  - ${SCRIPT_DIR}/logs/09_train_rerun_${SLURM_JOB_ID}.err"
fi
echo ""

conda deactivate
exit ${EXIT_CODE}
