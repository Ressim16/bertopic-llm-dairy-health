#!/bin/bash

#SBATCH --account=gratis
#SBATCH --job-name=5b_evaluate_runs
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mail-user=your.email@institution.ch
#SBATCH --mail-type=fail,end
#SBATCH --output=logs/11_evaluate_runs_%j.out
#SBATCH --error=logs/11_evaluate_runs_%j.err

# ============================================================
# Step 11: Evaluate Trained BERTopic Models (Initial Run and Re-run)
# ============================================================
# Computes silhouette, coherence (c_npmi), diversity, and coverage
# for both trained models:
#   - Step 05: full corpus, before manual topic curation
#   - Step 09: filtered corpus, after manual topic curation (re-run)
#
# Topic assignments are read from frozen CSVs produced by Steps 05/09
# rather than recomputed, so manual curation decisions are preserved.
# ============================================================

# ============================================================
# PATHS - EDIT THESE FOR YOUR ENVIRONMENT
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="$(dirname "${SCRIPT_DIR}")/output"
OUTPUT_DIR="${OUTPUT_BASE}/11_evaluate_runs"

TEXT_COLUMN="Abstract"
TOP_N_WORDS=10

CONDA_ENV="bertopic_gpu"

# ============================================================
# EXECUTION
# ============================================================

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONNOUSERSITE=1
unset PYTHONPATH

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "STEP 5b: EVALUATE BERTOPIC RUNS"
echo "=========================================="
echo "Date:         $(date)"
echo "Host:         $(hostname)"
echo "Job ID:       ${SLURM_JOB_ID:-N/A}"
echo "CPUs:         ${SLURM_CPUS_PER_TASK:-8}"
echo "Output base:  ${OUTPUT_BASE}"
echo "Output dir:   ${OUTPUT_DIR}"
echo ""
echo "Paths resolved from output base:"
echo "  Step 05 model:       ${OUTPUT_BASE}/05_bertopic_full/bertopic_model/"
echo "  Step 05 corpus:      ${OUTPUT_BASE}/02_truncate/abstracts_with_truncation.csv"
echo "  Step 05 embeddings:  ${OUTPUT_BASE}/03_embeddings/embeddings.npy"
echo "  Step 05 assignments: ${OUTPUT_BASE}/05_bertopic_full/topic_assignments_frozen.csv"
echo "  Step 09 model:       ${OUTPUT_BASE}/09_train_rerun/bertopic_model_rerun/"
echo "  Step 09 corpus:      ${OUTPUT_BASE}/07_filtered/abstracts_topic_filtered.csv"
echo "  Step 09 embeddings:  ${OUTPUT_BASE}/08_re_embed/embeddings_rerun.npy"
echo "  Step 09 assignments: ${OUTPUT_BASE}/09_train_rerun/topic_assignments_frozen_rerun.csv"
echo "=========================================="
echo ""

python "${SCRIPT_DIR}/11_evaluate_runs.py" \
    --output-base "${OUTPUT_BASE}" \
    --output-dir  "${OUTPUT_DIR}" \
    --text-column "${TEXT_COLUMN}" \
    --top-n-words ${TOP_N_WORDS}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS: Step 11 completed"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  - ${OUTPUT_DIR}/evaluation_results.json"
    echo "  - ${OUTPUT_DIR}/evaluation_results.csv"
    echo "  - ${OUTPUT_DIR}/step_11_evaluate_runs.log"
    echo ""
    echo "Results summary:"
    python - <<PYSUM
import json
with open("${OUTPUT_DIR}/evaluation_results.json") as f:
    results = json.load(f)
print(f"  {'Metric':<25} {'Initial run (05)':>18} {'Re-run (09)':>18}")
print(f"  {'-'*63}")
metrics = [
    ("n_documents",      "Documents"),
    ("n_topics",         "Topics"),
    ("n_outliers",       "Outliers"),
    ("coverage",         "Coverage"),
    ("silhouette",       "Silhouette"),
    ("coherence_c_npmi", "Coherence (c_npmi)"),
    ("diversity",        "Diversity"),
]
r05, r09 = results[0], results[1]
for key, label in metrics:
    v05 = r05.get(key)
    v09 = r09.get(key)
    fmt05 = f"{v05:>18}" if isinstance(v05, int) else (f"{v05:>18.4f}" if v05 is not None else f"{'N/A':>18}")
    fmt09 = f"{v09:>18}" if isinstance(v09, int) else (f"{v09:>18.4f}" if v09 is not None else f"{'N/A':>18}")
    print(f"  {label:<25} {fmt05} {fmt09}")
PYSUM
else
    echo "FAILED: Step 11 failed with exit code ${EXIT_CODE}"
    echo "=========================================="
    echo ""
    echo "Check logs:"
    echo "  - ${OUTPUT_DIR}/step_11_evaluate_runs.log"
    echo "  - ${SCRIPT_DIR}/logs/11_evaluate_runs_${SLURM_JOB_ID}.err"
fi
echo ""

conda deactivate
exit ${EXIT_CODE}