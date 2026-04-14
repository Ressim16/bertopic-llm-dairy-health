#!/bin/bash

#SBATCH --account=gratis
#SBATCH --job-name=phase1_pipeline
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mail-user=your.email@institution.ch
#SBATCH --mail-type=fail,end
#SBATCH --output=logs/phase1_pipeline_%j.out
#SBATCH --error=logs/phase1_pipeline_%j.err

# ============================================================
# Phase 1 Pipeline: Steps 01-06 (Before Manual Topic Review)
# ============================================================
#
# Prerequisites:
#   1. Completed Steps 00_01 and 00_02 (deduplication)
#   2. Copied merged_cleaned_auto_dedup.csv to input directory
#
# This script submits Steps 01-06 with proper dependencies.
# After completion, manual topic review is required before Phase 2.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="$(dirname "${SCRIPT_DIR}")/input"
OUTPUT_BASE="$(dirname "${SCRIPT_DIR}")/output"

echo "=========================================="
echo "PHASE 1 PIPELINE: STEPS 01-06"
echo "=========================================="
echo "Date: $(date)"
echo "=========================================="
echo ""

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================

echo "Pre-flight checks..."

# Check deduplicated dataset exists
DEDUP_FILE="${INPUT_DIR}/merged_cleaned_auto_dedup.csv"
if [ ! -f "${DEDUP_FILE}" ]; then
    echo "ERROR: Deduplicated dataset not found: ${DEDUP_FILE}"
    echo ""
    echo "Please complete the deduplication pipeline first:"
    echo "  1. Run 00_01_pairwise_comparison.sh"
    echo "  2. Label candidate pairs (is_duplicate column)"
    echo "  3. Run 00_02_threshold_selection.sh"
    echo "  4. Copy output to input directory:"
    echo "     cp ${OUTPUT_BASE}/00_02_threshold/merged_cleaned_auto_dedup.csv ${INPUT_DIR}/"
    exit 1
fi
echo "  ✓ Deduplicated dataset found"

# Check all scripts exist
SCRIPTS=(
    "01_filter.sh"
    "02_truncate.sh"
    "03_embed.sh"
    "04_grid_search.sh"
    "05_bertopic_full.sh"
    "06_llm_labeling.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ ! -f "${SCRIPT_DIR}/${script}" ]; then
        echo "  ✗ Missing: ${script}"
        exit 1
    fi
done
echo "  ✓ All scripts found"
echo ""

# ============================================================
# SUBMIT JOBS WITH DEPENDENCIES
# ============================================================

echo "Submitting jobs..."
echo ""

# Step 01: Filter (CPU)
JOB1_OUT=$(sbatch "${SCRIPT_DIR}/01_filter.sh")
JOB1_ID=$(echo "${JOB1_OUT}" | awk '{print $4}')
echo "  Step 01 (Filter):             Job ${JOB1_ID}"

# Step 02: Truncate (CPU) - depends on 01
JOB2_OUT=$(sbatch --dependency=afterok:${JOB1_ID} "${SCRIPT_DIR}/02_truncate.sh")
JOB2_ID=$(echo "${JOB2_OUT}" | awk '{print $4}')
echo "  Step 02 (Truncate):           Job ${JOB2_ID} (after ${JOB1_ID})"

# Step 03: Embed (GPU) - depends on 02
JOB3_OUT=$(sbatch --dependency=afterok:${JOB2_ID} "${SCRIPT_DIR}/03_embed.sh")
JOB3_ID=$(echo "${JOB3_OUT}" | awk '{print $4}')
echo "  Step 03 (Embed GPU):          Job ${JOB3_ID} (after ${JOB2_ID})"

# Step 04: Grid Search (CPU) - depends on 03
JOB4_OUT=$(sbatch --dependency=afterok:${JOB3_ID} "${SCRIPT_DIR}/04_grid_search.sh")
JOB4_ID=$(echo "${JOB4_OUT}" | awk '{print $4}')
echo "  Step 04 (Grid Search):        Job ${JOB4_ID} (after ${JOB3_ID})"

# Step 05: BERTopic Full (CPU) - depends on 04
JOB5_OUT=$(sbatch --dependency=afterok:${JOB4_ID} "${SCRIPT_DIR}/05_bertopic_full.sh")
JOB5_ID=$(echo "${JOB5_OUT}" | awk '{print $4}')
echo "  Step 05 (BERTopic Full):      Job ${JOB5_ID} (after ${JOB4_ID})"

# Step 06: LLM Labeling (GPU) - depends on 05
JOB6_OUT=$(sbatch --dependency=afterok:${JOB5_ID} "${SCRIPT_DIR}/06_llm_labeling.sh")
JOB6_ID=$(echo "${JOB6_OUT}" | awk '{print $4}')
echo "  Step 06 (LLM Labeling GPU):   Job ${JOB6_ID} (after ${JOB5_ID})"

echo ""
echo "=========================================="
echo "PHASE 1 SUBMITTED"
echo "=========================================="
echo ""
echo "Job chain: ${JOB1_ID} → ${JOB2_ID} → ${JOB3_ID} → ${JOB4_ID} → ${JOB5_ID} → ${JOB6_ID}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel:  scancel ${JOB1_ID},${JOB2_ID},${JOB3_ID},${JOB4_ID},${JOB5_ID},${JOB6_ID}"
echo ""
echo "=========================================="
echo "ESTIMATED RUNTIME"
echo "=========================================="
echo "  Step 01 (Filter):        ~10 min"
echo "  Step 02 (Truncate):      ~30 min"
echo "  Step 03 (Embed GPU):     ~2-4 hours"
echo "  Step 04 (Grid Search):   ~12-24 hours"
echo "  Step 05 (BERTopic):      ~1 hour"
echo "  Step 06 (LLM Label):     ~4-8 hours"
echo "  ─────────────────────────────────"
echo "  Total:                   ~20-38 hours"
echo "=========================================="
echo ""
echo "=========================================="
echo "⚠️  NEXT STEPS AFTER PHASE 1 COMPLETES"
echo "=========================================="
echo ""
echo "1. Review LLM labels:"
echo "   ${OUTPUT_BASE}/06_llm_labeling/topic_info_with_llm.csv"
echo ""
echo "2. For each topic, evaluate:"
echo "   - LLM_DairySpecificityScore (1-5)"
echo "   - LLM_PotentialNonDairyFocus"
echo "   - LLM_LinkToDairyCattleHealth"
echo ""
echo "3. Update TOPIC_CLASSIFICATION in:"
echo "   ${SCRIPT_DIR}/07_cluster_filtering.py"
echo ""
echo "4. Run Phase 2:"
echo "   sbatch ${SCRIPT_DIR}/run_phase2_pipeline.sh"
echo "=========================================="

exit 0