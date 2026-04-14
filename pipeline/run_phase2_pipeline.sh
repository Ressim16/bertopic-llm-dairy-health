#!/bin/bash

#SBATCH --account=gratis
#SBATCH --job-name=phase2_pipeline
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mail-user=your.email@institution.ch
#SBATCH --mail-type=fail,end
#SBATCH --output=logs/phase2_pipeline_%j.out
#SBATCH --error=logs/phase2_pipeline_%j.err

# ============================================================
# Phase 2 Pipeline: Steps 07-10 (After Manual Topic Review)
# ============================================================
#
# Run this AFTER you have:
#   1. Completed Steps 01-06
#   2. Reviewed LLM labels in 06_llm_labeling/topic_info_with_llm.csv
#   3. Updated TOPIC_CLASSIFICATION in 07_cluster_filtering.py
#
# This script submits Steps 07-10 with proper dependencies.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="$(dirname "${SCRIPT_DIR}")/output"

echo "=========================================="
echo "PHASE 2 PIPELINE: STEPS 07-10"
echo "=========================================="
echo "Date: $(date)"
echo "=========================================="
echo ""

# Pre-flight checks
echo "Pre-flight checks..."

# Check Step 06 output exists
LLM_OUTPUT="${OUTPUT_BASE}/06_llm_labeling/topic_info_with_llm.csv"
if [ ! -f "${LLM_OUTPUT}" ]; then
    echo "ERROR: Step 06 output not found: ${LLM_OUTPUT}"
    echo "Please complete Steps 01-06 first."
    exit 1
fi
echo "  ✓ Step 06 output found"

# Check Step 05 outputs exist
DOC_INFO="${OUTPUT_BASE}/05_bertopic_full/document_info_full.csv"
if [ ! -f "${DOC_INFO}" ]; then
    echo "ERROR: Step 05 output not found: ${DOC_INFO}"
    exit 1
fi
echo "  ✓ Step 05 output found"

# Check scripts exist
for step in 07 08 09 10; do
    script=$(ls ${SCRIPT_DIR}/*${step}*.sh 2>/dev/null | head -1)
    if [ -z "${script}" ]; then
        echo "ERROR: Script for step ${step} not found"
        exit 1
    fi
done
echo "  ✓ All scripts found"
echo ""

# Submit jobs
echo "Submitting jobs..."
echo ""

# Step 07: Cluster Filtering (CPU)
JOB7_OUT=$(sbatch "${SCRIPT_DIR}/07_cluster_filtering.sh")
JOB7_ID=$(echo "${JOB7_OUT}" | awk '{print $4}')
echo "  Step 07 (Cluster Filtering):  Job ${JOB7_ID}"

# Step 08: Re-embed (GPU) - depends on 07
JOB8_OUT=$(sbatch --dependency=afterok:${JOB7_ID} "${SCRIPT_DIR}/08_re_embed.sh")
JOB8_ID=$(echo "${JOB8_OUT}" | awk '{print $4}')
echo "  Step 08 (Re-embed GPU):       Job ${JOB8_ID} (after ${JOB7_ID})"

# Step 09: Load Rerun Model (CPU) - depends on 08
JOB9_OUT=$(sbatch --dependency=afterok:${JOB8_ID} "${SCRIPT_DIR}/09_train_rerun_model.sh")
JOB9_ID=$(echo "${JOB9_OUT}" | awk '{print $4}')
echo "  Step 09 (Rerun Model):        Job ${JOB9_ID} (after ${JOB8_ID})"

# Step 10: Label Topics (GPU) - depends on 09
JOB10_OUT=$(sbatch --dependency=afterok:${JOB9_ID} "${SCRIPT_DIR}/10_label_topics.sh")
JOB10_ID=$(echo "${JOB10_OUT}" | awk '{print $4}')
echo "  Step 10 (Final Labels GPU):   Job ${JOB10_ID} (after ${JOB9_ID})"

echo ""
echo "=========================================="
echo "PHASE 2 SUBMITTED"
echo "=========================================="
echo ""
echo "Job chain: ${JOB7_ID} → ${JOB8_ID} → ${JOB9_ID} → ${JOB10_ID}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel:  scancel ${JOB7_ID},${JOB8_ID},${JOB9_ID},${JOB10_ID}"
echo ""
echo "Estimated runtime: ~8-14 hours total"
echo "=========================================="

exit 0