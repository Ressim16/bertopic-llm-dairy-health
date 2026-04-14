#!/bin/bash

#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=06_llm_label
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --mail-user=your.email@institution.ch
#SBATCH --mail-type=fail,end
#SBATCH --output=logs/06_llm_labeling_%j.out
#SBATCH --error=logs/06_llm_labeling_%j.err

# ============================================================
# Step 06: Label Topics with LLM for Manual Review
# ============================================================
# This script uses a local Llama model to automatically label
# and summarize topics, with enhanced focus on dairy cattle
# health relevance. The output helps with manual review in
# deciding which topics to keep/remove in Step 07.
# ============================================================

# ============================================================
# PATHS - EDIT THESE FOR YOUR ENVIRONMENT
# ============================================================

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base output directory
OUTPUT_BASE="$(dirname "${SCRIPT_DIR}")/output"

# Input from Step 05
TOPIC_INFO="${OUTPUT_BASE}/05_bertopic_full/topic_info_full.csv"
DOCUMENT_INFO="${OUTPUT_BASE}/05_bertopic_full/document_info_full.csv"

# Input from Step 03
EMBEDDINGS="${OUTPUT_BASE}/03_embeddings/embeddings.npy"

# Output directory for this step
OUTPUT_DIR="${OUTPUT_BASE}/06_llm_labeling"

# Conda environment
CONDA_ENV="bertopic_gpu"

# ============================================================
# HUGGINGFACE TOKEN (REQUIRED FOR LLAMA MODEL)
# ============================================================
# Set your HuggingFace token for accessing Llama models
# Get token from: https://huggingface.co/settings/tokens
# Either export HF_TOKEN before running this script, or set it here:
#   export HF_TOKEN="your_token_here"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN is not set. Export it before running: export HF_TOKEN=your_token}"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

# ============================================================
# MODEL CONFIGURATION
# ============================================================
# Model to use (default: Llama 3.1 8B Instruct)
export LLAMA_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Use 4-bit quantization (0=no, 1=yes)
# Set to 1 if running on GPU with limited memory
export LLAMA_4BIT="0"

# ============================================================
# EXECUTION
# ============================================================

# Load modules
module load Anaconda3
module load GCC/11.2.0  # For CUDA support

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Log environment
echo "=========================================="
echo "STEP 06: LLM TOPIC LABELING"
echo "=========================================="
echo "Date:          $(date)"
echo "Host:          $(hostname)"
echo "Job ID:        ${SLURM_JOB_ID:-N/A}"
echo "GPU:           ${SLURM_GRES:-N/A}"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-8}"
echo "Topic info:    ${TOPIC_INFO}"
echo "Doc info:      ${DOCUMENT_INFO}"
echo "Embeddings:    ${EMBEDDINGS}"
echo "Output:        ${OUTPUT_DIR}"
echo "Model:         ${LLAMA_MODEL}"
echo "4-bit quant:   ${LLAMA_4BIT}"
echo "=========================================="
echo ""

# Check GPU availability
echo "Checking GPU..."
nvidia-smi || echo "Warning: nvidia-smi not available"
echo ""

# Verify PyTorch can see GPU
echo "Verifying PyTorch GPU access..."
python - <<'PYCHECK'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
else:
    print("WARNING: CUDA not available to PyTorch!")
PYCHECK
echo ""

# Check if input files exist
if [ ! -f "${TOPIC_INFO}" ]; then
    echo "ERROR: Topic info not found: ${TOPIC_INFO}"
    echo "Make sure Step 05 completed successfully."
    exit 1
fi

if [ ! -f "${DOCUMENT_INFO}" ]; then
    echo "ERROR: Document info not found: ${DOCUMENT_INFO}"
    echo "Make sure Step 05 completed successfully."
    exit 1
fi

if [ ! -f "${EMBEDDINGS}" ]; then
    echo "ERROR: Embeddings not found: ${EMBEDDINGS}"
    echo "Make sure Step 03 completed successfully."
    exit 1
fi

echo "LLM Labeling Configuration:"
echo "  - Uses local Llama model (no API calls)"
echo "  - Analyzes top-10 representative documents per topic"
echo "  - Generates structured labels with dairy health focus"
echo "  - Identifies potential non-dairy aspects"
echo "  - Provides dairy specificity score (1-5)"
echo ""
echo "This will help you decide which topics to Keep/Remove in Step 07"
echo ""
echo "Starting LLM labeling..."
echo "This may take 4-8 hours depending on GPU and number of topics..."
echo ""

# Execute Python script
python "${SCRIPT_DIR}/06_llm_labeling.py" \
    --topic-info "${TOPIC_INFO}" \
    --document-info "${DOCUMENT_INFO}" \
    --embeddings "${EMBEDDINGS}" \
    --output-dir "${OUTPUT_DIR}"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS: Step 06 completed"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  - ${OUTPUT_DIR}/topic_labels_llm.csv"
    echo "  - ${OUTPUT_DIR}/topic_info_with_llm.csv"
    echo "  - ${OUTPUT_DIR}/step_06_summary.json"
    echo "  - ${OUTPUT_DIR}/step_06_llm_labeling.log"
    echo ""
    
    # Show dairy specificity distribution
    if [ -f "${OUTPUT_DIR}/step_06_summary.json" ]; then
        echo "Dairy Specificity Distribution:"
        python - <<PYSUM
import json
try:
    with open("${OUTPUT_DIR}/step_06_summary.json") as f:
        summary = json.load(f)
    dist = summary.get("dairy_specificity_distribution", {})
    for score in range(5, 0, -1):
        count = dist.get(f"score_{score}", 0)
        print(f"  Score {score}: {count} topics")
except Exception as e:
    print(f"  Could not read summary: {e}")
PYSUM
        echo ""
    fi
    
    echo "Next steps:"
    echo "  1. Review ${OUTPUT_DIR}/topic_info_with_llm.csv"
    echo "  2. For each topic, check:"
    echo "     - LLM_DairySpecificityScore (1-5)"
    echo "     - LLM_PotentialNonDairyFocus"
    echo "     - LLM_LinkToDairyCattleHealth"
    echo "  3. Decide which topics to Keep/Remove"
    echo "  4. Update TOPIC_CLASSIFICATION in 07_cluster_filtering.py"
    echo "  5. Run Step 07 to filter dataset"
    echo ""
else
    echo "FAILED: Step 06 failed with exit code ${EXIT_CODE}"
    echo "=========================================="
    echo ""
    echo "Check logs:"
    echo "  - ${OUTPUT_DIR}/step_06_llm_labeling.log"
    echo "  - ${SCRIPT_DIR}/logs/06_llm_labeling_${SLURM_JOB_ID}.err"
    echo ""
    echo "Common issues:"
    echo "  - HuggingFace token not set or invalid"
    echo "  - Model not accessible (need Llama access approval)"
    echo "  - GPU not available or CUDA issues"
    echo "  - Out of memory (try setting LLAMA_4BIT=1)"
fi
echo ""

conda deactivate
exit ${EXIT_CODE}