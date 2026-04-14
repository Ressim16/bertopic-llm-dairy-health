#!/bin/bash

#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=10_label_topics
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-user=your.email@institution.ch
#SBATCH --mail-type=fail,end
#SBATCH --output=logs/10_label_topics_%j.out
#SBATCH --error=logs/10_label_topics_%j.err

# ============================================================
# Step 10: Label Topics with LLM
# ============================================================

# ============================================================
# PATHS - EDIT THESE FOR YOUR ENVIRONMENT
# ============================================================

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base output directory
OUTPUT_BASE="$(dirname "${SCRIPT_DIR}")/output"

# Inputs from previous steps
TOPIC_INFO="${OUTPUT_BASE}/09_train_rerun/topic_info_rerun.csv"
DOC_INFO="${OUTPUT_BASE}/09_train_rerun/document_info_rerun.csv"
EMBEDDINGS="${OUTPUT_BASE}/08_re_embed/embeddings_rerun.npy"

# Output directory for this step
OUTPUT_DIR="${OUTPUT_BASE}/10_label_topics"

# LLM settings
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
USE_4BIT=0  # 1 = use 4-bit quantization, 0 = full precision

# Labeling parameters
N_WORDS=15
TOP_K_DOCS=10
MAX_ABSTRACT_LENGTH=900

# Conda environment
CONDA_ENV="bertopic_gpu"

# ============================================================
# EXECUTION
# ============================================================

# Load environment
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV}"
module load GCC/11.2.0

# Set HuggingFace token (if not already in environment)
if [ -z "$HF_TOKEN" ]; then
    if [ -f ~/.tokens/huggingface ]; then
        export HF_TOKEN=$(cat ~/.tokens/huggingface)
        echo "Loaded HF_TOKEN from ~/.tokens/huggingface"
    elif [ -f ~/.cache/huggingface/token ]; then
        export HF_TOKEN=$(cat ~/.cache/huggingface/token)
        echo "Loaded HF_TOKEN from HuggingFace CLI cache"
    else
        echo "WARNING: HF_TOKEN not found. LLM download may fail."
        echo "Please run: huggingface-cli login"
    fi
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Log environment
echo "=========================================="
echo "STEP 10: LABEL TOPICS WITH LLM"
echo "=========================================="
echo "Date:           $(date)"
echo "Host:           $(hostname)"
echo "Job ID:         ${SLURM_JOB_ID:-N/A}"
echo "Topic info:     ${TOPIC_INFO}"
echo "Document info:  ${DOC_INFO}"
echo "Embeddings:     ${EMBEDDINGS}"
echo "Output:         ${OUTPUT_DIR}"
echo "Model:          ${MODEL_NAME}"
echo "4-bit quant:    ${USE_4BIT}"
echo "N words:        ${N_WORDS}"
echo "Top-K docs:     ${TOP_K_DOCS}"
echo "=========================================="
echo ""

# Check GPU
echo "GPU Information:"
nvidia-smi || echo "WARNING: nvidia-smi not available"
echo ""

# Verify PyTorch GPU access
echo "Verifying PyTorch GPU access:"
python - <<'PYCHECK'
import torch
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version:    {torch.version.cuda}")
    print(f"  GPU device:      {torch.cuda.get_device_name(0)}")
    print(f"  GPU count:       {torch.cuda.device_count()}")
PYCHECK
echo ""

# Check if input files exist
if [ ! -f "${TOPIC_INFO}" ]; then
    echo "ERROR: Topic info not found: ${TOPIC_INFO}"
    echo "Make sure Step 09 completed successfully."
    exit 1
fi

if [ ! -f "${DOC_INFO}" ]; then
    echo "ERROR: Document info not found: ${DOC_INFO}"
    echo "Make sure Step 09 completed successfully."
    exit 1
fi

if [ ! -f "${EMBEDDINGS}" ]; then
    echo "ERROR: Embeddings not found: ${EMBEDDINGS}"
    echo "Make sure Step 08 completed successfully."
    exit 1
fi

# Show input statistics
echo "Input statistics:"
python - <<PYSTATS
import pandas as pd
import numpy as np

topic_df = pd.read_csv("${TOPIC_INFO}")
doc_df = pd.read_csv("${DOC_INFO}")
emb = np.load("${EMBEDDINGS}")

n_topics = (topic_df['Topic'] != -1).sum()
print(f"  Topics to label:  {n_topics}")
print(f"  Total documents:  {len(doc_df):,}")
print(f"  Embedding shape:  {emb.shape}")
PYSTATS
echo ""

# Build command arguments
CMD_ARGS="--topic-info ${TOPIC_INFO}"
CMD_ARGS="${CMD_ARGS} --document-info ${DOC_INFO}"
CMD_ARGS="${CMD_ARGS} --embeddings ${EMBEDDINGS}"
CMD_ARGS="${CMD_ARGS} --output-dir ${OUTPUT_DIR}"
CMD_ARGS="${CMD_ARGS} --model-name ${MODEL_NAME}"
CMD_ARGS="${CMD_ARGS} --n-words ${N_WORDS}"
CMD_ARGS="${CMD_ARGS} --top-k-docs ${TOP_K_DOCS}"
CMD_ARGS="${CMD_ARGS} --max-abstract-length ${MAX_ABSTRACT_LENGTH}"

if [ ${USE_4BIT} -eq 1 ]; then
    CMD_ARGS="${CMD_ARGS} --use-4bit"
fi

# Execute Python script
echo "Starting topic labeling..."
echo ""
python "${SCRIPT_DIR}/10_label_topics.py" ${CMD_ARGS}

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "SUCCESS: Step 10 completed"
    echo "=========================================="
    echo ""
    echo "✅ All topics labeled successfully!"
    echo ""
    echo "Output files:"
    echo "  - ${OUTPUT_DIR}/topic_labels_llm_rerun.csv"
    echo "  - ${OUTPUT_DIR}/topic_info_with_llm_rerun.csv"
    echo "  - ${OUTPUT_DIR}/step_10_label_topics.log"
    echo ""
    echo "Summary:"
    python - <<PYSUM
import pandas as pd
df = pd.read_csv("${OUTPUT_DIR}/topic_labels_llm_rerun.csv")
print(f"  Topics labeled:     {len(df)}")
print(f"  Avg confidence:     {df['LLM_Confidence'].mean():.2f}")
print(f"  High confidence:    {(df['LLM_Confidence'] >= 4).sum()} topics")
PYSUM
    echo ""
    echo "=========================================="
    echo "🎉 PIPELINE COMPLETE! 🎉"
    echo "=========================================="
    echo ""
    echo "All 10 steps completed successfully!"
    echo ""
    echo "Final outputs:"
    echo "  1. Filtered abstracts:   ${OUTPUT_BASE}/01_filter/"
    echo "  2. Truncated abstracts:  ${OUTPUT_BASE}/02_truncate/"
    echo "  3. Initial embeddings:   ${OUTPUT_BASE}/03_embeddings/"
    echo "  4. Grid search results:  ${OUTPUT_BASE}/04_grid_search/"
    echo "  5. Initial clustering:   ${OUTPUT_BASE}/05_bertopic_full/"
    echo "  6. LLM labels (review):  ${OUTPUT_BASE}/06_llm_labels/"
    echo "  7. Filtered abstracts:   ${OUTPUT_BASE}/07_filtered/"
    echo "  8. Re-embeddings:        ${OUTPUT_BASE}/08_re_embed/"
    echo "  9. Final clustering:     ${OUTPUT_BASE}/09_train_rerun/"
    echo " 10. Final labels:         ${OUTPUT_BASE}/10_label_topics/"
    echo ""
else
    echo "FAILED: Step 10 failed with exit code ${EXIT_CODE}"
    echo "=========================================="
    echo ""
    echo "Check logs:"
    echo "  - ${OUTPUT_DIR}/step_10_label_topics.log"
    echo "  - ${SCRIPT_DIR}/logs/10_label_topics_${SLURM_JOB_ID}.err"
fi
echo ""

conda deactivate
exit ${EXIT_CODE}