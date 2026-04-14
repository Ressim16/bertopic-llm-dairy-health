#!/usr/bin/env python3
"""
Step 06: Label Topics with LLM for Manual Review

This script uses a local Llama model to automatically label and summarize each topic/cluster
from BERTopic. This helps with manual review by providing:
1. Refined topic titles
2. Cluster summaries
3. Factor-outcome associations
4. Dairy cattle health relevance assessment
5. Identification of potential non-dairy aspects

The LLM analysis helps you decide which topics to keep/remove in Step 07.

Inputs:
- topic_info_full.csv (from Step 05)
- document_info_full.csv (from Step 05)
- embeddings.npy (from Step 03)

Outputs:
- topic_labels_llm.csv (LLM-generated labels and summaries)
- topic_info_with_llm.csv (topic_info merged with LLM labels)
- step_06_llm_labeling.log (execution log)

Requirements:
- GPU (H100 or A100 recommended)
- HuggingFace token for Llama model access
- transformers, torch, json_repair packages
"""

import os
import sys
import re
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from json_repair import repair_json

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths (will be passed as command line arguments)
TOPIC_INFO_CSV = None
DOC_INFO_CSV = None
EMBEDDINGS_NPY = None
OUTPUT_DIR = None

# Model configuration
MODEL_NAME = os.environ.get("LLAMA_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
USE_4BIT = bool(int(os.environ.get("LLAMA_4BIT", "0")))
DTYPE = torch.bfloat16
DEVICE_MAP = "auto"

# Generation parameters
MAX_NEW_TOKENS = 1200
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 0
DO_SAMPLE = False

# Prompt parameters
N_WORDS = 15        # Number of topic keywords to include
TOPK_DOCS = 10      # Number of representative documents to analyze
MAX_ABS_CHARS = 900 # Maximum characters per abstract


def setup_logging(output_dir):
    """Setup logging to both file and console."""
    log_file = os.path.join(output_dir, "step_06_llm_labeling.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize rows of a matrix."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def parse_top_words(row: pd.Series, n_words: int) -> List[str]:
    """Extract top N words from a topic row."""
    # Try Representation column first
    if "Representation" in row and isinstance(row["Representation"], str) and row["Representation"].strip():
        parts = [w.strip() for w in row["Representation"].split(",")]
        return [p for p in parts if p][:n_words]
    
    # Try Top_n_words column
    if "Top_n_words" in row and isinstance(row["Top_n_words"], str) and row["Top_n_words"].strip():
        words = []
        for chunk in row["Top_n_words"].split(","):
            w = chunk.split(":")[0].strip()
            if w:
                words.append(w)
        return words[:n_words]
    
    # Fallback to Name column
    if "Name" in row and isinstance(row["Name"], str):
        return row["Name"].split()[:n_words]
    
    return []


def clean_abstract(text: str, max_len: int) -> str:
    """Clean and truncate abstract text."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return (text[:max_len] + "...") if len(text) > max_len else text


def pick_representative_documents(
    doc_df: pd.DataFrame,
    embeddings: np.ndarray,
    topic_id: int,
    k: int,
    max_abs_len: int
) -> List[str]:
    """
    Pick top-k most representative documents for a topic based on cosine similarity
    to the topic centroid.
    """
    # Get indices of documents in this topic
    topic_indices = doc_df.index[doc_df["Topic"] == topic_id].to_numpy()
    if topic_indices.size == 0:
        return []
    
    # Get embeddings for this topic
    topic_embeddings = embeddings[topic_indices]
    
    # Calculate centroid and normalize
    centroid = topic_embeddings.mean(axis=0, keepdims=True)
    centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
    
    # Calculate cosine similarities
    topic_emb_normalized = l2_normalize_rows(topic_embeddings)
    similarities = (topic_emb_normalized @ centroid.T).ravel()
    
    # Get top-k most similar documents
    top_k_local = np.argsort(-similarities)[:k]
    top_k_global = topic_indices[top_k_local]
    
    # Extract and clean documents
    documents = doc_df.loc[top_k_global, "Document"].astype(str).tolist()
    return [clean_abstract(doc, max_abs_len) for doc in documents]


def build_prompt_messages(topic_id: int, keywords: List[str], documents: List[str]) -> List[Dict[str, str]]:
    """
    Build the chat messages for the LLM with enhanced focus on dairy cattle health relevance.
    """
    keywords_str = ", ".join(keywords) if keywords else "(no keywords available)"
    documents_str = "\n".join(f"- {doc}" for doc in documents) if documents else "(no documents available)"
    
    system_prompt = (
        "You are an expert analyst of scientific abstracts in animal health and epidemiology, "
        "specialized in dairy cattle health indicators and risk-factor evidence synthesis.\n\n"
        
        "Your task is to analyze clusters of research abstracts and provide structured assessments "
        "with a specific focus on dairy cattle health relevance.\n\n"
        
        "CRITICAL RULES:\n"
        "1) Use ONLY the provided abstracts and keywords. Do NOT invent or hallucinate information.\n"
        "2) Be explicit about uncertainty - if you cannot determine something, say so.\n"
        "3) Distinguish between:\n"
        "   - Research specifically on dairy cattle\n"
        "   - Research on cattle in general (may include dairy)\n"
        "   - Research on other animals or contexts\n"
        "4) Identify aspects that may reduce dairy cattle health relevance\n"
        "5) Keep outputs concise, specific, and evidence-based.\n"
        "6) Return ONLY valid JSON between <json> and </json> tags.\n\n"
        
        "REQUIRED JSON FIELDS:\n"
        '  "refined_title": Short descriptive title (<= 10 words)\n'
        '  "most_relevant_search_topic": Single sentence describing the main research focus\n'
        '  "cluster_summary": Comprehensive synthesis (120-220 words) covering:\n'
        '    - Main research themes and questions\n'
        '    - Key findings or methods\n'
        '    - Target populations/species mentioned\n'
        '  "factor_associations": Array of 3-8 objects, each with:\n'
        '    - "factor": The independent variable or exposure\n'
        '    - "outcome_or_result": The dependent variable or outcome\n'
        '    - "direction": "positive" | "negative" | "none" | "unclear"\n'
        '    - "brief_evidence": Supporting evidence (<= 20 words)\n'
        '  "dairy_health_indicator_or_proxy": Array of 1-5 relevant health indicators or proxies\n'
        '  "link_to_dairy_cattle_health": 1-2 sentences explaining HOW this cluster relates to dairy cattle health specifically\n'
        '  "potential_non_dairy_focus": 2-4 sentences explaining aspects that may REDUCE dairy cattle health relevance:\n'
        '    - Is this research on other species (pigs, poultry, humans)?\n'
        '    - Is this general livestock/animal health rather than dairy-specific?\n'
        '    - Is this about beef cattle rather than dairy?\n'
        '    - Is this methodological research without dairy application?\n'
        '    - Are there environmental/economic factors not specific to dairy?\n'
        '    - Is the geographic/production context not applicable to dairy?\n'
        '  "dairy_specificity_score": Integer 1-5 where:\n'
        '    5 = Explicitly about dairy cattle health\n'
        '    4 = Mostly dairy-related with minor non-dairy aspects\n'
        '    3 = Mixed dairy and non-dairy cattle research\n'
        '    2 = General animal health that may apply to dairy\n'
        '    1 = Not dairy-specific or primarily other contexts\n'
        '  "tags": Array of 3-8 lowercase descriptive tags\n'
        '  "confidence": Integer 1-5 (quality of this analysis)\n\n'
        
        "OUTPUT FORMAT:\n"
        "Return ONLY the JSON object between <json> and </json> tags.\n"
        "No markdown, no explanatory text, no preamble."
    )
    
    user_prompt = f"""Topic/Cluster ID: {topic_id}

Topic Keywords:
{keywords_str}

Representative Documents (top-{len(documents)} by centroid similarity):
{documents_str}

ANALYSIS INSTRUCTIONS:

1. IDENTIFY the main research focus and themes
2. SYNTHESIZE findings across documents
3. EXTRACT factor-outcome associations with direction
4. DETERMINE dairy cattle health relevance:
   - Is this explicitly about dairy cattle?
   - Could this apply to dairy cattle health?
   - What aspects are NOT dairy-specific?
5. ASSESS dairy specificity (1-5 scale)
6. PROVIDE evidence-based, concise outputs

Remember: Be critical about dairy cattle specificity. Many topics may seem relevant but could be:
- About other animal species
- General animal health concepts
- Beef cattle rather than dairy
- Human health or environmental research
- Methodological papers

OUTPUT the complete JSON structure between <json> and </json> tags."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


def extract_json_from_response(text: str) -> str:
    """Extract JSON content from LLM response."""
    # Try to find content between <json> tags
    match = re.search(r"<json>(.*?)</json>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Try to find a JSON object
    start = text.find("{")
    if start == -1:
        return text
    
    depth = 0
    for i, char in enumerate(text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    
    return text


def parse_llm_json(text: str, logger) -> Dict[str, Any]:
    """Parse JSON from LLM response with fallback mechanisms."""
    json_str = extract_json_from_response(text)
    
    # Try direct parsing
    try:
        return json.loads(json_str)
    except Exception as e:
        logger.warning(f"Direct JSON parse failed: {e}")
    
    # Try JSON repair
    try:
        repaired = repair_json(json_str)
        return json.loads(repaired)
    except Exception as e:
        logger.warning(f"JSON repair failed: {e}")
    
    return {}


def normalize_llm_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize and validate LLM output."""
    def as_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            return [item.strip() for item in x.split(",") if item.strip()]
        return [x]
    
    # Normalize factor associations
    associations = data.get("factor_associations", [])
    if not isinstance(associations, list):
        associations = []
    
    clean_associations = []
    for assoc in associations[:8]:  # Limit to 8
        if not isinstance(assoc, dict):
            continue
        
        direction = str(assoc.get("direction", "unclear")).strip().lower()
        if direction not in {"positive", "negative", "none", "unclear"}:
            direction = "unclear"
        
        clean_associations.append({
            "factor": str(assoc.get("factor", "")).strip(),
            "outcome_or_result": str(assoc.get("outcome_or_result", "")).strip(),
            "direction": direction,
            "brief_evidence": str(assoc.get("brief_evidence", "")).strip()[:160]
        })
    
    # Validate dairy specificity score
    dairy_score = data.get("dairy_specificity_score", 3)
    try:
        dairy_score = int(dairy_score)
        dairy_score = max(1, min(5, dairy_score))
    except (ValueError, TypeError):
        dairy_score = 3

    # Validate confidence
    confidence = data.get("confidence", 3)
    try:
        confidence = int(confidence)
        confidence = max(1, min(5, confidence))
    except (ValueError, TypeError):
        confidence = 3
    
    normalized = {
        "refined_title": str(data.get("refined_title", "unlabeled-topic")).strip()[:100],
        "most_relevant_search_topic": str(data.get("most_relevant_search_topic", "")).strip()[:250],
        "cluster_summary": str(data.get("cluster_summary", "")).strip()[:2000],
        "factor_associations": clean_associations,
        "dairy_health_indicator_or_proxy": as_list(data.get("dairy_health_indicator_or_proxy", []))[:5],
        "link_to_dairy_cattle_health": str(data.get("link_to_dairy_cattle_health", "")).strip()[:500],
        "potential_non_dairy_focus": str(data.get("potential_non_dairy_focus", "")).strip()[:800],
        "dairy_specificity_score": dairy_score,
        "tags": [str(t).strip().lower() for t in as_list(data.get("tags", [])) if str(t).strip()][:10],
        "confidence": confidence
    }
    
    return normalized


def initialize_model(logger):
    """Initialize the LLM pipeline."""
    logger.info(f"Initializing model: {MODEL_NAME}")
    logger.info(f"Using 4-bit quantization: {USE_4BIT}")
    logger.info(f"Device map: {DEVICE_MAP}")
    
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    
    if USE_4BIT:
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE
        )
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map=DEVICE_MAP,
            torch_dtype=DTYPE,
            token=hf_token
        )
        
        return pipeline("text-generation", model=model, tokenizer=tokenizer, device_map=DEVICE_MAP)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map=DEVICE_MAP,
            torch_dtype=DTYPE,
            token=hf_token
        )
        
        return pipeline("text-generation", model=model, tokenizer=tokenizer, device_map=DEVICE_MAP)


def generate_topic_labels(gen_pipeline, messages: List[Dict[str, str]], logger) -> Dict[str, Any]:
    """Generate labels for a topic using the LLM."""
    tokenizer = gen_pipeline.tokenizer
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Try up to 5 times with exponential backoff
    for attempt in range(5):
        try:
            output = gen_pipeline(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = output[0]["generated_text"]
            parsed = parse_llm_json(generated_text, logger)
            return normalize_llm_output(parsed)
            
        except Exception as e:
            logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
            if attempt < 4:
                time.sleep(2 ** attempt)
    
    # Fallback if all attempts fail
    return normalize_llm_output({
        "refined_title": "generation-failed",
        "cluster_summary": "LLM generation failed after multiple attempts",
        "confidence": 1,
        "dairy_specificity_score": 1
    })


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Label BERTopic topics with LLM for manual review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--topic-info", required=True, help="Path to topic_info_full.csv")
    parser.add_argument("--document-info", required=True, help="Path to document_info_full.csv")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings.npy")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Set global paths
    global TOPIC_INFO_CSV, DOC_INFO_CSV, EMBEDDINGS_NPY, OUTPUT_DIR
    TOPIC_INFO_CSV = args.topic_info
    DOC_INFO_CSV = args.document_info
    EMBEDDINGS_NPY = args.embeddings
    OUTPUT_DIR = args.output_dir
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logging(OUTPUT_DIR)
    
    try:
        logger.info("="*70)
        logger.info("STEP 06: LLM TOPIC LABELING")
        logger.info("="*70)
        logger.info(f"Topic info:    {TOPIC_INFO_CSV}")
        logger.info(f"Document info: {DOC_INFO_CSV}")
        logger.info(f"Embeddings:    {EMBEDDINGS_NPY}")
        logger.info(f"Output dir:    {OUTPUT_DIR}")
        logger.info(f"Model:         {MODEL_NAME}")
        logger.info("="*70)
        
        # Load data
        logger.info("Loading topic info...")
        topic_df = pd.read_csv(TOPIC_INFO_CSV)
        
        logger.info("Loading document info...")
        doc_df = pd.read_csv(DOC_INFO_CSV, usecols=["Document", "Topic"])
        
        logger.info("Loading embeddings...")
        embeddings = np.load(EMBEDDINGS_NPY)
        
        # Validate alignment
        if embeddings.shape[0] != len(doc_df):
            raise ValueError(
                f"Embeddings ({embeddings.shape[0]} rows) don't match documents ({len(doc_df)} rows). "
                "They must be aligned."
            )
        
        # Normalize embeddings for faster cosine similarity
        logger.info("Normalizing embeddings...")
        embeddings = l2_normalize_rows(embeddings)
        
        # Prepare topics to process (exclude outlier topic -1)
        topics_to_process = topic_df[topic_df["Topic"] != -1].copy()
        topics_to_process["Topic"] = topics_to_process["Topic"].astype(int)
        topics_to_process = topics_to_process.sort_values("Topic")
        
        logger.info(f"Processing {len(topics_to_process)} topics (excluding outlier topic -1)")
        
        # Initialize model
        logger.info("Initializing LLM...")
        gen_pipeline = initialize_model(logger)
        logger.info("Model ready")
        
        # Process each topic
        results = []
        start_time = time.time()
        
        for idx, (_, row) in enumerate(topics_to_process.iterrows(), 1):
            topic_id = int(row["Topic"])
            logger.info(f"Processing topic {topic_id} ({idx}/{len(topics_to_process)})...")
            
            # Extract keywords
            keywords = parse_top_words(row, N_WORDS)
            
            # Get representative documents
            documents = pick_representative_documents(
                doc_df=doc_df,
                embeddings=embeddings,
                topic_id=topic_id,
                k=TOPK_DOCS,
                max_abs_len=MAX_ABS_CHARS
            )
            
            # Generate labels
            messages = build_prompt_messages(topic_id, keywords, documents)
            labels = generate_topic_labels(gen_pipeline, messages, logger)
            
            # Store results
            results.append({
                "Topic": topic_id,
                "LLM_RefinedTitle": labels["refined_title"],
                "LLM_MostRelevantSearchTopic": labels["most_relevant_search_topic"],
                "LLM_ClusterSummary": labels["cluster_summary"],
                "LLM_FactorAssociations_JSON": json.dumps(labels["factor_associations"], ensure_ascii=False),
                "LLM_DairyHealthIndicatorOrProxy": ", ".join(labels["dairy_health_indicator_or_proxy"]),
                "LLM_LinkToDairyCattleHealth": labels["link_to_dairy_cattle_health"],
                "LLM_PotentialNonDairyFocus": labels["potential_non_dairy_focus"],
                "LLM_DairySpecificityScore": labels["dairy_specificity_score"],
                "LLM_Tags": ", ".join(labels["tags"]),
                "LLM_Confidence": labels["confidence"],
                "TopWordsUsed": ", ".join(keywords),
                "NumDocsAnalyzed": len(documents)
            })
            
            # Log progress
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = (len(topics_to_process) - idx) * avg_time
            logger.info(f"  Dairy specificity: {labels['dairy_specificity_score']}/5")
            logger.info(f"  Estimated time remaining: {remaining/60:.1f} minutes")
        
        # Save results
        logger.info("Saving results...")
        
        results_df = pd.DataFrame(results)
        labels_csv = os.path.join(OUTPUT_DIR, "topic_labels_llm.csv")
        results_df.to_csv(labels_csv, index=False)
        logger.info(f"Saved: {labels_csv}")
        
        # Merge with original topic_info
        merged_df = topic_df.merge(results_df, on="Topic", how="left")
        merged_csv = os.path.join(OUTPUT_DIR, "topic_info_with_llm.csv")
        merged_df.to_csv(merged_csv, index=False)
        logger.info(f"Saved: {merged_csv}")
        
        # Generate summary
        total_time = time.time() - start_time
        
        summary = {
            "generated_at": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "topics_processed": len(results),
            "total_time_seconds": round(total_time, 2),
            "avg_time_per_topic": round(total_time / len(results), 2),
            "dairy_specificity_distribution": {
                "score_5": int((results_df["LLM_DairySpecificityScore"] == 5).sum()),
                "score_4": int((results_df["LLM_DairySpecificityScore"] == 4).sum()),
                "score_3": int((results_df["LLM_DairySpecificityScore"] == 3).sum()),
                "score_2": int((results_df["LLM_DairySpecificityScore"] == 2).sum()),
                "score_1": int((results_df["LLM_DairySpecificityScore"] == 1).sum())
            },
            "output_files": {
                "labels": labels_csv,
                "merged": merged_csv,
                "log": os.path.join(OUTPUT_DIR, "step_06_llm_labeling.log")
            }
        }
        
        summary_json = os.path.join(OUTPUT_DIR, "step_06_summary.json")
        with open(summary_json, 'w') as f:
            json.dump(summary, indent=2, fp=f)
        logger.info(f"Saved: {summary_json}")
        
        # Print summary
        logger.info("")
        logger.info("="*70)
        logger.info("SUMMARY")
        logger.info("="*70)
        logger.info(f"Topics processed:    {len(results)}")
        logger.info(f"Total time:          {total_time/60:.1f} minutes")
        logger.info(f"Avg time per topic:  {total_time/len(results):.1f} seconds")
        logger.info("")
        logger.info("Dairy Specificity Distribution:")
        for score in range(5, 0, -1):
            count = summary["dairy_specificity_distribution"][f"score_{score}"]
            logger.info(f"  Score {score}: {count} topics ({count/len(results)*100:.1f}%)")
        logger.info("")
        logger.info("Next step: Review topic_info_with_llm.csv and classify topics as Keep/Remove for Step 07")
        logger.info("="*70)
        logger.info("SUCCESS: Step 06 completed")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()