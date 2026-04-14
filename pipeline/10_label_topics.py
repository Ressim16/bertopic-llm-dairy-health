#!/usr/bin/env python3
"""
Step 10: Label & Summarize Topics Using Local Llama 3.1 Instruct Model

Generates rich topic labels and metadata using LLM analysis:
- Refined topic titles
- Cluster summaries
- Factor-outcome associations
- Dairy health indicators
- Domain-specific tags

Inputs:
    - topic_info_rerun.csv: Topic information from Step 09
    - document_info_rerun.csv: Document assignments from Step 09  
    - embeddings_rerun.npy: Embeddings from Step 08

Outputs:
    - topic_labels_llm_rerun.csv: LLM-generated labels
    - topic_info_with_llm_rerun.csv: Merged with original topic info
    - step_10_label_topics.log: Execution log

Author: Reda Zahri
Date: 2025
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from json_repair import repair_json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

def configure_logging(log_file: Path = None) -> logging.Logger:
    """Configure logging with file and console output."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)


# ============================================================
# EMBEDDING & DOCUMENT UTILITIES
# ============================================================

def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize rows of matrix."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def clean_abstract(text: str, max_len: int) -> str:
    """Clean and truncate abstract text."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return (text[:max_len] + "...") if len(text) > max_len else text


def parse_top_words(row: pd.Series, n_words: int) -> List[str]:
    """Extract top N words from topic representation."""
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
    
    # Fallback to Name
    if "Name" in row and isinstance(row["Name"], str):
        return row["Name"].split()[:n_words]
    
    return []


def pick_top_k_docs_by_cosine(
    doc_df: pd.DataFrame,
    embeddings: np.ndarray,
    topic_id: int,
    k: int,
    max_abs_len: int,
) -> List[str]:
    """
    Select top-K documents closest to topic centroid by cosine similarity.
    
    Args:
        doc_df: DataFrame with Topic and Document columns
        embeddings: Document embeddings (normalized)
        topic_id: Topic ID to analyze
        k: Number of documents to return
        max_abs_len: Maximum abstract length
        
    Returns:
        List of cleaned abstract strings
    """
    # Get indices for this topic
    topic_mask = doc_df["Topic"] == topic_id
    topic_indices = doc_df.index[topic_mask].to_numpy()
    
    if topic_indices.size == 0:
        return []
    
    # Get topic embeddings and compute centroid
    topic_emb = embeddings[topic_indices]
    centroid = topic_emb.mean(axis=0, keepdims=True)
    centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
    
    # Normalize topic embeddings
    topic_emb_normalized = l2_normalize_rows(topic_emb)
    
    # Compute cosine similarities
    similarities = (topic_emb_normalized @ centroid.T).ravel()
    
    # Get top-K
    top_k_local_indices = np.argsort(-similarities)[:k]
    top_k_global_indices = topic_indices[top_k_local_indices]
    
    # Extract documents
    documents = doc_df.loc[top_k_global_indices, "Document"].astype(str).tolist()
    
    return [clean_abstract(doc, max_abs_len) for doc in documents]


# ============================================================
# LLM PROMPT CONSTRUCTION
# ============================================================

def build_chat_messages(
    topic_id: int,
    words: List[str],
    example_docs: List[str]
) -> List[Dict[str, str]]:
    """
    Build chat messages for LLM topic labeling.
    
    Args:
        topic_id: Topic ID
        words: Top topic keywords
        example_docs: Representative documents
        
    Returns:
        List of chat message dictionaries
    """
    words_str = ", ".join(words) if words else "(no words available)"
    docs_block = "\n".join(f"- {doc}" for doc in example_docs) if example_docs else "(no examples available)"
    
    system_prompt = (
        "You are an expert analyst of scientific abstracts in animal health and epidemiology, "
        "specialized in dairy cattle health indicators and risk-factor evidence synthesis.\n\n"
        
        "CRITICAL ANTI-HALLUCINATION RULES:\n"
        "1) Use ONLY information explicitly stated in the provided abstracts and keywords.\n"
        "2) Do NOT infer, assume, or add information not present in the abstracts.\n"
        "3) If information is unclear or not stated, use 'unclear' or leave blank.\n"
        "4) For dairy health indicators: ONLY suggest if explicitly mentioned or directly measurable.\n"
        "5) If unsure about any field, use conservative/minimal responses.\n"
        "6) Return ONLY JSON between <json> and </json>.\n\n"
        
        "Return JSON fields:\n"
        '  "refined_title" (5-10 words, based only on abstracts),\n'
        '  "most_relevant_search_topic" (1 sentence, factual only),\n'
        '  "cluster_summary" (120-220 words, evidence-based only),\n'
        '  "factor_associations" (3-8 objects with: "factor", "outcome_or_result", '
        '  "direction" (positive|negative|none|unclear), "brief_evidence" with exact quotes/facts),\n'
        '  "dairy_health_indicator_or_proxy" (1-5 items, ONLY if explicitly measurable from abstracts, otherwise empty list),\n'
        '  "link_to_dairy_cattle_health" (1-2 sentences, ONLY if direct evidence exists, otherwise state "Not explicitly stated"),\n'
        '  "potential_non_dairy_focus" (2-3 sentences, aspects not specific to dairy),\n'
        '  "tags" (3-8 lowercase, directly from abstracts),\n'
        '  "confidence" (1-5, be honest about uncertainty).\n\n'
        
        "No markdown, no speculation, no invented facts."
    )
    
    user_prompt = f"""Topic/Cluster ID: {topic_id}

Topic keywords:
{words_str}

TOP-{len(example_docs)} most central documents (cosine similarity to cluster centroid):
{docs_block}

Instructions:
- Analyze ONLY the information provided above.
- Extract the most relevant search topic based on actual abstract content.
- Summarize the cluster using only evidence from the abstracts.
- Identify factor-outcome associations with direction ONLY if clearly stated.
- For dairy health indicators: list ONLY if directly mentioned or clearly measurable.
- For link to dairy cattle health: provide ONLY if explicitly stated in abstracts. If not clear, write "Not explicitly stated in abstracts."
- Explain aspects that may NOT be dairy-specific.
- Be conservative. When in doubt, use "unclear" or minimal responses.
- Output JSON only between <json> and </json>.
"""
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]


# ============================================================
# JSON PARSING & NORMALIZATION
# ============================================================

def extract_json_from_text(text: str) -> str:
    """Extract JSON content from LLM output."""
    # Try to find content between <json> tags
    match = re.search(r"<json>(.*?)</json>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Try to find first complete JSON object
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


def parse_json_best_effort(text: str) -> Dict[str, Any]:
    """Parse JSON with error recovery."""
    json_text = extract_json_from_text(text)
    
    # Try standard parsing
    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        pass
    
    # Try json_repair
    try:
        repaired = repair_json(json_text)
        return json.loads(repaired)
    except Exception:
        pass
    
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
    for assoc in associations[:8]:  # Max 8
        if not isinstance(assoc, dict):
            continue
        
        direction = str(assoc.get("direction", "unclear")).strip().lower()
        if direction not in {"positive", "negative", "none", "unclear"}:
            direction = "unclear"
        
        clean_associations.append({
            "factor": str(assoc.get("factor", "")).strip()[:200],
            "outcome_or_result": str(assoc.get("outcome_or_result", "")).strip()[:200],
            "direction": direction,
            "brief_evidence": str(assoc.get("brief_evidence", "")).strip()[:300],
        })
    
    # Build normalized output
    output = {
        "refined_title": str(data.get("refined_title", "unlabeled-topic")).strip()[:100],
        "most_relevant_search_topic": str(data.get("most_relevant_search_topic", "")).strip()[:250],
        "cluster_summary": str(data.get("cluster_summary", "")).strip(),
        "factor_associations": clean_associations,
        "dairy_health_indicator_or_proxy": as_list(data.get("dairy_health_indicator_or_proxy", []))[:5],
        "link_to_dairy_cattle_health": str(data.get("link_to_dairy_cattle_health", "")).strip()[:400],
        "potential_non_dairy_focus": str(data.get("potential_non_dairy_focus", "")).strip()[:500],
        "tags": [str(t).strip().lower() for t in as_list(data.get("tags", [])) if str(t).strip()][:10],
        "confidence": int(data.get("confidence", 3)) if str(data.get("confidence", "")).isdigit() else 3,
    }
    
    # Truncate summary if too long
    if len(output["cluster_summary"].split()) > 250:
        output["cluster_summary"] = " ".join(output["cluster_summary"].split()[:250])
    
    # Clamp confidence to 1-5
    output["confidence"] = max(1, min(5, output["confidence"]))
    
    return output


# ============================================================
# LLM INITIALIZATION & GENERATION
# ============================================================

def init_llm_pipeline(model_name: str, use_4bit: bool, logger: logging.Logger):
    """Initialize LLM pipeline with optional 4-bit quantization."""
    logger.info(f"Loading model: {model_name}")
    logger.info(f"4-bit quantization: {use_4bit}")
    
    # Get HuggingFace token
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    
    if use_4bit:
        logger.info("Setting up 4-bit quantization...")
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=hf_token
        )
        
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    else:
        logger.info("Loading model in full precision...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=hf_token
        )
        
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    
    logger.info("Model loaded successfully")
    return pipe


def generate_topic_label(
    gen_pipeline,
    messages: List[Dict[str, str]],
    logger: logging.Logger,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Generate topic label using LLM with retry logic.
    
    Args:
        gen_pipeline: HuggingFace pipeline
        messages: Chat messages
        logger: Logger instance
        max_retries: Maximum number of retries
        
    Returns:
        Normalized LLM output dictionary
    """
    tokenizer = gen_pipeline.tokenizer
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    last_error = None
    for attempt in range(max_retries):
        try:
            output = gen_pipeline(
                prompt,
                max_new_tokens=1000,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            
            raw_text = output[0]["generated_text"]
            parsed = parse_json_best_effort(raw_text)
            return normalize_llm_output(parsed)
            
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    
    # Return fallback on failure
    logger.error(f"All attempts failed: {last_error}")
    return normalize_llm_output({
        "refined_title": "unlabeled-topic",
        "cluster_summary": f"Generation failed after {max_retries} retries: {last_error}",
        "confidence": 1
    })


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Step 10: Label topics with LLM (Llama 3.1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python 10_label_topics.py \\
        --topic-info /path/to/09_re_cluster/topic_info_rerun.csv \\
        --document-info /path/to/09_re_cluster/document_info_rerun.csv \\
        --embeddings /path/to/08_re_embed/embeddings_rerun.npy \\
        --output-dir /path/to/output/10_label_topics \\
        --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --use-4bit
        """
    )
    
    parser.add_argument('--topic-info', type=Path, required=True,
                       help='Topic info CSV from Step 09')
    parser.add_argument('--document-info', type=Path, required=True,
                       help='Document info CSV from Step 09')
    parser.add_argument('--embeddings', type=Path, required=True,
                       help='Embeddings NPY from Step 08')
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Output directory')
    parser.add_argument('--model-name', type=str,
                       default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                       help='LLM model name')
    parser.add_argument('--use-4bit', action='store_true',
                       help='Use 4-bit quantization')
    parser.add_argument('--n-words', type=int, default=15,
                       help='Number of topic words to use (default: 15)')
    parser.add_argument('--top-k-docs', type=int, default=10,
                       help='Number of example documents (default: 10)')
    parser.add_argument('--max-abstract-length', type=int, default=900,
                       help='Maximum abstract length (default: 900)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logger = configure_logging(args.output_dir / "step_10_label_topics.log")
    
    # Log header
    logger.info("=" * 70)
    logger.info("STEP 10: LABEL TOPICS WITH LLM")
    logger.info("=" * 70)
    logger.info(f"Topic info:    {args.topic_info}")
    logger.info(f"Document info: {args.document_info}")
    logger.info(f"Embeddings:    {args.embeddings}")
    logger.info(f"Output dir:    {args.output_dir}")
    logger.info(f"Model:         {args.model_name}")
    logger.info(f"4-bit quant:   {args.use_4bit}")
    logger.info("=" * 70)
    logger.info("")
    
    try:
        # Validate inputs
        for path in [args.topic_info, args.document_info, args.embeddings]:
            if not path.exists():
                raise FileNotFoundError(f"Input not found: {path}")
        
        # Load data
        logger.info("Loading data...")
        topic_df = pd.read_csv(args.topic_info)
        doc_df = pd.read_csv(args.document_info, usecols=lambda c: c in {"Document", "Topic"})
        logger.info(f"Loaded {len(topic_df)} topics, {len(doc_df)} documents")
        
        # Load embeddings
        logger.info("Loading embeddings...")
        embeddings = np.load(args.embeddings)
        logger.info(f"Loaded embeddings: {embeddings.shape}")
        
        # Validate alignment
        if embeddings.shape[0] != len(doc_df):
            raise ValueError(
                f"Embeddings ({embeddings.shape[0]}) and documents ({len(doc_df)}) "
                f"must have same length"
            )
        
        # Normalize embeddings
        embeddings = l2_normalize_rows(embeddings)
        logger.info("")
        
        # Filter to non-outlier topics
        topics_to_label = topic_df[topic_df["Topic"] != -1].copy()
        topics_to_label["Topic"] = topics_to_label["Topic"].astype(int)
        topics_to_label = topics_to_label.sort_values("Topic")
        logger.info(f"Topics to label: {len(topics_to_label)}")
        logger.info("")
        
        # Initialize LLM
        gen_pipeline = init_llm_pipeline(args.model_name, args.use_4bit, logger)
        logger.info("")
        
        # Process topics
        results = []
        for idx, (_, row) in enumerate(topics_to_label.iterrows(), 1):
            topic_id = int(row["Topic"])
            logger.info(f"Processing topic {topic_id} ({idx}/{len(topics_to_label)})...")
            
            # Extract topic words
            words = parse_top_words(row, args.n_words)
            
            # Get example documents
            example_docs = pick_top_k_docs_by_cosine(
                doc_df,
                embeddings,
                topic_id,
                args.top_k_docs,
                args.max_abstract_length
            )
            
            # Build messages
            messages = build_chat_messages(topic_id, words, example_docs)
            
            # Generate label
            llm_output = generate_topic_label(gen_pipeline, messages, logger)
            
            # Store result
            results.append({
                "Topic": topic_id,
                "LLM_RefinedTitle": llm_output["refined_title"],
                "LLM_MostRelevantSearchTopic": llm_output["most_relevant_search_topic"],
                "LLM_ClusterSummary": llm_output["cluster_summary"],
                "LLM_FactorAssociations_JSON": json.dumps(
                    llm_output["factor_associations"],
                    ensure_ascii=False
                ),
                "LLM_DairyHealthIndicatorOrProxy": ", ".join(
                    [str(x) for x in llm_output["dairy_health_indicator_or_proxy"]]
                ),
                "LLM_LinkToDairyCattleHealth": llm_output["link_to_dairy_cattle_health"],
                "LLM_PotentialNonDairyFocus": llm_output["potential_non_dairy_focus"],
                "LLM_Tags": ", ".join(llm_output["tags"]),
                "LLM_Confidence": llm_output["confidence"],
                "TopWordsUsed": ", ".join(words),
                "TopKDocsUsed": len(example_docs),
            })
        
        # Save results
        logger.info("")
        logger.info("Saving results...")
        
        # Save labels CSV
        labels_df = pd.DataFrame(results)
        labels_path = args.output_dir / "topic_labels_llm_rerun.csv"
        labels_df.to_csv(labels_path, index=False)
        logger.info(f"Saved labels: {labels_path}")
        
        # Merge with topic info
        merged_df = topic_df.merge(labels_df, on="Topic", how="left")
        merged_path = args.output_dir / "topic_info_with_llm_rerun.csv"
        merged_df.to_csv(merged_path, index=False)
        logger.info(f"Saved merged info: {merged_path}")
        
        # Log completion
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 10 COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        logger.info(f"Processed {len(results)} topics successfully")
        logger.info("")
        logger.info("Output files:")
        logger.info(f"  - {labels_path}")
        logger.info(f"  - {merged_path}")
        logger.info(f"  - {args.output_dir}/step_10_label_topics.log")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"ERROR: {e}", exc_info=True)
        logger.error("Step 10 FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())