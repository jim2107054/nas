import re
import time
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from src.metric_utils import rouge_l_f1

try:
    from normalizer import normalize
except ImportError:
    # Fallback normalizer if csebuetnlp/normalizer is not installed in local environment
    def normalize(text: str) -> str:
        return str(text)

SYSTEM_PROMPT = "আপনি একজন অভিজ্ঞ চিকিৎসক। রোগীর প্রশ্ন মনোযোগ দিয়ে পড়ুন এবং চিকিৎসাগতভাবে সঠিক, স্পষ্ট ও সহানুভূতিশীল উত্তর বাংলায় দিন।"

def load_model_for_inference(model_path: str = "/kaggle/working/final_model") -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the merged model and tokenizer in bfloat16 for evaluation or submission."""
    print(f"Loading merged model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("Model loaded and set to eval mode.")
    return model, tokenizer

def generate_candidates(
    model, 
    tokenizer, 
    patient_input: str, 
    k: int = 4, 
    max_new_tokens: int = 260, 
    min_new_tokens: int = 60, 
    temperature: float = 0.7, 
    top_p: float = 0.9
) -> list[str]:
    """Generate k candidate responses for a single patient query."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": patient_input}
    ]
    
    try:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback to manual Alpaca-style prompt
        prompt_text = (
            f"### System:\n{SYSTEM_PROMPT}\n\n"
            f"### Instruction:\n{patient_input}\n\n"
            f"### Response:\n"
        )
        
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=True,
            num_return_sequences=k,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
    candidates = []
    for out in outputs:
        new_tokens = out[input_len:]
        candidate = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        candidates.append(candidate)
        
    return candidates

def length_ok(text: str, target_range: tuple[int, int] = (65, 145)) -> bool:
    """Check if the response length in words falls within the target range."""
    word_count = len(str(text).split())
    return target_range[0] <= word_count <= target_range[1]

def mbr_select(candidates: list[str]) -> str:
    """Select the most canonical candidate using Minimum Bayes Risk over pairwise ROUGE-L."""
    if len(candidates) <= 1:
        return candidates[0] if candidates else ""
        
    # Filter to length_ok candidates if any exist, otherwise use all candidates
    ok_candidates = [c for c in candidates if length_ok(c)]
    pool = ok_candidates if ok_candidates else candidates
    
    if len(pool) == 1:
        return pool[0]
        
    best_candidate = pool[0]
    best_score = -1.0
    
    for i, cand_i in enumerate(pool):
        scores = []
        for j, cand_j in enumerate(pool):
            if i != j:
                scores.append(rouge_l_f1(cand_i, cand_j))
        mean_score = sum(scores) / len(scores) if scores else 0.0
        if mean_score > best_score:
            best_score = mean_score
            best_candidate = cand_i
            
    return best_candidate

def clean_output(text: str) -> str:
    """Clean the generated response by removing echoed prompts and extra whitespaces."""
    # Collapse multiple spaces and newlines
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    
    # Remove system prompt leaks if any
    escaped_sys = re.escape(SYSTEM_PROMPT)
    cleaned = re.sub(escaped_sys, "", cleaned, flags=re.IGNORECASE).strip()
    
    # Remove template tags if model generates them
    cleaned = re.sub(r"###\s*(System|Instruction|Input|Response|User|Assistant):", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.replace("<s>", "").replace("</s>", "").strip()
    
    return cleaned

def run_inference(model, tokenizer, df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """Generate medical responses for the given dataset with MBR selection and progress updates."""
    results = []
    start_time = time.time()
    
    total_rows = len(df)
    print(f"Starting inference on {total_rows} rows (k={k})...")
    
    for idx, row in df.iterrows():
        row_id = row["id"]
        patient_input = row["input"]
        
        # 1. Generate candidates
        candidates = generate_candidates(model, tokenizer, patient_input, k=k)
        
        # 2. Minimum Bayes Risk selection
        selected = mbr_select(candidates)
        
        # 3. Clean output
        output_text = clean_output(selected)
        
        results.append({
            "id": row_id,
            "output": output_text
        })
        
        # Print progress every 50 rows
        count = idx + 1
        if count % 50 == 0 or count == total_rows:
            elapsed = time.time() - start_time
            avg_time = elapsed / count
            rem_time = avg_time * (total_rows - count)
            
            # Format time
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            rem_str = f"{int(rem_time // 60)}m {int(rem_time % 60)}s"
            print(f"Processed {count}/{total_rows} rows | Elapsed: {elapsed_str} | Est. Remaining: {rem_str}")
            
    return pd.DataFrame(results)

def load_banglat5_for_inference(model_path: str = "/kaggle/working/banglat5_final") -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    """Load the fine-tuned BanglaT5 model and tokenizer in eval mode on GPU."""
    print(f"Loading BanglaT5 model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
    ).to(device)
    
    model.eval()
    print(f"BanglaT5 model loaded and set to eval mode on {device}.")
    return model, tokenizer

def generate_banglat5(
    model, 
    tokenizer, 
    patient_input: str, 
    num_beams: int = 5, 
    no_repeat_ngram_size: int = 3, 
    length_penalty: float = 0.6, 
    max_new_tokens: int = 300, 
    min_new_tokens: int = 40
) -> str:
    """Generate medical response using BanglaT5 with beam search and text normalization."""
    # 1. Apply Bangla text normalization
    norm_input = normalize(str(patient_input).strip())
    
    # 2. Tokenize input
    inputs = tokenizer(
        norm_input,
        max_length=384,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)
    
    # 3. Generate via beam search
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            early_stopping=True,
            do_sample=False
        )
        
    # 4. Decode and clean
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_output(raw_output)

def run_inference_banglat5(
    model, 
    tokenizer, 
    df: pd.DataFrame, 
    batch_size: int = 16,
    num_beams: int = 5,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 0.6,
    max_new_tokens: int = 300,
    min_new_tokens: int = 40
) -> pd.DataFrame:
    """Run batched inference for BanglaT5 over a DataFrame with ['id', 'input'] columns."""
    results = []
    total_rows = len(df)
    start_time = time.time()
    
    num_batches = (total_rows + batch_size - 1) // batch_size
    print(f"Starting BanglaT5 batched inference on {total_rows} rows (batch_size={batch_size}, total_batches={num_batches})...")
    
    for batch_idx in range(num_batches):
        start_i = batch_idx * batch_size
        end_i = min(start_i + batch_size, total_rows)
        batch_df = df.iloc[start_i:end_i]
        
        batch_ids = batch_df["id"].tolist()
        batch_inputs = [normalize(str(x).strip()) for x in batch_df["input"].tolist()]
        
        # Batched tokenization
        encoded = tokenizer(
            batch_inputs,
            max_length=384,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                length_penalty=length_penalty,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                early_stopping=True,
                do_sample=False
            )
            
        decoded_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for row_id, raw_out in zip(batch_ids, decoded_batch):
            results.append({
                "id": row_id,
                "output": clean_output(raw_out)
            })
            
        current_batch_num = batch_idx + 1
        if current_batch_num % 5 == 0 or current_batch_num == num_batches:
            elapsed = time.time() - start_time
            avg_per_batch = elapsed / current_batch_num
            rem_time = avg_per_batch * (num_batches - current_batch_num)
            
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            rem_str = f"{int(rem_time // 60)}m {int(rem_time % 60)}s"
            processed_count = min(end_i, total_rows)
            print(f"Batch {current_batch_num}/{num_batches} ({processed_count}/{total_rows} rows) | Elapsed: {elapsed_str} | Est. Remaining: {rem_str}")
            
    return pd.DataFrame(results)

