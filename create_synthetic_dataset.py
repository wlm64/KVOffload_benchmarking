import pandas as pd
import json
import random
import string
from transformers import AutoTokenizer
from vllm import TokensPrompt  # assuming you're using vLLM
import numpy as np
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="sharegpt")
parser.add_argument("--KT", type = int, default=None)
parser.add_argument("--N", type = int, default=5000)

args = parser.parse_args()

if args.dataset == "sharegpt":
    csv_path = "sharegpt_effective_prefill.csv"
elif args.dataset == "narrativeqa":
    csv_path = "narrativeqa_token_counts_all_splits.csv"
elif args.dataset == "docfinqa":
    csv_path = "docfinqa_token_counts_all_splits.csv"

print(csv_path)

for csv in [csv_path]: #, "narrativeqa_token_counts_all_splits.csv", "docfinqa_token_counts_all_splits.csv"]:
    # --- Config ---
    input_csv = csv
    output_jsonl = f"synthetic_prompts_{csv.split('_')[0]}.jsonl"
    compute_jsonl = f"synthetic_prompts_{csv.split('_')[0]}_compute.jsonl"
    memory_jsonl = f"synthetic_prompts_{csv.split('_')[0]}_memory.jsonl"
    N = args.N                        # Number of samples to generate
    
    # --- Load dataset ---
    df = pd.read_csv(input_csv)
    
    # Sample N rows randomly
    sampled = df.sample(n=N, replace=False, random_state=1)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B-Instruct-2507")
    
    vocab_words = ["the","be","to","of","and","a","in","that","have","I","it","for","not","on","with","he",
        "as","you","do","at","this","but","his","by","from","they","we","say","her","she","or",
        "an","will","my","one","all","would","there","their","what","so","up","out","if","about",
        "who","get","which","go","me","when","make","can","like","time","no","just","him","know",
        "take","people","into","year","your","good","some","could","them","see","other","than",
        "then","now","look","only","come","its","over","think","also","back","after","use","two",
        "how","our","work","first","well","way","even","new","want","because","any","these","give",
        "day","most","us","is","am","are","was","were","been","has","had","does","did","say","says",
        "said","goes","went","make","makes","made","know","knows","knew","think","thinks","thought",
        "see","sees","saw","come","comes","came","take","takes","took","want","wants","wanted"]
    
    # --- Build prompts ---
    prompts = []

    compute_prompts = []
    memory_prompts = []
    
    for _, row in sampled.iterrows():
        c_tokens = int(row['context_tokens'])
        q_tokens = int(row['question_tokens'])
        if (c_tokens + q_tokens) < 262144:
            question = " ".join(np.random.choice(vocab_words, size=q_tokens - 1))
            # --- Decode back into text ---
            context = "Hi" * (c_tokens - 1) #tokenizer.decode([1] * int(context_tokens), skip_special_tokens=True)
            prompt = f"{context} {question}"             
            prompts.append({"prompt": prompt})
            if args.KT:
                if c_tokens / q_tokens <= args.KT:
                    compute_prompts.append({"prompt": prompt})
                else:
                    memory_prompts.append({"prompt": prompt})
    
    # --- Save to JSONL ---
    with open(output_jsonl, "w") as f:
        for p in prompts:
            json.dump(p, f)
            f.write("\n")
    if args.KT:
        with open(compute_jsonl, "w") as f:
            for p in compute_prompts:
                json.dump(p, f)
                f.write("\n")
        with open(memory_jsonl, "w") as f:
            for p in memory_prompts:
                json.dump(p, f)
                f.write("\n")  
    
    print(f"✅ Saved {len(prompts)} synthetic prompts to {output_jsonl}")
