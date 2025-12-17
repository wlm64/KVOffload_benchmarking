import pandas as pd
import json
import random
import string
from transformers import AutoTokenizer
from vllm import TokensPrompt  # assuming you're using vLLM
import numpy as np
import os

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

for model in ["Qwen/Qwen3-30B-A3B-Instruct-2507"]:
    os.makedirs(f"./offload_datasets/{model}", exist_ok=True)
    for context_tokens in [0, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        for prefill in [16, 32, 64, 128, 256, 512, 1024, 2048]:
            print(model + str(context_tokens) + str(prefill))
            output_jsonl = f"./offload_datasets/{model}/synthetic_prompts_{context_tokens}_{prefill}.jsonl"
            N = 500                        # Number of samples to generate
            tokenizer = AutoTokenizer.from_pretrained(model)

            # --- Build prompts ---
            prompts = []
        
            for i in range(N):

                # Build context and question text
                # input_tokens = np.random.choice(valid_ids, size=prefill).tolist()
                # # --- Decode back into text ---
                # question = tokenizer.decode(input_tokens, skip_special_tokens=True)

                question = " ".join(np.random.choice(vocab_words, size=prefill - 1))
                # --- Decode back into text ---
                context = "Hi" * (context_tokens - 1) #tokenizer.decode([1] * int(context_tokens), skip_special_tokens=True)
                prompt = f"{context} {question}"             
                prompts.append({"prompt": prompt})
                #import pdb; pdb.set_trace()
            # --- Save to JSONL ---
            with open(output_jsonl, "w") as f:
                for p in prompts:
                    json.dump(p, f)
                    f.write("\n")

            print(f"✅ Saved {len(prompts)} synthetic prompts to {output_jsonl}")
