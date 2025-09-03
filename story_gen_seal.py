# story_gen_seal.py
import os, json, pickle, torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
# Use the same custom class you used in your SEAL eval
#from modeling_utils.modeling_gemma2 import Gemma2ForCausalLM
#from modeling_utils.modeling_qwen2 import Qwen2ForCausalLM  
from modeling_utils.modeling_llama import LlamaForCausalLM   
import torch._dynamo as dynamo

dynamo.config.capture_dynamic_output_shape_ops = True
dynamo.config.suppress_errors = True

def main(CONFIG):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model_id = CONFIG["llm_model_id"]
    cache_dir = CONFIG.get("cache_dir", None)

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Model (SEAL-capable class) ---
    model_cls = LlamaForCausalLM   
    model = model_cls.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    model.eval()
    model.config.use_cache = True

    # --- Steering vector ---
    with open(CONFIG["steering_vec_path"], "rb") as f:
        steer_vec = pickle.load(f)
    steer_vec = steer_vec.to(model.device)

    model.set_steering_flag(
        steering_flag=CONFIG.get("seal_enable", True),
        steering_layer=CONFIG.get("steering_layer", 20),
        steer_vec=steer_vec,
        steer_coef=CONFIG.get("steer_coef", 1.0),
        tokenizer=tokenizer,
    )

    if getattr(model, "start_new_round", None):
        model.start_new_round()

    # --- Dataset & prompt builder ---
    dataset = load_dataset("euclaise/writingprompts", split="test[:100]")
    results = []

    def build_messages(sample):
        prompt_txt = sample.get("prompt")
        user_prompt = "Write a story based on the following prompt: \n" + (prompt_txt or "")
        sample["_prompt_used"] = user_prompt
        return [{"role": "user", "content": user_prompt}]

    # --- Generation loop (SEAL only) ---
    for i, row in enumerate(tqdm(dataset, desc="SEAL: Generating Stories")):
        if i >= CONFIG["num_samples_to_generate"]:
            break

        messages = build_messages(row)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if getattr(model, "start_new_round", None):
            model.start_new_round()
        inputs = tokenizer(
            prompt_text, return_tensors="pt", truncation=True,
            max_length=CONFIG.get("max_input_len", 1024)
        ).to(model.device)

        with torch.inference_mode():
            if getattr(model, "start_new_round", None):
                model.start_new_round()
            out = model.generate(
                **inputs,
                min_new_tokens=10,
                max_new_tokens=CONFIG["generation_max_len"],
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)

        results.append({
            "problem": row,
            "prompt_used": row.get("_prompt_used", ""),
            "output_seal": decoded,
            "seal_settings": {
                "enabled": CONFIG.get("seal_enable", True),
                "steering_layer": CONFIG.get("steering_layer", 20),
                "steer_coef": CONFIG.get("steer_coef", 1.0),
            }
        })

    # --- Save ---
    out_path = CONFIG["output_json_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    CONFIG = {
        "llm_model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "cache_dir": "/data/philipchen",

        # Dataset / I/O
        "num_samples_to_generate": 100,
        "generation_max_len": 512,              
        "max_input_len": 1024,
        "output_json_path": "WritingPrompts/llama/llama_writingprompts_seal_100.json",

        # SEAL
        "steering_vec_path": "/data/philipchen/llama/steering_vector.pkl",
        "steering_layer": 20,                    
        "steer_coef": 1.0,                 
        "seal_enable": True,                    
    }
    main(CONFIG)
