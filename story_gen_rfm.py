import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
import json
from tqdm import tqdm
from neural_controllers import NeuralController

def main(CONFIG):
    print("Loading base LLM in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    llm = AutoModelForCausalLM.from_pretrained(
        CONFIG["llm_model_id"],
        quantization_config=bnb_config,
        torch_dtype=CONFIG["dtype"],
        device_map="auto",
        cache_dir="/data/ubuntu/huggingface_cache"
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["llm_model_id"])
    # ensure a pad token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    llm.config.use_cache = True
    llm.eval()

    # RFM controller
    controller = NeuralController(
        model=llm,
        tokenizer=tokenizer,
        control_method="rfm",
        n_components=CONFIG.get("rfm_n_components", 5),
        rfm_iters=CONFIG.get("rfm_iters", 10),
        batch_size=CONFIG.get("rfm_batch_size", 2),
    )
    controller.load(concept="climaqa", model_name=CONFIG["llm_model_id"])
    print("Controller loaded.")

    print(f"Loading dataset: {CONFIG['dataset_name']}")
    results_to_save = []

    if CONFIG['dataset_name'] == 'WritingPrompts':
        dataset = load_dataset('euclaise/writingprompts', split='test[:100]')
        save_results = True
        def get_prompt_messages(sample):
            prompt_txt = sample.get('prompt')
            user_prompt = "Write a story based on the following prompt: \n" + (prompt_txt or "")
            sample['_prompt_used'] = user_prompt
            return [{"role": "user", "content": user_prompt}]
    elif CONFIG['dataset_name'] == "GSM8k":
        dataset = load_dataset('openai/gsm8k', 'main', split='test')
        save_results = True
        def get_prompt_messages(sample):
            return [{"role": "user", "content": sample['question']}]
    elif CONFIG['dataset_name'] == 'ClimaQA':
        dataset = load_dataset('UCSD-genie/ClimaQA', 'Gold', split='ffq')
        save_results = True
        def get_prompt_messages(sample):
            return [{"role": "user", "content": sample["Question"]}]
    else:
        raise ValueError("Unknown dataset name.")

    # control layers once
    control_layers = list(range(-1, -32, -1))

    # RFM-only generation loop
    for i, row in enumerate(tqdm(dataset, desc="Generating Samples")):
        if i >= CONFIG["num_samples_to_generate"]:
            break

        messages = get_prompt_messages(row)
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        with torch.no_grad():
            out_rfm = controller.generate(
                prompt_text,
                max_new_tokens=CONFIG["generation_max_len"],
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                layers_to_control=control_layers,
                control_coef=0.2,
                do_sample=False,
            )
        decoded_rfm = out_rfm if isinstance(out_rfm, str) else str(out_rfm)

        if save_results:
            results_to_save.append({
                "problem": row,
                "prompt_used": row.get("_prompt_used", ""),
                "output_rfm": decoded_rfm,
                "rfm_layers": control_layers,
                "rfm_coef": 0.2
            })

    # save
    if save_results:
        out_dir = os.path.dirname(CONFIG['output_json_path'])
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        print(f"\nSaving results to {CONFIG['output_json_path']}...")
        with open(CONFIG['output_json_path'], 'w') as f:
            json.dump(results_to_save, f, indent=4, ensure_ascii=False)
        print("Save complete.")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    CONFIG = {
        "llm_model_id": "meta-llama/Llama-3.1-8B-Instruct",   # use model that matches your RFM checkpoint
        "sft_model_path": "/data/llama-sft-gsm8k-final",

        "lm_dim": 4096,
        "state_dim": 4600,

        "dataset_name": "WritingPrompts",
        "num_samples_to_generate": 100,
        "generation_max_len": 512,  # 200–400 words
        "output_json_path": "WritingPrompts/llama/llama_writingprompts_rfm_100.json",
        "dtype": torch.bfloat16,

        "rfm_n_components": 5,
        "rfm_iters": 10,
        "rfm_batch_size": 2,
    }
    main(CONFIG)
    torch.cuda.empty_cache()
