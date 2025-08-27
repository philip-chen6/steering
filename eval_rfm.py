import re
import time
import os
import torch

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from neural_controllers import NeuralController
from evaluate import load as load_metric
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import evaluate
import pickle

from train_rfm import setup_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ["HF_HOME"] = "/opt/huggingface_cache"


RANDOM_SEED = 44
model_id = "google/gemma-2-9b-it"
# pipeline = transformers.pipeline(
#         "text-generation",
#         model=model_id,
#         model_kwargs={"torch_dtype": torch.bfloat16},
#         device_map="auto",
#         cache_dir="/opt/huggingface_cache",
#     )

bleu = load_metric("bleu")
rouge = load_metric("rouge")
bertscore = load_metric("bertscore")

log_file = "/data/philipchen/gemma/gemma_rfm.txt"
def log(msg):
    with open(log_file, "a") as f:
        f.write(msg + "\n")
    print(msg)
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")




#COT
def extract_mcq_response(text): 
    return text[-2]
def extract_frq_response(text): return text.strip()
def extract_cloze_response(text): return text.strip().split(" ")[-1]

def extract_final_answer(text):
    match = re.search(r'Answer:\s*([A-Da-z0-9\-]+)', text)
    if match:
        return match.group(1).strip()
    return text.strip().split()[-1]  # fallback
# ZERO SHOT
def extract_mcq_response(text): 
    return text
def extract_frq_response(text): return text.strip()
def extract_cloze_response(text): return text.strip()
def clean_response(text):
    # Remove <|...|> special tokens
    return re.sub(r"<\|.*?\|>", "", text).strip()

def extract_assistant_response(decoded):
    split_token = "<|start_header_id|>assistant<|end_header_id|>"
    if split_token in decoded:
        return clean_response(decoded.split(split_token)[-1])
    return clean_response(decoded)

# def generate_answer(formatted_question):
    

#     messages = [
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": formatted_question},
#     ]

#     outputs = pipeline(
#         messages,
#         max_new_tokens=1048,
#     )
#     response = outputs[0]["generated_text"][-1]
#     return response['content']

'''def generate_answer(formatted_question):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": formatted_question}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt", padding=True).to('cuda')

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=256
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.strip()
    return response'''

def generate_answer(formatted_question): 
    chat = [{"role": "user", "content": formatted_question}]
    prompt = tokenizer.apply_chat_template(
    chat, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=156, do_sample=False)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_part = decoded.split("model\n")[-1]
        
    return response_part.strip()
def evaluate_climaqa(test_data, controller):
    """
    Evaluate ClimaQA performance per task type.
    """

    task_metrics = {
        "mcq": {"preds": [], "refs": [], "correct": 0, "total": 0},
        "frq": {"preds": [], "refs": [], "correct": 0, "total": 0},
        "cloze": {"preds": [], "refs": [], "correct": 0, "total": 0},
    }

    control_layers = list(range(-1, -28, -1))
    start_time = time.time()

    for i, row in enumerate(tqdm(test_data, desc="Evaluating RFM on ClimaQA...")):
        task_type = row["type"]
        prompt = row["prompt"]
        gold = row["answer"].strip()

        formatted_prompt = controller.format_prompt(prompt)
        output = controller.generate(
            formatted_prompt,
            max_new_tokens = 512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            layers_to_control=control_layers,
            control_coef=0.2,
            do_sample=False,
        )
        output = extract_assistant_response(output)

        if task_type == "mcq":
            #pred = extract_mcq_response(output)
            pred = extract_final_answer(output)
        elif task_type == "frq":
            pred = extract_frq_response(output)
        else:
            #pred = extract_cloze_response(output)
            pred = extract_final_answer(output)

        task_metrics[task_type]["preds"].append(pred)
        task_metrics[task_type]["refs"].append(gold)
        task_metrics[task_type]["total"] += 1

        # Accuracy scoring
        if task_type == "cloze":
            if gold.lower() in pred.lower():
                task_metrics[task_type]["correct"] += 1
        else:
            if pred.strip().lower() == gold.strip().lower():
                task_metrics[task_type]["correct"] += 1

        # Logging
        log(f"Example {i+1}")
        log(f"Q: {prompt}")
        log(f"Raw model output: {output}")
        log(f"Gold: {gold}")
        log(f"Pred: {pred}")
        log("")

    elapsed = time.time() - start_time
    results = {}

    for task in ["mcq", "frq", "cloze"]:
        preds = task_metrics[task]["preds"]
        refs = task_metrics[task]["refs"]
        refs_nested = [[r] for r in refs]

        if len(preds) == 0:
            continue  # skip if task not present

        acc = task_metrics[task]["correct"] / task_metrics[task]["total"]
        bleu_score = bleu.compute(predictions=preds, references=refs_nested)["bleu"]
        rouge_score = rouge.compute(predictions=preds, references=refs)["rougeL"]
        bert_score = bertscore.compute(predictions=preds, references=refs, lang="en")["f1"]
        avg_bertscore = sum(bert_score) / len(bert_score)

        log("=" * 40)
        log(f"{task.upper()} Task Accuracy: {acc:.4f}")
        log(f"{task.upper()} BLEU: {bleu_score:.4f}")
        log(f"{task.upper()} ROUGE-L: {rouge_score:.4f}")
        log(f"{task.upper()} BERTScore (avg F1): {avg_bertscore:.4f}")
        log("=" * 40)

        results[task] = {
            "accuracy": acc,
            "bleu": bleu_score,
            "rougeL": rouge_score,
            "bertscore": avg_bertscore,
        }

    results["runtime"] = elapsed
    return results

if __name__ == "__main__":
    with open("/data/philipchen/gemma/rfm_test_data_raw.pkl", "rb") as f:
        test_data_raw = pickle.load(f)
    eval_data = []
    for ex in test_data_raw:
        eval_data.append({"prompt": ex["prompt_cot"], "answer": ex["answer"], "type": ex["type"]})
    tokenizer, model, device = setup_model()

    # Load the controller
    controller = NeuralController(
        model = model,
        tokenizer=tokenizer,
        control_method="rfm",
        n_components=5,
        rfm_iters=10,
        batch_size=2,
    )

    # Load the controller state
    filename = '/data/philipchen/gemma/gemma-2-9b-it'
    controller.load(concept="climaqa", model_name=model_id)

    print(f"Controller loaded from {filename}")

    # Prepare the dataset
    
    # Evaluate accuracy
    results = evaluate_climaqa(
        test_data=eval_data,
        controller=controller,
    )

