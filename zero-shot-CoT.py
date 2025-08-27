import time
import re
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load as load_metric
import torch
torch.cuda.empty_cache()
import os
import re


# Load model and tokenizer
#model_id = "Qwen/Qwen2-7B-Instruct"
model_id = "google/gemma-2-9b-it"

'''model_id = "meta-llama/Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )'''

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="/opt/huggingface_cache", device_map="auto")

# Load evaluation metrics
bleu = load_metric("bleu")
rouge = load_metric("rouge")
bertscore = load_metric("bertscore")
#Formatters and extractors for Zero-Shot
# Format MCQ prompt
def format_mcq_prompt(question, options):
    prompt = f"Answer the following multiple choice question with the letter of the correct option (A, B, C, or D). State your answer as the first character so I can extract it. Do not explain.\n\n{question.strip()}\n"
    for i, option in enumerate(options):
        letter = chr(65 + i)
        prompt += f"{letter}. {option.strip()}\n"
    prompt += "\nAnswer:"
    return prompt


def format_frq_prompt(question):
    return f"Answer the following question in one or two sentences. \n\n{question.strip()}\nAnswer:"

#CLOZE
def format_cloze_prompt(question):
    return f"Fill in the blank denoted by <MASK> in the sentence below with the correct word. Return just the word.\n\n{question.strip()}\nAnswer:"
# Zero Shot extractors
def extract_mcq_response(text): 
    return text
def extract_frq_response(text): return text.strip()
def extract_cloze_response(text): return text.strip()


#CoT Extractors and Formatters
# Extractors CoT
'''def extract_mcq_response(text): 
    return text[-2]
def extract_frq_response(text): return text.strip()
def extract_cloze_response(text): return text.strip().split(" ")[-1]

def extract_final_answer(text):
    match = re.search(r'Answer:\s*([A-Da-z0-9\-]+)', text)
    if match:
        return match.group(1).strip()
    return text.strip().split()[-1]  # fallback
def format_mcq_prompt(question, options):
    prompt = f"""Answer the following multiple choice question (A, B, C, or D) with step by step reasoning including transition and reflection thoughts if needed, then state your final answer as 'Answer: <LETTER>' on a new line.\n\n{question.strip()}\n"""
    for i, option in enumerate(options):
        letter = chr(65 + i)
        prompt += f"{letter}. {option.strip()}\n"
    return prompt
def format_frq_prompt(question):
    return f"""\n\n{question.strip()}\nLet's think step by step, including transition and reflection thoughts if needed. \n"""
def format_cloze_prompt(question):
    return f"""Fill in the blank denoted by <MASK> in the sentence. Think step by step including transition and reflection thoughts if needed, then write your final answer on a new line starting with 'Answer:'.\n\n{question.strip()}\n"""
'''

# Logger setup
log_file = "/data/philipchen/gemma/gemma_zero_shot.txt"
def log(msg):
    with open(log_file, "a") as f:
        f.write(msg + "\n")
    print(msg)

# Generate model response using chat template GEMMA
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

#QWEN

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
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = response.strip()
    return response'''


#LLAMA
'''def generate_answer(formatted_question):
    

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": formatted_question},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    response = outputs[0]["generated_text"][-1]
    return response['content']'''


# Evaluation logic
def evaluate_split(dataset_split, task_type):
    references, predictions = [], []
    correct = 0
    start_time = time.time()

    for i, row in enumerate(dataset_split):
        q = row["Question"]
        a = row["Answer"]
        options = row.get("Options", None)

        # Build prompt & extract ground truth
        if task_type == "mcq":
            raw_options = row["Options"]
            options = re.findall(r"[a-d]\)\s*(.*?)\s*(?=(?:[a-d]\)|$))", raw_options, flags=re.DOTALL)
            options = [opt.split("\n")[0].strip() for opt in options]
            prompt = format_mcq_prompt(q, options)
            gold = a
        elif task_type == "frq":
            prompt = format_frq_prompt(q)
            gold = a.strip()
        elif task_type == "cloze":
            prompt = format_cloze_prompt(q)
            gold = a.strip()

        # Generate and extract
        output = generate_answer(prompt)
        if task_type == "mcq":
            #zero shot pred = extract_mcq_response(output)
            pred = extract_frq_response(output)
        elif task_type == "frq":
            pred = extract_frq_response(output)
        else:
            #zero shot pred = extract_cloze_response(output)
            pred = extract_frq_response(output)


        # Logging
        log(f"Example {i+1}")
        log(f"Q: {q}")
        if task_type == "mcq":
            log(f"Options parsed: {options}")
        log(f"Raw model output: {output}")
        log(f"Gold: {gold}")
        log(f"Pred: {pred}")
        log("")

        # Score
        references.append([gold])
        predictions.append(pred)

        if task_type == "cloze":
            # Allow substring matching for cloze since model may embed correct word in a sentence
            if gold.lower() in pred.lower():
                correct += 1
        else:
            if pred.strip().lower() == gold.strip().lower():
                correct += 1


    total = len(dataset_split)
    acc = correct / total
    elapsed = time.time() - start_time

    bleu_score = bleu.compute(predictions=predictions, references=references)["bleu"]
    rouge_score = rouge.compute(
        predictions=predictions,
        references=[ref[0] for ref in references],
        rouge_types=["rougeL"]
    )["rougeL"]
    bert_score = bertscore.compute(
        predictions=predictions,
        references=[ref[0] for ref in references],
        lang="en"
    )["f1"]
    avg_bertscore = sum(bert_score) / len(bert_score)

    log("=" * 40)
    log(f"{task_type.upper()} Task Accuracy: {acc:.4f}")
    log(f"{task_type.upper()} BLEU: {bleu_score:.4f}")
    log(f"{task_type.upper()} ROUGE-L: {rouge_score:.4f}")
    log(f"{task_type.upper()} BERTScore (avg F1): {avg_bertscore:.4f}")
    log(f"{task_type.upper()} Runtime (s): {elapsed:.2f}")
    log("=" * 40)
    log("")

    return {
        "accuracy": acc,
        "bleu": bleu_score,
        "rougeL": rouge_score,
        "bertscore": avg_bertscore,
        "runtime": elapsed,
    }

# Load data
dataset = load_dataset("UCSD-GENIE/ClimaQA", "Gold")
mcq_data = dataset["mcq"]
frq_data = dataset["ffq"]
cloze_data = dataset["cloze"]

# Clear previous logs and run evaluation
import os
os.makedirs("/data/philipchen", exist_ok=True)
#open(log_file, "w").close()


# Run evaluation
evaluate_model_mcq = evaluate_split(mcq_data, "mcq")
evaluate_model_frq = evaluate_split(frq_data, "frq")
evaluate_model_cloze = evaluate_split(cloze_data, "cloze")

