import torch
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from model import get_fine_tune_model
from config import get_ds_config, get_lora_config
from tqdm import tqdm
import evaluate
import random
import numpy as np


# Reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Load Config
ds_config = get_ds_config()
lora_config = get_lora_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = ds_config['model_name']

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Load Evaluation Dataset
print("\nLoading evaluation dataset...")
ds = load_dataset(ds_config['dataset_name'], split="train[:1%]")
examples = ds.select(range(100))  # fixed subset for fair comparison

input_texts = []
reference_outputs = []
for example in examples:
    input_text = f"{ds_config['Q_role']}: {example[ds_config['Q_role']]} {ds_config['A_role']}:"
    ref_output = example[ds_config['A_role']]
    input_texts.append(input_text)
    reference_outputs.append(ref_output)

# Load Models
print("\nLoading pretrained model...")
pretrained_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
pretrained_model.eval()

print("\nLoading LoRA fine-tuned model...")
fine_tuned_model = get_fine_tune_model(ds_config=ds_config, lora_config=lora_config).to(device)
fine_tuned_model.eval()

# Evaluation Utilities
def generate_outputs(model, inputs, max_new_tokens=64):
    preds = []
    for text in tqdm(inputs, desc="Generating"):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=ds_config['max_length']).to(device)
        with torch.no_grad():
            output = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,             
                temperature=0.0
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        preds.append(decoded)
    return preds

def compute_perplexity(model, inputs):
    losses = []
    for text in tqdm(inputs, desc="Calculating Perplexity"):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=ds_config['max_length']).to(device)
        labels = enc.input_ids.clone()
        with torch.no_grad():
            output = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels)
        losses.append(output.loss.item())
    return math.exp(sum(losses) / len(losses))

# Evaluation
print("\nEvaluating Pretrained Model...")
pretrain_preds = generate_outputs(pretrained_model, input_texts)
pretrain_ppl = compute_perplexity(pretrained_model, input_texts)

print("\nEvaluating LoRA Fine-tuned Model...")
lora_preds = generate_outputs(fine_tuned_model, input_texts)
lora_ppl = compute_perplexity(fine_tuned_model, input_texts)

# Compute Metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

pretrain_bleu = bleu.compute(predictions=pretrain_preds, references=[[r] for r in reference_outputs])['bleu']
lora_bleu = bleu.compute(predictions=lora_preds, references=[[r] for r in reference_outputs])['bleu']

pretrain_rouge = rouge.compute(predictions=pretrain_preds, references=reference_outputs)['rougeL']
lora_rouge = rouge.compute(predictions=lora_preds, references=reference_outputs)['rougeL']

# Display Results
print("\n===== Performance Comparison =====")
print(f"Pretrained Model:")
print(f"  Perplexity : {pretrain_ppl:.2f}")
print(f"  BLEU       : {pretrain_bleu * 100:.2f}")
print(f"  ROUGE-L    : {pretrain_rouge * 100:.2f}")

print(f"\nLoRA Fine-tuned Model:")
print(f"  Perplexity : {lora_ppl:.2f}")
print(f"  BLEU       : {lora_bleu * 100:.2f}")
print(f"  ROUGE-L    : {lora_rouge * 100:.2f}")
print("==================================")
