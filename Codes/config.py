# Set various hyperparameters o the model
import torch
from pathlib import Path

# set the LoRA parameters
def get_lora_config():
    return{
        "rank" : 64,
        "alpha" : 128,
        "layer1" : "q_proj",
        "layer2" : "v_proj",
        "layer3" : 'o_proj',
        "lora_a" : "A",
        "lora_b" : "B"
    }

# set the dataset parameters
def get_ds_config():
    return{
        "model_name" : "meta-llama/Llama-3.2-1B",
        "dataset_name" : "Open-Orca/OpenOrca",
        "batch_size" : 4,
        "max_length" : 512,
        "Q_role" : "question",
        "A_role" : "response",
        "lastep_weights_folder" : "LoRA_weights_bf16",
        "best_weights_folder" : "best_LoRA_weights_fp",
        "model_basename" : "fine_tune_LoRA_llama3_bf16",
    }

# set Training parameters
def get_train_config():
    return {
        'lr' : 1e-4,
        'eps' : 1e-9,
        'num_epochs' : 5,
        "experiment_name" : "/home/ubuntu/Llama_proj_v2/runs/Llama3_bf16",
        'mixed_precision' : True,
        "use_fsdp": False,
        'mixed_precision_type1' : torch.float16,
        'mixed_precision_type2' : torch.bfloat16,
        'lr_scheduler': "cosine",
        'warmup_steps': 500,
    }

def get_last_checkpoint_file_path(ds_config, epoch):
    last_weights_folder = ds_config['lastep_weights_folder']
    model_basename = ds_config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.')/last_weights_folder/model_filename)

def get_best_checkpoint_file_path(ds_config):
    best_weights_folder = ds_config['best_weights_folder']
    model_filename = f"best_fine_tune_LoRA_llama3_fp.pt"
    return str(Path('.')/best_weights_folder/model_filename)
