# load the pretrained model and define the fine-tuning pre-trained model
import torch
import os
import torch.nn as nn
from transformers import AutoModelForCausalLM
from config import get_best_checkpoint_file_path

# define the LoRA module
# The weights relationship can be expressed as: W' = W + (alpha/rank)AB
# A and B are the LoRA matrices are learnable in the training step

class LoRABlock(nn.Module):
    def __init__(self, original_layer, rank, alpha):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha

        # define matrices A and B
        self.A = nn.Parameter(torch.zeros(original_layer.out_features, rank))
        self.B = nn.Parameter(torch.zeros(original_layer.in_features, rank))

        # Initialize A and B
        nn.init.kaiming_uniform_(self.A, a=5**0.5)  
        nn.init.normal_(self.B, mean=0.0, std=1e-3)

        # define scaling
        self.scaling = alpha / rank

    def forward(self, x):
        lora_updated = (x @ self.B) @ self.A.T * self.scaling
        return self.original_layer(x) + lora_updated


# get the pretrained model
def get_model(ds_config):
    return AutoModelForCausalLM.from_pretrained(ds_config['model_name'])

# define the fine_tune model
def get_fine_tune_model(ds_config, lora_config):
    pretrained_model = get_model(ds_config)
    target_layers = [lora_config['layer1'], lora_config['layer2'], lora_config['layer3']]

    for name, module in pretrained_model.named_modules():
        if any(layer in name for layer in target_layers) and isinstance(module, nn.Linear):
            print(f"Replacing {name} with LoRA")

            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = pretrained_model.get_submodule(parent_name)

            lora = LoRABlock(module, lora_config['rank'], lora_config['alpha'])
            setattr(parent_module, child_name, lora)

    # Verify LoRA modules placed
    for name, module in pretrained_model.named_modules():
        if isinstance(module, LoRABlock):
            print(f"LoRA is placed at {name}")
    
    best_checkpoint_path = get_best_checkpoint_file_path(ds_config)
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        pretrained_model.load_state_dict(checkpoint['lora_state_dict'], strict=False)
    if 'lora_state_dict' in checkpoint:
        pretrained_model.load_state_dict(checkpoint['lora_state_dict'], strict=False)
        print(f"Loaded fine-tuned LoRA weights from {best_checkpoint_path}")
    else:
        print(f"Warning: No lora_state_dict found in checkpoint!")

    return pretrained_model