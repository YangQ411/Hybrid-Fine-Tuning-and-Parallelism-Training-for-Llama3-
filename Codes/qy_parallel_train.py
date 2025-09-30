# training
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from tqdm import tqdm
import math

from model import get_fine_tune_model
from config import get_lora_config, get_ds_config, get_train_config, get_last_checkpoint_file_path, get_best_checkpoint_file_path
from dataset import BuildDataset, build_tokenizer

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp import ShardingStrategy


from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import warnings
import time
import traceback
import sys

def collate_fn(batch):
    output = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            output[key] = torch.stack(values)
        elif isinstance(values[0], list):
            output[key] = torch.tensor(values)
        else:
            raise ValueError(f"Unsupported type for collate: {type(values[0])}")
    return output

# set weights
def weights_setting(model, lora_config):
    # freeze all param in the original_pretrained model
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze LoRA weights
    for name, param in model.named_parameters():
        if lora_config['lora_a'] in name or lora_config['lora_b'] in name:
            param.requires_grad=True

    # print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters (LoRA only): {trainable_params}")

# save the LorA-only parameters
def extract_lora_state_dict(model, lora_config):
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad and (lora_config['lora_a'] in name or lora_config['lora_b'] in name):
            lora_state_dict[name] = param.detach().cpu()
    return lora_state_dict

# get dataset, build the dataloader
def get_ds(ds_config, rank, world_size):
     # load dataset
     raw_ds = load_dataset(ds_config["dataset_name"], split="train[:30%]")

     # load tokenizer
     tokenizer = build_tokenizer(ds_config)
     dataset = BuildDataset(
          raw_dataset=raw_ds,
          tokenizer_q=tokenizer,
          tokenizer_as=tokenizer,
          role_user=ds_config["Q_role"],
          role_ass=ds_config["A_role"],
          max_length=ds_config["max_length"]
     )

     # split the train and valdation
     train_size = int(0.9 * len(dataset))
     val_size = len(dataset) - train_size
     train_ds, val_ds = random_split(dataset, [train_size, val_size])

     # Build the dataloader
     train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
     val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
     
     train_dataloader = DataLoader(train_ds, batch_size=ds_config['batch_size'], sampler=train_sampler, drop_last=True, num_workers=4, collate_fn=collate_fn)
     val_dataloader = DataLoader(val_ds, batch_size=ds_config['batch_size'], sampler=val_sampler, drop_last=False, num_workers=4, collate_fn=collate_fn )

     print(f"Dataset Loaded:  {len(train_ds)} training samples, {len(val_ds)} validation samples.")

     # return dataloader and tokenizer
     return train_dataloader, val_dataloader, tokenizer, train_sampler

# validation function
def validation(model, val_dataloader, epoch, tokenizer, loss_fn, device, writer, global_step, use_mix_precision, mixed_type):
    model.eval()
    
    total_loss = torch.tensor(0.0, device=device)
    total_tokens_num = torch.tensor(0.0, device=device)
    
    batch_iterator = tqdm(val_dataloader, desc=f"Processing Epoch {epoch:02d}")

    with torch.no_grad():
        for batch in batch_iterator:
            with torch.cuda.amp.autocast(enabled=use_mix_precision, dtype=mixed_type):
                val_input = batch['input_ids'].to(device)
                val_label = batch['labels'].to(device)
                attention_mask_val = batch['attention_mask'].to(device)

                val_output = model(input_ids=val_input, labels=val_label, attention_mask=attention_mask_val)

                # compute the loss
                val_logits = val_output.logits
                val_logits = val_logits.view(-1, val_logits.size(-1))
                val_label = val_label.view(-1)
                val_loss = loss_fn(val_logits, val_label)

                # compute number of tokens
                num_tokens = (val_label != tokenizer.pad_token_id).sum()

                total_loss += val_loss * num_tokens
                total_tokens_num += num_tokens

    # all_reduce only can be used for Tensor
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tokens_num, op=dist.ReduceOp.SUM)

    if dist.get_rank() == 0:
        aver_loss = total_loss.item() / total_tokens_num.item()
        perplexity = math.exp(aver_loss)
        writer.add_scalar('validation loss', aver_loss, global_step)
        writer.add_scalar('perplexity', perplexity, global_step)
        writer.flush()
        print(f"Validation Loss: {aver_loss:.4f}, Perplexity: {perplexity:.2f}")
        return aver_loss

    return None

def custom_wrap_policy(module, recurse, nonwrapped_numel):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
    if isinstance(module, LlamaDecoderLayer):
        return size_based_auto_wrap_policy(
            module=module,
            recurse=recurse,
            nonwrapped_numel=nonwrapped_numel,
            min_num_params=1e7
        )
    
    # Wrap embedding layer
    if isinstance(module, torch.nn.Embedding):
         return True
    
    return False  
    

def train(model, train_config, lora_config, ds_config):
    dist.init_process_group('nccl')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)

    # load the dataset
    train_dataloader, val_dataloader, tokenizer, train_sampler = get_ds(ds_config, rank, dist.get_world_size())
    
    torch.cuda.empty_cache()

    # load the model
    model = model.to(device)

    if train_config.get("use_fsdp", False):
         # Setup FSDP config         
         fsdp_config = {
              "cpu_offload": CPUOffload(offload_params=False),
              "use_orig_params": True,
              "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
              "sharding_strategy": ShardingStrategy.FULL_SHARD,
         }

         def fsdp_wrapper(module, **kwargs):
              return FSDP(module, device_id=device, auto_wrap_policy=custom_wrap_policy, **fsdp_config)
         
         # FSDP wrapping
         with enable_wrap(wrapper_cls=fsdp_wrapper):
              model = wrap(model)
              
         if dist.get_rank() == 0:
              print("[INFO] Model wrapped with FullyShardedDataParallel (FSDP)")
    else:
         # DDP wrapping
         model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
         if dist.get_rank() == 0:
              print("[INFO] Model wrapped with DistributedDataParallel (DDP)")

    # set weights
    weights_setting(model, lora_config)

    # Tensorboard to visualize the loss the graphic chart
    writer = SummaryWriter(train_config['experiment_name'])

    # define optimizer, and only LoAR parameters are trainable in the training
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=train_config['lr'], eps=train_config['eps'])

    scheduler = None
    if train_config.get("lr_scheduler", None) == "cosine":
     from torch.optim.lr_scheduler import CosineAnnealingLR
     scheduler = CosineAnnealingLR(
          optimizer,
          T_max=train_config['num_epochs'], 
          eta_min=1e-6  
     )

    # define loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1).to(device)

    # control using the mixed precision
    use_mixed_precision = train_config['mixed_precision']
    FP16 = train_config['mixed_precision_type1']
    BF16 = train_config['mixed_precision_type2']

    # initial state that can help us reset the model
    initial_epoch = 0
    global_step = 0
    scaler = torch.amp.GradScaler(enabled=use_mixed_precision)
    best_val_loss = float('inf')

    # training loop
    total_start_time = time.time()
    for epoch in range(initial_epoch, train_config['num_epochs']):
        epoch_start_time = time.time()
        model.train()
        train_sampler.set_epoch(epoch)
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
          input_ids = batch['input_ids'].to(device)
          labels = batch['labels'].to(device)
          attention_mask = batch["attention_mask"].to(device)

          with torch.cuda.amp.autocast(enabled=use_mixed_precision, dtype=FP16):
                     # forward propagation
                     output = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
                     logits = output.logits
                     logits = logits.view(-1, logits.size(-1))
                     labels = labels.view(-1)
                     loss = loss_fn(logits, labels)

          # backward propagation
          optimizer.zero_grad()
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()
          if scheduler:
                 scheduler.step()

          batch_iterator.set_postfix({f"loss" : f"{loss.item():6.3f}"})
          # log the loss
          if dist.get_rank() == 0:
               writer.add_scalar('training loss', loss.item(), global_step)
               writer.flush()
                
                
          global_step += 1

        # log epoch time
        if rank == 0:
             epoch_end_time = time.time()
             epoch_time = epoch_end_time - epoch_start_time
             print(f"[Epoch {epoch}] Time: {epoch_time:.2f} seconds\n")
             writer.add_scalar('epoch_time_seconds', epoch_time, epoch)
             
             # log the throughput
             throughput_epoch = len(train_dataloader.dataset) / epoch_time
             writer.add_scalar('training_throughput_samples_per_sec', throughput_epoch, epoch)
             
             # log the GPU memory usage
             # # convert bytes ---> GB
             
             gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
             writer.add_scalar('training_max_memory_allocated_GB', gpu_memory, epoch)
             # reset for next epoch
             torch.cuda.reset_peak_memory_stats() 

        if rank == 0:
             print(f"Running validation at epoch {epoch:02d}...")

        # validation
        validation_loss = validation(model=model, 
                                     val_dataloader=val_dataloader, 
                                     epoch=epoch,
                                     tokenizer=tokenizer,
                                     loss_fn=loss_fn, 
                                     device=device,
                                     writer=writer,
                                     global_step=global_step,
                                     use_mix_precision=use_mixed_precision,
                                     mixed_type=FP16
                                     )

        torch.cuda.empty_cache()
        
        # save the last checkpoint in each epoch
        if rank == 0:
              # extract LoRA state
              last_lora_state = extract_lora_state_dict(model, lora_config)
              last_sigle_gpu_checkpoint = get_last_checkpoint_file_path(ds_config=ds_config, epoch=epoch)
              Path(last_sigle_gpu_checkpoint).parent.mkdir(parents=True, exist_ok=True)
              torch.save({
                   'epoch': epoch,
                   'lora_state_dict': last_lora_state,
                   'global_step': global_step
                   }, last_sigle_gpu_checkpoint)
        
        # save the checkpoint of the best performance model
        if rank == 0 and validation_loss < best_val_loss:
            last_lora_state = extract_lora_state_dict(model, lora_config)
            best_val_loss = validation_loss
            best_single_gpu_checkpoint = get_best_checkpoint_file_path(ds_config=ds_config)
            Path(best_single_gpu_checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                  {
                       'epoch': epoch,
                        'lora_state_dict': last_lora_state,
                        'global_step': global_step
                  }, best_single_gpu_checkpoint)
            print(f"Best model updated at epoch {epoch:02d}, saved to disk.\n")

    if dist.get_rank() == 0:
        total_end_time = time.time()
        total_train_time = total_end_time - total_start_time
        print(f"Total Time: {total_train_time}")
        writer.add_scalar('total_training_time', total_train_time)

    print(f"Training completed successfully! Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
     warnings.filterwarnings("ignore")
     try:
          lora_config = get_lora_config()
          ds_config = get_ds_config()
          train_config = get_train_config()
          model = get_fine_tune_model(ds_config=ds_config, lora_config=lora_config)
          train(model, train_config=train_config, lora_config=lora_config, ds_config=ds_config)
     except Exception as e:
          print(f"[RANK {os.environ.get('LOCAL_RANK', '?')}] Caught Exception:")
          traceback.print_exc()
          sys.exit(1)