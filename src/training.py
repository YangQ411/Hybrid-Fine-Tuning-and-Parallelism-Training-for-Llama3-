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

from torch.utils.tensorboard import SummaryWriter
import warnings
from pathlib import Path
import time


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

# Load LoRA weights later (optional)
def load_lora_weights(model, lora_state_dict):
    model_state = model.state_dict()
    model_state.update(lora_state_dict)
    model.load_state_dict(model_state)

# get dataset, build the dataloader
def get_ds(ds_config):
    
    # load the dataset
    raw_ds = load_dataset(ds_config["dataset_name"], split="train")

    # get the tokenizer and build ds for dataloader
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

    # build the dataloader
    train_dataloader = DataLoader(train_ds, batch_size=ds_config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_ds, batch_size=ds_config["batch_size"], shuffle=False, collate_fn=collate_fn)

    print(f"Dataset Loaded: {len(train_ds)} training, {len(val_ds)} validation samples.")
    return train_dataloader, val_dataloader, tokenizer

# validation function
def validation(model, val_dataloader, epoch, tokenizer, loss_fn, device, writer, global_step, use_mix_precision, mixed_type):
     model.eval()
     total_loss = 0.0
     total_tokens_num = 0
     batch_iterator = tqdm(val_dataloader, desc=f"Processing Epoch {epoch:02d}")
     
     with torch.no_grad():
          for batch in batch_iterator:
               with torch.cuda.amp.autocast(enabled=use_mix_precision, dtype=mixed_type):
                    val_input =  batch["input_ids"].to(device)
                    val_label = batch['labels'].to(device)
                    attention_mask_val = batch["attention_mask"].to(device)
                    
                    val_output = model(input_ids=val_input, labels=val_label, attention_mask=attention_mask_val)
                    
                    # compute the loss
                    val_logits = val_output.logits
                    val_logits = val_logits.view(-1, val_logits.size(-1))
                    val_label = val_label.view(-1)
                    val_loss = loss_fn(val_logits, val_label)
                    
               # compute number of tokens
               num_tokens = (val_label != tokenizer.pad_token_id).sum().item()

               total_loss += val_loss.item() * num_tokens
               total_tokens_num += num_tokens

     # calculate the average loss of this epoch          
     aver_loss = total_loss / total_tokens_num
     perplexity = math.exp(aver_loss)
     
     # log the loss and perplexity
     writer.add_scalar('validation loss', aver_loss, global_step)
     writer.add_scalar('perplexity', perplexity, global_step)
     writer.flush()
     print(f"Validation Loss: {aver_loss:.4f}, Perplexity: {perplexity:.2f}")
     return aver_loss
    

def train(model, train_config, lora_config, ds_config):
    # define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device", device)

    # load the dataset
    train_dataloader, val_dataloader, tokenizer = get_ds(ds_config)

    # load the model
    model = model.to(device)

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
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
             input_ids = batch['input_ids'].to(device)
             labels = batch['labels'].to(device)
             attention_mask = batch["attention_mask"].to(device)

             with torch.cuda.amp.autocast(enabled=use_mixed_precision, dtype=BF16):
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
             writer.add_scalar('training loss', loss.item(), global_step)
             writer.flush()
                
                
             global_step += 1

        # log epoch time
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f"[Epoch {epoch}] Time: {epoch_time:.2f} seconds\n")
        writer.add_scalar('epoch_time_seconds', epoch_time, epoch)

        # log the throughput
        throughput_epoch = len(train_dataloader.dataset) / epoch_time
        writer.add_scalar('training_throughput_samples_per_sec', throughput_epoch, epoch)

        # log the GPU memory usage
        # convert bytes ---> GB
        gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)
        writer.add_scalar('training_max_memory_allocated_GB', gpu_memory, epoch)
        # reset for next epoch
        torch.cuda.reset_peak_memory_stats() 

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
                                     mixed_type=BF16
                                     )
        
        # extract LoRA state
        last_lora_state = extract_lora_state_dict(model, lora_config)

        # save the LoRA_only checkpoint in each epoch
        last_sigle_gpu_checkpoint = get_last_checkpoint_file_path(ds_config=ds_config, epoch=epoch)
        Path(last_sigle_gpu_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'lora_state_dict': last_lora_state,
            'global_step': global_step
        }, last_sigle_gpu_checkpoint)
        
        # save the checkpoint of the best performance of LorA
        best_single_gpu_checkpoint = get_best_checkpoint_file_path(ds_config=ds_config)
        Path(best_single_gpu_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        if validation_loss < best_val_loss:
             best_val_loss = validation_loss
             torch.save(
                  {
                       'epoch': epoch,
                        'lora_state_dict': last_lora_state,
                        'global_step': global_step
                  }, best_single_gpu_checkpoint)
             print(f"Best model updated at epoch {epoch:02d}, saved to disk.\n")

    # log the total time
    total_end_time = time.time()
    total_train_time = total_end_time - total_start_time
    print(f"Total Time: {total_train_time}")
    writer.add_scalar('total_training_time', total_train_time)


    print(f"Training completed successfully! Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
     warnings.filterwarnings("ignore")
     lora_config = get_lora_config()
     ds_config = get_ds_config()
     train_config = get_train_config()
     fine_tune_llama3b = get_fine_tune_model(ds_config=ds_config, lora_config=lora_config)
     train(fine_tune_llama3b, train_config=train_config, lora_config=lora_config, ds_config=ds_config)