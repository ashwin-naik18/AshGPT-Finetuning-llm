from config import *
import torch
from model import GPT
from pathlib import Path
from torch.nn.utils import clip_grad_norm_
import time
from datetime import timedelta
from shard_loader import ShardDataset
from tokenizer import enc
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from tokenizer import *


device = "cuda" if torch.cuda.is_available() else "cpu"

enabled = device == "cuda"


def save_checkpoint(
    model, epoch, best_val_loss, optimiser,scaler, scheduler
):
    status = {
        "model" : model.state_dict(),
        "epoch" : epoch,
        "best_loss" : best_val_loss,
        "scaler" : scaler.state_dict(),
        "optimiser" : optimiser.state_dict(),
        "scheduler" : scheduler.state_dict()
    }
    
    torch.save(status, f"{save_dir}/best_model.pt")
    
    print("Best Model Saved successfully..")



def load_checkpoint(model, optimiser, filename, scaler, scheduler):
    checkpoint = torch.load(
        filename,
        map_location=device
    )
    
    model.load_state_dict(
        checkpoint["model"]
    )
    
    optimiser.load_state_dict(
        checkpoint["optimiser"]
    )
    
    scaler.load_state_dict(
        checkpoint["scaler"]
    )
    
    scheduler.load_state_dict(
        checkpoint["scheduler"]
    )
    
    epoch = checkpoint["epoch"]
    
    best_val_loss = checkpoint["best_loss"]
    
    return epoch, best_val_loss
    


def estimate_loss(model, val_dataset: ShardDataset):
    model.eval()
    
    losses = []
    
    for shard_index in val_dataset.get_shard_order():
        
        data = val_dataset.load_shard(
            shard_index.item()
        )
        
        for x, y in val_dataset.create_batches(
            data, 
            BATCH_SIZE
        ):
            x = x.to(device, non_blocking = True)
            y = y.to(device, non_blocking = True)
            
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type = device,
                    enabled= enabled
                ):
                    _, loss = model(x, y)
                    
            losses.append(loss.item())
            
            if (len(losses) >= EVAL_STEPS):
                del data
                model.train()
                return sum(losses) / len(losses)
            
        del data
        
    model.train()
    return sum(losses) / len(losses)
            
     
                

def main():
    
    
    train_dir = Path("/content/openhermes_chunks/train")
    val_dir = Path("/content/openhermes_chunks/val")
    
    train_dataset = ShardDataset(train_dir)
    val_dataset = ShardDataset(val_dir)
    
        
    executor = ThreadPoolExecutor(max_workers=1)
        
    vocab_size = enc.n_vocab
    
    model = GPT(vocab_size, EMBEDDING_DIM, BLOCK_SIZE, NUM_HEAD, NUM_LAYER).to(device)
    
    check_point = torch.load(
        f"{save_dir}/gpt_512.pt",
        map_location=device
    )
    
    model.load_state_dict(
        check_point['model'],
        strict=True
    )
    
    model = model.to(device)
    
    model = torch.compile(model)


    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4
    )
    
    T_Max = EPOCH * TRAIN_STEP_PER_CHUNK * len(train_dataset.shard_files)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_Max)
    
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=enabled
    )   
    
        
    print("Model Loaded successfully..")
        
    start = 0
    best_val_loss = float('inf')


    print("MODEL Parameters : ", model.count_parameters())

        

    model.train()
    
    start_time = time.time()

    for epoch in range(start, EPOCH):
        print(f"\nEpoch {epoch+1}/{EPOCH}")
        
        order = train_dataset.get_shard_order()

        
        future = executor.submit(
            train_dataset.load_shard,
            order[0].item()
        )
        
        for shard_position, shard_index in enumerate(order):
            
            print(
                f"{shard_position + 1} / {len(order)}"
            )
            
            data = future.result()  
            
            if shard_position + 1 < len(order):
                future = executor.submit(
                    train_dataset.load_shard,
                    order[shard_position +  1].item()
                )
                
            data = train_dataset.shuffle_shard(data)
            
            
            for step, (x, y) in enumerate(
                train_dataset.create_batches(data, BATCH_SIZE)
            ): 
                
                if step >= TRAIN_STEP_PER_CHUNK:
                    break
                              
                x = x.to(device, non_blocking = True)
                y = y.to(device, non_blocking = True)      
                
                optimiser.zero_grad(set_to_none=True)
                
                with torch.amp.autocast( device_type = device, enabled = enabled) :
                                    
                    _, loss = model(x, y)
                
                
                scaler.scale(loss).backward()
                
                scaler.unscale_(optimiser)
                
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimiser)

                scaler.update()
                
                scheduler.step()
                
                completed_steps = (
                    epoch * len(order) * TRAIN_STEP_PER_CHUNK
                    + shard_position * TRAIN_STEP_PER_CHUNK
                    + step
                )
                
                if completed_steps % 50 == 0:
                    elapsed = time.time() - start_time

                    time_per_step = elapsed / (completed_steps + 1)
                    
                    total_steps =  EPOCH * TRAIN_STEP_PER_CHUNK * len(order)

                    remaining_steps = total_steps - completed_steps - 1

                    eta_seconds = remaining_steps * time_per_step

                    eta = timedelta(seconds=int(eta_seconds))
                    progress = (completed_steps+1) / T_Max * 100    
                    
                    print("=" * 50)

                    print(f"Global Step: {(completed_steps + 1)}/{T_Max}")
                    print(f"Loss       : {loss.item():.4f}")
                    print(f"LR         : {optimiser.param_groups[0]['lr']:.8f}")
                    print(f"Progress   : {progress:.2f}%")

                    print(f"ETA        : {eta}")

                    print("=" * 50)                    
                     
                    
            del data
            
            
        val_loss = estimate_loss(model, val_dataset)
        
        print(f"Validation Loss : {val_loss:.4f}")                    

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            save_checkpoint(model, epoch + 1, best_val_loss, optimiser, scaler, scheduler)        

    executor.shutdown(wait=True)