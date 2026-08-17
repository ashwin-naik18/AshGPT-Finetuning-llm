from config import *
import torch
from model import GPT
from pathlib import Path
from torch.nn.utils import clip_grad_norm_
import time
from datetime import timedelta
from dataset import ChunkDataset
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
    


def estimate_loss(model, val_loader):
    model.eval()
    
    losses = torch.zeros(EVAL_STEPS)
    
    
    for i, (x, y) in enumerate(val_loader):
        
        x = x.to(device, non_blocking = True)
        y = y.to(device, non_blocking = True)
        
        if i >= EVAL_STEPS:
            break
        
        with torch.no_grad():
            with torch.amp.autocast( device_type = device, enabled = enabled ):
                
                _, loss = model(x, y)          
        
        losses[i] = loss.item()
        
    model.train()
    
    return losses.mean().item()

def main():
    
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
    
    chunk_files = sorted(
            Path(save_dir).glob("chunk_*.pt"),
            key= lambda p : int(p.stem.split('_')[1])
        )
        
        
    val_data = torch.load(
        f"{save_dir}/val_chunk.pt"
    )
    
    val_dataset = ChunkDataset(
        val_data,
        BLOCK_SIZE
    )
    
    val_dataset_loader =  DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=enabled
    )

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4
    )
    
    T_Max = EPOCH * TRAIN_STEP_PER_CHUNK * len(chunk_files)
    
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
        
        future = executor.submit(
            torch.load,
            chunk_files[0]
        )
        
        for chunk_idx, chunk in enumerate(chunk_files):
            print(
                f"Chunk {chunk_idx+1}/{len(chunk_files)}"
            )
            
            data = future.result()  
            
            if chunk_idx + 1 < len(chunk_files):
                future = executor.submit(
                    torch.load,
                    chunk_files[chunk_idx + 1]
                )    
            
            train_dataset = ChunkDataset(data, BLOCK_SIZE)
                            
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                pin_memory=enabled
            )
                
            
            for step, (x, y) in enumerate(train_dataloader): 
                
                if step >= TRAIN_STEP_PER_CHUNK:
                    break
                              
                x = x.to(device, non_blocking = True)
                y = y.to(device, non_blocking = True)      
                
                
                with torch.amp.autocast(device_type=device, enabled=enabled) :
                                    
                    _, loss = model(x, y)
                
                optimiser.zero_grad(set_to_none=True)
                
                scaler.scale(loss).backward()
                
                scaler.unscale_(optimiser)
                
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimiser)

                scaler.update()
                
                scheduler.step()
                
                completed_steps = (
                    epoch * len(chunk_files) * TRAIN_STEP_PER_CHUNK
                    + chunk_idx * TRAIN_STEP_PER_CHUNK
                    + step
                )
                
                if completed_steps % 50 == 0:
                    elapsed = time.time() - start_time

                    time_per_step = elapsed / (completed_steps + 1)
                    
                    total_steps =  EPOCH * TRAIN_STEP_PER_CHUNK * len(chunk_files)

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
                     
                    
            del train_dataloader
            del train_dataset
            del data
            
            
        val_loss = estimate_loss(model, val_dataset_loader)
        
        print(f"Validation Loss : {val_loss:.4f}")                    

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            save_checkpoint(model, epoch + 1, best_val_loss, optimiser, scaler, scheduler)        

    executor.shutdown(wait=True)