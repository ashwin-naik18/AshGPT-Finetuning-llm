import torch
from pathlib import Path
from datasets import load_dataset
from tokenizer import encode
from config import *
from setup import *

print("Dataset Loading.....")

dataset = load_dataset("teknium/OpenHermes-2.5", split="train")

print("Dataset Loaded....")


def format_dataset(conversation):
    formatted = ""
    
    for message in conversation:
        role = message['from']
        content = message['value'].strip()
        
        if role == "human":
            formatted += f"<|user|>{content}"
            
        elif role == "gpt":
            formatted += f"<|assistant|>{content}<|end|>"
        
        else:
            continue
            
    return encode(formatted)


def split_long_conversations(tokens):
    chunks = []
    
    for i in range(0, len(tokens), BLOCK_SIZE):
        chunk = tokens[i: i + BLOCK_SIZE]
        chunks.append(chunk)
        
    return chunks


def pack_conversations(conversations):
    chunk = []
    current_chunk = []
    
    for conversation in conversations:
        tokens = format_dataset(conversation)
        
        if len(tokens) >= BLOCK_SIZE:
            if current_chunk:
                chunk.append(current_chunk)
                
            c = split_long_conversations(tokens)
            chunk.extend(c)
            continue
        
        
        if len(current_chunk) + len(tokens) <= BLOCK_SIZE:
            current_chunk.extend(tokens)
            
            
        else:
            chunk.append(current_chunk)
            current_chunk = tokens.copy()
            
    if current_chunk:
        chunk.append(current_chunk)
        
    return chunk