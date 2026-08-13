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
    t = len(tokens)
    r = t // BLOCK_SIZE
    for i in range(0, r * 512, BLOCK_SIZE):
        chunk = tokens[i: i + BLOCK_SIZE]
        chunks.append(chunk)
        
    leftover = tokens[t-(t % BLOCK_SIZE):] # or simply (block_size * r + 1)
    return chunks, leftover


# def pack_conversations(conversations):
#     chunk = []
#     current_chunk = []
    
#     for conversation in conversations:
#         tokens = format_dataset(conversation)
        
#         if len(tokens) > BLOCK_SIZE:
            
#             if current_chunk:
#                 chunk.append(current_chunk)
#                 current_chunk = []
                
                
#             completed_chunk, leftover = split_long_conversations(tokens)
            
#             chunk.extend(completed_chunk)
            
#             if (len(current_chunk) + len(leftover)) <= BLOCK_SIZE:
#                 current_chunk.extend(leftover)
                
#             else:
#                 chunk.append(current_chunk)
#                 current_chunk = leftover
            
#             continue
        
        
#         if len(current_chunk) + len(tokens) <= BLOCK_SIZE:
#             current_chunk.extend(tokens)
            
            
#         else:
#             chunk.append(current_chunk)
#             current_chunk = tokens.copy()
            
#     if current_chunk:
#         chunk.append(current_chunk)
        
#     return chunk

def pack_conversations(conversations):
    MIN_CHUNK_SIZE = 512
    
    buffer = []

    for conversation in conversations:

        tokens = format_dataset(conversation)

        if not tokens:
            continue

        buffer.extend(tokens)

        while len(buffer) >= BLOCK_SIZE:
            yield buffer[:BLOCK_SIZE]
            buffer = buffer[BLOCK_SIZE:]

    if len(buffer) >= MIN_CHUNK_SIZE:
        yield buffer