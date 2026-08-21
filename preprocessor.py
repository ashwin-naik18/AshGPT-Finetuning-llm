from google.colab import drive

drive.mount("/content/drive")

import json
from pathlib import Path

import torch
from datasets import load_dataset

from tokenizer import encode, EOT_ID
from config import *


print("Dataset Loading.....")

dataset = load_dataset(DATASET_NAME, split="train")

print(f"Dataset Loaded.... ({len(dataset)} conversations)")



def format_conversation(messages):
    
    input_ids = []
    labels = []

    for message in messages:
        role = message.get("from")
        content = message.get("value", "").strip()

        if not content:
            continue

        if role in ("human", "system"):
            tokens = encode(f"<|user|>{content}")
            input_ids.extend(tokens)
            labels.extend([-100] * len(tokens))

        elif role == "gpt":
            tokens = encode(f"<|assistant|>{content}<|end|>")
            input_ids.extend(tokens)
            labels.extend(tokens)

        else:
            continue

    if not input_ids:
        return input_ids, labels

    input_ids.append(EOT_ID)
    labels.append(-100)

    assert len(input_ids) == len(labels)
    return input_ids, labels



def process_split(
    hf_split,
    output_dir,
    split_name,
    block_size=BLOCK_SIZE,
    shard_size=SHARD_SIZE,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_buffer, label_buffer = [], []
    
    shard_inputs, shard_labels = [], []
    
    shard_index = 0
    total_blocks = 0
    manifest = []

    total_conversations = 0
    total_tokens = 0
    total_trainable = 0
    long_conversations = 0
    min_length = None
    max_length = 0


    def flush_shard():
        
        nonlocal shard_inputs, shard_labels, shard_index
        
        if not shard_inputs:
            return
        
        filename = f"{split_name}_shard_{shard_index:05d}.pt"
        path = output_dir / filename
        torch.save(
            {
                "input_ids": torch.tensor(shard_inputs, dtype=torch.long),
                "labels": torch.tensor(shard_labels, dtype=torch.long),
            },
            path,
        )
        manifest.append({"file": filename, "blocks": len(shard_inputs)})
        print(f"[{split_name}] Saved {filename} ({len(shard_inputs)} blocks)")
        shard_inputs, shard_labels = [], []
        shard_index += 1

    for example in hf_split:
        input_ids, labels = format_conversation(example["conversations"])

        if not input_ids:
            continue

        length = len(input_ids)
        
        total_conversations += 1
        total_tokens += length
        total_trainable += sum(l != -100 for l in labels)
        
        if length > block_size:
            long_conversations += 1
            
        min_length = length if min_length is None else min(min_length, length)
        max_length = max(max_length, length)


        input_buffer.extend(input_ids)
        label_buffer.extend(labels)

        while len(input_buffer) >= block_size:
            input_chunk = input_buffer[:block_size]
            label_chunk = label_buffer[:block_size]
            input_buffer = input_buffer[block_size:]
            label_buffer = label_buffer[block_size:]

            if sum(l != -100 for l in label_chunk) < MIN_TRAINABLE_TOKENS:
                continue

            shard_inputs.append(input_chunk)
            shard_labels.append(label_chunk)
            total_blocks += 1

            if len(shard_inputs) >= shard_size:
                flush_shard()

    flush_shard()  

    assistant_pct = (total_trainable / total_tokens * 100) if total_tokens else 0

    stats = {
        "conversations": total_conversations,
        "total_tokens": total_tokens,
        "trainable_tokens": total_trainable,
        "assistant_token_percentage": assistant_pct,
        "long_conversations": long_conversations,
        "min_length": min_length,
        "max_length": max_length,
    }

    return {
        "split": split_name,
        "shards": shard_index,
        "blocks": total_blocks,
        "manifest": manifest,
        "statistics": stats,
    }



def main():
    
    print("\nCreating train/validation split...")
    
    split_dataset = dataset.train_test_split(
        test_size=1 - TRAIN_RATIO, seed=SEED, shuffle=True
    )
    
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    print(f"Train conversations: {len(train_dataset)}")
    print(f"Validation conversations: {len(val_dataset)}")

    output_dir = Path("/content/drive/MyDrive/openhermes_chunks")
    
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nProcessing TRAIN dataset...")
    
    train_result = process_split(
        train_dataset,
        output_dir / "train",
        split_name="train",
        block_size=BLOCK_SIZE,
        shard_size=SHARD_SIZE,
    )

    print("\nProcessing VALIDATION dataset...")
    val_result = process_split(
        val_dataset,
        output_dir / "val",
        split_name="val",
        block_size=BLOCK_SIZE,
        shard_size=SHARD_SIZE,
    )

    for k, v in train_result["statistics"].items():
        print(f"{k}: {v}")

    for k, v in val_result["statistics"].items():
        print(f"{k}: {v}")

    metadata = {
        "dataset": DATASET_NAME,
        "block_size": BLOCK_SIZE,
        "train_ratio": TRAIN_RATIO,
        "seed": SEED,
        "shard_size": SHARD_SIZE,
        "train_result": train_result,
        "validation_result": val_result,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


    print(f"Train blocks: {train_result['blocks']}")
    print(f"Validation blocks: {val_result['blocks']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()