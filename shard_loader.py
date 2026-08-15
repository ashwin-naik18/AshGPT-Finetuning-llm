from pathlib import Path
import torch



class ShardDataset:
    
    def __init__(self, shard_dir):
        self.shard_dir = shard_dir
        
        self.shard_files = sorted(
            self.shard_dir.glob("*.pt")
        )
        
        print(f"Found {len(self.shard_files)} shards")
        

    def get_shard_order(self):
        return torch.randperm(len(self.shard_files))
    
    
    def load_shard(self, shard_index):
        shard_path = self.shard_files[shard_index]
        
        data = torch.load(shard_path)
        
        return data
    
    def shuffle_shard(self, data):
        
        indices = torch.randperm(
            len(data['input_ids'])
        )
        
        return {
            "input_ids" : data["input_ids"][indices],
            "labels" : data["labels"][indices]
        }
        
    
    def create_batches(self, data, batch_size):
        
        input_ids = data['input_ids']
        label = data['labels']
        
        for i in range(0, len(input_ids), batch_size):
            input_batch = input_ids[i: i + batch_size]
            label_batch = input_ids[i: i + batch_size]
            
            if len(input_batch) < batch_size:
                break
            
            yield input_batch, label_batch
            
        
    def process_shard(self, shard_index, batch_size):
        
        data = self.load_shard(shard_index)
        
        data = self.shuffle_shard(data)
        
        for input_batches, label_batch in self.create_batches(data, batch_size):
            yield input_batches, label_batch
            