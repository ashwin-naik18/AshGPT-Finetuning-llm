import torch
from model import *
from config import *
from tokenizer import *
import torch.nn.functional as F
from setup import *

new_model = GPT(enc.n_vocab, EMBEDDING_DIM, BLOCK_SIZE, NUM_HEAD, NUM_LAYER)

print(new_model.position_embedding.weight.shape)

check_point = torch.load(
    f"{save_dir}/best_model.pt",
    map_location=device
)

print(check_point.keys())

old_state = check_point["model"]
old_pos = old_state["_orig_mod.position_embedding.weight"]

print(old_pos.shape)

old_pos = old_pos.T.unsqueeze(0)

new_pos = F.interpolate(
    old_pos,
    size=BLOCK_SIZE,
    mode="linear",
    align_corners=True
)

new_pos = new_pos.squeeze(0).T

new_state = {}

old_vocab_size = old_state[
    "_orig_mod.token_embedding.weight"
].shape[0]

for key, value in old_state.items():
    
    key = key.replace("_orig_mod.", "")
    
    if key == 'position_embedding.weight':
        new_state[key] = new_pos
        
    elif key == "token_embedding.weight":
        new_weight = torch.randn(
            enc.n_vocab,
            EMBEDDING_DIM
        ) * 0.02
        
        new_weight[: old_vocab_size] = value
        
        new_state[key] = new_weight
        
    
    elif key == "lm.lm.weight":
        new_weight = torch.randn(
            enc.n_vocab,
            EMBEDDING_DIM
        ) * 0.02
        
        new_weight[:old_vocab_size] = value
        
        new_state[key] = new_weight
        
    elif key == "lm.lm.bias":
        new_bias = torch.zeros(enc.n_vocab)
        
        new_bias[: old_vocab_size] = value
        
        new_state[key] = new_bias
        
    elif key.endswith(".mask"):
        continue
    
    else:
        new_state[key] = value
        
    
new_state['position_embedding.weight'] = new_pos


result = new_model.load_state_dict(
    new_state,
    strict=False
)


torch.save(
    {
        "model" : new_model.state_dict()
    },
    f"{save_dir}/gpt_512.pt"
)