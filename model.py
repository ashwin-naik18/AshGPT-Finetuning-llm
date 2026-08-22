import torch
import torch.nn as nn
import torch.nn.functional as F 


device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)


class Head(nn.Module):
    def __init__(self, embedding_dim, head_size, block_size):
        super().__init__()
        
        
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )
        
        self.dropout = nn.Dropout(0.1)
        
        
    def forward(self, x):
        B, T, C = x.shape
        
            
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        score = q @ k.transpose(-2, -1)
        
        score = score / (k.shape[-1] ** 0.5)
        
        mask = self.mask[:T, :T]
        
        score = score.masked_fill(mask == 0, float('-inf'))
        
        weights = torch.softmax(score, dim=-1)
        
        weights = self.dropout(weights)
        
        output = weights @ v
        
        return output
    
    
class MultiHead(nn.Module):
    def __init__(self, num_head, head_size, embedding_dim, block_size) :
        super().__init__()
        
        self.heads = nn.ModuleList(
            [Head(embedding_dim, head_size, block_size) for _ in range(num_head)]
        )
        
        self.proj = nn.Linear(num_head * head_size, embedding_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        output = [head(x) for head in self.heads]
        
        output = torch.cat(output, dim=-1) 
        
        output = self.proj(output)  
        
        output = self.dropout(output)
        
        return output       
    

class FFN(nn.Module):
    def __init__(self, embedding_dim) :
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            
            nn.GELU(),
            
            nn.Linear(embedding_dim * 4, embedding_dim),
            
            nn.Dropout(0.1)
        )        
        
    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, embedding_dim, num_head, block_size) :
        super().__init__()
        
        head_size = embedding_dim // num_head
        
        self.sa = MultiHead(num_head, head_size, embedding_dim, block_size)
        
        self.ffwd = FFN(embedding_dim)
        
        self.ln1 = nn.LayerNorm(embedding_dim)
        
        self.ln2 = nn.LayerNorm(embedding_dim)
        
        
    def forward(self, x):
        x  = x +  self.sa(self.ln1(x))
        
        x = x + self.ffwd(self.ln2(x))
        
        return x
        
        
class LMHead(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()

        self.lm = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        output = self.lm(x)
        return output   


class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size, num_head, num_layer) :
        super().__init__()
        self.block_size = block_size
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim) 
        
        self.position_embedding = nn.Embedding(block_size, embedding_dim) 
        
        self.blocks = nn.ModuleList(
            [Block(embedding_dim, num_head, block_size) for _ in range(num_layer)]
        )
                
        self.lm = LMHead(embedding_dim, vocab_size)
        
        self.ln_f = nn.LayerNorm(embedding_dim)
        
        self.apply(self._init_weights)
        
    
    def count_parameters(self):
        return sum(
            p.numel() 
            for p in self.parameters()
            if p.requires_grad
        )
        
    
    def _init_weights(self, module):
        
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean = 0.0,
                std = 0.02
            )
            
            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):
            
            nn.init.normal_(
                module.weight,
                mean=0.0, 
                std=0.02    
            )
                
        
    def forward(self, x, y = None):
        B, T = x.shape
        
        token = self.token_embedding(x)
        
        positions = torch.arange(T, device=x.device)
        
        pos = self.position_embedding(positions)
        
        output = token + pos
        
        for bloc in self.blocks:
            output = bloc(output)
            
        output = self.ln_f(output)
                
        logits = self.lm(output)
        
        loss = None
        
        if y is not None:
            
            logits = logits[:, :-1, :]
            y = y[:, 1:]
            
            B, T, C = logits.shape
            
            logits = logits.reshape(B * T , C)
            y = y.reshape(B * T)
            
            loss = F.cross_entropy(logits, y)
            
            return logits, loss

        return logits, loss