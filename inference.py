

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from dataclasses import dataclass
import os


@dataclass
class Config:
    embedding_dim: int = 768
    vocab_size: int = 50257 
    block_size: int = 1024   
    n_head: int = 12
    n_layer: int = 12


class MultiheadAttention(nn.Module):
    def __init__(self, mha_config: Config):
        super().__init__()
        assert mha_config.embedding_dim % mha_config.n_head == 0
        self.n_head = mha_config.n_head
        self.embedding_dim = mha_config.embedding_dim

        self.attn = nn.Linear(mha_config.embedding_dim, 3 * mha_config.embedding_dim)
        self.c_proj = nn.Linear(mha_config.embedding_dim, mha_config.embedding_dim)

        self.register_buffer("bias", torch.tril(torch.ones(mha_config.block_size, mha_config.block_size))
                                     .view(1, 1, mha_config.block_size, mha_config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        assert C == self.embedding_dim, f"Input embedding dim ({C}) doesn't match model embedding dim ({self.embedding_dim})"

  
        qkv = self.attn(x)
        q, k, v = qkv.split(self.embedding_dim, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 


        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v 

        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, mlp_config: Config):
        super().__init__()
        self.l1 = nn.Linear(mlp_config.embedding_dim, 4 * mlp_config.embedding_dim)
        self.activation = nn.GELU(approximate='tanh') 
        self.l2 = nn.Linear(4 * mlp_config.embedding_dim, mlp_config.embedding_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, block_config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(block_config.embedding_dim)
        self.attn = MultiheadAttention(block_config)
        self.ln2 = nn.LayerNorm(block_config.embedding_dim)
        self.mlp = MLP(block_config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, gpt_config: Config):
        super().__init__()
        self.config = gpt_config 

        self.wte = nn.Embedding(gpt_config.vocab_size, gpt_config.embedding_dim) 
        self.wpe = nn.Embedding(gpt_config.block_size, gpt_config.embedding_dim) 
        self.blocks = nn.ModuleList([TransformerBlock(gpt_config) for _ in range(gpt_config.n_layer)])
        self.ln_f = nn.LayerNorm(gpt_config.embedding_dim)
        self.lm_head = nn.Linear(gpt_config.embedding_dim, gpt_config.vocab_size, bias=False)

        
        self.wte.weight = self.lm_head.weight


        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
     
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        if idx.dim() == 1: 
            idx = idx.unsqueeze(0)
        B, T = idx.size()

        assert T <= self.config.block_size, \
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}. Input should be truncated."

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) 

        tok_emb = self.wte(idx) 
        pos_emb = self.wpe(pos) 

        x = tok_emb + pos_emb 

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) 
        return logits


def generate_tokens_from_model(model: GPT, tokenizer, device: torch.device, prompt: str, max_new_tokens: int, temperature: float, top_k: int, block_size: int):

    model.eval() 
    

    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    print(f"Prompt: \"{prompt}\"")
    print("Generating: ", end='', flush=True)
    

    generated_sequence_ids = list(idx[0].tolist())

    for _ in range(max_new_tokens):

        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        
        with torch.no_grad():
            logits = model(idx_cond) 
        logits = logits[:, -1, :] / temperature
        
        #
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))

            logits[logits < v[:, [-1]]] = -float('Inf') 
            

        probs = F.softmax(logits, dim=-1) 
        

        idx_next = torch.multinomial(probs, num_samples=1) 
        if idx_next.item() == tokenizer.eot_token:
            print("\n<|endoftext|> token generated.")
            break
            

        idx = torch.cat((idx, idx_next), dim=1) 
        generated_sequence_ids.append(idx_next.item())


        try:
            new_token_str = tokenizer.decode([idx_next.item()])
            print(new_token_str, end='', flush=True)
        except: 
             print("", end='', flush=True)


    print("\n--- End of Generation ---")
    return tokenizer.decode(generated_sequence_ids)


if __name__ == "__main__":

    MODEL_CONFIG = Config(
        embedding_dim=768,
        vocab_size=50257,
        block_size=32,  
        n_head=12,
        n_layer=12
    )

    DEVICE = torch.device('cpu')
    MODEL_PATH = '../Downloads/tinystories.pth' 


    PROMPT_TEXT = "Once upon a time, in a land far away,"
    MAX_NEW_TOKENS = 100
    TEMPERATURE = 0.8
    TOP_K = 50  


    print(f"Using device: {DEVICE}")


    try:
        tokenizer = tiktoken.get_encoding('gpt2')
        print("Tokenizer loaded (gpt2).")
    except Exception as e:
        print(f"Failed to initialize tiktoken tokenizer: {e}")
        print("Please ensure tiktoken is installed correctly (`pip install tiktoken`).")
        exit(1)


    model = GPT(MODEL_CONFIG).to(DEVICE)
    print(f"Model instantiated with block_size: {MODEL_CONFIG.block_size}.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")



    print(f"Attempting to load model state_dict from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please ensure the path is correct and the model file exists.")
        exit(1)

    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        

        if 'model' in state_dict and isinstance(state_dict['model'], dict):
            print("Found 'model' key in state_dict, loading that.")
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
            print("Found 'state_dict' key in state_dict, loading that.")
            state_dict = state_dict['state_dict']
            
        model.load_state_dict(state_dict)
        print("Model state_dict loaded successfully.")
    except RuntimeError as e:
        print(f"RuntimeError loading state_dict: {e}")
        print("This often means a mismatch between the model architecture defined in the script and the one in the checkpoint.")
        print(f"Ensure the Config parameters (especially block_size={MODEL_CONFIG.block_size}, vocab_size, n_layer, n_head, embedding_dim) match the saved model.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        exit(1)

    model.eval() 



    generated_text_output = generate_tokens_from_model(
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        prompt=PROMPT_TEXT,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        block_size=MODEL_CONFIG.block_size
    )
    
    print("\nFull Generated Text:")
    print(generated_text_output)
