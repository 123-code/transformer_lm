import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from dataclasses import dataclass

#clase de configuracion
@dataclass
class Config:
    embedding_dim:int = 768
    vocab_size:int = 50257
    block_size:int = 1024
    n_head = 12
    n_layer = 12


sample_text = "This is an example sentence. This sentence is used to train the GPT model"
class GPT(nn.Module):
    def __init__(self,Config):
        super().__init__()
        self.config = Config

        self.wte = nn.Embedding(Config.vocab_size,Config.embedding_dim)
        self.wpe = nn.Embedding(Config.block_size,Config.embedding_dim)

        self.blocks = nn.ModuleList([TransformerBlock(Config) for _ in range(Config.n_layer)])

        self.ln_f = nn.LayerNorm(Config.embedding_dim)
        self.lm_head = nn.Linear(Config.embedding_dim,Config.vocab_size,bias=False)

    def forward(self,idx):
        B,T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0,T,dtype=torch.long,device=idx.device)
        positional_embedding = self.wpe(pos)
        token_embedding = self.wte(idx)
        x = token_embedding + positional_embedding

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits 
    

class MultiheadAttention(nn.Module):
    def __init__(self,Config):
        super().__init__()
        #se añade una capa lineal, por que necesitamos que los input embeddings aprendadn funciones no lineales
        self.n_head = Config.n_head
    
        self.attn = nn.Linear(Config.embedding_dim,3*Config.embedding_dim)
        self.c_proj = nn.Linear(Config.embedding_dim,Config.embedding_dim)
        self.register_buffer("bias",torch.tril(torch.ones(Config.block_size,Config.block_size)).
                             view(1,1,Config.block_size,Config.block_size))
    def forward(self,x):
        # b es el numero de secuencias que se procesan al mismo tiempo, 
        # T es la dimension( en tokens) de cada una de las secuencias 
        # C es la cantidad de vectores con los que se representa a cada token


        B,T,C = x.size()
        qkv = self.attn(x)
        #dividimos al tensor en los tres vectores q,k,v de size embedding_dim
        # el .transpose se usa para poder 
        q,k,v = qkv.split(Config.embedding_dim,dim=2)
        #reshapings para multihead attention
        # con.view, cambiamos la forma del tensor para que se divida en el numero de attention heads, y cada attention head se divida en c//n_head B y T se mantienen iguales
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)

        attn = (q@k.transpose(2,-1)) * (1.0/math.sqrt(k.size(-1)))
        #aplicar una mascara para evitar que el modelo "mire hacia adelante"
        attn = attn.masked_fill(self.bias[:,:,:T,T]==0,float('-inf'))
        attn = F.softmax(attn,dim=-1)
        y = attn @ v
        # despues de calcular la atencion multicabeza, debemos reorganizar las llos resultados en una sola operacion
        #luego con .view(B,T,C) fusionamos las cabezas de atencion en un solo embedding. combinandolas en una sola dimension C
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.c_proj(y)
        return y
    

#clase de red neuronal
class MLP(nn.Module):
    def __init__(self,Config):
        super().__init__()
        self.l1 = nn.Linear(Config.embedding_dim,4*Config.embedding_dim)
        self.activation  = nn.GELU(approximate='tanh')
        self.l2 = nn.Linear(4*Config.embedding_dim,Config.embedding_dim)

    def forward(self,x):
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,Config):
        super().__init__()
        #capa de normalizacion antes de la atencion
        self.ln1 = nn.LayerNorm(Config.embedding_dim)
        self.attn = MultiheadAttention(Config)
        self.ln_2 = nn.LayerNorm(Config.embedding_dim)
        self.mlp = MLP(Config)
    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln_2(x))
        return x





encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode(sample_text)
tokens = torch.tensor(tokens,dtype=torch.long)

block_size = 8
batch_size = 4

def get_batch(tokens, block_size, batch_size):
    # Seleccionar índices aleatorios para los lotes
    ix = torch.randint(len(tokens) - block_size, (batch_size,))
    x = torch.stack([tokens[i:i+block_size] for i in ix])
    y = torch.stack([tokens[i+1:i+block_size+1] for i in ix])
    return x, y

model = GPT(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
num_epochs = 1000

for epoch in range(num_epochs):
    x,y = get_batch(tokens,block_size,batch_size)
    x,y = x.to(device),y.to(device)

    logits = model(x)
    print(logits)

    loss = criterion(logits.view(-1,Config.vocab_size),y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        