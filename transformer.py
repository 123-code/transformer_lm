import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from dataclasses import dataclass


#cargamos los datos
class DataLoader:
    def __init__(self,B,T):
        self.B = B
        self.T = T

        with open('input.txt','r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.current_position = 0

    def next_batch(self):
        B,T = self.B,self.T
        buf = self.tokens[self.current_position: self.current_position+ B*T+1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)

        self.current_position += B*T

        if self.current_position + (B*T + 1) >len(self.tokens):
            self.current_position = 0
        return x,y



#clase de configuracion
@dataclass
class Config:
    embedding_dim:int = 768
    vocab_size:int = 50257
    block_size:int = 1024
    n_head = 12
    n_layer = 12


class GPT(nn.Module):
    def __init__(self,Config):
        super().__init__()
        self.config = Config

        self.wte = nn.Embedding(Config.vocab_size,Config.embedding_dim)
   
   
        self.wpe = nn.Embedding(Config.block_size,Config.embedding_dim)

        self.blocks = nn.ModuleList([TransformerBlock(Config) for _ in range(Config.n_layer)])

        self.ln_f = nn.LayerNorm(Config.embedding_dim)
        self.lm_head = nn.Linear(Config.embedding_dim,Config.vocab_size,bias=False)
        self.apply(self.init_weights)


    def init_weights(self,module):
        if isinstance(module,nn.Linear):
            std = 0.02
            if hasattr(module,'NANOGPT_SCALE_INIT'):
                std *= (2*self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
    
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
        #print(logits)

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
       # print(qkv)

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
        attn = attn.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))
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




block_size = 8
batch_size = 4

train_loader = DataLoader(B=4,T=32)
model = GPT(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model = torch.compile(model)





criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
num_epochs = 100

for epoch in range(num_epochs):
    x,y = train_loader.next_batch()
    x,y = x.to(device),y.to(device)
    optimizer.zero_grad()

    logits = model(x)


    loss = criterion(logits.view(-1,Config.vocab_size),y.view(-1))
   
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        
model_path = "nano_gpt.pth"
torch.save(model.state_dict(),model_path)


def generate(model,idx,max_new_tokens,temperature=1.0,top_k=None):
    for x in range(max_new_tokens):

        idx_cond = idx[:,-model.config.block_size:]
        logits = model(idx_cond)

        logits = logits[:,-1,:]/temperature

        if top_k is not None:
            v,_ = torch.topk(logits,top_k)
            logits[logits<v[:,[-1]]] = -float('Inf')

        probs = F.softmax(logits,dim=-1)
        idx_next = torch.multinomial(probs,num_samples=1)
        idx = torch.cat((idx,idx_next),dim=1)
    return idx
model.load_state_dict(torch.load('nano_gpt.pth'))
model.eval()
encoding = tiktoken.get_encoding('gpt2')
initial_tokens = encoding.encode("This is a test sentence.")
initial_tokens = torch.tensor([initial_tokens],dtype=torch.long).to(device)
generated = generate(model,initial_tokens,100)
print(generated)
decoded_text = encoding.decode(generated[0].tolist())
print(decoded_text)

print(sum(x.numel() for x in model.parameters() if x.requires_grad))
#generated_text = encoding.decode(generated[0].tolist())
