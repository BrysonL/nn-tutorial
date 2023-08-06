import torch
import torch.nn as nn
import time
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000 # how many training iterations?
eval_interval = 50 # how often to print the loss?
learning_rate = 3e-4 # how quickly to learn?
device = 'cuda' if torch.cuda.is_available() else 'mps' # MPS is the device for M1 Macs
eval_iters = 200 # how many times to run the validation loop to estimate loss?
n_embd = 384 # embedding dimensions
n_head = 6 # number of attention heads
n_layer = 6 # number of layers in the network
dropout = 0.2 # dropout rate
# ------------

torch.manual_seed(1337) # for reproducibility

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('tiny-shakespeare.txt', 'r', encoding='utf-8') as f: # this is what i named the file
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # turn on eval mode so we don't mess with batch stats
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters): # run eval_iters samples when estimating
            X, Y = get_batch(split)
            logits, loss = model(X, Y) # run the model and record the loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # turn back on training mode
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # each element in the embedding has a key that says "i have this data" (ELI5)
        self.query = nn.Linear(n_embd, head_size, bias=False) # each element in the embedding has a query that says "i am looking for this data" (ELI5)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # fast way to calculate the triangular mask

        self.dropout = nn.Dropout(dropout) # randomly zero out some inputs to prevent overfitting

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")

        # query-key dot product scaled by the square root of head size to make stdev and mean look normal
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        # mask out the upper triangular part of the matrix
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        # exponentiate and normalize to get weights
        # weights are the affinities between the query and the key
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # apply dropout to the weights
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs) instead of (B,T,C) 
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    # multiple heads let the model attend to different parts of the input
    # and also go faster because we can do the computation in parallel

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd) # each head will have head_size outputs, and we need to convert them back to embedding size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    # this apparently makes it easier to track complex relationships

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # blow up inputs
            nn.ReLU(), # non-linearity
            nn.Linear(4 * n_embd, n_embd), # project back to embedding size
            nn.Dropout(dropout), # zero out some inputs
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) # analogous to batch norm, but for the embedding dimension
        self.ln2 = nn.LayerNorm(n_embd) # better because it doesn't create dependence between inputs

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # create embedding table for each char
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # create embedding for each position (1->block_size) i.e. the model will learn the importance of each position in the sequence
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # stack of transformer blocks defined above
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) # final linear layer to convert back to the probability of each char in vocab

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # run the inputs through the network and get logits (probabilities of next token being each item in vocab)
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        # if we're training, compute the loss, otherwise, no need
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # generate text of a given length from a given prompt
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

t1 = time.time() # track how long this takes

model = GPTLanguageModel()
m = model.to(device) # send the model to the device defined above (for me that's the MPS apple silicon)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        t2 = time.time()

        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # print elapsed time (min)
        print(f"epoch time: {(t2 - t1)/60:.2f} min")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

# print how long it took
t2 = time.time()
print(f"took {t2 - t1:.2f} seconds")

### RESULTS ###
# 10M parameters
# train loss 0.8535, val loss 1.5677
# validation loss flattened off much before the 5000th iter (my terminal logs only go back to 3500, but that's got train loss 1.0086, val loss 1.4979)
# took about 3.5hrs on 16GB M1 Pro