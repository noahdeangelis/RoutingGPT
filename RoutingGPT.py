import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer
from datasets import load_dataset

#
# Hyperparameters
#
VOCAB_SIZE = 50257

EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 8
MAX_SEQ_LEN = 512
TOP_K_RATIO = 0.25 # Keep the top 25% of tokens as attention keys/values.
USE_AMP = False # Use AMP for faster training and lower memory usage (if supported by your GPU).
BATCH_SIZE = 4
LEARNING_RATE = 3e-4
MIN_LR = 3e-5
TRAIN_STEPS = 25000
USE_COMPILE = True # torch.compile

#
# Device stuff
#
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

#
# Routing attention
#
class RoutingAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, top_k_ratio=0.25):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.top_k_ratio = top_k_ratio

        self.router = nn.Linear(embed_dim, 1) # The "router" linear layer. [B, T]

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim) # Output projection

    def forward(self, x):
        B, T, C = x.shape

        # num_kv_tokens = max(2, int(T * self.top_k_ratio)) # Calculate the number of tokens to attend to based on the top_k_ratio
        num_kv_tokens = min(T, max(2, int(T * self.top_k_ratio)))
        route_logits = self.router(x).squeeze(-1) # [B, T, 1] -> [B, T]
        _, topk_indices = torch.topk(route_logits, num_kv_tokens, dim=-1) # [B, num_kv_tokens]

        # Project to Q, K, V
        q = self.q_proj(x)
        selected_x = torch.gather(x, 1, topk_indices.unsqueeze(-1).expand(-1, -1, C))
        k = self.k_proj(selected_x)
        v = self.v_proj(selected_x)

        # -> [B, H, T, D]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, num_kv_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, num_kv_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Causal mask
        # Prevent each query from attending to key/value tokens from the future.
        query_positions = torch.arange(T, device=x.device).view(1, 1, T, 1) # [1, 1, T, 1]
        key_positions = (topk_indices.unsqueeze(1).unsqueeze(2))
        causal_mask = (key_positions <= query_positions)

        # Actually compute attention then merge heads and project back to the original embedding dimension.
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)
        context = (context.transpose(1, 2).contiguous().view(B, T, C))
        out = self.out_proj(context)

        return out

#
# Transformer block
#
class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, top_k_ratio=0.25):
        super().__init__()
        self.attn = RoutingAttention(embed_dim, num_heads, top_k_ratio)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
#
# GPT
#
class RoutingGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, top_k_ratio=0.25):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, top_k_ratio) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        self.lm_head.weight = (self.token_embedding.weight) # Weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        positions = torch.arange(T, device=idx.device).unsqueeze(0) # [1, T]
        x = self.token_embedding(idx) + self.positional_embedding(positions) # [B, T, C]

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        logits = self.lm_head(x) # [B, T, vocab_size]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss

#
# Training functions
#
def data_gen(tokenizer, dataset, max_len, batch_size):
    buffer = []
    for item in dataset:
        text = item['text']
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(tokenizer.eos_token_id)
        buffer.extend(tokens)

        required = (max_len + 1) * batch_size
        while len(buffer) >= required:
            batch = []

            for _ in range(batch_size):
                chunk = buffer[:max_len + 1]
                buffer = buffer[max_len + 1:]
                batch.append(chunk)

            batch = torch.tensor(batch, dtype=torch.long)

            x = batch[:, :-1]
            y = batch[:, 1:]

            yield x, y

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#
# Training
#
def train(model, tokenizer, dataset, max_len, batch_size, learning_rate, train_steps):
    model.train() # Make sure the model is in training mode
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_steps,
        eta_min=MIN_LR
    )

    if USE_AMP:
        scaler = torch.amp.GradScaler(DEVICE)

    for step in range(train_steps):
        start_time = time.time()
        x, y = next(dataset)
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        # You can replace logits with _ if you want
        if USE_AMP:
            with torch.amp.autocast(DEVICE):
                logits, loss = model(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        end_time = time.time()
        elapsed_time = end_time - start_time

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Step {step + 1}/{train_steps}, Loss: {loss.item():.4f}, LR: {current_lr:.2e} Time: {elapsed_time:.2f}s")
    print("\nTraining complete!!!!")
    if USE_COMPILE:
        torch.save(model._orig_mod.state_dict(), "model.pth")
    else:
        torch.save(model.state_dict(), "model.pth")

#
# Main
#
def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    print("Loading TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    data_iterator = data_gen(tokenizer, dataset, MAX_SEQ_LEN, BATCH_SIZE)

    model = RoutingGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN, TOP_K_RATIO).to(DEVICE)
    print(f"RoutingGPT parameters: {count_parameters(model) / 1e6:.2f}M")

    if USE_COMPILE:
        model = torch.compile(model)

    train(model, tokenizer, data_iterator, MAX_SEQ_LEN, BATCH_SIZE, LEARNING_RATE, TRAIN_STEPS)

if __name__ == "__main__":
    main()