import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """
    Causal self-attention module used in Transformer architectures.
    
    Args:
        config (object): Configuration object containing:
            - n_embd (int): Embedding size.
            - n_head (int): Number of attention heads.
            - dropout (float): Dropout probability.
            - bias (bool): Whether to include bias in linear layers.
            - block_size (int): Size of the attention block.
    
    Attributes:
        attn (nn.Linear): Linear layer for computing query, key, and value.
        proj (nn.Linear): Linear layer for projecting attention outputs.
        attn_dropout (nn.Dropout): Dropout layer for attention scores.
        proj_dropout (nn.Dropout): Dropout layer for projected outputs.
        n_head (int): Number of attention heads.
        n_embd (int): Embedding size.
        dropout (float): Dropout probability.
        flash (bool): Flag indicating whether to use Flash Attention.
        bias (torch.Tensor): Causal mask for attention.
    
    Shape:
        - Input: (B, T, C)
        - Output: (B, T, C)
        
        - B: Batch size
        - T: Sequence length
        - C: Embedding dimension
    """

    def __init__(self, config) -> None:
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        
        # Linear layer for computing query, key, and value
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Linear layer for projecting attention outputs
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Check for Flash Attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        # Causal mask for attention
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        """
        Forward pass of the CausalSelfAttention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        B, T, C = x.shape
        q, k, v = self.attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(2, 1).contiguous().view(B, T, C)

        y = self.proj_dropout(self.proj(y))

        return y


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module used in Transformer architectures.
    
    Args:
        config (object): Configuration object containing:
            - n_embd (int): Embedding size.
            - dropout (float): Dropout probability.
    
    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        gelu (nn.GELU): GELU activation function.
        fc2 (nn.Linear): Second fully connected layer.
        dropout (nn.Dropout): Dropout layer.
    
    Shape:
        - Input: (B, T, C)
        - Output: (B, T, C)
        
        - B: Batch size
        - T: Sequence length
        - C: Embedding dimension
    """

    def __init__(self, config) -> None:
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """
        Forward pass of the Mlp module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        fc1_out = self.fc1(x)
        gelu_out = self.gelu(fc1_out)
        fc2_out = self.fc2(gelu_out)
        out = self.dropout(fc2_out)
        return out

class Block(nn.Module):
    """
    Transformer Block module used in Transformer architectures.
    
    Args:
        config (object): Configuration object containing:
            - n_embd (int): Embedding size.
            - dropout (float): Dropout probability.
    
    Attributes:
        attn (CausalSelfAttention): Causal self-attention module.
        mlp (Mlp): MLP module.
        ln1 (nn.LayerNorm): Layer normalization for the first layer.
        ln2 (nn.LayerNorm): Layer normalization for the second layer.
    
    Shape:
        - Input: (B, T, C)
        - Output: (B, T, C)
        
        - B: Batch size
        - T: Sequence length
        - C: Embedding dimension
    """

    def __init__(self, config) -> None:
        super(Block, self).__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = Mlp(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        """
        Forward pass of the Block module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, T, C).
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Echo(nn.Module):
    """
    Echo model based on the Transformer architecture for sequence generation tasks.

    Args:
        config (object): Configuration object containing model parameters:
            - vocab_size (int): Size of the vocabulary.
            - block_size (int): Maximum length of a sequence.
            - n_embd (int): Embedding size.
            - n_layer (int): Number of layers in the Transformer.
            - dropout (float): Dropout probability.
            - bias (bool): Whether to include bias in linear layers.

    Attributes:
        transformer (nn.ModuleDict): Transformer components including:
            - wte (nn.Embedding): Token embedding layer.
            - wpe (nn.Embedding): Positional embedding layer.
            - drop (nn.Dropout): Dropout layer.
            - h (nn.ModuleList): List of Transformer blocks.
            - ln_final (nn.LayerNorm): Layer normalization for final output.
        lm_head (nn.Linear): Linear layer for language model head.

    Shape:
        - Input (Forward): (B, T)
        - Output (Forward): (B, T, vocab_size)
        
        - B: Batch size
        - T: Sequence length
        - vocab_size: Size of the vocabulary

    Example:
        ```python
        config = Config(vocab_size=10000, block_size=128, n_embd=256, n_layer=6, dropout=0.1, bias=True)
        model = Echo(config)
        idx = torch.randint(0, 10000, (1, 128))
        logits, _ = model(idx)
        ```

    """

    def __init__(self, config):
        super(Echo, self).__init__()
        assert config.vocab_size != 0, "Vocabulary size cannot be zero."
        assert config.block_size != 0, "Block size cannot be zero."
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_final = nn.LayerNorm(config.n_embd, bias=config.bias)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('fc2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Report number of parameters
        print("Number of parameters: %.2fM" % (self.num_params() / 1e6,))

    def num_params(self, non_embedding=True):
        """
        Calculate the number of trainable parameters in the model.

        Args:
            non_embedding (bool): Whether to exclude embedding parameters.

        Returns:
            int: Total number of trainable parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """
        Initialize the weights of Linear and Embedding layers.

        Args:
            module (nn.Module): The module to initialize weights.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass of the Echo model.

        Args:
            idx (torch.Tensor): Input tensor of shape (B, T).
            targets (torch.Tensor): Target tensor of shape (B, T) for training.

        Returns:
            torch.Tensor: Logits tensor of shape (B, T, vocab_size).
            torch.Tensor or None: Loss tensor if targets are provided, None otherwise.
        """
        device = idx.device
        b, t = idx.shape
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_final(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens based on a given input sequence.

        Args:
            idx (torch.Tensor): Input tensor of shape (B, T).
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Softmax temperature for sampling.
            top_k (int): Top-k sampling parameter.

        Returns:
            torch.Tensor: Generated tensor of shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long, crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def save_model(self, filepath):
        """
        Save the model's state dict to a file.

        Args:
            filepath (str): Path to save the model.
        """
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath):
        """
        Load the model's state dict from a file.

        Args:
            filepath (str): Path to load the model.
        """
        self.load_state_dict(torch.load(filepath, map_location=torch.device(self.config.device)))
