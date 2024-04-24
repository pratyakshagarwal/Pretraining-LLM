import torch
import textwrap
from model import Echo
from data import get_batch, itos, stoi, encode, decode
from dataclasses import dataclass

@dataclass
class EchoConfig:
    batch_size: int = 64
    block_size: int = 256
    vocab_size: int =  len(itos) # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    lr = 3e-4
    eval_iters = 200
    eval_interval = 500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    max_iters = 6000


@torch.no_grad()
def estimate_loss():
    """
    Estimate the loss on training and validation data using the model.

    Returns:
        dict: Dictionary containing the mean loss for 'train' and 'val' splits.
    """
    out = {}  # Dictionary to store the results
    model.eval()  # Set the model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)  # Placeholder for losses
        for k in range(config.eval_iters):
            # Get a batch of data
            xb, yb = get_batch(split)
            # Calculate logits and loss
            logits, loss = model(xb, yb)
            # Store the loss in the losses tensor
            losses[k] = loss.item()
        # Compute the mean loss for the split
        out[split] = losses.mean()
    model.train()  # Set the model back to training mode
    return out



config = EchoConfig()
model = Echo(config).to(config.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

# train step i already trained the model 
# for iter in range(1, config.max_iters+1):
#     # Evaluate and print loss on training and validation data at specified intervals
#     if iter % config.eval_interval == 0:
#         losses = estimate_loss()
#         print(f'step: {iter} train loss: {losses["train"]:.4f} val loss: {losses["val"]:.4f}')
        
#     # Save model checkpoints at specified intervals
#     if iter % 3000 == 0:
#         model.save_model(f'chatbot_model_pretrained{iter}.pth')
        
#     # Get a batch of training data
#     xb, yb = get_batch('train')

#     # Forward pass
#     logits, loss = model(xb, yb)

#     # Backward pass and optimization
#     optimizer.zero_grad(set_to_none=True)  # Clear gradients
#     loss.backward()  # Backpropagation
#     optimizer.step()  # Optimization step

model.load_model('chatbot_model_pretrained6000.pth')

prompt = input("Enter the query: ")
context = torch.tensor(encode(prompt), dtype=torch.long, device=config.device).view(1, -1)
print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))