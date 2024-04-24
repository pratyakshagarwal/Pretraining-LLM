## **Echo Model for Sequence Generation**
This repository contains a PyTorch implementation of the **Echo model**, based on the Transformer architecture. The Echo model is designed for sequence generation tasks, such as language modeling, text completion, and dialogue generation.

### **Overview**
The Echo model consists of the following components:

- **CausalSelfAttention**: A module for causal self-attention, used within each block of the Transformer.
- **Mlp**: Multi-Layer Perceptron module used within each block of the Transformer.
- **Block**: Transformer Block module, comprising the CausalSelfAttention and Mlp modules.
- **Echo**: The main model that ties everything together, with token and positional embeddings, multiple Transformer blocks, and a final linear layer for language modeling.

### **Usage**

To use the Echo model, you can follow these steps:

1. Instantiate the Model:
```bash
from model import Echo, EchoConfig

# Define configuration
config = EchoConfig(
    vocab_size=YOUR_VOCAB_SIZE,
    block_size=YOUR_BLOCK_SIZE,
    n_embd=YOUR_EMBEDDING_SIZE,
    n_layer=YOUR_NUM_LAYERS,
    dropout=YOUR_DROPOUT_RATE
)

# Create the Echo model
model = Echo(config)
```

2. Load Pretrained Model (optional):
```bash
model.load_model('path_to_pretrained_model.pth')
```


3. Generate Text
```bash
# Encode the prompt
prompt = "Enter your prompt here"
context = torch.tensor(encode(prompt), dtype=torch.long, device=config.device).view(1, -1)

# Generate new text
generated_text = decode(model.generate(context, max_new_tokens=1000)[0].tolist())
print(generated_text)
```


### **Example**

Let's take an example prompt and generate text using the Echo model:
**prompt:**
```bash
prompt = "Sally Forrest, an actress's wife for parts and categories,who friends Tahi.com:"
```
**Generated Text**
```bash
The script for the upcoming episode of Tahi.com is finalized and ready for publication. Sally Forrest, an actress known for her roles in various categories, including drama and comedy, shares her thoughts on the upcoming season. "I have a deep love for chocolate," she says with a laugh. "It's a season of indulgence, and I can't wait to see what delicious treats the show has in store for us."

As for her character's journey, Sally hints at a dramatic turn. "There's a plot twist that will have viewers on the edge of their seats," she teases. "I can't reveal too much, but let's just say it involves a long-lost relative and a mysterious letter."

Away from the set, Sally enjoys spending time with her husband, who is a well-known director in the industry. "We've been working on a passion project together," she shares. "It's a film that explores the complexities of family and identity. I'm excited to see how audiences respond."

In addition to her acting career, Sally is also passionate about philanthropy. "I believe in using my platform for good," she explains. "I'm involved in several charitable organizations that support causes close to my heart."

As for future projects, Sally is keeping her options open. "I'm always looking for new challenges," she says. "Whether it's a gritty drama or a lighthearted comedy, I'm ready to dive in and bring a character to life."

With her talent and determination, Sally Forrest is sure to continue making a mark in the world of entertainment. Stay tuned for more updates on her exciting journey.

```

### **Training**
The model can also be trained on custom datasets using the provided training script train.py. Simply run the script with the desired configuration and data.

### **Requirements**
- Python 3.x
- PyTorch
- pandas

### **Reference**
This implementation is inspired by the original Transformer model introduced in the paper:

**"Attention Is All You Need"** by Vaswani et al. (2017)
For more details on the Echo model and its components, refer to the source code files model.py and data.py.

### **License**
This project is licensed under the MIT License. Feel free to use and modify it for your needs.
