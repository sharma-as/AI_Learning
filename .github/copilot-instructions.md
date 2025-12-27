# AI Coding Instructions for AI_Learning

This repository contains deep learning implementations and educational explorations of transformer architectures, tokenizers, and fine-tuning techniques.

## Project Overview

This is an **educational AI learning repository** documenting implementations of:
- **Attention mechanisms** (scaled dot-product self-attention)
- **Language models** (GPT-1, GPT-2 from scratch)
- **Tokenization** (Byte-Pair Encoding/BPE)
- **Fine-tuning methods** (LoRA, QLoRA)
- **Advanced techniques** (Reflexion, ToolFormer)

## Key Architecture Patterns

### 1. **PyTorch Module Structure**
- All neural network components inherit from `nn.Module`
- Use `@dataclass` for config objects (see `GPTConfig` in [day2_gpt1.ipynb](day2_gpt1.ipynb))
- Implement `forward()` method with proper tensor shape handling
- Register buffers for static tensors (e.g., causal masks): `self.register_buffer("mask", mask)`

### 2. **Transformer Building Blocks**
The codebase follows a bottom-up implementation strategy:

1. **Attention layers** ([day1_attention_tinytransformer.py](day1_attention_tinytransformer.py)): Implements scaled dot-product attention with query/key/value projections
2. **Causal masking**: Used in GPT models to prevent attending to future tokens (see `CausalSelfAttention` in [day2_gpt1.ipynb](day2_gpt1.ipynb))
3. **Block structure**: Combines attention + MLP with LayerNorm (pre-norm architecture)
4. **Stack assembly**: Transformer stacks multiple blocks with embeddings

### 3. **Tokenization Approach**
- **Byte-level tokenization**: Uses UTF-8 byte encoding + special tokens (EOS=256, CLS=257)
- **BPE (Byte-Pair Encoding)**: Implements merge operations to build subword vocabularies ([Day16_BPE_Tokenizer.ipynb](Day16_BPE_Tokenizer.ipynb))
- Custom `ByteTokenizer` class handles encode/decode with error handling

## Development Workflows

### Running Notebooks
```bash
# Activate environment
source .venv/bin/activate

# Run individual cell in VS Code (use Run Cell button) or execute full notebook
jupyter notebook <filename>.ipynb
```

### Training & Checkpoints
- Pre-trained model checkpoints: `gpt1_tiny_pretrain.pt`, `gpt1_tiny_finetune.pt`
- Models use `torch.save()` / `torch.load()` (standard PyTorch serialization)
- Configuration + state_dict pattern (config in dataclass, weights in checkpoint)

### Dependencies
Install from [requirements.txt](requirements.txt):
```bash
pip install -r requirements.txt
```
Core packages: PyTorch, transformers, huggingface_hub, langchain

## Project-Specific Conventions

### Tensor Conventions
- **Shapes**: Use `(batch, seq_len, embed_dim)` consistently
- **Attention outputs**: Return tuple `(output, weights)` for debugging
- **Multi-head reshape**: `(B, T, C) → (B, T, n_head, head_dim) → (B, n_head, T, head_dim)`

### Masking Patterns
- Create causal masks once, register as buffer: `self.register_buffer("mask", mask)`
- Apply with `masked_fill(condition==0, float('-inf'))` before softmax
- Example: [CausalSelfAttention](day2_gpt1.ipynb#L51-L80)

### Configuration as Dataclass
- All hyperparameters in `@dataclass` (e.g., `block_size`, `n_layer`, `n_head`, `n_embd`, `dropout`)
- Pass config object to all module constructors
- Modify experiments by creating new config instances

### Error Handling in Tokenization
- Use `errors='ignore'` in encode/decode to handle invalid UTF-8
- `ByteTokenizer` gracefully returns empty string on decode failures
- BPE merges use regex patterns for robustness ([Day16_BPE_Tokenizer.ipynb](Day16_BPE_Tokenizer.ipynb#L8-L50))

## Cross-Component Communication

1. **Tokenizer → Model**: Text → token IDs (vocab_size must match model)
2. **Model → Output**: Logits over vocab (shape: `(batch, seq_len, vocab_size)`)
3. **Training loop pattern**: 
   - Tokenize text → IDs
   - Create batches with causal masking
   - Forward pass → loss
   - Backward + optimizer step

## When Adding Features

- Add tokenization utilities to [Day16_BPE_Tokenizer.ipynb](Day16_BPE_Tokenizer.ipynb)
- Extend attention in [day1_attention_tinytransformer.py](day1_attention_tinytransformer.py) or as new notebook
- Use existing `GPTConfig` pattern for hyperparameter management
- Always test tensor shapes with `assert x.shape == expected_shape`
- Document causal masking behavior when modifying attention

