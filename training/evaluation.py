import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from torch.amp import autocast
from configs.llm_config import BlueberryConfig


def evaluate_model(model: nn.Module, val_loader: DataLoader, config: BlueberryConfig):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= config.eval_steps:
                break

            if isinstance(batch, dict):
                x = batch["input_ids"]
                y = batch["labels"]
                attention_mask = batch.get("attention_mask")
            elif isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    x, attention_mask, y = batch
                elif len(batch) == 2:
                    x, y = batch
                    attention_mask = None
                else:
                    raise ValueError(f"Unexpected batch structure with {len(batch)} elements.")
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            x, y = x.to(device), y.to(device)

            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with autocast('cuda', dtype=torch.bfloat16, enabled=config.use_amp):
                # Dense model evaluation
                logits = model(x)
                # Shift for causal LM: predict next token
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = y[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, config.vocab_size),
                    shift_labels.view(-1)
                )

            # Count tokens correctly (we lose one token per sequence due to shifting)
            num_tokens = shift_labels.numel()
            total_loss += loss.item() * num_tokens
            
            total_tokens += num_tokens

            predictions = shift_logits.argmax(dim=-1)
            total_correct += (predictions == shift_labels).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {
        'val_loss': avg_loss, 
        'val_accuracy': accuracy, 
        'val_perplexity': perplexity
    }
