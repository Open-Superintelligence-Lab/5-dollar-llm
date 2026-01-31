import torch
import torch.nn as nn
from configs.llm_config import BlueberryConfig
from models.llm import MinimalLLM

def test_hyper_connections():
    print("Testing Hyper-connections...")
    
    # Configure model with hyper-connections
    config = BlueberryConfig(
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=64,
        use_hyper_connections=True,
        hyper_rate=4,
        hyper_dynamic=True,
        compile_model=False
    )
    
    model = MinimalLLM(config)
    batch_size = 2
    seq_len = 16
    
    # Create dummy input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    print("Running forward pass...")
    logits = model(input_ids)
    
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Unexpected logits shape"
    
    # Backward pass
    print("Running backward pass...")
    loss = logits.mean()
    loss.backward()
    
    # Check if hyper-connection parameters have gradients
    found_hc_grad = False
    for name, param in model.named_parameters():
        if "hc" in name and param.grad is not None:
            found_hc_grad = True
            break
            
    assert found_hc_grad, "No gradients found for hyper-connection parameters"
    print("Success! Hyper-connections forward and backward passes work as expected.")

if __name__ == "__main__":
    test_hyper_connections()
