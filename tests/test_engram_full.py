
import sys
import os
import torch
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.getcwd())

from models.engram import EngramModule, TokenizerCompression
from transformers import AutoTokenizer

@dataclass
class MockConfig:
    d_model: int = 64
    engram_ngrams = [2, 3]
    engram_vocab_size = 1000
    engram_dim = 64
    engram_num_heads = 2

def test_engram_module():
    print("Initializing components...")
    tokenizer_name = "HuggingFaceTB/SmolLM2-135M"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    compression = TokenizerCompression(tokenizer)
    
    config = MockConfig()
    engram = EngramModule(config, compression)
    
    # Fake Inputs
    B, T = 2, 10
    input_ids = torch.randint(0, tokenizer.vocab_size, (B, T))
    hidden_states = torch.randn(B, T, config.d_model)
    
    print("\nRunning Engram Forward Pass...")
    out = engram(hidden_states, input_ids)
    
    print(f"Input Hidden Shape: {hidden_states.shape}")
    print(f"Output Shape: {out.shape}")
    
    assert out.shape == hidden_states.shape, "Output shape mismatch"
    print("PASS: Shapes match")
    
    # Check consistency (Static memory test)
    # If we pass same sequence, retrieval part (before gating depends on hidden) should be same?
    # Gating depends on hidden_states, so output depends on hidden_states.
    # To check static memory, we'd need to mock hidden states or check internal tensors.
    # Let's just run twice with same inputs and expect same output.
    
    out2 = engram(hidden_states, input_ids)
    if torch.allclose(out, out2):
        print("PASS: Deterministic output")
    else:
        print("FAIL: Output not deterministic")

if __name__ == "__main__":
    test_engram_module()
