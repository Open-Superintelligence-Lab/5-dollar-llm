import torch
import torch.nn.functional as F
import time
from configs.llm_config import BlueberryConfig
from models.llm import MinimalLLM
import pandas as pd

def measure_memory(config_class, name):
    print(f"\n📏 Measuring GPU usage for: {name}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    config = config_class()
    # Ensure standard seq len for comparison
    config.max_seq_len = 2048
    config.use_amp = True
    
    device = torch.device("cuda")
    model = MinimalLLM(config).to(device)
    
    # Simple optimizer setup for memory tracking
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Use AMP for realistic measurement
    scaler = torch.amp.GradScaler('cuda')
    
    # Run 5 steps
    for step in range(5):
        # Generate dummy batch
        x = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_len), device=device)
        y = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_len), device=device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if step == 0:
            # First step often has compilation/initialization overhead
            pass
            
    max_memory = torch.cuda.max_memory_allocated() / (1024**3) # GB
    reserved_memory = torch.cuda.max_memory_reserved() / (1024**3) # GB
    
    # Cleanup
    del model
    del optimizer
    del x, y, logits, loss
    torch.cuda.empty_cache()
    
    return {
        "Architecture": name,
        "Max Allocated (GB)": round(max_memory, 2),
        "Max Reserved (GB)": round(reserved_memory, 2),
        "Params (M)": round(sum(p.numel() for p in MinimalLLM(config).parameters()) / 1e6, 1)
    }

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot measure GPU usage.")
        exit(1)
        
    configs = [
        (BlueberryConfig, "Blueberry-B8 (Default)"),
    ]
    
    results = []
    for config_class, name in configs:
        try:
            res = measure_memory(config_class, name)
            results.append(res)
        except Exception as e:
            print(f"❌ Failed to measure {name}: {e}")
            
    df = pd.DataFrame(results)
    print("\n" + "="*50)
    print(f"📊 PEAK GPU MEMORY USAGE (Batch {BlueberryConfig.batch_size}, Seq 2048, BF16)")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)
