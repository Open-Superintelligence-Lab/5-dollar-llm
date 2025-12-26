import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from models.llm import MinimalLLM
from configs.multimodal_config import MultimodalConfig
from optimizers.muon import Muon # Leveraging the project's optimizer
import os
from tqdm import tqdm

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MultimodalConfig()
    
    # 1. Load Model
    model = MinimalLLM(config).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    # 2. Dataset
    # We expect data prepared by prepare_multimodal_data.py
    data_path = "processed_data/multimodal_data.jsonl"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run data/prepare_multimodal_data.py first.")
        return
        
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    def collate_fn(batch):
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        labels = [torch.tensor(item["labels"]) for item in batch]
        
        # Simple padding
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return {"input_ids": input_ids, "labels": labels}

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    
    # 3. Optimizer
    # MinimalLLM has tied weights. setup_muon_optimizer handles parameter separation.
    from training.trainer import setup_muon_optimizer
    optimizers = setup_muon_optimizer(model, config)
    
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    
    print("Starting training...")
    for epoch in range(20): # 10x total training effort (50k samples * 20 epochs)
        pbar = tqdm(dataloader)
        for i, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(input_ids)
                
                # Shift logits and labels for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss = F.cross_entropy(shift_logits.view(-1, config.vocab_size), shift_labels.view(-1))
            
            # Since we use Muon which is a custom optimizer, let's see if we need scaler for it.
            # Usually custom optimizers like Muon don't use scaler directly in the same way.
            # But standard backward works fine.
            loss.backward()
            
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
            
            pbar.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
            
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/multimodal_llm_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()
