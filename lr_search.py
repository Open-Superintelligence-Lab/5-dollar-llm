import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from models.llm import MinimalLLM
from configs.multimodal_config import MultimodalConfig
from training.trainer import setup_muon_optimizer
import os
from tqdm import tqdm
import copy

def run_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_config = MultimodalConfig()
    
    # 1. Dataset
    data_path = "processed_data/multimodal_data.jsonl"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run data/prepare_multimodal_data.py first.")
        return
        
    dataset = load_dataset("json", data_files=data_path, split="train")
    
    def collate_fn(batch):
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        labels = [torch.tensor(item["labels"]) for item in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "labels": labels}

    dataloader = DataLoader(dataset, batch_size=base_config.batch_size, shuffle=True, collate_fn=collate_fn)
    
    lrs = [0.01, 0.02, 0.04, 0.08]
    results = {}
    
    # Save initial weights to reset for each LR
    initial_model = MinimalLLM(base_config).to(device)
    initial_state = copy.deepcopy(initial_model.state_dict())

    search_steps = 100
    
    for lr in lrs:
        print(f"\n🔍 Testing LR: {lr}")
        model = MinimalLLM(base_config).to(device)
        model.load_state_dict(initial_state)
        
        # Override LR in config
        temp_config = copy.copy(base_config)
        temp_config.muon_lr = lr
        
        optimizers = setup_muon_optimizer(model, temp_config)
        
        model.train()
        losses = []
        
        pbar = tqdm(range(search_steps))
        data_iter = iter(dataloader)
        
        for step in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
                
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, temp_config.vocab_size), shift_labels.view(-1))
            
            loss.backward()
            for opt in optimizers:
                opt.step()
                opt.zero_grad()
            
            losses.append(loss.item())
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
        results[lr] = sum(losses[-20:]) / 20 # Mean of last 20 steps
        print(f"✅ LR {lr} Final Loss (Avg last 20): {results[lr]:.4f}")

    best_lr = min(results, key=results.get)
    print(f"\n🏆 Best Learning Rate: {best_lr}")
    return best_lr

if __name__ == "__main__":
    run_search()
