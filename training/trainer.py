import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
from typing import List, Optional, Callable, Dict, Any
from configs.llm_config import Blueberry80GBConfig
from models.llm import MoEMinimalLLM
from optimizers.muon import Muon
from training.evaluation import evaluate_model
from utils.helpers import set_seed


class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience: int = 30, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_step = 0
        
    def __call__(self, val_loss: float, step: int) -> bool:
        """Returns True if training should stop"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_step = step
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\n⏹️  Early stopping triggered at step {step}")
                print(f"   Best loss: {self.best_loss:.4f} at step {self.best_step}")
                return True
            return False


def setup_muon_optimizer(model: nn.Module, config: Blueberry80GBConfig):
    """Setup Muon optimizer with hybrid approach"""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    print(f"  Muon parameters: {sum(p.numel() for p in muon_params):,}")
    print(f"  AdamW parameters: {sum(p.numel() for p in adamw_params):,}")

    muon_optimizer = Muon(muon_params, lr=config.muon_lr, momentum=config.muon_momentum)
    adamw_optimizer = torch.optim.AdamW(
        adamw_params,
        lr=config.adamw_lr,
        weight_decay=config.weight_decay
    )

    return [muon_optimizer, adamw_optimizer]


def train_model(
    model: nn.Module,
    config: Blueberry80GBConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizers: List[torch.optim.Optimizer],
    schedulers: Optional[List] = None,
    early_stopper: Optional[EarlyStopping] = None,
    output_dir: Optional[str] = None,
    experiment_name: Optional[str] = None,
    plot_fn: Optional[Callable] = None,
    extra_config: Optional[Dict[str, Any]] = None,
    target_train_loss: Optional[float] = None,
) -> Any:
    """
    Generic training function that can be used by experiments.
    
    Args:
        model: Model to train
        config: Model configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizers: List of optimizers
        schedulers: Optional list of learning rate schedulers
        early_stopper: Optional early stopping handler
        output_dir: Optional directory to save outputs
        experiment_name: Optional experiment name for logging
        plot_fn: Optional custom plotting function(metrics_history, output_path)
        extra_config: Optional dict of extra config to save with metrics
    
    Returns:
        model, final_metrics, metrics_history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    if schedulers is None:
        schedulers = []

    # Training metrics tracking
    train_start_time = time.time()
    metrics_history = {
        'steps': [],
        'val_losses': [],
        'val_aux_losses': [],
        'val_accuracies': [],
        'val_perplexities': [],
        'elapsed_times': [],
        'learning_rates': [],
    }

    # Training loop
    model.train()
    step = 0
    desc = f"Training {experiment_name}" if experiment_name else "Training"
    pbar = tqdm(total=config.max_steps, desc=desc)
    
    stopped_early = False

    while step < config.max_steps:
        for batch_idx, batch in enumerate(train_loader):
            if step >= config.max_steps:
                break

            # Handle different batch formats
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

            # Forward pass
            if config.use_amp:
                with autocast('cuda', dtype=torch.bfloat16):
                    logits, aux_loss = model(x, return_aux_loss=True)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = y[:, 1:].contiguous()
                    ce_loss = F.cross_entropy(
                        shift_logits.view(-1, config.vocab_size),
                        shift_labels.view(-1)
                    )

                    total_loss = ce_loss
                    if aux_loss is not None:
                        total_loss = total_loss + aux_loss

                    loss = total_loss / config.gradient_accumulation_steps
                loss.backward()
            else:
                logits, aux_loss = model(x, return_aux_loss=True)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = y[:, 1:].contiguous()
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, config.vocab_size),
                    shift_labels.view(-1)
                )

                total_loss = ce_loss
                if aux_loss is not None:
                    total_loss = total_loss + aux_loss

                loss = total_loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    for optimizer in optimizers:
                        optimizer.step()
                        optimizer.zero_grad()
                    for scheduler in schedulers:
                        scheduler.step()

            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = ce_loss.item()
                    perplexity = math.exp(min(current_loss, 20))
                    current_lr = schedulers[0].get_last_lr()[0] if schedulers else optimizers[0].param_groups[0]['lr']

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'aux': f'{aux_loss.item() if aux_loss is not None else 0:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr': f'{current_lr:.5f}'
                })

                # Target train loss check
                if target_train_loss is not None and current_loss <= target_train_loss:
                    print(f"\n🎯 Target train loss {target_train_loss} reached at step {step}!")
                    stopped_early = True
                    break

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                elapsed_time = (time.time() - train_start_time) / 60
                current_lr = schedulers[0].get_last_lr()[0] if schedulers else optimizers[0].param_groups[0]['lr']
                
                # Track metrics
                metrics_history['steps'].append(step)
                metrics_history['val_losses'].append(eval_metrics['val_loss'])
                metrics_history['val_aux_losses'].append(eval_metrics['val_aux_loss'])
                metrics_history['val_accuracies'].append(eval_metrics['val_accuracy'])
                metrics_history['val_perplexities'].append(eval_metrics['val_perplexity'])
                metrics_history['elapsed_times'].append(elapsed_time)
                metrics_history['learning_rates'].append(current_lr)
                
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Aux Loss: {eval_metrics['val_aux_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}, "
                      f"LR: {current_lr:.5f}")
                
                # Early stopping check
                if early_stopper is not None:
                    if early_stopper(eval_metrics['val_loss'], step):
                        stopped_early = True
                        break

            step += 1
            if step % 20 == 0:
                pbar.update(20)
        
        if stopped_early:
            break

    pbar.close()

    # Final evaluation (if not stopped early)
    if not stopped_early or step == config.max_steps:
        final_eval = evaluate_model(model, val_loader, config)
        elapsed_time = (time.time() - train_start_time) / 60
        current_lr = schedulers[0].get_last_lr()[0] if schedulers else optimizers[0].param_groups[0]['lr']
        
        metrics_history['steps'].append(step)
        metrics_history['val_losses'].append(final_eval['val_loss'])
        metrics_history['val_aux_losses'].append(final_eval['val_aux_loss'])
        metrics_history['val_accuracies'].append(final_eval['val_accuracy'])
        metrics_history['val_perplexities'].append(final_eval['val_perplexity'])
        metrics_history['elapsed_times'].append(elapsed_time)
        metrics_history['learning_rates'].append(current_lr)
    else:
        # Use best metrics if stopped early
        best_idx = metrics_history['val_losses'].index(min(metrics_history['val_losses']))
        final_eval = {
            'val_loss': metrics_history['val_losses'][best_idx],
            'val_accuracy': metrics_history['val_accuracies'][best_idx],
            'val_perplexity': metrics_history['val_perplexities'][best_idx],
        }
    
    total_time = (time.time() - train_start_time) / 60
    
    print(f"\n📊 Final Results:")
    print(f"   Val Loss: {final_eval['val_loss']:.4f}")
    print(f"   Val Aux Loss: {final_eval['val_aux_loss']:.4f}")
    print(f"   Val Accuracy: {final_eval['val_accuracy']:.4f}")
    print(f"   Val Perplexity: {final_eval['val_perplexity']:.2f}")
    print(f"   Total Time: {total_time:.2f} min")
    if stopped_early:
        print(f"   ⚠️  Training stopped early at step {step}")
    
    # Save outputs if directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = output_path / "metrics.json"
        metrics_data = {
            'final_metrics': final_eval,
            'total_time_minutes': total_time,
            'stopped_early': stopped_early,
            'actual_steps': step,
            'history': metrics_history,
        }
        if extra_config:
            metrics_data['experiment_config'] = extra_config
            
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"   📁 Metrics saved to {metrics_file}")
        
        # Plot metrics using custom function or default
        if plot_fn:
            plot_fn(metrics_history, output_path)
        else:
            plot_training_metrics(metrics_history, output_path)
        
        # Save model checkpoint
        checkpoint_path = output_path / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'metrics': final_eval,
            'step': step,
        }, checkpoint_path)
        print(f"   💾 Model saved to {checkpoint_path}")
    
    return model, final_eval, metrics_history


def plot_training_metrics(metrics_history: Dict, output_path: Path):
    """Default plotting function for training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Metrics', fontsize=14, fontweight='bold')
    
    # Plot 1: Val Loss vs Time
    ax = axes[0, 0]
    ax.plot(metrics_history['elapsed_times'], metrics_history['val_losses'], 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Time')
    ax.grid(True, alpha=0.3)
    
    # Highlight best point
    if metrics_history['val_losses']:
        best_idx = metrics_history['val_losses'].index(min(metrics_history['val_losses']))
        ax.plot(metrics_history['elapsed_times'][best_idx], 
                metrics_history['val_losses'][best_idx], 
                'r*', markersize=15, label=f'Best: {metrics_history["val_losses"][best_idx]:.4f}')
        ax.legend()
    
    # Plot 2: Val Loss vs Steps
    ax = axes[0, 1]
    ax.plot(metrics_history['steps'], metrics_history['val_losses'], 'g-o', linewidth=2, markersize=4)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Steps')
    ax.grid(True, alpha=0.3)
    if metrics_history['val_losses']:
        best_idx = metrics_history['val_losses'].index(min(metrics_history['val_losses']))
        ax.plot(metrics_history['steps'][best_idx], 
                metrics_history['val_losses'][best_idx], 
                'r*', markersize=15)
    
    # Plot 3: Val Accuracy vs Steps
    ax = axes[1, 0]
    ax.plot(metrics_history['steps'], metrics_history['val_accuracies'], 'purple', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy vs Steps')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate vs Steps
    ax = axes[1, 1]
    ax.plot(metrics_history['steps'], metrics_history['learning_rates'], 'orange', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_path / "metrics_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   📊 Plots saved to {plot_path}")


def train_moe_model(
    config: Blueberry80GBConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Optional[str] = None,
    experiment_name: Optional[str] = None,
    load_weights_path: Optional[str] = None,
    target_train_loss: Optional[float] = None
):
    """
    Train the MoE model with default Muon optimizer setup.
    This is a convenience wrapper around the generic train_model function.
    """
    print(f"\n🚀 Training model with {getattr(config, 'num_experts', 'N/A')} experts (top-{getattr(config, 'expert_top_k', 'N/A')})")

    # Initialize model
    set_seed(42)
    model = MoEMinimalLLM(config)
    
    if load_weights_path:
        print(f"Loading pretrained weights from {load_weights_path}...")
        checkpoint = torch.load(load_weights_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        keys = model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded: {keys}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    active_params = sum(p.numel() for n, p in model.named_parameters()
                       if 'expert' not in n)
    expert_params = total_params - active_params

    print(f"  📊 Total parameters: {total_params:,}")
    print(f"  📊 Active parameters: {active_params:,}")
    print(f"  📊 Expert parameters: {expert_params:,}")
    print(f"  📊 Parameter efficiency: {active_params/total_params:.1%} active per forward pass")

    # Setup optimizers
    optimizers = setup_muon_optimizer(model, config)

    # Compile the model if requested (PyTorch 2.0+)
    if config.compile_model:
        print("🚀 Compiling model with torch.compile...")
        # Reduce compilation overhead for MoE by not enforcing fullgraphs
        # mode='max-autotune' gives best perf but takes longest to compile
        # mode='reduce-overhead' is good for small batches
        try:
            model = torch.compile(model)
            print("✅ Model compiled successfully")
        except Exception as e:
            print(f"⚠️ Model compilation failed: {e}")
            print("Running in eager mode instead.")

    # Learning rate schedule with cosine decay
    schedulers = []
    warmup_steps = max(1, int(config.max_steps * config.warmup_ratio))
    for optimizer in optimizers:
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    # Use the generic training function
    model, final_eval, metrics_history = train_model(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        early_stopper=None,
        output_dir=output_dir,
        experiment_name=experiment_name,
        plot_fn=None,
        extra_config=None,
        target_train_loss=target_train_loss,
    )

    return model, final_eval, metrics_history
