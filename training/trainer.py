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
from configs.llm_config import BlueberryConfig
from models.llm import MinimalLLM
from optimizers.muon import Muon
from training.evaluation import evaluate_model
from utils.helpers import set_seed, format_time


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



def setup_muon_optimizer(model: nn.Module, config: BlueberryConfig):
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
    config: BlueberryConfig,
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
    log_every: int = 100,
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

    current_loss_val = 0.0

    # Training metrics tracking
    # Synchronize CUDA to ensure accurate timing (no queued operations)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_start_time = time.time()
    metrics_history = {
        'steps': [],
        'val_losses': [],
        'val_accuracies': [],
        'val_perplexities': [],
        'elapsed_times': [],
        'learning_rates': [],
    }

    # Training loop
    model.train()
    step = 0
    tokens_seen = 0
    desc = f"Training {experiment_name}" if experiment_name else "Training"
    pbar = tqdm(total=config.train_tokens, desc=desc, unit="tokens")
    
    stopped_early = False

    while tokens_seen < config.train_tokens:
        for batch_idx, batch in enumerate(train_loader):
            if tokens_seen >= config.train_tokens:
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
            
            # Count tokens in this batch (approx: batch_size * seq_len)
            batch_tokens = x.numel()

            # Forward pass (optimized to avoid large contiguous copies of logits)
            if config.use_amp:
                with autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)
                    # Shift labels instead of logits to save ~3GB VRAM
                    # We set the last token to -100 so cross_entropy ignores it
                    shift_labels = torch.full_like(y, -100)
                    shift_labels[:, :-1] = y[:, 1:]
                    
                    ce_loss = F.cross_entropy(
                        logits.view(-1, config.vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                    loss = ce_loss / config.gradient_accumulation_steps
                loss.backward()
            else:
                logits = model(x)
                shift_labels = torch.full_like(y, -100)
                shift_labels[:, :-1] = y[:, 1:]
                
                ce_loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                loss = ce_loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad()
                for scheduler in schedulers:
                    scheduler.step()

            # Track current loss as a scalar only every 100 steps to avoid sync bottleneck
            if step % 100 == 0 or step == 0:
                current_loss_val = ce_loss.item()
                if target_train_loss is not None and current_loss_val <= target_train_loss:
                    print(f"\n🎯 Target train loss {target_train_loss} reached at step {step}!")
                    stopped_early = True
                
            # Logging
            if step % log_every == 0 or stopped_early:
                with torch.no_grad():
                    # Calculate accuracy using the shifted labels mask
                    predictions = logits.argmax(dim=-1)
                    mask = (shift_labels != -100)
                    accuracy = (predictions[mask] == shift_labels[mask]).float().mean().item()
                    
                    # Use the scalar value we polled above
                    perplexity = math.exp(min(current_loss_val if 'current_loss_val' in locals() else ce_loss.item(), 20))
                    current_lr = schedulers[0].get_last_lr()[0] if schedulers else optimizers[0].param_groups[0]['lr']

                # Update progress bar
                tokens_per_step = config.batch_size * config.max_seq_len * config.gradient_accumulation_steps
                est_total_steps = config.train_tokens // tokens_per_step
                
                pbar.set_postfix({
                    'step': f'{step}/{est_total_steps}',
                    'loss': f'{current_loss_val:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'lr': f'{current_lr:.5f}'
                })
                # Console print for visibility
                if step % (log_every * 10) == 0 or stopped_early:
                    print(f" [Step {step}] Loss: {current_loss_val:.4f} | Acc: {accuracy:.3f} | LR: {current_lr:.6f}")
            
            pbar.update(batch_tokens)
            tokens_seen += batch_tokens

            if stopped_early:
                current_loss_val = ce_loss.item()
                break

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                eval_metrics = evaluate_model(model, val_loader, config)
                elapsed_time = (time.time() - train_start_time) / 60
                current_lr = schedulers[0].get_last_lr()[0] if schedulers else optimizers[0].param_groups[0]['lr']
                
                # Track metrics
                metrics_history['steps'].append(step)
                metrics_history['val_losses'].append(eval_metrics['val_loss'])
                metrics_history['val_accuracies'].append(eval_metrics['val_accuracy'])
                metrics_history['val_perplexities'].append(eval_metrics['val_perplexity'])
                metrics_history['elapsed_times'].append(elapsed_time)
                metrics_history['learning_rates'].append(current_lr)
                
                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}, "
                      f"LR: {current_lr:.5f}")
                
                # Early stopping check
                if early_stopper is not None:
                    if early_stopper(eval_metrics['val_loss'], step):
                        current_loss_val = ce_loss.item()
                        stopped_early = True
                        break

            step += 1
        
        # If we finished the inner loop but didn't stop early, 
        # ensure we have the most recent loss from the very last batch
        if not stopped_early and 'ce_loss' in locals():
            current_loss_val = ce_loss.item()

        if stopped_early:
            break

    pbar.close()

    # Final evaluation (if not stopped early)
    if not stopped_early or tokens_seen >= config.train_tokens:
        final_eval = evaluate_model(model, val_loader, config)
        final_eval['train_loss'] = current_loss_val
        elapsed_time = (time.time() - train_start_time) / 60
        current_lr = schedulers[0].get_last_lr()[0] if schedulers else optimizers[0].param_groups[0]['lr']
        
        metrics_history['steps'].append(step)
        metrics_history['val_losses'].append(final_eval['val_loss'])
        metrics_history['val_accuracies'].append(final_eval['val_accuracy'])
        metrics_history['val_perplexities'].append(final_eval['val_perplexity'])
        metrics_history['elapsed_times'].append(elapsed_time)
        metrics_history['learning_rates'].append(current_lr)
    else:
        # Use best metrics if stopped early
        if metrics_history['val_losses']:
            best_idx = metrics_history['val_losses'].index(min(metrics_history['val_losses']))
            final_eval = {
                'val_loss': metrics_history['val_losses'][best_idx],
                'val_accuracy': metrics_history['val_accuracies'][best_idx],
                'val_perplexity': metrics_history['val_perplexities'][best_idx],
                'train_loss': current_loss_val if 'current_loss_val' in locals() else 0.0,
            }
        else:
            final_eval = {
                'val_loss': current_loss_val if 'current_loss_val' in locals() else 0.0,
                'val_accuracy': accuracy if 'accuracy' in locals() else 0.0,
                'val_perplexity': perplexity if 'perplexity' in locals() else 0.0,
                'train_loss': current_loss_val if 'current_loss_val' in locals() else 0.0,
            }
    
    # Synchronize CUDA to ensure all operations are complete before ending timer
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time_seconds = time.time() - train_start_time
    
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
            'total_time_minutes': total_time_seconds / 60,
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
    
    return {
        'model': model,
        'final_metrics': final_eval,
        'metrics_history': metrics_history,
        'training_time': total_time_seconds,
        'steps': step,
        'tokens_seen': tokens_seen,
        'train_loss': current_loss_val if 'current_loss_val' in locals() else 0.0,
    }


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

def warmup_compiled_kernels(
    model: nn.Module,
    config: BlueberryConfig,
    train_loader: DataLoader,
    device: torch.device,
    num_steps: int = 3
) -> None:
    """
    Warm up all compiled kernels (forward, backward, optimizer).
    Caller is responsible for resetting state afterwards.
    """
    print(f"🔥 Warming up kernels ({num_steps} steps)...")
    model.train()
    
    # Temporary optimizer to warm up optimizer kernels too
    temp_optimizers = setup_muon_optimizer(model, config)
    
    warmup_iter = iter(train_loader)
    
    for _ in range(num_steps):
        try:
            batch = next(warmup_iter)
        except StopIteration:
            warmup_iter = iter(train_loader)
            batch = next(warmup_iter)
        
        # Parse batch
        if isinstance(batch, dict):
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
        else:
            x, y = batch[0].to(device), batch[-1].to(device)
        
        # Forward + Backward
        if config.use_amp:
            with autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, config.vocab_size),
                    y[:, 1:].reshape(-1)
                )
            loss.backward()
        else:
            logits = model(x)
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, config.vocab_size),
                y[:, 1:].reshape(-1)
            )
            loss.backward()
        
        # Optimizer step (warms up optimizer kernels)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        for opt in temp_optimizers:
            opt.step()
            opt.zero_grad()
    
    torch.cuda.synchronize()
    
    # Cleanup temp optimizers
    del temp_optimizers
    torch.cuda.empty_cache()
    
    print("✅ Kernels compiled and cached")

def train_minimal_llm(
    config: BlueberryConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Optional[str] = None,
    experiment_name: Optional[str] = None,
    load_weights_path: Optional[str] = None,
    target_train_loss: Optional[float] = None,
):
    print(f"\n🚀 Training dense model")
    setup_start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ============================================
    # 1. Initialize model with fixed seed
    # ============================================
    set_seed(42)
    model = MinimalLLM(config)
    model = model.to(device)
    
    # Load pretrained weights if specified
    if load_weights_path:
        print(f"Loading pretrained weights from {load_weights_path}...")
        checkpoint = torch.load(load_weights_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

    # ============================================
    # 2. Save initial state BEFORE any forward pass
    # ============================================
    initial_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Total parameters: {total_params:,}")

    # ============================================
    # 3. Compile model (if requested)
    # ============================================
    if config.compile_model:
        print("🚀 Compiling model with torch.compile...")
        # Keep a reference to the original model for state restoration
        orig_model = model
        try:
            model = torch.compile(model)
            print("✅ Model compiled successfully")
            
            # ============================================
            # 4. Warm up kernels (dirties model state)
            # ============================================
            warmup_compiled_kernels(model, config, train_loader, device, num_steps=3)
            
            # ============================================
            # 5. Reset model to initial state
            # ============================================
            # Restore state ensuring we use the original model keys to avoid calling load_state_dict on the wrapper
            orig_model.load_state_dict(initial_model_state)
            print("🔄 Model weights reset to initial state")
            
        except Exception as e:
            print(f"⚠️ Compilation failed: {e}")
            print("Continuing in eager mode.")
            # Fallback to original model
            model = orig_model
            # Ensure state is clean
            model.load_state_dict(initial_model_state)
    
    # Free the backup
    del initial_model_state
    torch.cuda.empty_cache()

    # ============================================
    # 6. Create FRESH optimizers (no accumulated state)
    # ============================================
    optimizers = setup_muon_optimizer(model, config)

    # ============================================
    # 7. Create FRESH schedulers
    # ============================================
    # Tokens per optimization step
    tokens_per_opt = config.batch_size * config.max_seq_len * config.gradient_accumulation_steps
    total_steps = config.train_tokens // tokens_per_opt
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))
    schedule_type = getattr(config, 'schedule_type', 'cosine')
    
    schedulers = []
    for optimizer in optimizers:
        if schedule_type == 'cosine':
            def lr_lambda(current_step, warmup=warmup_steps, total=total_steps):
                if current_step < warmup:
                    return current_step / warmup
                progress = (current_step - warmup) / max(1, total - warmup)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        elif schedule_type == 'linear':
            def lr_lambda(current_step, warmup=warmup_steps, total=total_steps):
                if current_step < warmup:
                    return current_step / warmup
                progress = (current_step - warmup) / max(1, total - warmup)
                return max(0.1, 1.0 - progress)
        else:  # constant
            def lr_lambda(current_step, warmup=warmup_steps):
                return current_step / warmup if current_step < warmup else 1.0
        
        schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda))

    # ============================================
    # 8. Reset RNG for reproducible training
    # ============================================
    set_seed(42)
    
    setup_time = time.time() - setup_start
    print(f"⚙️ Setup & Compilation complete in {setup_time:.2f}s")
    print("-" * 70)

    # ============================================
    # 9. Train from scratch (fresh iterator created internally)
    # ============================================
    # Clear GPU cache and synchronize to ensure consistent starting state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    train_start = time.time()
    
    results = train_model(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        early_stopper=None,
        output_dir=None,
        experiment_name=experiment_name,
        plot_fn=None,
        extra_config=None,
        target_train_loss=target_train_loss,
        log_every=getattr(config, 'log_every', 100),
    )
    
    total_training_time = results['training_time']
    total_wall_time = setup_time + total_training_time
    final_eval = results['final_metrics']
    metrics_history = results['metrics_history']
    step = results['steps']
    tokens_seen = results['tokens_seen']

    # ============================================
    # 10. Unified Saving & Reporting
    # ============================================
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive metrics
        metrics_file = output_path / "metrics.json"
        metrics_data = {
            'final_metrics': final_eval,
            'setup_time_seconds': setup_time,
            'active_training_time_seconds': total_training_time,
            'total_wall_time_seconds': total_wall_time,
            'total_time_minutes': total_wall_time / 60,
            'actual_steps': step,
            'history': metrics_history,
        }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
            
        # Save model
        checkpoint_path = output_path / "model.pt"
        torch.save({
            'model_state_dict': results['model'].state_dict(),
            'config': config,
            'metrics': final_eval,
        }, checkpoint_path)
        
        # Plot
        plot_training_metrics(metrics_history, output_path)
    
    # Final Output
    print("\n" + "="*70)
    print(" SPEEDRUN RESULTS")
    print("="*70)
    print(f"Warmup & Setup:                  {format_time(setup_time)}")
    print(f"Training Time (⏱️ Speedrun):      {format_time(total_training_time)}")
    print(f"Total Tokens:                    {tokens_seen:,}")
    print("-" * 70)
    print(f"Final Train Loss:                {final_eval.get('train_loss', 0.0):.4f}")
    print(f"Final Val Loss:                  {final_eval['val_loss']:.4f}")
    print(f"Final Val Accuracy:              {final_eval['val_accuracy']:.4f}")
    print("="*70 + "\n")

    return {
        'model': results['model'],
        'metrics': final_eval,
        'history': metrics_history,
        'setup_time': setup_time,
        'training_time': total_training_time,
        'steps': step,
        'tokens_seen': tokens_seen
    }