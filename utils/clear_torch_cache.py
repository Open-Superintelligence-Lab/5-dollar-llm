import os
import shutil
import torch
from pathlib import Path

def clear_torch_inductor_cache():
    """Clears the torch.compile (Inductor) cache to force re-compilation."""
    
    # 1. Try to get cache dir from torch._inductor.config if available
    try:
        import torch._inductor.config as inductor_config
        cache_dir = getattr(inductor_config, 'cache_dir', None)
    except (ImportError, AttributeError):
        cache_dir = None

    import getpass
    possible_paths = [
        cache_dir,
        os.path.join(os.path.expanduser("~"), ".cache", "torch", "torchinductor"),
        os.path.join("/tmp", f"torchinductor_{getpass.getuser()}"),
    ]
    
    cleared = False
    for path_str in possible_paths:
        if path_str:
            path = Path(path_str)
            if path.exists():
                print(f"🗑️ Found cache at: {path}")
                try:
                    shutil.rmtree(path)
                    print(f"✅ Successfully cleared {path}")
                    cleared = True
                except Exception as e:
                    print(f"❌ Failed to clear {path}: {e}")

    # 3. Handle triton cache specifically if it exists
    triton_cache = Path(os.path.expanduser("~")) / ".triton" / "cache"
    if triton_cache.exists():
        print(f"🗑️ Found Triton cache at: {triton_cache}")
        try:
            shutil.rmtree(triton_cache)
            print(f"✅ Successfully cleared {triton_cache}")
            cleared = True
        except Exception as e:
            print(f"❌ Failed to clear {triton_cache}: {e}")

    if not cleared:
        print("🤷 No torch/inductor cache directories found.")
    else:
        print("\n✨ All compilation caches cleared. Next 'torch.compile' will trigger full re-compilation.")

if __name__ == "__main__":
    clear_torch_inductor_cache()
