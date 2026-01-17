
import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata
from typing import Dict, Optional, List
from transformers import PreTrainedTokenizer

class TokenizerCompression(nn.Module):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Compresses the vocabulary by mapping semantically equivalent tokens 
        (e.g., "Apple" and " apple") to the same canonical ID.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        
        # Build the mapping
        # We use a helper to get the mapping logic, then register as buffer
        raw_to_canonical, num_ids = self._build_compression_map()
        self.register_buffer("raw_to_canonical", raw_to_canonical)
        self.num_canonical = num_ids
        
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for semantic equivalence.
        1. NFKC normalization
        2. Lowercase
        3. Strip whitespace (to merge " Apple" and "Apple")
        """
        text = unicodedata.normalize('NFKC', text)
        text = text.lower()
        text = text.strip()
        return text

    def _build_compression_map(self):
        """
        Iterates through the entire vocabulary and assigns canonical IDs.
        Returns:
            mapping: torch.Tensor from [vocab_size] -> [canonical_id]
            num_ids: int count of unique canonical IDs
        """
        norm_to_id: Dict[str, int] = {}
        # Temporarily create tensor on CPU, will be moved when registered
        mapping = torch.zeros(self.vocab_size, dtype=torch.long)
        next_id = 0
        
        for i in range(self.vocab_size):
            try:
                # Use tokenizer decode directly
                text = self.tokenizer.decode([i], skip_special_tokens=False)
            except Exception:
                text = ""
                
            norm_text = self._normalize_text(text)
            
            if norm_text not in norm_to_id:
                norm_to_id[norm_text] = next_id
                next_id += 1
            
            mapping[i] = norm_to_id[norm_text]
            
        print(f"TokenizerCompression: Reduced vocab from {self.vocab_size} to {next_id} canonical tokens "
              f"({(1 - next_id/self.vocab_size)*100:.1f}% reduction)")
              
        return mapping, next_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq_len] tensor of raw token IDs
        Returns:
            canonical_ids: [batch, seq_len] tensor of compressed IDs
        """
        return self.raw_to_canonical[input_ids]

class MultiHeadNgramHash(nn.Module):
    def __init__(self, n: int, num_heads: int, table_size: int, seed: int = 42):
        """
        Maps sequences of canonical tokens to hash indices using a sliding window.
        
        Args:
            n: N-gram size (e.g., 2 or 3)
            num_heads: Number of independent hash functions
            table_size: Size of the embedding table (modulus)
            seed: Random seed for hash weights
        """
        super().__init__()
        self.n = n
        self.num_heads = num_heads
        self.table_size = table_size
        
        # Initialize random weights for hashing
        # Structure: We need distinct weights for each position in the n-gram 
        # and for each hash head.
        gen = torch.Generator()
        gen.manual_seed(seed)
        
        # Weights: [n, num_heads]
        # We use large random integers. Python ints handle arbitrary size, 
        # but PyTorch Tensors are limited to int64. 
        # We ensure they are non-negative.
        self.register_buffer(
            "hash_weights",
            torch.randint(
                low=1, 
                high=2**31 - 1, # Keep within safe int32 range for multiplication safety before 64-bit sum
                size=(n, num_heads), 
                generator=gen,
                dtype=torch.int64
            )
        )
        
        # XOR Mixing Masks: [num_heads]
        self.register_buffer(
            "xor_masks",
            torch.randint(
                low=0,
                high=2**31 - 1,
                size=(num_heads,),
                generator=gen,
                dtype=torch.int64
            )
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [Batch, Seq] tensor of canonical IDs (int64)
            
        Returns:
            hash_indices: [Batch, Seq, NumHeads] in range [0, table_size)
        """
        B, T = input_ids.shape
        
        # 1. Pad inputs to handle the start of the sequence (Suffix N-Grams)
        # For an N-gram ending at t, we need tokens t-(N-1) to t.
        # So we pad N-1 zeros at the left.
        if self.n > 1:
            padding = torch.zeros((B, self.n - 1), dtype=input_ids.dtype, device=input_ids.device)
            padded = torch.cat([padding, input_ids], dim=1) # [B, T + N - 1]
        else:
            padded = input_ids
            
        # 2. Compute Hashes
        # We want [B, T, Heads]
        # Implementation: Broadcast sum over the N positions
        
        # hashes accumulator
        hashes = torch.zeros((B, T, self.num_heads), dtype=torch.int64, device=input_ids.device)
        
        # Loop over the N positions in the window
        for i in range(self.n):
            # Extract the slice of the sequence corresponding to position i in the N-ngrams
            # If N=3, i=0 corresponds to lag 2 (x_{t-2}), i=1 -> x_{t-1}, i=2 -> x_t
            # The slice from padded is simply [0 : T], [1 : T+1], etc.
            
            # Slice: [B, T]
            input_slice = padded[:, i : i + T]
            
            # Weight: [Heads]
            w = self.hash_weights[i]
            
            # Add to hash: slice[B, T, 1] * w[Heads] -> [B, T, Heads]
            hashes += input_slice.unsqueeze(-1) * w
            
        # 3. Apply XOR mask for nonlinearity
        hashes = hashes ^ self.xor_masks
        
        # 4. Modulo to get table indices
        indices = hashes % self.table_size
        
        return indices

class EngramModule(nn.Module):
    def __init__(
        self, 
        config, 
        compression_module: TokenizerCompression
    ):
        """
        Engram Conditional Memory Module.
        
        Retrieves static embeddings via hashed N-grams and fuses them with the 
        dynamic residual stream using context-aware gating.
        
        Args:
            config: Configuration object containing:
                - engram_ngrams: List[int] (e.g., [2, 3])
                - engram_vocab_size: int (e.g., 200000)
                - engram_dim: int (optional, default d_model)
                - engram_num_heads: int (e.g., 2)
                - d_model: int
            compression_module: Shared TokenizerCompression instance
        """
        super().__init__()
        self.d_model = config.d_model
        
        # Default config fallbacks if not present
        self.ngrams = getattr(config, "engram_ngrams", [2, 3])
        self.table_size = getattr(config, "engram_vocab_size", 200000)
        self.num_heads = getattr(config, "engram_num_heads", 2)
        
        target_et_dim = getattr(config, "engram_dim", self.d_model)
        total_sources = len(self.ngrams) * self.num_heads
        
        # Ensure divisible
        self.dim_per_head = target_et_dim // total_sources
        self.et_dim = self.dim_per_head * total_sources 
        
        if self.et_dim != target_et_dim:
            print(f"Engram: Adjusted engram_dim from {target_et_dim} to {self.et_dim} to be divisible by {total_sources}")

        self.compression = compression_module
        
        # Hashing Modules
        # One hasher per N-gram order.
        self.hash_modules = nn.ModuleDict({
            str(n): MultiHeadNgramHash(n, self.num_heads, self.table_size) 
            for n in self.ngrams
        })
        
        # Embedding Tables
        # Distinct tables for each N-gram order and each Head
        # Keys: "n{n}_h{k}"
        self.embeddings = nn.ModuleDict()
        for n in self.ngrams:
            for h in range(self.num_heads):
                key = f"n{n}_h{h}"
                self.embeddings[key] = nn.Embedding(self.table_size, self.dim_per_head)
                
        # Projections for Gating
        # k_t = W_K e_t, v_t = W_V e_t
        self.wk = nn.Linear(self.et_dim, self.d_model, bias=False)
        self.wv = nn.Linear(self.et_dim, self.d_model, bias=False)
        
        # Norms
        self.q_norm = nn.RMSNorm(self.d_model)
        self.k_norm = nn.RMSNorm(self.d_model)
        
        # Convolution Refinement
        # Depthwise causal convolution with kernel 4, dilation = max(N)
        # Init with padding=0 because we will pad manually
        self.max_n = max(self.ngrams) if self.ngrams else 1
        self.conv = nn.Conv1d(
            in_channels=self.d_model, 
            out_channels=self.d_model, 
            kernel_size=4, 
            groups=self.d_model, # Depthwise
            padding=0, 
            dilation=self.max_n
        )
        
        self.silu = nn.SiLU()
        self._init_weights()

    def _init_weights(self):
        # Initialize embeddings with normal/uniform
        for mod in self.embeddings.values():
            nn.init.normal_(mod.weight, mean=0.0, std=0.02)
        
        # Zero init conv to start as identity-like behavior combined with residual
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, D] - Backbone hidden states
            input_ids: [B, T] - Raw input tokens
        
        Returns:
            output: [B, T, D] - Tensor to be added to residual stream
        """
        B, T, D = hidden_states.shape
        
        # 1. Retrieval
        # Canonicalize
        c_ids = self.compression(input_ids) # [B, T]
        
        retrieved_parts = []
        
        for n in self.ngrams:
            # Hash: [B, T, Heads]
            hasher = self.hash_modules[str(n)]
            indices = hasher(c_ids)
            
            for h in range(self.num_heads):
                # Lookup
                # indices[..., h]: [B, T]
                h_indices = indices[:, :, h]
                emb_module = self.embeddings[f"n{n}_h{h}"]
                part = emb_module(h_indices) # [B, T, dim_per_head]
                retrieved_parts.append(part)
        
        # Concatenate: [B, T, et_dim]
        et = torch.cat(retrieved_parts, dim=-1)
        
        # 2. Context-Aware Gating
        # q = h_t (from backbone)
        # k = W_K e_t
        # v = W_V e_t
        
        q = hidden_states
        k = self.wk(et)
        v = self.wv(et)
        
        # Norms
        # Paper says: Norm(h)^T Norm(k) / sqrt(d)
        q_norm_out = self.q_norm(q) # [B, T, D]
        k_norm_out = self.k_norm(k) # [B, T, D]
        
        # Compute Gate alpha (scalar per token)
        # (q * k).sum(dim=-1)
        score = (q_norm_out * k_norm_out).sum(dim=-1, keepdim=True) / (self.d_model ** 0.5) # [B, T, 1]
        alpha = torch.sigmoid(score)
        
        # Gated Value
        v_gated = alpha * v # [B, T, D]
        
        # 3. Refinement Module
        # Y = SiLU(Conv(RMS(v_gated))) + v_gated
        # Conv needs [B, D, T]
        v_in = v_gated.transpose(1, 2) # [B, D, T]
        
        # Apply Causal Padding (Left only)
        # Kernel=4, Dilation=max_n
        # Padding required = (Kernel-1) * Dilation
        pad_amount = (4 - 1) * self.max_n
        v_padded = F.pad(v_in, (pad_amount, 0)) # Pad left
        
        conv_out = self.conv(v_padded) # [B, D, T]
        conv_out = conv_out.transpose(1, 2) # [B, T, D]
        
        # Refined output
        y = self.silu(conv_out) + v_gated
        
        return y
