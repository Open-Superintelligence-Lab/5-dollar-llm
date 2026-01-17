"""
[Engram Architecture v2 Implementation for Blueberry]
Adapted from user provided demo.
"""

from typing import List, Optional
import math
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex
from sympy import isprime
from configs.llm_config import BlueberryConfig

class CompressedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        
        self.lookup_table, self.num_new_token = self._build_lookup_table()
    
    def __len__(self):
        return self.num_new_token
    
    def _build_lookup_table(self):
        old2new = {}
        key2new = {}          
        new_tokens = []

        vocab_size = len(self.tokenizer)
        # Handle case where tokenizer length > vocab size in config (e.g. added tokens)
        # Limiting to actual vocab size iterated
        
        # Optimization: Don't iterate all if too slow? 
        # For now we keep the loop but it runs once at init.
        
        lookup = np.zeros(vocab_size, dtype=np.int64)
        
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            
            # Simple check for unknown/replacement chars
            if "" in text:
                key = str(tid) # Fallback to unique char
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
            lookup[tid] = nid

        return lookup, len(new_tokens)
    
    def _compress(self, input_ids):
        # input_ids: list or numpy array
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        
        # Safety for OOV
        mask_valid = (arr < len(self.lookup_table)) & pos_mask
        valid_ids = arr[mask_valid]
        out[mask_valid] = self.lookup_table[valid_ids]
        return out   
    
    def __call__(self, input_ids):
        return self._compress(input_ids)
            
class ShortConv(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        kernel_size: int = 4, 
        dilation: int = 1, 
        norm_eps: float = 1e-5,
        activation: bool = True,
    ):
        super().__init__()
        self.activation = activation
        
        # Depthwise conv for standard residual stream [B, D, T]
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        self.norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B,T,D)
        Output: (B,T,D)
        """
        B, T, D = x.shape
        
        # RMS Norm
        x_norm = self.norm(x) # [B, T, D]
        
        # Conv expects [B, Channels, Length]
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        # Causal slicing (remove right padding)
        y_bct = y_bct[..., :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)
            
        y = y_bct.transpose(1, 2).contiguous()
        return y
    
def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1

class NgramHashMapping:
    def __init__(
        self, 
        config: BlueberryConfig,
        tokenizer,
    ):
        self.max_ngram_size = max(config.engram_ngrams) if config.engram_ngrams else 3
        # Use single vocab size from config or similar
        # Creating a list of sizes for ngrams if needed, or reusing
        self.vocab_size_per_ngram = [config.engram_vocab_size] * (self.max_ngram_size - 1)
        
        self.n_embed_per_ngram = config.engram_dim or config.d_model
        
        # config.engram_num_heads
        self.n_head_per_ngram = config.engram_num_heads
        self.layer_ids = config.engram_layers
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        self.compressed_tokenizer = CompressedTokenizer(tokenizer)            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        
        # Map pad_id to compressed space
        if 0 <= self.pad_id < len(self.compressed_tokenizer.lookup_table):
             self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])
        else:
             self.pad_id = 0

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        self.layer_multipliers = {}
        seed = 42

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}
        
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                
                # Check index bounds
                idx = ngram - 2
                if idx < len(self.vocab_size_per_ngram):
                    vocab_size = self.vocab_size_per_ngram[idx]
                else:
                    vocab_size = 200000

                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1
                
                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
            
        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0: return x
            shifted = np.pad(x, ((0, 0), (k, 0)),
                                mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            
            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        
        # Stack: [B, T, NumHashes] where NumHashes = (N-1) * Heads
        if not all_hashes:
             return np.zeros((B, T, 0), dtype=np.int64)
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids_numpy, layer_id):
        # input_ids_numpy: [B, T]
        # returns hashes for specific layer
        compressed_ids = self.compressed_tokenizer(input_ids_numpy)
        return self._get_ngram_hashes(compressed_ids, layer_id=layer_id)

class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        
        offsets = [0]
        # For torch.cumsum/sum safety with list
        current = 0
        for n in list_of_N[:-1]:
            current += n
            offsets.append(current)
        
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        
        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T, Heads]
        shifted_input_ids = input_ids + self.offsets
        output = self.embedding(shifted_input_ids) # [B, T, Heads, D]
        return output
    
class EngramModuleV2(nn.Module):
    def __init__(self, config: BlueberryConfig, tokenizer, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        
        # We assume the hash_mapping is shared or light enough to create. 
        # Ideally it should be created once and shared, but to match the v2 structure we create it here.
        # Warning: CompressedTokenizer build time might be high if repeated.
        # But MinimalLLM instantiates modules per layer. 
        # We should create the Mapping ONCE in MinimalLLM and pass it down.
        # But for this file refactor, let's allow passing the pre-built mapping OR build it.
        # Actually, let's adhere to the structure where `engram_v2.py` owns the class.
        
        # NOTE: For efficiency, the tokenizer hashing object should be unique.
        
        # Let's assume passed `tokenizer` is the HF tokenizer.
        # We will initialize `NgramHashMapping` inside, but since it has overhead, 
        # we might want to pass it in if we were optimizing. 
        # For now, create it inside.
        
        self.hash_mapping = NgramHashMapping(config, tokenizer)

        # Config vars
        # Heads per ngram * (MaxN - 1) total heads used in hashes
        # Dimensions
        heads_per_ngram = config.engram_num_heads
        max_n = max(config.engram_ngrams) if config.engram_ngrams else 3
        # Total distinct hash heads = (max_n - 1) * heads_per_ngram  (e.g. 2-gram + 3-gram = 2 slots)
        
        total_hash_heads = (max_n - 1) * heads_per_ngram
        
        # Embedding Dimension per head
        # In v2: n_embed_per_ngram // n_head_per_ngram
        # We use engram_dim (total) // total_hash_heads? 
        # Or engram_dim per N-gram order?
        # V2 logic: n_embed_per_ngram is total width per N-gram order.
        # D = n_embed_per_ngram // n_head_per_ngram.
        
        target_dim_per_ngram = config.engram_dim or config.d_model
        dim_per_head = target_dim_per_ngram // heads_per_ngram
        
        vocab_sizes = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y]
        
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N = vocab_sizes,
            D = dim_per_head
        )
        
        self.short_conv = ShortConv(
            hidden_size = config.d_model,
            kernel_size = 4,
            dilation    = max_n,
        )
        
        # Total engram concatenated dim
        # (MaxN - 1) * (Heads * D_head) = (MaxN - 1) * TargetDimPerNgram
        engram_hidden_size = (max_n - 1) * heads_per_ngram * dim_per_head
        
        # Output PROJ
        self.value_proj = nn.Linear(engram_hidden_size, config.d_model)
        
        # Gating
        self.key_proj = nn.Linear(engram_hidden_size, config.d_model)
        self.norm1 = nn.RMSNorm(config.d_model) # For Key
        self.norm2 = nn.RMSNorm(config.d_model) # For Query (Hidden)
    
    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, T, D]
        input_ids: [B, T] (Torch Tensor)
        """
        # 1. CPU / Numpy Hashing
        # This is the synchronization point.
        device = hidden_states.device
        
        # Check if we can do this faster?
        # input_ids to CPU
        ids_cpu = input_ids.detach().cpu().numpy()
        
        # Hash
        hashes_np = self.hash_mapping.hash(ids_cpu, self.layer_id) # [B, T, Heads]
        
        # To Tensor
        hash_input_ids = torch.from_numpy(hashes_np).to(device)
        
        # 2. Embedding
        # [B, T, Heads, D_head]
        embeddings = self.multi_head_embedding(hash_input_ids)
        # Flatten heads -> [B, T, TotalEngramDim]
        embeddings = embeddings.flatten(start_dim=-2)
        
        # 3. Gating
        # Key
        key = self.key_proj(embeddings) # [B, T, D]
        normed_key = self.norm1(key)
        
        # Query
        normed_query = self.norm2(hidden_states)
        
        # Gate score: (K * Q).sum
        # [B, T]
        gate = (normed_key * normed_query).sum(dim=-1, keepdim=True) / math.sqrt(self.config.d_model)
        
        # Activation (Specific to v2 demo: Abs -> Sqrt -> Sign -> Sigmoid)
        # gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        # Wait, abs().sqrt() * sign() is just signed sqrt. 
        # Then sigmoid.
        gate = (gate.abs().clamp_min(1e-6).sqrt() * gate.sign()).sigmoid()
        
        # 4. Value Fusion
        value = self.value_proj(embeddings)
        gated_value = gate * value
        
        # 5. Convolution Refinement
        output = gated_value + self.short_conv(gated_value)
        
        return output

            