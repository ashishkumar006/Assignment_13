import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import pytorch_lightning as pl
from typing import Optional, Tuple
from tqdm import tqdm


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the frequency tensor for complex exponentials (cis) with given dimensions."""
    # Compute the frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)  # [end, dim/2]
    
    # Create cosine and sine components
    freqs_cos = torch.cos(freqs)  # [end, dim/2]
    freqs_sin = torch.sin(freqs)  # [end, dim/2]
    
    return freqs_cos, freqs_sin


def apply_rotary_emb(xq, xk, cos, sin):
    """Apply rotary embeddings to the query and key tensors using the given cosine and sine values."""
    # Get dimensions
    batch_size, n_heads, seq_len, d_head = xq.shape
    
    # Make sure d_head is even
    assert d_head % 2 == 0, "Dimension of head must be even for rotary embeddings"
    
    # Get cosine and sine for the current sequence length
    cos = cos[:seq_len]  # [seq_len, d_head//2]
    sin = sin[:seq_len]  # [seq_len, d_head//2]
    
    # Reshape for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_head//2]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_head//2]
    
    # More efficient implementation - avoid unnecessary tensor creation
    # Split heads into even and odd dimensions
    xq_even = xq[:, :, :, 0::2]
    xq_odd = xq[:, :, :, 1::2]
    xk_even = xk[:, :, :, 0::2]
    xk_odd = xk[:, :, :, 1::2]
    
    # Apply rotation using the rotation matrix
    xq_out = torch.empty_like(xq, device=xq.device)
    xk_out = torch.empty_like(xk, device=xk.device)
    
    # Compute directly into output tensors
    xq_out[:, :, :, 0::2] = xq_even * cos - xq_odd * sin
    xq_out[:, :, :, 1::2] = xq_even * sin + xq_odd * cos
    xk_out[:, :, :, 0::2] = xk_even * cos - xk_odd * sin
    xk_out[:, :, :, 1::2] = xk_even * sin + xk_odd * cos
    
    return xq_out, xk_out


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config["num_key_value_heads"]
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config["max_position_embeddings"]
        self.rope_theta = config["rope_theta"]

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Precompute the frequency tensor for complex exponentials
        freqs_cos, freqs_sin = precompute_freqs_cis(self.head_dim, self.max_position_embeddings, self.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        
        # Check if Flash Attention is available
        self.use_flash_attn = False
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.use_flash_attn = True
        except ImportError:
            pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        # Apply rotary embeddings to the query and key states
        query_states, key_states = apply_rotary_emb(
            query_states, key_states, self.freqs_cos, self.freqs_sin
        )

        if past_key_value is not None:
            # Reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat k/v heads if num_key_value_heads < num_attention_heads
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Check if we can use Flash Attention (must have correct shape, CUDA device, and no attention mask for causal case)
        can_use_flash_attn = (
            self.use_flash_attn
            and query_states.is_cuda
            and query_states.dtype in (torch.float16, torch.bfloat16)
            and key_states.shape[2] == query_states.shape[2]  # same seq length for q and k
            and (attention_mask is None)  # either no mask or simple causal mask
        )

        if can_use_flash_attn:
            # [batch, heads, seqlen, head_dim] -> [batch, seqlen, heads, head_dim]
            q = query_states.transpose(1, 2).contiguous()
            k = key_states.transpose(1, 2).contiguous()
            v = value_states.transpose(1, 2).contiguous()
            
            # Flash Attention expects inputs in [batch, seqlen, heads, head_dim]
            attn_output = self.flash_attn_func(q, k, v, causal=True)
            
            # Convert back to [batch, seqlen, hidden_size]
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        else:
            # Regular attention implementation
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config["rms_norm_eps"])
        
        # Residual scaling
        self.residual_scale = 1.0 / math.sqrt(config["num_hidden_layers"] * 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        # Use CUDA stream for better parallelism when available
        if hidden_states.is_cuda and torch.cuda.is_available():
            # First normalization and attention
            norm_hidden = self.input_layernorm(hidden_states)
            
            # Process in current stream
            with torch.cuda.stream(torch.cuda.current_stream()):
                attn_output, past_key_value = self.self_attn(
                    hidden_states=norm_hidden,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            # Apply residual connection
            hidden_states = hidden_states + self.residual_scale * attn_output

            # Second normalization and MLP
            norm_hidden = self.post_attention_layernorm(hidden_states)
            
            # Process in current stream
            with torch.cuda.stream(torch.cuda.current_stream()):
                mlp_output = self.mlp(norm_hidden)

            # Apply residual connection
            hidden_states = hidden_states + self.residual_scale * mlp_output
        else:
            # Original implementation for CPU
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            attn_output, past_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
            # Apply residual scaling
            hidden_states = residual + self.residual_scale * attn_output

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(hidden_states)
            
            # Apply residual scaling
            hidden_states = residual + self.residual_scale * mlp_output

        return hidden_states, past_key_value


class SmolLM2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.num_hidden_layers = config["num_hidden_layers"]
        
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(self.num_hidden_layers)])
        self.norm = RMSNorm(self.hidden_size, eps=config["rms_norm_eps"])
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections
        for layer in self.layers:
            nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=config["initializer_range"] / math.sqrt(2 * self.num_hidden_layers))
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=config["initializer_range"] / math.sqrt(2 * self.num_hidden_layers))

    def _init_weights(self, module):
        std = self.config["initializer_range"]
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # Efficiently handle device placement for inputs
        device = self.embed_tokens.weight.device  # Get model's device
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
            # Ensure input_ids is on the correct device
            input_ids = input_ids.to(device)
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            # Ensure inputs_embeds is on the correct device
            inputs_embeds = inputs_embeds.to(device)
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        else:
            attention_mask = attention_mask.to(device)

        # Convert attention mask to causal mask for attention
        if attention_mask.dim() == 2:
            # Use more efficient memory operation for causal mask
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(inputs_embeds.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(inputs_embeds.dtype).min
        else:
            extended_attention_mask = attention_mask.to(device)

        # Initialize past_key_values if needed
        if past_key_values is None:
            past_key_values = tuple([None] * self.num_hidden_layers)
        elif isinstance(past_key_values, tuple) and len(past_key_values) > 0 and past_key_values[0] is not None:
            # Make sure past key values are on the right device
            past_key_values = tuple(
                tuple(p.to(device) for p in layer_past)
                for layer_past in past_key_values
            )

        # Initialize lists to collect outputs
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        hidden_states = inputs_embeds
        
        # Use CUDA streams for better parallelism in the forward pass
        if hidden_states.is_cuda and torch.cuda.is_available():
            main_stream = torch.cuda.current_stream()
            
            for idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # Process in main stream
                hidden_states, layer_past_key_value = layer(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

                if use_cache:
                    next_decoder_cache += (layer_past_key_value,)

                if output_attentions:
                    all_self_attns += (None,)
        else:
            # Regular forward pass for CPU
            for idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                hidden_states, layer_past_key_value = layer(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

                if use_cache:
                    next_decoder_cache += (layer_past_key_value,)

                if output_attentions:
                    all_self_attns += (None,)

        # Apply final layer norm
        hidden_states = self.norm(hidden_states)

        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "past_key_values": next_decoder_cache,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }


class SmolLM2LMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = SmolLM2Model(config)
        self.lm_head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        
        # Tie weights
        if config.get("tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = output["last_hidden_state"]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": output["past_key_values"],
            "hidden_states": output["hidden_states"],
            "attentions": output["attentions"],
        }

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=True,
        **kwargs
    ):
        """Generate text based on input_ids."""
        # Set generation parameters
        use_cache = True
        
        # Prepare input variables
        batch_size = input_ids.shape[0]
        cur_len = input_ids.shape[1]
        max_length = max_length if max_length is not None else self.config["max_position_embeddings"]
        
        # Keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(batch_size).fill_(1)
        
        # Initialize past key values
        past = None
        
        # Create a tqdm progress bar for generation
        pbar = tqdm(total=max_length-cur_len, desc="Generating")
        
        while cur_len < max_length:
            # Prepare inputs
            model_inputs = {
                "input_ids": input_ids,
                "past_key_values": past,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
            
            # Forward pass
            outputs = self.forward(**model_inputs)
            next_token_logits = outputs["logits"][:, -1, :]
            past = outputs["past_key_values"]
            
            # Apply temperature and repetition penalty
            next_token_logits = next_token_logits / temperature
            
            # Repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                        # If score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty
            
            # Get next token using sampling or greedy
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = torch.finfo(next_token_logits.dtype).min

                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for batch_idx in range(batch_size):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = torch.finfo(next_token_logits.dtype).min
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)
            
            # Add code that checks if eos_token_id is reached
            # (not implemented here as it depends on tokenizer)
            
            # Add token to sequence and increment length
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1
                )
            
            # Update progress bar
            pbar.update(1)
        
        # Close progress bar
        pbar.close()
        
        return input_ids


class SmolLM2LightningModule(pl.LightningModule):
    def __init__(self, config, tokenizer=None):
        super().__init__()
        self.config = config
        self.model = SmolLM2LMHeadModel(config)
        self.tokenizer = tokenizer
        self.save_hyperparameters()
        
        self.train_step_count = 0
        self.validation_step_count = 0
        
        # For sample generations during training
        self.last_sample_step = 0
        self.sample_interval = 500  # Generate sample every 500 steps
        
        # Set automatic precision based on hardware
        if torch.cuda.is_available():
            # Try to use more efficient memory format when on GPU
            self.memory_format = torch.channels_last if torch.cuda.is_available() else torch.contiguous_format
            # Enable CUDA graph optimization if supported
            self._use_cuda_graph = hasattr(torch, 'cuda') and torch.cuda.is_available() and hasattr(torch.cuda, 'make_graphed_callables')
            
            # Check if we're on a T4 GPU
            gpu_name = torch.cuda.get_device_name(0).lower()
            self._is_t4 = 't4' in gpu_name
            if self._is_t4:
                print("Detected NVIDIA T4 GPU - enabling T4 specific optimizations")
                # T4 optimizations will be applied in on_fit_start
        else:
            self.memory_format = torch.contiguous_format
            self._use_cuda_graph = False
            self._is_t4 = False
        
    def forward(self, **inputs):
        return self.model(**inputs)
    
    def on_fit_start(self):
        """Setup optimizations when training begins."""
        if torch.cuda.is_available():
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            
            # T4 GPUs use Turing architecture which has good FP16 support but no native BF16
            # They benefit from cudnn benchmarking and tensor core usage
            if self._is_t4:
                # Optimize for T4's tensor cores with FP16
                torch.backends.cudnn.benchmark = True
                # T4 doesn't benefit from TF32 (only Ampere and newer)
                
                # Try to load APEX for optimized CUDA kernels if available
                try:
                    global amp
                    from apex import amp
                    print("Successfully loaded APEX for optimized T4 performance")
                except ImportError:
                    print("APEX not available - for better T4 performance consider installing NVIDIA APEX")
            else:
                # For other GPUs like Ampere and newer, TF32 helps
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                if hasattr(torch.backends.cuda, 'matmul'):
                    torch.backends.cuda.matmul.allow_tf32 = True
    
    def training_step(self, batch, batch_idx):
        # Move batch to the right device first to avoid unnecessary transfers
        if torch.cuda.is_available():
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
        )
        loss = outputs["loss"]
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.train_step_count += 1
        
        # Generate text sample every sample_interval steps
        if self.train_step_count - self.last_sample_step >= self.sample_interval:
            self.last_sample_step = self.train_step_count
            self.generate_sample(batch["input_ids"][:1])
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Move batch to the right device first to avoid unnecessary transfers
        if torch.cuda.is_available():
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"],
            )
        loss = outputs["loss"]
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.validation_step_count += 1
        return loss
    
    def generate_sample(self, input_ids, max_length=100):
        """Generate and log a text sample during training."""
        # Generate text
        with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                )
        
        # Decode text if tokenizer is available
        if self.tokenizer:
            input_text = self.tokenizer.decode(input_ids[0])
            generated_text = self.tokenizer.decode(generated_ids[0])
            
            # Print to console
            self.print(f"\n--- Sample at step {self.train_step_count} ---")
            self.print(f"Input: {input_text[:50]}...")
            self.print(f"Generated: {generated_text}")
            
            # Log to our text file via callback
            try:
                for callback in self.trainer.callbacks:
                    if hasattr(callback, 'log_sample_generation'):
                        callback.log_sample_generation(
                            step=self.train_step_count,
                            input_text=input_text[:50] + "...",
                            generated_text=generated_text
                        )
                        break
            except Exception as e:
                self.print(f"Could not log sample to file: {e}")
    
    def configure_optimizers(self):
        # Use fused Adam implementation for better GPU performance if available
        try:
            if torch.cuda.is_available():
                from torch.optim.adamw import AdamW
                use_fused = True
            else:
                use_fused = False
                AdamW = torch.optim.AdamW
        except:
            use_fused = False
            AdamW = torch.optim.AdamW
        
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.get("learning_rate", 3e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
            betas=(self.config.get("adam_beta1", 0.9), self.config.get("adam_beta2", 0.95)),
            eps=self.config.get("adam_eps", 1e-8),
            fused=use_fused if hasattr(AdamW, 'fused') else False  # Use fused implementation if available
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.get("lr_decay_steps", 5000),
            eta_min=self.config.get("min_decay_lr", 0),
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }