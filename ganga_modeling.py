import os
import argparse
import random
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, Optional, Tuple, Literal

from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_attention_mask_for_sdpa
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers import MistralModel, MistralConfig, AutoModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING

from configuration_hinvec import BidirectionalMistralConfig, HinvecConfig, HinvecAConfig

from loguru import logger

class BidirectionalMistralModel(MistralModel):
    config_class = BidirectionalMistralConfig
    
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False
        self._attn_implementation = "eager"

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_attention_mask_for_sdpa(
                attention_mask, inputs_embeds.dtype
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_attention_mask(
                attention_mask, inputs_embeds.dtype,
            )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()
    
class HinvecAEmbedding(PreTrainedModel):
    config_class = HinvecAConfig
    
    def __init__(self, config: HinvecAConfig):
        super(HinvecAEmbedding, self).__init__(config)
        
        #self.embedding_model = embedding_model
        self.embedding_model = BidirectionalMistralModel(config.bidirectional_config)
        self.d_model = config.d_model
        self.hidden_dim = config.hidden_size
        self.attn_dim = config.hidden_size // 2
        self.scale = math.sqrt(self.hidden_dim)
        
        # Learnable Key matrix: d_model x hidden_dim
        self.key_states = nn.Parameter(torch.randn(config.d_model, self.attn_dim))
        
        # Optional: projection layers for Q and V if needed
        self.query_proj = nn.Linear(self.hidden_dim, self.attn_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.attn_dim)
        
        # Output projection to get d_model x hidden_dim
        self.output_proj = nn.Linear(self.attn_dim, self.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.key_states)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        
        # Initialize biases to zero
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, 
                output_attentions=False, token_type_ids=None, output_hidden_states=False):
        
        # Get embeddings from the embedding model
        outputs = self.embedding_model(input_ids, 
                                       attention_mask=attention_mask,
                                       past_key_values=past_key_values,
                                       use_cache=use_cache,
                                       output_attentions=output_attentions,
                                       token_type_ids=token_type_ids,
                                       output_hidden_states=output_hidden_states)
        
        embeddings = outputs.last_hidden_state
        embeddings_hidden_states = outputs.hidden_states
        
        batch_size, seq_len, d_model = embeddings.shape
        input_shape = embeddings.shape[:-1]  # (batch_size, seq_len)
        
        # Store all hidden states if requested
        all_hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states = embeddings_hidden_states + (embeddings,) if embeddings_hidden_states else (embeddings,)
        
        # Project embeddings to Q and V (using embeddings as source for both)
        # Q: (batch_size, seq_len, hidden_dim)
        query_states = self.query_proj(embeddings)
        
        # V: (batch_size, seq_len, d_model) 
        value_states = self.value_proj(embeddings)
        
        # Handle past_key_values if provided
        if past_key_values is not None:
            # In this implementation, we could extend key_states based on past values
            # For now, we keep the current implementation
            pass
        
        # Compute attention scores using scaled dot-product attention
        # Q: (batch_size, num_heads, seq_len, head_dim)
        # K: (batch_size, num_heads, d_model, head_dim)
        # Scores: (batch_size, num_heads, seq_len, d_model)
        attention_scores = torch.matmul(query_states, self.key_states.transpose(-1, -2)) / math.sqrt(self.attn_dim)
        
        # Apply attention mask if provided
        batch_size, seq_len, d_model_dim = attention_scores.shape
            
        if attention_scores.dim() == 3:  # (batch_size, seq_len, d_model)
            extended_mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            extended_mask = extended_mask.expand(-1, -1, d_model_dim)  # (batch_size, seq_len, d_model)
        else:  # (batch_size, num_heads, seq_len, d_model)
            # Convert attention mask to proper shape
            # attention_mask: (batch_size, seq_len) -> (batch_size, 1, seq_len, 1)
            extended_mask = attention_mask.view(batch_size, 1, seq_len, 1)
            # Expand to match attention scores: (batch_size, num_heads, seq_len, d_model)
            extended_mask = extended_mask.expand(-1, self.attn_dim, -1, self.d_model)

        attention_scores = attention_scores.masked_fill(extended_mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores.transpose(1, 2), dim=-1)
        
        # Apply dropout
        if self.training:
            attention_weights = self.dropout(attention_weights)

        # Reshape value_states to match attention computation
        value_states = value_states.view(batch_size, self.attn_dim, seq_len)
        
        # Compute attended output
        # (batch_size, num_heads, seq_len, d_model) @ (batch_size, num_heads, d_model, seq_len) 
        # -> (batch_size, num_heads, seq_len, seq_len)
        # Then @ (batch_size, num_heads, seq_len, value_dim) -> (batch_size, num_heads, seq_len, value_dim)
        
        # Transpose for correct matrix multiplication
        attn_output = torch.matmul(attention_weights, value_states.transpose(1,2))
        # attn_output: (batch_size, num_heads, d_model, value_dim)
        
        # Reshape back to original format
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, self.d_model, -1)
        
        # Final projection to get desired output dimension
        hidden_states = self.output_proj(attn_output)
        
        # Store final hidden states if requested
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,) if all_hidden_states else (hidden_states,)
        
        # Prepare past_key_values for next iteration if use_cache is True
        next_cache = None
        if use_cache:
            # Store current key-value pair for caching
            next_cache = ((self.key_states, value_states),) if past_key_values is None else past_key_values + ((self.key_states, value_states),)
        
        # Prepare attention outputs
        all_self_attns = None
        if output_attentions:
            # Average attention weights across heads for output
            avg_attention_weights = attention_weights.mean(dim=1)
            all_self_attns = (avg_attention_weights,)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def get_attention_weights(self, input_ids, attention_mask=None):
        """
        Get only the attention weights without computing the full forward pass.
        
        Returns:
            attention_weights: Attention weights of shape (batch_size, seq_len, d_model)
        """
        _, attention_weights = self.forward(input_ids, attention_mask)
        return attention_weights
    
class HinvecEmbedding(PreTrainedModel):
    config_class = HinvecConfig
    
    def __init__(self, config: HinvecConfig):
        super(HinvecEmbedding, self).__init__(config)
        
        self.embedding_model = BidirectionalMistralModel(config.bidirectional_config)
        self.hidden_dim = config.hidden_size
        
        # Learnable Key matrix: d_model x hidden_dim
        self.threshold = nn.Parameter(torch.tensor(0.5, dtype=torch.float16))
        self.redundancy_vector = nn.Parameter(torch.randn(1, config.hidden_size))

        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.redundancy_vector)
    
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, 
                output_attentions=False, token_type_ids=None, output_hidden_states=False):
        
        # Get embeddings from the embedding model
        outputs = self.embedding_model(input_ids, 
                                       attention_mask=attention_mask,
                                       past_key_values=past_key_values,
                                       use_cache=use_cache,
                                       output_attentions=output_attentions,
                                       token_type_ids=token_type_ids,
                                       output_hidden_states=output_hidden_states)
        
        embeddings = outputs.last_hidden_state

        redundancy_vector = self.redundancy_vector

        # Normalize embeddings and redundancy_vector
        normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
        normalized_vector = F.normalize(redundancy_vector, p=2, dim=-1)

        # Compute cosine similarity
        # result shape: (batch_size, seq_len)
        cosine_sim = torch.matmul(normalized_embeddings, normalized_vector.transpose(0,1))

        # Optionally invert the similarity if you're interested in dissimilar positions
        relevant_pos = attention_mask & (cosine_sim.reshape(cosine_sim.shape[:-1]) < self.threshold).int()

        outputs['relevant_pos'] = relevant_pos
        return outputs

class EmbeddingModel(nn.Module):
    """
    Wrapper model that outputs embeddings from the base model
    """
    def __init__(self, base_model, pooling_type="mean"):
        super().__init__()
        self.base_model = base_model
        self.pooling_type = pooling_type
        
    def forward(self, input_ids, attention_mask, ids=None, token_type_ids=None):

        outputs = self.base_model(input_ids=input_ids, 
                                  attention_mask=attention_mask,
                                  )
        
        # Use the appropriate pooling strategy
        if self.pooling_type == "mean":
            # Mean pooling over all tokens (accounting for padding via attention mask)
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float().to(token_embeddings.device)
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        elif self.pooling_type == "selective":
            token_embeddings = outputs.last_hidden_state
            outputs.relevant_pos[:,-1] = 1
            input_mask_expanded = outputs.relevant_pos.unsqueeze(-1).expand(token_embeddings.size()).float().to(token_embeddings.device)
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        elif self.pooling_type == "mean_with_attn":
            token_embeddings = outputs.last_hidden_state
            embeddings = token_embeddings.mean(dim=1)

        elif self.pooling_type == "cls":
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0]

        elif self.pooling_type == "eos":
            # Use [EOS] token embedding
            embeddings = outputs.last_hidden_state[:, -1]

        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
            
        return embeddings
        
AutoModel.register(BidirectionalMistralConfig, BidirectionalMistralModel)
AutoModel.register(HinvecAConfig, HinvecAEmbedding)
AutoModel.register(HinvecConfig, HinvecEmbedding)

BidirectionalMistralConfig.register_for_auto_class("AutoModel")