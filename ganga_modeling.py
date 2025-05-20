import os
import argparse
import random
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif self.pooling_type == "cls":
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0]
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
            
        return embeddings
        
    