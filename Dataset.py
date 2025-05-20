import json
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MultiTaskDataset(Dataset):
    """
    A custom dataset that loads multiple .jsonl files and implements a special
    batching strategy where:
    - 1st epoch: Each batch contains data from a single task
    - Later epochs: Each batch contains mixed data from different tasks
    """
    
    def __init__(self, data, tokenizer, max_source_length = 512, max_target_length=512):
        """
        Initialize the dataset by loading all the jsonl files.
        
        Args:
            data_dir (str): Directory containing the data files
            task_files (list): List of filenames to load (if None, load all .jsonl files)
            batch_size (int): Size of each batch
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length    
    
    def __len__(self):
        """Return the number of batches"""

        return len(self.data)
    
    def __getitem__(self, idx):
        """Return a single tokenized example"""
        item = self.data[idx]
        
        input_encoding = self.tokenizer(
            item['source'],
            padding=False,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors=None
        )
        
        output_encoding = self.tokenizer(
            item['target'],
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors=None
        )

        input_encoding['ids'] = item['id']
        input_encoding['input_ids'] = torch.tensor(input_encoding['input_ids'])
        input_encoding['attention_mask'] = torch.tensor(input_encoding['attention_mask'])
        input_encoding['token_type_ids'] = torch.tensor(input_encoding['token_type_ids'])
        input_encoding['output_ids'] = torch.tensor(output_encoding['input_ids'])
        input_encoding['output_attention_mask'] = torch.tensor(output_encoding['attention_mask'])
        input_encoding['labels'] = torch.tensor([1])

        return input_encoding