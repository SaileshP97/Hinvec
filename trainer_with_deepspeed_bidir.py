import os
import argparse
import random
import json
import math
from loguru import logger

from Dataset import MultiTaskDataset
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoConfig,
)
from transformers.trainer_callback import EarlyStoppingCallback
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, dispatch_model, infer_auto_device_map

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import wandb

from ganga_modeling import EmbeddingModel, BidirectionalMistralModel, BidirectionalMistralConfig

random.seed(42)
torch.manual_seed(42)

os.makedirs('./logs', exist_ok=True)

logger.add("./logs/logfile.log", level="INFO", rotation="1 MB", retention="7 days", compression="zip")

def get_jsonl(file):
    data = []
    with open(file, 'r') as f:
        for sample in f:
            data.append(json.loads(sample))
    return data

class CustomDataCollator(DataCollatorForSeq2Seq):

    def __init__(self, tokenizer, padding=True, pad_to_multiple_of=None, return_tensors="pt"):
        # Ensure the tokenizer has padding_side set to 'left' here
        tokenizer.padding_side = 'left'
        super().__init__(tokenizer, padding=padding, pad_to_multiple_of=pad_to_multiple_of, return_tensors=return_tensors)
        
    def __call__(self, features, return_tensors=None):
        # First get the standard collated batch

        batch_size = len(features)

        mask = torch.ones(batch_size, batch_size)
        for i in range(batch_size):
            for j in range(batch_size):
                # Set mask to 0 for samples with the same id (excluding self)
                if i != j and features[i]['ids'] == features[j]['ids']:
                    mask[i][j] = 0
        
        # Store the mask in each feature
        for i in range(batch_size):
            features[i]['ids'] = torch.tensor(mask[i])

        max_output_length = max([len(features[i]['output_ids']) for i in range(batch_size)])
        
        for i in range(batch_size):
        
            # Pad to max length if it's a list
            pad_length = max_output_length - len(features[i]['output_ids'])
            features[i]['output_ids'] = torch.tensor([0] * pad_length + list(features[i]['output_ids']))
            
            # Do the same for output_attention_mask if it exists
            if 'output_attention_mask' in features[i]:
                features[i]['output_attention_mask'] = torch.tensor([0] * pad_length + list(features[i]['output_attention_mask']))
        
        batch = super().__call__(features, return_tensors)

        return batch

def set_model_and_tokenizer(args):
    logger.info(f"Loading {args.model_name} Model")

    if args.use_devicemap_auto:
        
        base_model = AutoModel.from_pretrained(args.model_name,
                                        torch_dtype=torch.bfloat16,
                                        #attn_implementation="flash_attention_2",
                                        device_map="auto",
                                        )
    else:
        
        base_model = AutoModel.from_pretrained(args.model_name,
                                            torch_dtype=torch.bfloat16,
                                            #attn_implementation="flash_attention_2",
                                            #device_map="auto"
                                            ).to("cuda")
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = 'left'
    
    # Apply LoRA if enabled
    if args.use_lora:
        logger.info("Setting up LoRA for parameter-efficient fine-tuning")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias="none"
        )
        
        if args.use_8bit:
            logger.info("Preparing model for 8-bit training")
            base_model = prepare_model_for_kbit_training(base_model)
            base_model.print_trainable_parameters()
        
        base_model = get_peft_model(base_model, lora_config)
    else:
        base_model.enable_input_require_grads()
    
    # Create embedding model wrapper
    base_model.config.use_cache = False

    original_config = AutoConfig.from_pretrained(args.model_name)
    bidir_config = BidirectionalMistralConfig(**original_config.to_dict())
    bidir_model = BidirectionalMistralModel(bidir_config)
    bidir_model.load_state_dict(base_model.state_dict())
    bidir_model = bidir_model.to(torch.float16)

    bidir_model = EmbeddingModel(bidir_model, pooling_type=args.pooling_type)

    device_map = infer_auto_device_map(
        bidir_model,
        max_memory={0: "0.65GB",
                    1: "0.65GB", 
                    2: "0.65GB", 
                    3: "0.65GB"},
        no_split_module_classes=["MistralDecoderLayer"],
        dtype=torch.float16
    )

    model = dispatch_model(bidir_model, device_map=device_map)

    return model, tokenizer

def InfoNCELoss(query_embeddings, key_embeddings, ids, temperature=0.05):

    batch_size = query_embeddings.shape[0]
    
    # Normalize embeddings
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    key_embeddings = F.normalize(key_embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(query_embeddings, key_embeddings.transpose(0, 1)) / temperature
    sim_matrix = torch.exp(sim_matrix)
    
    losses = torch.zeros(batch_size, device=sim_matrix.device)
    for i in range(batch_size):

        denominator = torch.sum(sim_matrix[i] * ids[i])
        numerator = sim_matrix[i][i]
        losses[i] = -torch.log(numerator / denominator)
        
    return losses.mean()

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        query_embeddings = model(inputs.input_ids, inputs.attention_mask)
        key_embeddings = model(inputs.output_ids, inputs.output_attention_mask).detach()
        
        loss = InfoNCELoss(query_embeddings, key_embeddings, inputs['ids'])
        
        return (loss, query_embeddings) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        

        model.eval()
        with torch.no_grad():
            query_embeddings = model(inputs.input_ids, inputs.attention_mask)
            key_embeddings = model(inputs.output_ids, inputs.output_attention_mask).detach()
        
        loss = InfoNCELoss(query_embeddings, key_embeddings, inputs['ids'])

        torch.cuda.empty_cache()

        model.train()
        
        return loss, None, None


def main():
    parser = argparse.ArgumentParser(description="Fine Tuning Embedding Model with InfoNCE Loss using DeepSpeed")

    # Model arguments
    parser.add_argument("--model_name", type=str, required=True, help="Model to be fine-tuned")
    parser.add_argument("--pooling_type", type=str, default="mean", choices=["mean", "cls"], help="Pooling strategy for embeddings")
    parser.add_argument("--use_devicemap_auto", action="store_true", help="Use devicemap auto for model sharding")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU for training")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--temperature", type=float, default=0.02, help="Temperature parameter for InfoNCE loss")
    parser.add_argument("--logging_steps", type=int, default=1, help="Log every X steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Steps per epoch for scheduler (calculated if None)")
    parser.add_argument("--eval_steps", type=int, default=500, help="Steps after which evaluation is done.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient_accumulation_steps.")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./new_training_data", help="Directory with JSONL files")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--validation_split", type=float, default=0.01, help="Validation data fraction")
    
    # Optimization arguments
    parser.add_argument("--use_fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization for training")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora_r", type=int, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, help="LoRA dropout probability")
    parser.add_argument("--lora_target_modules", type=str, help="Comma-separated list of target modules for LoRA")

    # wandb use_wandb
    parser.add_argument("--use_wandb", action="store_false", help="Enable wandb")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for model checkpoints")
    
    # Add this line to accept --local_rank argument from DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Setup model
    model, tokenizer = set_model_and_tokenizer(args)
    tokenizer.padding_side = 'left'

    # Prepare datasets
    train_dataset = get_jsonl(f"{args.data_dir}/train_data.jsonl")
    val_dataset = get_jsonl(f"{args.data_dir}/val_data.jsonl")

    train_dataset = MultiTaskDataset(train_dataset, tokenizer, args.max_length, args.max_length)
    val_dataset = MultiTaskDataset(val_dataset, tokenizer, args.max_length, args.max_length)


    if args.use_wandb:
        wandb.init(
                    project = 'HinVec',
                    name = f"{args.model_name}-{args.pooling_type}-{args.batch_size}-{args.num_epochs}",
                    config=vars(args)
                )
        wandb.watch(model, log="all", log_freq=1)

    output_dir = f"{args.output_dir}-{args.pooling_type}-{args.batch_size}-epoch-{args.num_epochs}"

    # Define training arguments - include DeepSpeed config
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        logging_dir="./logs",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=not args.use_fp16,
        fp16=args.use_fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=1.0,
        report_to="wandb",
    )

    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=CustomDataCollator(tokenizer, 
                                         padding=True,
                                         pad_to_multiple_of=8,
                                         return_tensors="pt"),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Start training
    trainer.train()

    trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()