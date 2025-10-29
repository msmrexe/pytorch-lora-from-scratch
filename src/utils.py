"""
Utility functions and classes for the project, including logging,
dataset handling, and metric saving.
"""

import logging
import json
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoTokenizer

def setup_logging(log_dir="logs", log_file="training.log"):
    """Configures logging to both console and file."""
    log_path = os.path.join(log_dir, log_file)
    os.makedirs(log_dir, exist_ok=True)
    
    # Base logging config
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_path}")

def save_metrics(metrics, output_dir, file_name="results.json"):
    """Saves a dictionary of metrics to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, file_name)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics successfully saved to {file_path}")
    except IOError as e:
        logging.error(f"Failed to save metrics to {file_path}: {e}")

def load_processed_data(data_dir):
    """Loads processed train and test dataframes."""
    try:
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        
        # Load label map
        with open(os.path.join(data_dir, "label_map.json"), 'r', encoding='utf-8') as f:
            label_map = json.load(f)
            
        num_labels = len(label_map)
        logging.info(f"Loaded processed data. Train: {len(train_df)}, Test: {len(test_df)}, NumLabels: {num_labels}")
        return train_df, test_df, num_labels
    except FileNotFoundError as e:
        logging.error(f"Error loading processed data from {data_dir}: {e}")
        raise

class ComplaintDataset(Dataset):
    """
    A custom Dataset class for handling consumer complaints text data
    with BERT tokenization.
    """
    
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        try:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
            )

            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            logging.error(f"Error tokenizing text at index {idx}: {text} | Error: {e}")
            # Return a dummy item or raise error
            return None # Dataloader collate_fn should handle Nones if any
