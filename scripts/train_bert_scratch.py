"""
Experiment 1: Train a Small BERT Model from Scratch.

This script initializes a small BERT model with a custom configuration
and trains it on the processed complaint data.
"""

import sys
import os
import argparse
import logging

import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertForSequenceClassification, AdamW, BertTokenizer

# Add the project root directory to the Python path
# This allows us to import from the 'src' folder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.utils import setup_logging, load_processed_data, ComplaintDataset, save_metrics
from src.training_module import train_epoch, evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Small BERT model from scratch.")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory with processed data.")
    parser.add_argument("--output-dir", type=str, default="models/bert_scratch", help="Directory to save model and results.")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--max-len", type=int, default=512, help="Max sequence length for tokenizer.")
    # BERT Config args
    parser.add_argument("--bert-hidden-size", type=int, default=256)
    parser.add_argument("--bert-num-heads", type=int, default=4)
    parser.add_argument("--bert-num-layers", type=int, default=4)
    parser.add_argument("--bert-intermediate-size", type=int, default=512)
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(log_dir=args.output_dir, log_file="train_bert_scratch.log")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load data
    try:
        train_df, test_df, num_labels = load_processed_data(args.data_dir)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Create datasets and dataloaders
    train_dataset = ComplaintDataset(
        texts=train_df["Processed_complaint"].values,
        labels=train_df["Product_encoded"].values,
        tokenizer=tokenizer,
        max_len=args.max_len
    )
    test_dataset = ComplaintDataset(
        texts=test_df["Processed_complaint"].values,
        labels=test_df["Product_encoded"].values,
        tokenizer=tokenizer,
        max_len=args.max_len
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Define model configuration
    config = BertConfig(
        num_labels=num_labels,
        hidden_size=args.bert_hidden_size,
        num_attention_heads=args.bert_num_heads,
        intermediate_size=args.bert_intermediate_size,
        num_hidden_layers=args.bert_num_layers
    )
    
    model = BertForSequenceClassification(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Initialized Small-BERT model. Trainable parameters: {total_params}")
    
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Training loop
    best_f1 = 0.0
    for epoch in range(args.num_epochs):
        logging.info(f"--- Epoch {epoch+1}/{args.num_epochs} ---")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        logging.info(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        
        test_loss, test_acc, test_prec, test_recall, test_f1 = evaluate(model, test_loader, device)
        logging.info(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")
        
        if test_f1 > best_f1:
            best_f1 = test_f1
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            logging.info(f"New best model saved with F1: {best_f1:.4f}")

    # Final evaluation and metrics saving
    logging.info("Training complete. Final evaluation...")
    test_loss, test_acc, test_prec, test_recall, test_f1 = evaluate(model, test_loader, device)
    
    metrics = {
        "model": "bert_from_scratch",
        "trainable_params": total_params,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_precision": test_prec,
        "test_recall": test_recall,
        "test_f1_score": test_f1
    }
    save_metrics(metrics, args.output_dir)
    logging.info("Process finished.")

if __name__ == "__main__":
    main()
