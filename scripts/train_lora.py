"""
Experiment 3: LoRA Fine-Tuning of TinyBERT.

This script loads a pre-trained TinyBERT model, injects custom
LoRA layers (from-scratch implementation), and fine-tunes only
the LoRA parameters and the classifier head.
"""

import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from src.utils import setup_logging, load_processed_data, ComplaintDataset, save_metrics
from src.lora import apply_lora_to_bert, mark_only_lora_as_trainable, print_trainable_parameters
from src.training_module import train_epoch, evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of TinyBERT.")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory with processed data.")
    parser.add_argument("--output-dir", type=str, default="models/tinybert_lora", help="Directory to save model and results.")
    parser.add_argument("--base-model-name", type=str, default="prajjwal1/bert-tiny", help="Base model from Hugging Face.")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--max-len", type=int, default=512, help="Max sequence length for tokenizer.")
    # LoRA args
    parser.add_argument("--lora-r", type=int, default=8, help="Rank 'r' for LoRA matrices.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="Alpha scaling for LoRA.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(log_dir=args.output_dir, log_file="train_lora.log")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load data
    try:
        train_df, test_df, num_labels = load_processed_data(args.data_dir)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    
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

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_name, 
        num_labels=num_labels
    ).to(device)
    
    logging.info(f"Loaded base model {args.base_model_name}.")
    
    # --- Apply LoRA ---
    model = apply_lora_to_bert(model, r=args.lora_r, lora_alpha=args.lora_alpha)
    model = mark_only_lora_as_trainable(model)
    total_params = print_trainable_parameters(model)
    # --- --- --- --- ---
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

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
            # Note: This saves the whole model. For true PEFT, we would save
            # only the adapter weights, but this is simpler for this project.
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            logging.info(f"New best model saved with F1: {best_f1:.4f}")

    # Final evaluation and metrics saving
    logging.info("Training complete. Final evaluation...")
    test_loss, test_acc, test_prec, test_recall, test_f1 = evaluate(model, test_loader, device)
    
    metrics = {
        "model": "tinybert_lora",
        "trainable_params": total_params,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
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
