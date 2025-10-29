"""
Shared Training and Evaluation Module.

Contains the core train_epoch and evaluate functions used by all
training scripts (BERT-from-scratch, Full-Finetune, LoRA).
"""

import torch
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def train_epoch(model, data_loader, optimizer, device):
    """Performs one training epoch."""
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        try:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            progress_bar.set_postfix(loss=loss.item())

        except Exception as e:
            logging.error(f"Error during training batch: {e}")
            continue # Skip batch on error

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy

def evaluate(model, data_loader, device):
    """Performs evaluation on a dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
    
    with torch.no_grad():
        for batch in progress_bar:
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            except Exception as e:
                logging.error(f"Error during evaluation batch: {e}")
                continue # Skip batch on error

    avg_loss = total_loss / len(data_loader)
    
    try:
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        return avg_loss, 0.0, 0.0, 0.0, 0.0

    return avg_loss, accuracy, precision, recall, f1
