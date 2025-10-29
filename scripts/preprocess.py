"""
Data Preprocessing Script.

Loads the raw consumer complaint dataset, samples it, merges and filters
classes, cleans text, and splits it into train/test sets, saving
the results to an output directory.
"""

import sys
import os

import argparse
import re
import json
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add the project root directory to the Python path
# This allows us to import from the 'src' folder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.utils import setup_logging

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess consumer complaint data.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the raw complaints_small.csv file.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save processed files.")
    parser.add_argument("--sample-frac", type=float, default=0.1, help="Fraction of data to sample.")
    parser.add_argument("--class-threshold", type=int, default=1000, help="Minimum samples per class to keep.")
    parser.add_argument("--min-text-length", type=int, default=10, help="Minimum number of words for a complaint to be kept.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data to use for the test set.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility.")
    return parser.parse_args()

def preprocess_text(text):
    """Cleans and preprocesses a single text string."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.replace('\n', ' ').replace('\r', '').strip()
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special chars
    text = ' '.join(text.split())  # Remove extra spaces
    return text

def main():
    """Main data processing pipeline."""
    args = parse_args()
    setup_logging(log_file="preprocessing.log")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        logging.info(f"Loading raw data from {args.input_file}...")
        data = pd.read_csv(args.input_file)
        logging.info(f"Initial dataset shape: {data.shape}")
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_file}")
        return
    
    # Drop missing values
    data = data.dropna(subset=['Product', 'Consumer complaint narrative'])
    logging.info(f"Shape after dropping NaNs: {data.shape}")

    # Sample data
    logging.info(f"Sampling {args.sample_frac * 100}% of the data...")
    sampled_data = data.sample(frac=args.sample_frac, random_state=args.random_state)
    logging.info(f"Sampled data shape: {sampled_data.shape}")

    # Merge classes
    class_mapping = {
        "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
        "Credit reporting or other personal consumer reports": "Credit reporting",
        "Credit reporting": "Credit reporting",
        "Payday loan, title loan, or personal loan": "Payday or personal loan",
        "Payday loan, title loan, personal loan, or advance loan": "Payday or personal loan",
        "Payday loan": "Payday or personal loan",
        "Money transfer, virtual currency, or money service": "Money transfer or virtual currency",
        "Money transfers": "Money transfer or virtual currency",
        "Virtual currency": "Money transfer or virtual currency",
        "Bank account or service": "Checking or savings account",
        "Debt or credit management": "Debt collection",
        "Credit card": "Credit card or prepaid card",
        "Prepaid card": "Credit card or prepaid card",
    }
    sampled_data["Product"] = sampled_data["Product"].replace(class_mapping)
    logging.info("Merged similar classes.")

    # Filter by major classes
    class_counts = sampled_data["Product"].value_counts()
    major_classes = class_counts[class_counts >= args.class_threshold].index
    filtered_data = sampled_data[sampled_data["Product"].isin(major_classes)]
    logging.info(f"Filtered to {len(major_classes)} major classes. New shape: {filtered_data.shape}")

    # Label Encoding
    label_encoder = LabelEncoder()
    filtered_data = filtered_data.copy()
    filtered_data.loc[:, "Product_encoded"] = label_encoder.fit_transform(filtered_data["Product"])
    
    # Save label map
    label_map = {cls: int(code) for cls, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
    map_path = os.path.join(args.output_dir, "label_map.json")
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=4)
    logging.info(f"Saved label map to {map_path}")

    # Text Preprocessing
    logging.info("Applying text preprocessing...")
    filtered_data.loc[:, "Processed_complaint"] = filtered_data["Consumer complaint narrative"].apply(preprocess_text)

    # Remove short complaints
    min_len = args.min_text_length
    filtered_data = filtered_data[filtered_data["Processed_complaint"].apply(lambda x: len(x.split()) >= min_len)]
    logging.info(f"Removed complaints shorter than {min_len} words. Final shape: {filtered_data.shape}")

    # Split data
    logging.info(f"Splitting data into train/test sets (test_size={args.test_size})...")
    train_df, test_df = train_test_split(
        filtered_data,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=filtered_data["Product_encoded"]
    )

    # Save processed data
    train_path = os.path.join(args.output_dir, "train.csv")
    test_path = os.path.join(args.output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logging.info(f"Processed data saved. Train: {train_path}, Test: {test_path}")
    logging.info("Preprocessing complete.")

if __name__ == "__main__":
    main()
