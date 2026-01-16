import argparse
import pickle
import pandas as pd
import numpy as np
from model_utils import HierarchicalPasswordClassifier

def load_data(filepath):
    """
    Loads data from a text file.
    Expected format: Space-separated file with header "password strength"
    """
    print(f"Loading dataset from {filepath}...")
    try:
        # REMOVED names=['password', 'strength'] to let pandas infer header automatically
        df = pd.read_csv(filepath, sep=' ', on_bad_lines='skip')
        
        # Standardize column names to lowercase to avoid case-sensitivity issues
        df.columns = df.columns.str.strip().str.lower()
        
        # Check if required columns exist
        if 'password' not in df.columns or 'strength' not in df.columns:
            print("Error: Dataset must contain 'password' and 'strength' columns in the header.")
            exit(1)

        df = df.dropna()
        
        # Ensure labels are integers
        df["strength"] = df['strength'].astype(int)
        
        # Ensure passwords are treated as strings (in case of purely numeric passwords like '123456')
        df["password"] = df["password"].astype(str)

        return df["password"].values, df["strength"].values
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Train the Hierarchical Password Classifier.")
    parser.add_argument("-i", "--input", required=True, help="Path to the training dataset (txt file).")
    parser.add_argument("-o", "--output", required=True, help="Path to save the trained model (.pkl).")
    args = parser.parse_args()

    # 1. Load Data
    X, y = load_data(args.input)
    print(f"Dataset loaded: {len(X)} samples.")

    # 2. Initialize and Train Model
    print("Training Hierarchical Model (BoW + Logistic Regression)...")
    clf = HierarchicalPasswordClassifier()
    clf.fit(X, y)
    print("Training completed.")

    # 3. Save Model
    print(f"Saving model to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(clf, f)
    print("Done.")

if __name__ == '__main__':
    main()