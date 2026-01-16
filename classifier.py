import argparse
import pickle
import sys
import pandas as pd
from model_utils import HierarchicalPasswordClassifier

LABELS = {0: "WEAK", 1: "NORMAL", 2: "STRONG"}

def load_model(model_path):
    """Load the trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def process_file(model, input_file, output_file):
    """Batch process passwords from a text file."""
    print(f"Processing passwords from {input_file}...")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            passwords = [line.strip() for line in f if line.strip()]
        
        if not passwords:
            print("Input file is empty.")
            return

        predictions = model.predict(passwords)
        
        print(f"Writing results to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for pwd, label in zip(passwords, predictions):
                f.write(f"{pwd} {label}\n")
        print("Done.")
        
    except Exception as e:
        print(f"Error processing file: {e}")

def interactive_mode(model):
    """Run the classifier in interactive mode."""
    print("\n=== Interactive Mode ===")
    print("Type a password and press Enter to classify.")
    print("Press Ctrl+C to exit.\n")
    
    while True:
        try:
            pwd = input("Password> ").strip()
            if not pwd: continue
            
            # Predict expects a list
            pred_label = model.predict([pwd])[0]
            label_name = LABELS.get(pred_label, "UNKNOWN")
            
            print(f"Result: [{pred_label}] {label_name}")
            print("-" * 30)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    # professional help message formatting
    parser = argparse.ArgumentParser(
        description="Classify passwords using the Pre-trained Hierarchical Model.",
        epilog="""
Modes of Operation:
  1. Interactive Mode:
     Run without -i and -o arguments to manually enter passwords.
     Example: python3 classifier.py -m model.pkl

  2. File Mode:
     Provide input and output files for batch processing.
     Example: python3 classifier.py -m model.pkl -i input.txt -o output.txt
""",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("-m", "--model", required=True, help="Path to the trained .pkl model file.")
    parser.add_argument("-i", "--input", help="Input text file containing passwords (one per line).")
    parser.add_argument("-o", "--output", help="Output file to save classification results.")
    
    args = parser.parse_args()

    # Load Model
    clf = load_model(args.model)

    # Determine mode based on arguments
    if args.input and args.output:
        process_file(clf, args.input, args.output)
    elif args.input or args.output:
        print("Error: Both -i (input) and -o (output) must be provided for File Mode.")
        parser.print_help()
    else:
        interactive_mode(clf)

if __name__ == '__main__':
    main()