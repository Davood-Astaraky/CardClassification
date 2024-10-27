import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.utils import print_versions
from train import start_training
from evaluate import main as evaluate_main  

def main():
    # Argument parser for selecting mode
    parser = argparse.ArgumentParser(description="Select mode: train or evaluate.")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train", help="Choose 'train' to train the model or 'evaluate' to run evaluation.")
    
    # Use parse_known_args to ignore any extra arguments (such as Jupyter’s own)
    args, _ = parser.parse_known_args()

    # Print version information
    print_versions()

    # Mode selection
    if args.mode == "train":
        print("Starting the training process...")
        start_training()
    elif args.mode == "evaluate":
        print("Starting the evaluation process...")
        evaluate_main()

if __name__ == "__main__":
    main()
