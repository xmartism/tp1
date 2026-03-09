"""
Dummy Evaluation Script

Usage:
    python3 evaluate.py --model-name <name> --output-file <model_output.txt> --results-file results.csv
"""

import argparse
import csv
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="Name of the model being evaluated")
    parser.add_argument("--output-file", required=True, help="Path to the model output file")
    parser.add_argument("--results-file", required=True, help="Path to the results CSV file")
    args = parser.parse_args()

    # Dummy evaluation: just print a message
    evaluation_message = f"Model {args.model_name} was evaluated"
    print(evaluation_message)

    # Write into results.csv
    results_path = Path(args.results_file)
    file_exists = results_path.exists()

    with open(results_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(["model_name", "output_file", "result"])

        writer.writerow([args.model_name, args.output_file, evaluation_message])

    print(f"Results written to {args.results_file}")


if __name__ == "__main__":
    main()