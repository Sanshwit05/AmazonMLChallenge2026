"""
Convenience entry point for the entity resolution pipeline.

Usage:
  # Run EDA first
  uv run python run.py eda

  # Train the model
  uv run python run.py train --max-s1 50000

  # Generate test predictions (requires trained model)
  uv run python run.py predict

  # Full end-to-end (train + predict)
  uv run python run.py full --max-s1 50000

  # Validate submission
  uv run python run.py validate
"""

import argparse
import os
import subprocess
import sys

# Ensure the project root is in the Python path for imports to resolve
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.eda import run_eda  # type: ignore
from src.pipeline import run_training, run_prediction  # type: ignore

def main():
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline Runner")
    parser.add_argument(
        "command",
        choices=["eda", "train", "predict", "full", "validate"],
        help="Command to run",
    )
    parser.add_argument("--max-s1", type=int, default=50000,
                       help="Max S1 entities for training (default: 50000)")
    parser.add_argument("--chunk-size", type=int, default=50000,
                       help="Chunk size for prediction (default: 50000)")
    args = parser.parse_args()

    if args.command == "eda":
        run_eda()

    elif args.command == "validate":
        # Run the official validation script
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        validate_script = os.path.join(
            project_root, "student_resource", "utils", "validate_submission.py"
        )
        matching = os.path.join(project_root, "output", "matching_results.tsv")
        candidate = os.path.join(project_root, "output", "candidate_pairs.tsv")
        test_dir = os.path.join(project_root, "student_resource", "dataset", "test")

        cmd = [
            sys.executable, validate_script,
            "--matching", matching,
            "--candidate", candidate,
            "--test-dir", test_dir,
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=os.path.join(project_root, "student_resource"))
        sys.exit(result.returncode)

    else:
        if args.command in ("train", "full"):
            model, feat_cols, threshold = run_training(max_s1=args.max_s1)

        if args.command in ("predict", "full"):
            if args.command == "full":
                run_prediction(
                    model=model, feat_cols=feat_cols, threshold=threshold,
                    chunk_size=args.chunk_size,
                )
            else:
                run_prediction(chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
