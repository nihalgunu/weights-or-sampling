#!/usr/bin/env python3
"""
GenEval evaluation wrapper.
Evaluates generated images using the GenEval benchmark (object detection based).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "repos/geneval"))


def run_geneval_evaluation(
    image_dir: str,
    output_file: str,
    detector_path: str,
) -> dict:
    """
    Run GenEval evaluation on generated images.

    Returns dict with:
    - overall_score: float
    - per_category_scores: dict
    - detailed_results: list
    """

    image_dir = Path(image_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    geneval_dir = PROJECT_ROOT / "repos/geneval"

    # Check if geneval repo exists
    if not geneval_dir.exists():
        print("Error: GenEval repo not found. Please run setup.sh first.")
        return create_dummy_results()

    # Try to import geneval modules
    try:
        sys.path.insert(0, str(geneval_dir))
        from evaluation import evaluate_images, summary_scores

        print(f"Evaluating images in: {image_dir}")

        # Run evaluation
        results_jsonl = output_file.with_suffix('.jsonl')

        # Call evaluate_images script
        eval_cmd = [
            sys.executable,
            str(geneval_dir / "evaluation/evaluate_images.py"),
            str(image_dir),
            "--outfile", str(results_jsonl),
            "--model-path", detector_path,
        ]

        print(f"Running: {' '.join(eval_cmd)}")
        result = subprocess.run(eval_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Warning: evaluate_images.py returned non-zero: {result.stderr}")

        # Get summary scores
        if results_jsonl.exists():
            summary_cmd = [
                sys.executable,
                str(geneval_dir / "evaluation/summary_scores.py"),
                str(results_jsonl),
            ]

            summary_result = subprocess.run(summary_cmd, capture_output=True, text=True)
            print(summary_result.stdout)

            # Parse results
            return parse_geneval_results(results_jsonl, summary_result.stdout)
        else:
            print("Warning: Results file not created")
            return create_dummy_results()

    except ImportError as e:
        print(f"Warning: Could not import geneval modules: {e}")
        print("Running alternative evaluation...")
        return run_alternative_evaluation(image_dir, output_file)


def parse_geneval_results(results_jsonl: Path, summary_output: str) -> dict:
    """Parse GenEval evaluation results."""

    results = {
        "overall_score": 0.0,
        "per_category_scores": {},
        "detailed_results": [],
        "num_images": 0,
    }

    # Parse detailed results
    if results_jsonl.exists():
        with open(results_jsonl, 'r') as f:
            for line in f:
                try:
                    results["detailed_results"].append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        results["num_images"] = len(results["detailed_results"])

    # Parse summary output for scores
    # GenEval summary format varies, try to extract key metrics
    lines = summary_output.strip().split('\n')
    for line in lines:
        line = line.strip()
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            try:
                value = float(value.strip().rstrip('%'))
                if 'overall' in key or 'total' in key or 'accuracy' in key:
                    results["overall_score"] = value
                else:
                    results["per_category_scores"][key] = value
            except ValueError:
                continue

    return results


def run_alternative_evaluation(image_dir: Path, output_file: Path) -> dict:
    """
    Alternative evaluation when GenEval modules are not available.
    Uses a simpler approach based on image analysis.
    """

    print("Running simplified evaluation (GenEval not available)")

    results = {
        "overall_score": 0.0,
        "per_category_scores": {},
        "detailed_results": [],
        "num_images": 0,
        "note": "Simplified evaluation - GenEval not available"
    }

    # Count images
    image_files = list(Path(image_dir).glob("*.png")) + list(Path(image_dir).glob("*.jpg"))
    results["num_images"] = len(image_files)

    # Placeholder scores (should be replaced with actual evaluation)
    results["overall_score"] = -1.0  # Indicates evaluation not run
    results["per_category_scores"] = {
        "single_object": -1.0,
        "two_objects": -1.0,
        "counting": -1.0,
        "colors": -1.0,
        "position": -1.0,
        "color_attribution": -1.0,
    }

    return results


def create_dummy_results() -> dict:
    """Create dummy results for testing."""
    return {
        "overall_score": -1.0,
        "per_category_scores": {
            "single_object": -1.0,
            "two_objects": -1.0,
            "counting": -1.0,
            "colors": -1.0,
            "position": -1.0,
            "color_attribution": -1.0,
        },
        "detailed_results": [],
        "num_images": 0,
        "error": "Evaluation not run properly"
    }


def compute_fid(image_dir: str, reference_dir: str = None) -> float:
    """
    Compute FID score against reference images.

    Note: Requires pytorch-fid or clean-fid package.
    """
    try:
        from pytorch_fid import fid_score

        if reference_dir is None:
            # Use COCO or other reference
            print("Warning: No reference directory provided for FID")
            return -1.0

        fid = fid_score.calculate_fid_given_paths(
            [image_dir, reference_dir],
            batch_size=50,
            device='cuda',
            dims=2048,
        )
        return fid
    except ImportError:
        print("Warning: pytorch-fid not installed. Skipping FID computation.")
        return -1.0


def main():
    parser = argparse.ArgumentParser(description="Run GenEval evaluation")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing generated images")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSON file for results")
    parser.add_argument("--detector_path", type=str, required=True,
                        help="Path to GenEval detector models")
    parser.add_argument("--compute_fid", action="store_true",
                        help="Also compute FID score")
    parser.add_argument("--fid_reference", type=str, default=None,
                        help="Reference directory for FID computation")

    args = parser.parse_args()

    # Run evaluation
    results = run_geneval_evaluation(
        args.image_dir,
        args.output_file,
        args.detector_path,
    )

    # Optionally compute FID
    if args.compute_fid:
        fid = compute_fid(args.image_dir, args.fid_reference)
        results["fid"] = fid

    # Load generation metadata for timing info
    metadata_path = Path(args.image_dir) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if "generation_times" in metadata:
                times = metadata["generation_times"]
                results["avg_sec_per_image"] = sum(times) / len(times) if times else 0
                results["total_generation_time"] = sum(times)

    # Save results
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  Overall Score: {results.get('overall_score', 'N/A')}")
    print(f"  Num Images: {results.get('num_images', 'N/A')}")
    if 'avg_sec_per_image' in results:
        print(f"  Avg Time/Image: {results['avg_sec_per_image']:.2f}s")
    print(f"\nPer-category scores:")
    for cat, score in results.get('per_category_scores', {}).items():
        print(f"  {cat}: {score}")


if __name__ == "__main__":
    main()
