#!/usr/bin/env python3
"""
W&B logging utility for experiment results.
"""

import argparse
import json
from pathlib import Path

import wandb


def log_results_to_wandb(
    results_file: str,
    run_name: str,
    wandb_project: str,
    wandb_entity: str = None,
):
    """Log evaluation results to Weights & Biases."""

    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Initialize W&B run
    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        job_type="evaluation",
        config={"run_name": run_name},
    )

    # Log overall metrics
    metrics = {
        "geneval/overall": results.get("overall_score", -1),
        "geneval/num_images": results.get("num_images", 0),
    }

    # Log per-category scores
    for cat, score in results.get("per_category_scores", {}).items():
        metrics[f"geneval/{cat}"] = score

    # Log timing
    if "avg_sec_per_image" in results:
        metrics["timing/sec_per_image"] = results["avg_sec_per_image"]
    if "total_generation_time" in results:
        metrics["timing/total_seconds"] = results["total_generation_time"]

    # Log FID if available
    if "fid" in results:
        metrics["quality/fid"] = results["fid"]

    wandb.log(metrics)

    # Log sample images if available
    image_dir = Path(results_file).parent.parent / "images" / run_name
    if image_dir.exists():
        images = list(image_dir.glob("*.png"))[:10]  # Log first 10 images
        if images:
            wandb.log({
                "sample_images": [
                    wandb.Image(str(img), caption=img.stem)
                    for img in images
                ]
            })

    # Create summary table
    summary_data = [[run_name, results.get("overall_score", -1)]]
    for cat, score in results.get("per_category_scores", {}).items():
        summary_data[0].extend([score])

    # Log as artifact
    artifact = wandb.Artifact(
        name=f"results_{run_name}",
        type="evaluation_results",
    )
    artifact.add_file(results_file)
    run.log_artifact(artifact)

    wandb.finish()

    print(f"Results logged to W&B: {wandb_project}/{run_name}")


def main():
    parser = argparse.ArgumentParser(description="Log results to W&B")
    parser.add_argument("--results_file", type=str, required=True,
                        help="Path to results JSON file")
    parser.add_argument("--run_name", type=str, required=True,
                        help="Name of the run")
    parser.add_argument("--wandb_project", type=str, default="tfg-rl-baselines",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (username/team)")

    args = parser.parse_args()

    log_results_to_wandb(
        args.results_file,
        args.run_name,
        args.wandb_project,
        args.wandb_entity,
    )


if __name__ == "__main__":
    main()
