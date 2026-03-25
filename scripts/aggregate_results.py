#!/usr/bin/env python3
"""
Aggregate results from all experimental runs into a final summary table.
"""

import argparse
import csv
import json
from pathlib import Path

import wandb


def load_run_results(results_dir: str) -> dict:
    """Load results from all runs."""

    results_dir = Path(results_dir)
    all_results = {}

    # Expected run files
    run_names = [
        "run1_baseline",
        "run2_tfg",
        "run3_flowgrpo",
        "run4_flowgrpo_tfg",
    ]

    for run_name in run_names:
        result_file = results_dir / f"{run_name}.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                all_results[run_name] = json.load(f)
        else:
            print(f"Warning: Results for {run_name} not found")
            all_results[run_name] = None

    return all_results


def create_summary_table(all_results: dict) -> list:
    """Create summary table from all results."""

    # Table header
    header = [
        "Run",
        "GenEval (%)",
        "Single Obj",
        "Two Obj",
        "Counting",
        "Colors",
        "Position",
        "Color Attr",
        "FID",
        "Sec/Img",
    ]

    # Human-readable run names
    run_labels = {
        "run1_baseline": "1. Base SD3.5-M",
        "run2_tfg": "2. Base + TFG",
        "run3_flowgrpo": "3. FlowGRPO",
        "run4_flowgrpo_tfg": "4. FlowGRPO + TFG",
    }

    rows = [header]

    for run_name in ["run1_baseline", "run2_tfg", "run3_flowgrpo", "run4_flowgrpo_tfg"]:
        results = all_results.get(run_name)

        if results is None:
            row = [run_labels[run_name]] + ["N/A"] * (len(header) - 1)
        else:
            per_cat = results.get("per_category_scores", {})
            row = [
                run_labels[run_name],
                f"{results.get('overall_score', -1):.1f}",
                f"{per_cat.get('single_object', -1):.1f}",
                f"{per_cat.get('two_objects', -1):.1f}",
                f"{per_cat.get('counting', -1):.1f}",
                f"{per_cat.get('colors', -1):.1f}",
                f"{per_cat.get('position', -1):.1f}",
                f"{per_cat.get('color_attribution', -1):.1f}",
                f"{results.get('fid', -1):.1f}",
                f"{results.get('avg_sec_per_image', -1):.2f}",
            ]

        rows.append(row)

    return rows


def save_csv(rows: list, output_file: str):
    """Save results as CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def print_table(rows: list):
    """Print formatted table to console."""

    # Calculate column widths
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]

    # Print header
    header = rows[0]
    print("-" * (sum(widths) + len(widths) * 3 + 1))
    print("| " + " | ".join(str(h).ljust(w) for h, w in zip(header, widths)) + " |")
    print("-" * (sum(widths) + len(widths) * 3 + 1))

    # Print data rows
    for row in rows[1:]:
        print("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |")

    print("-" * (sum(widths) + len(widths) * 3 + 1))


def log_to_wandb(rows: list, wandb_project: str):
    """Log summary table to W&B."""

    run = wandb.init(
        project=wandb_project,
        name="summary_table",
        job_type="aggregation",
    )

    # Create W&B table
    columns = rows[0]
    data = rows[1:]

    table = wandb.Table(columns=columns, data=data)
    wandb.log({"results_summary": table})

    # Also log as comparison chart
    for i, col in enumerate(columns[1:], 1):
        chart_data = [[row[0], float(row[i].replace('N/A', '-1'))]
                      for row in data]
        wandb.log({
            f"comparison/{col}": wandb.plot.bar(
                wandb.Table(data=chart_data, columns=["Run", col]),
                "Run", col, title=f"{col} by Run"
            )
        })

    wandb.finish()


def compute_analysis(all_results: dict) -> dict:
    """Compute analysis of results (complementarity, etc.)."""

    analysis = {}

    # Get scores
    scores = {}
    for run_name, results in all_results.items():
        if results:
            scores[run_name] = results.get("overall_score", -1)

    # TFG-only lift
    if "run1_baseline" in scores and "run2_tfg" in scores:
        if scores["run1_baseline"] > 0 and scores["run2_tfg"] > 0:
            analysis["tfg_lift"] = scores["run2_tfg"] - scores["run1_baseline"]
            analysis["tfg_lift_pct"] = (analysis["tfg_lift"] / scores["run1_baseline"]) * 100

    # FlowGRPO-only lift
    if "run1_baseline" in scores and "run3_flowgrpo" in scores:
        if scores["run1_baseline"] > 0 and scores["run3_flowgrpo"] > 0:
            analysis["flowgrpo_lift"] = scores["run3_flowgrpo"] - scores["run1_baseline"]
            analysis["flowgrpo_lift_pct"] = (analysis["flowgrpo_lift"] / scores["run1_baseline"]) * 100

    # Complementarity: Does TFG help on top of FlowGRPO?
    if "run3_flowgrpo" in scores and "run4_flowgrpo_tfg" in scores:
        if scores["run3_flowgrpo"] > 0 and scores["run4_flowgrpo_tfg"] > 0:
            analysis["tfg_on_flowgrpo_lift"] = scores["run4_flowgrpo_tfg"] - scores["run3_flowgrpo"]
            analysis["complementary"] = analysis["tfg_on_flowgrpo_lift"] > 1.0  # >1% improvement

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Aggregate results from all runs")
    parser.add_argument("--results_dir", type=str, default="outputs/results",
                        help="Directory containing result JSON files")
    parser.add_argument("--output_file", type=str, default="outputs/results/final_table.csv",
                        help="Output CSV file")
    parser.add_argument("--wandb_project", type=str, default="tfg-rl-baselines",
                        help="W&B project for logging summary")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Skip W&B logging")

    args = parser.parse_args()

    # Load all results
    print("Loading results...")
    all_results = load_run_results(args.results_dir)

    # Create summary table
    print("\nCreating summary table...")
    rows = create_summary_table(all_results)

    # Print to console
    print("\n" + "=" * 80)
    print("FINAL RESULTS TABLE")
    print("=" * 80)
    print_table(rows)

    # Compute analysis
    analysis = compute_analysis(all_results)
    if analysis:
        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)
        if "tfg_lift" in analysis:
            print(f"TFG-only lift: {analysis['tfg_lift']:.1f}% ({analysis['tfg_lift_pct']:.1f}% relative)")
        if "flowgrpo_lift" in analysis:
            print(f"FlowGRPO-only lift: {analysis['flowgrpo_lift']:.1f}% ({analysis['flowgrpo_lift_pct']:.1f}% relative)")
        if "complementary" in analysis:
            if analysis["complementary"]:
                print(f"TFG + FlowGRPO are COMPLEMENTARY (additional {analysis['tfg_on_flowgrpo_lift']:.1f}% from TFG)")
            else:
                print(f"TFG + FlowGRPO are REDUNDANT (only {analysis['tfg_on_flowgrpo_lift']:.1f}% from TFG)")

    # Save CSV
    save_csv(rows, args.output_file)
    print(f"\nResults saved to: {args.output_file}")

    # Save analysis
    analysis_file = Path(args.output_file).with_name("analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {analysis_file}")

    # Log to W&B
    if not args.no_wandb:
        try:
            log_to_wandb(rows, args.wandb_project)
            print(f"\nSummary logged to W&B: {args.wandb_project}")
        except Exception as e:
            print(f"Warning: Could not log to W&B: {e}")


if __name__ == "__main__":
    main()
