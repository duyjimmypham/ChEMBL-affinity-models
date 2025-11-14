#!/usr/bin/env python
"""
Visualize ROC/PR curves from results/{target_id}_metrics.json.

Usage:
    python scripts/plot_metrics.py CHEMBL203 [--model log_reg]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt


def load_metrics(target_id: str, results_dir: str = "results") -> Dict[str, Any]:
    """Load metrics JSON for a given target."""
    results_path = Path(results_dir)
    metrics_path = results_path / f"{target_id}_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_model(metrics: Dict[str, Any], model_name: Optional[str] = None) -> tuple[str, Dict[str, Any]]:
    """Choose which model's curves to plot."""
    models = metrics.get("models", {})
    if not models:
        raise ValueError("No models found in metrics JSON (missing 'models' field).")

    if model_name is not None:
        if model_name not in models:
            available = ", ".join(models.keys())
            raise ValueError(
                f"Requested model '{model_name}' not found. "
                f"Available: {available}"
            )
        return model_name, models[model_name]

    # Default: use best_model.name if present, otherwise the first key
    best = metrics.get("best_model")
    if isinstance(best, dict):
        best_name = best.get("name")
        if best_name in models:
            return best_name, models[best_name]

    # Fallback: deterministic order
    name = sorted(models.keys())[0]
    return name, models[name]


def plot_roc_curve(
    target_id: str,
    model_name: str,
    roc_curve: Dict[str, Any],
    labeling_strategy: Optional[str],
    out_dir: Path,
) -> Path:
    """Plot and save ROC curve."""
    fpr = roc_curve.get("fpr")
    tpr = roc_curve.get("tpr")
    if fpr is None or tpr is None:
        raise ValueError("ROC curve data must have 'fpr' and 'tpr' arrays.")

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    if labeling_strategy:
        title = f"{target_id} | {model_name} ROC ({labeling_strategy})"
    else:
        title = f"{target_id} | {model_name} ROC"
    plt.title(title)

    out_path = out_dir / f"{target_id}_{model_name}_roc.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path


def plot_pr_curve(
    target_id: str,
    model_name: str,
    pr_curve: Dict[str, Any],
    labeling_strategy: Optional[str],
    out_dir: Path,
) -> Path:
    """Plot and save Precision–Recall curve."""
    precision = pr_curve.get("precision")
    recall = pr_curve.get("recall")
    if precision is None or recall is None:
        raise ValueError("PR curve data must have 'precision' and 'recall' arrays.")

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    if labeling_strategy:
        title = f"{target_id} | {model_name} PR ({labeling_strategy})"
    else:
        title = f"{target_id} | {model_name} PR"
    plt.title(title)

    out_path = out_dir / f"{target_id}_{model_name}_pr.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ROC/PR curves for a trained target-specific model "
                    "using results/{target_id}_metrics.json."
    )
    parser.add_argument(
        "target_id",
        help="ChEMBL target ID, e.g. CHEMBL203",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        default=None,
        help="Model name to plot (e.g. log_reg, random_forest, xgboost). "
             "Defaults to best_model.name from metrics JSON.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing {target_id}_metrics.json (default: results)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to write PNG files. "
             "Default: results/{target_id}_plots",
    )

    args = parser.parse_args()

    # Load metrics
    metrics = load_metrics(args.target_id, results_dir=args.results_dir)

    labeling_strategy = metrics.get("labeling_strategy")
    class_counts = metrics.get("class_counts", {})
    n_raw = metrics.get("n_raw_rows")
    n_filtered = metrics.get("n_filtered_rows")
    n_total = metrics.get("n_molecules_total")
    n_labeled = metrics.get("n_molecules_labeled")

    # Choose which model to visualize
    model_name, model_metrics = choose_model(metrics, model_name=args.model_name)
    roc_curve = model_metrics.get("roc_curve")
    pr_curve = model_metrics.get("pr_curve")

    # Figure out output directory
    if args.out_dir is None:
        # Default: keep plots alongside results, per-target
        out_dir = Path(args.results_dir) / f"{args.target_id}_plots"
    else:
        out_dir = Path(args.out_dir)

    # Basic summary in stdout
    print(f"Target: {args.target_id}")
    print(f"Labeling strategy: {labeling_strategy}")
    print(f"Class counts: {class_counts}")
    print(
        f"Data: raw_rows={n_raw}, filtered_rows={n_filtered}, "
        f"molecules_total={n_total}, molecules_labeled={n_labeled}"
    )
    print(f"Selected model: {model_name}")

    test_roc = model_metrics.get("test_roc_auc")
    test_pr = model_metrics.get("test_pr_auc")
    if test_roc is not None and test_pr is not None:
        print(f"Test ROC-AUC: {test_roc:.3f}, Test PR-AUC: {test_pr:.3f}")
    else:
        print("No test ROC/PR AUC found in metrics for this model.")

    # Plot ROC
    if roc_curve:
        try:
            roc_path = plot_roc_curve(
                args.target_id,
                model_name,
                roc_curve,
                labeling_strategy,
                out_dir,
            )
            print(f"ROC curve saved to: {roc_path}")
        except Exception as e:
            print(f"Skipping ROC plot due to error: {e}")
    else:
        print("No ROC curve data found in metrics for this model.")

    # Plot PR
    if pr_curve:
        try:
            pr_path = plot_pr_curve(
                args.target_id,
                model_name,
                pr_curve,
                labeling_strategy,
                out_dir,
            )
            print(f"PR curve saved to: {pr_path}")
        except Exception as e:
            print(f"Skipping PR plot due to error: {e}")
    else:
        print("No PR curve data found in metrics for this model.")


if __name__ == "__main__":
    main()
