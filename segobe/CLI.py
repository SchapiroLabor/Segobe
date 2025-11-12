#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import tifffile

from segobe.evaluator import SegmentationEvaluationBatch
from segobe.plotter import plot_error_types, plot_barplot
from segobe import __version__


def get_args():
    parser = argparse.ArgumentParser(
        description="Run segmentation evaluation on a batch of segmentation masks."
    )
    parser.add_argument(
        "-i",
        "--input_csv",
        type=str,
        required=True,
        help="CSV file with columns: sampleID, ref_mask, eval_mask, category",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output metrics and plots",
    )
    parser.add_argument(
        "-b",
        "--basename",
        type=str,
        required=True,
        help="Unique basename used when saving metrics and plots",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching (0-1, default: 0.5)",
    )
    parser.add_argument(
        "--graph_iou_threshold",
        type=float,
        default=0.1,
        help="Graph IoU threshold for error detection (0-1, default: 0.1)",
    )
    parser.add_argument(
        "--unmatched_cost",
        type=float,
        default=0.4,
        help="Cost for unmatched objects (0-1, default: 0.4)",
    )
    parser.add_argument(
        "--cost_matrix_metric",
        type=str,
        default="iou",
        help="Metric used for cost matrix calculation (default: iou)",
        choices=["iou", "dice", "moc"],
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save plots of metrics and error types",
    )

    parser.add_argument("--version", action="version", version=f"Segobe {__version__}")

    return parser.parse_args()


def main():
    args = get_args()

    # Validate thresholds
    for threshold_name in ["iou_threshold", "graph_iou_threshold", "unmatched_cost"]:
        value = getattr(args, threshold_name)
        if not (0 <= value <= 1):
            raise ValueError(f"{threshold_name} must be between 0 and 1, got {value}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Read input CSV
    df = pd.read_csv(args.input_csv)

    print(args.cost_matrix_metric)
    # Run batch evaluation
    batch_eval = SegmentationEvaluationBatch(
        df,
        iou_threshold=args.iou_threshold,
        graph_iou_threshold=args.graph_iou_threshold,
        unmatched_cost=args.unmatched_cost,
        cost_matrix_metric=args.cost_matrix_metric,
    )
    results_df = batch_eval.run()

    # Save full results CSV
    csv_path = os.path.join(args.output_dir, f"{args.basename}_metrics.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

    # Summarize by category
    summary_df = batch_eval.summarize_by_category()
    summary_csv_path = os.path.join(args.output_dir, f"{args.basename}_summary.csv")
    summary_df.to_csv(summary_csv_path)
    print(f"Saved summary to {summary_csv_path}")

    # Optionally save plots
    if args.save_plots:
        plots_dir = os.path.join(args.output_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        barplot_path = os.path.join(plots_dir, f"{args.basename}_metrics_barplot.png")
        plot_barplot(summary_df, save_path=barplot_path)
        print(f"Saved barplot to {barplot_path}")

        # Optionally generate error type plots for each sample
        for _, row in df.iterrows():
            gt_mask = tifffile.imread(row["ref_mask"])
            pred_mask = tifffile.imread(row["eval_mask"])
            metrics = results_df.loc[results_df["sampleID"] == row["sampleID"]].to_dict(
                "records"
            )[0]
            # check if output_dir / plots exists

            plot_error_types(
                gt_mask,
                pred_mask,
                metrics,
                name=f"{row['sampleID']}_{row['category']}",
                save=True,
                save_path=os.path.join(
                    plots_dir,
                    f"{args.basename}_{row['sampleID']}_{row['category']}_error_types.png",
                ),
                suptitle=True,
            )


if __name__ == "__main__":
    main()
