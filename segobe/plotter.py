#!/usr/bin/env python3
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from skimage import segmentation
from segobe.utils import filter_mask_by_ids

"""Plot summary statistics from evaluated results."""


def plot_barplot(metrics_df, save_path=None):
    metrics_df.columns = ["_".join(col).strip() for col in metrics_df.columns.values]
    models = metrics_df.index.tolist()
    x = np.arange(len(models))
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        x,
        metrics_df["iou_mean_mean"],
        bar_width,
        yerr=metrics_df["iou_mean_std"],
        label="IoU",
    )
    ax.bar(
        x + bar_width,
        metrics_df["dice_mean_mean"],
        bar_width,
        yerr=metrics_df["dice_mean_std"],
        label="Dice",
    )
    ax.bar(
        x + bar_width * 2,
        metrics_df["precision_mean"],
        bar_width,
        yerr=metrics_df["precision_std"],
        label="Precision",
    )
    ax.bar(
        x + bar_width * 3,
        metrics_df["recall_mean"],
        bar_width,
        yerr=metrics_df["recall_std"],
        label="Recall",
    )
    ax.bar(
        x + bar_width * 4,
        metrics_df["f1_score_mean"],
        bar_width,
        yerr=metrics_df["f1_score_std"],
        label="F1",
    )
    ax.set_xticks(x + 2 * bar_width)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_error_types(
    gt_mask,
    pred_mask,
    metrics,
    name,
    save=False,
    save_path=None,
    suptitle=False,
    legend=False,
):

    # Extract categories
    tp_gt = metrics["TTP_gt"]
    tp_preds = metrics["TTP_preds"]
    fp_list = metrics["FP_list"]
    fn_list = metrics["FN_list"]
    split_gt = [s["gt"] for s in metrics["split_details"]]
    split_preds = [s["preds"] for s in metrics["split_details"]]
    merge_gt = [m["gts"] for m in metrics["merge_details"]]
    merge_preds = [m["pred"] for m in metrics["merge_details"]]
    cat_gt = [c["gts"] for c in metrics["catastrophe_details"]]
    cat_preds = [c["preds"] for c in metrics["catastrophe_details"]]

    # Colors
    blue = "#1f77b4"
    yellow = "#ffdb58"
    green = "#32CD32"

    # Categories: (Title, GT Mask, Pred Mask)
    categories = [
        ("Ground Truth", filter_mask_by_ids(gt_mask, np.unique(gt_mask)[1:]), None),
        ("Prediction", None, filter_mask_by_ids(pred_mask, np.unique(pred_mask)[1:])),
        (
            "True Positives",
            filter_mask_by_ids(gt_mask, tp_gt),
            filter_mask_by_ids(pred_mask, tp_preds),
        ),
        ("False Negatives", filter_mask_by_ids(gt_mask, fn_list), None),
        ("False Positives", None, filter_mask_by_ids(pred_mask, fp_list)),
        (
            "Merges",
            filter_mask_by_ids(gt_mask, merge_gt),
            filter_mask_by_ids(pred_mask, merge_preds),
        ),
        (
            "Splits",
            filter_mask_by_ids(gt_mask, split_gt),
            filter_mask_by_ids(pred_mask, split_preds),
        ),
        (
            "Catastrophes",
            filter_mask_by_ids(gt_mask, cat_gt),
            filter_mask_by_ids(pred_mask, cat_preds),
        ),
    ]

    fig, axes = plt.subplots(1, 8, figsize=(20, 4))
    if suptitle:
        plt.suptitle(
            f"{name}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, "
            f"F1={metrics['f1_score']:.2f}, splits={metrics['splits']}, merges={metrics['merges']}, "
            f"catastrophes={metrics['catastrophes']}, TP={metrics['true_positives']}, "
            f"FP={metrics['false_positives']}, FN={metrics['false_negatives']}",
            fontsize=14,
        )
    else:
        print(
            f"{name}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, "
            f"F1={metrics['f1_score']:.2f}, splits={metrics['splits']}, merges={metrics['merges']}, "
            f"catastrophes={metrics['catastrophes']}, TP={metrics['true_positives']}, "
            f"FP={metrics['false_positives']}, FN={metrics['false_negatives']}"
        )

    for idx, (title, gt, pred) in enumerate(categories):
        ax = axes[idx]
        ax.axis("off")
        ax.set_title(title, fontsize=14)

        # Black background
        ax.imshow(np.zeros_like(gt_mask, dtype=np.uint8), cmap="gray")

        # Overlap: green where both gt and pred are present
        if gt is not None and pred is not None:
            overlap = (gt > 0) & (pred > 0)
            only_gt = (gt > 0) & (~overlap)
            only_pred = (pred > 0) & (~overlap)

            ax.imshow(
                np.ma.masked_where(~overlap, overlap),
                cmap=ListedColormap([green]),
                alpha=0.8,
            )
            ax.imshow(
                np.ma.masked_where(~only_gt, only_gt),
                cmap=ListedColormap([blue]),
                alpha=0.8,
            )
            ax.imshow(
                np.ma.masked_where(~only_pred, only_pred),
                cmap=ListedColormap([yellow]),
                alpha=0.8,
            )

            ax.contour(
                segmentation.find_boundaries(gt, mode="outer"),
                colors=blue,
                linewidths=0.5,
            )
            ax.contour(
                segmentation.find_boundaries(pred, mode="outer"),
                colors=yellow,
                linewidths=0.5,
            )

        elif gt is not None:
            ax.imshow(
                np.ma.masked_where(gt == 0, gt), cmap=ListedColormap([blue]), alpha=0.8
            )
            ax.contour(
                segmentation.find_boundaries(gt, mode="outer"),
                colors=blue,
                linewidths=0.5,
            )

        elif pred is not None:
            ax.imshow(
                np.ma.masked_where(pred == 0, pred),
                cmap=ListedColormap([yellow]),
                alpha=0.8,
            )
            ax.contour(
                segmentation.find_boundaries(pred, mode="outer"),
                colors=yellow,
                linewidths=0.5,
            )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.05, wspace=0.05)

    # Legend
    legend_elements = [
        Patch(facecolor=blue, edgecolor=blue, label="Ground Truth"),
        Patch(facecolor=yellow, edgecolor=yellow, label="Prediction"),
        Patch(facecolor=green, edgecolor=green, label="Overlap"),
    ]
    if legend:
        fig.legend(
            handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.05)
        )

    if save:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
