#!/usr/bin/env python3
import numpy as np
import pandas as pd
import tifffile
import networkx as nx
from scipy.stats import hmean
from scipy.optimize import linear_sum_assignment


class SegmentationEvaluator:
    """Evaluate segmentation between a reference (GT) and evaluation (target) mask."""

    def __init__(
        self,
        gt_mask,
        pred_mask,
        iou_threshold=0.5,
        graph_iou_threshold=0.1,
        unmatched_cost=0.4,
        cost_matrix_metric="iou",
    ):
        self.gt = gt_mask
        self.pred = pred_mask
        self.gt_ids = np.unique(self.gt)[1:]
        self.pred_ids = np.unique(self.pred)[1:]
        self.n_gt = len(self.gt_ids)
        self.n_pred = len(self.pred_ids)
        self.n = self.n_gt + self.n_pred

        self.iou_threshold = iou_threshold
        self.graph_iou_threshold = graph_iou_threshold
        self.unmatched_cost = unmatched_cost
        self.cost_matrix_metric = cost_matrix_metric
        self.get_metric_matrix()

    def get_metric_matrix(self):
        iou_matrix = np.zeros((len(self.gt_ids), len(self.pred_ids)))
        dice_matrix = np.zeros((len(self.gt_ids), len(self.pred_ids)))
        moc_matrix = np.zeros((len(self.gt_ids), len(self.pred_ids)))
        iou_graph = nx.Graph()
        for i, gt_id in enumerate(self.gt_ids):
            gt_labmask = self.gt == gt_id
            for j, pred_id in enumerate(self.pred_ids):
                pred_labmask = self.pred == pred_id
                intersection = np.logical_and(gt_labmask, pred_labmask).sum()
                union = np.logical_or(gt_labmask, pred_labmask).sum()
                gt_size = gt_labmask.sum()
                pred_size = pred_labmask.sum()
                iou = intersection / union if union > 0 else 0
                dice = (
                    (2 * intersection) / (gt_size + pred_size)
                    if (gt_size + pred_size) > 0
                    else 0
                )
                moc = (
                    (intersection / gt_size + intersection / pred_size) / 2
                    if gt_size > 0 and pred_size > 0
                    else 0
                )
                iou_matrix[i, j] = iou
                dice_matrix[i, j] = dice
                moc_matrix[i, j] = moc
                if iou > self.graph_iou_threshold:
                    iou_graph.add_edge(f"gt_{gt_id}", f"pred_{pred_id}", weight=1 - iou)
        self.iou_matrix = iou_matrix
        self.dice_matrix = dice_matrix
        self.moc_matrix = moc_matrix
        self.iou_graph = iou_graph

    def calculate_cost_matrices(self):
        self.cost_matrix_iou = self.construct_cost_matrix(self.iou_matrix)
        self.cost_matrix_dice = self.construct_cost_matrix(self.dice_matrix)
        self.cost_matrix_moc = self.construct_cost_matrix(self.moc_matrix)

    def construct_cost_matrix(self, metric_matrix):
        cost_matrix = np.ones((self.n, self.n))
        cost_matrix[: self.n_gt, : self.n_pred] = (
            1 - metric_matrix
        )  # A: Top-left block = real costs
        cost_matrix[self.n_gt :, self.n_pred :] = (
            1 - metric_matrix
        ).T  # D: Bottom-right = transpose of A
        top_right = self.unmatched_cost * np.eye(self.n_gt) + (1 - np.eye(self.n_gt))
        cost_matrix[: self.n_gt, self.n_pred :] = (
            top_right  # B: Top-right = unmatched GTs
        )
        bottom_left = self.unmatched_cost * np.eye(self.n_pred) + (
            1 - np.eye(self.n_pred)
        )
        cost_matrix[self.n_gt :, : self.n_pred] = (
            bottom_left  # C: Bottom-left = unmatched preds
        )
        return cost_matrix

    def specify_cost_matrix(self):
        if self.cost_matrix_metric == "iou":
            return self.construct_cost_matrix(self.iou_matrix)
        elif self.cost_matrix_metric == "dice":
            return self.construct_cost_matrix(self.dice_matrix)
        elif self.cost_matrix_metric == "moc":
            return self.construct_cost_matrix(self.moc_matrix)
        else:
            raise ValueError("Metric must be one of 'iou', 'dice', or 'moc'.")

    def evaluate(self):

        # Construct cost matrix
        cost_matrix = self.specify_cost_matrix()

        # Solve assignment
        order_res = linear_sum_assignment(cost_matrix)
        order_mat = np.zeros_like(cost_matrix)
        order_mat[order_res] = 1

        row_ind, col_ind = np.nonzero(order_mat[: self.n_gt, : self.n_pred])

        matched_pairs, iou_list, dice_list = [], [], []
        matched_gt, matched_pred = set(), set()

        for i, j in zip(row_ind, col_ind):
            if self.iou_matrix[i, j] >= self.iou_threshold:
                gt_id, pred_id = self.gt_ids[i], self.pred_ids[j]
                matched_pairs.append((gt_id, pred_id))
                matched_gt.add(gt_id)
                matched_pred.add(pred_id)

                gt_mask = self.gt == gt_id
                pred_mask = self.pred == pred_id
                intersection = np.logical_and(gt_mask, pred_mask).sum()
                dice = (
                    (2 * intersection) / (gt_mask.sum() + pred_mask.sum())
                    if (gt_mask.sum() + pred_mask.sum()) > 0
                    else 0
                )

                iou_list.append(self.iou_matrix[i, j])
                dice_list.append(dice)

        # Error classification
        splits, merges, catastrophes = 0, 0, 0
        split_details, merge_details, catastrophe_details = [], [], []

        # Remove matched nodes
        self.iou_graph.remove_nodes_from(
            [
                n
                for n in self.iou_graph.nodes
                if (n.startswith("pred_") and int(n.split("_")[1]) in matched_pred)
                or (n.startswith("gt_") and int(n.split("_")[1]) in matched_gt)
            ]
        )

        for component in nx.connected_components(self.iou_graph):
            gts = {n for n in component if n.startswith("gt_")}
            preds = {n for n in component if n.startswith("pred_")}
            ng, np_ = len(gts), len(preds)
            gt_nodes = [int(g.split("_")[1]) for g in gts]
            pred_nodes = [int(p.split("_")[1]) for p in preds]

            if ng == 1 and np_ > 1:
                splits += 1
                split_details.append({"gt": gt_nodes[0], "preds": sorted(pred_nodes)})
            elif ng > 1 and np_ == 1:
                merges += 1
                merge_details.append({"pred": pred_nodes[0], "gts": sorted(gt_nodes)})
            elif ng > 1 and np_ > 1:
                catastrophes += 1
                catastrophe_details.append(
                    {"gts": sorted(gt_nodes), "preds": sorted(pred_nodes)}
                )

        # Compute metrics
        tp = len(matched_pairs)
        fp = len(self.pred_ids) - len(matched_pred)
        fn = len(self.gt_ids) - len(matched_gt)
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = hmean([precision, recall]) if precision > 0 and recall > 0 else 0

        fp_preds = set(self.pred_ids) - set(matched_pred)
        fn_gts = set(self.gt_ids) - set(matched_gt)
        ttp_gts = set(matched_gt)
        ttp_preds = set(matched_pred)
        self.metrics = {
            "iou_mean": np.mean(iou_list) if iou_list else 0,
            "iou_list": iou_list,
            "dice_mean": np.mean(dice_list) if dice_list else 0,
            "dice_list": dice_list,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "splits": splits,
            "merges": merges,
            "catastrophes": catastrophes,
            "split_details": split_details,
            "merge_details": merge_details,
            "catastrophe_details": catastrophe_details,
            "iou_graph": self.iou_graph,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "n_gt_labels": self.n_gt,
            "n_pred_labels": self.n_pred,
            "FP_list": list(fp_preds),
            "FN_list": list(fn_gts),
            "TTP_gt": list(ttp_gts),
            "TTP_preds": list(ttp_preds),
            "total_cells": len(self.gt_ids),
            "total_pred_cells": len(self.pred_ids),
        }
        return self.metrics


class SegmentationEvaluationBatch:
    """Run evaluation for a dataframe of segmentation pairs."""

    def __init__(
        self,
        df,
        plotting=False,
        iou_threshold=0.5,
        graph_iou_threshold=0.1,
        unmatched_cost=0.4,
        cost_matrix_metric="iou",
    ):
        self.df = df.copy()
        self.iou_threshold = iou_threshold
        self.graph_iou_threshold = graph_iou_threshold
        self.unmatched_cost = unmatched_cost
        self.plotting = plotting
        self.cost_matrix_metric = cost_matrix_metric

    def run(self):
        results = []
        for idx, row in self.df.iterrows():
            print(f"Evaluating {row['sampleID']}...")
            gt = tifffile.imread(row["ref_mask"])
            pred = tifffile.imread(row["eval_mask"])
            evaluator = SegmentationEvaluator(
                gt,
                pred,
                self.iou_threshold,
                self.graph_iou_threshold,
                self.unmatched_cost,
                self.cost_matrix_metric,
            )
            metrics = evaluator.evaluate()
            results.append({**row, **metrics})
        self.results = pd.DataFrame(results)
        return self.results

    def summarize_by_category(self):
        grouped = self.results.groupby("category").agg(
            {
                "iou_mean": ["mean", "std"],
                "dice_mean": ["mean", "std"],
                "precision": ["mean", "std"],
                "recall": ["mean", "std"],
                "f1_score": ["mean", "std"],
                "splits": "sum",
                "merges": "sum",
                "catastrophes": "sum",
            }
        )
        self.summary = grouped
        return self.summary
