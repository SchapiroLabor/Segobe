#!/usr/bin/env python3
import numpy as np
import pandas as pd
import tifffile
from collections import Counter, defaultdict, deque
from scipy.stats import hmean
from scipy.optimize import linear_sum_assignment


class SegmentationEvaluator:
    """
    Memory-efficient cell matcher.
    Handles merges, splits, catastrophes.
    Ensures unmatched cells appear in IoU/Dice/MOC matrices.
    """

    def __init__(
        self,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        iou_threshold: float = 0.5,
        graph_iou_threshold: float = 0.1,
        unmatched_cost: float = 0.4,
    ):
        assert gt_mask.shape == pred_mask.shape
        self.gt = gt_mask
        self.pred = pred_mask
        self.iou_threshold = iou_threshold
        self.graph_iou_threshold = graph_iou_threshold
        self.unmatched_cost = unmatched_cost

        # label sets
        self.gt_ids = np.unique(self.gt)
        self.pred_ids = np.unique(self.pred)
        self.gt_ids = self.gt_ids[self.gt_ids != 0]
        self.pred_ids = self.pred_ids[self.pred_ids != 0]

        # Compute cell sizes
        gt_counts = np.bincount(self.gt.ravel())
        pred_counts = np.bincount(self.pred.ravel())
        self.gt_sizes = {int(i): int(gt_counts[i]) for i in self.gt_ids}
        self.pred_sizes = {int(i): int(pred_counts[i]) for i in self.pred_ids}

        # Compute sparse intersections
        self.intersections = self._compute_intersections(self.gt, self.pred)

        # Build pair_df (all intersecting object pairs)
        rows = []
        for (g, p), inter in self.intersections.items():
            gsz = self.gt_sizes[g]
            psz = self.pred_sizes[p]
            union = gsz + psz - inter
            iou = inter / union if union > 0 else 0.0
            dice = (2 * inter) / (gsz + psz)
            moc = 0.5 * (inter / gsz + inter / psz)

            rows.append((g, p, inter, iou, dice, moc))

        if rows:
            self.pair_df = pd.DataFrame(
                rows, columns=["gt", "pred", "intersection", "iou", "dice", "moc"]
            )
        else:
            self.pair_df = pd.DataFrame(
                columns=["gt", "pred", "intersection", "iou", "dice", "moc"]
            )

    def _compute_intersections(self, gt: np.ndarray, pred: np.ndarray) -> Counter:
        """Get intersections through encoding"""
        max_pred = np.int64(pred.max() + 1)  # int64 scalar

        # multiply only in int64 temporarily
        joint = np.multiply(gt, max_pred, dtype=np.int64)  # result is int64
        joint += pred  # pred is int32, addition is safe

        vals, counts = np.unique(joint, return_counts=True)
        gt_ids = vals // max_pred
        pred_ids = vals % max_pred

        c = Counter()
        for g, p, cnt in zip(gt_ids, pred_ids, counts):
            if g != 0 and p != 0:
                c[(int(g), int(p))] = int(cnt)
        return c

    #
    # Get adjacency graph
    # TODO: can we filter out cells below graph threshold already?
    def _adjacency_graph(self):
        adj_gt_to_pred = defaultdict(set)
        adj_pred_to_gt = defaultdict(set)
        for row in self.pair_df.itertuples(index=False):
            if row.iou > 0:
                adj_gt_to_pred[int(row.gt)].add(int(row.pred))
                adj_pred_to_gt[int(row.pred)].add(int(row.gt))
        return adj_gt_to_pred, adj_pred_to_gt

    #
    # Get connected components (not using networkx)
    #
    def _connected_components_prematching(self, adj_gt2p, adj_p2gt):
        visited_g = set()
        visited_p = set()

        for g0 in adj_gt2p.keys():
            if g0 in visited_g:
                continue

            comp_g = set()
            comp_p = set()
            q = deque([("g", g0)])
            visited_g.add(g0)

            while q:
                t, x = q.popleft()

                if t == "g":
                    comp_g.add(x)
                    for p in adj_gt2p[x]:
                        if p not in visited_p:
                            visited_p.add(p)
                            q.append(("p", p))

                else:  # p → g
                    comp_p.add(x)
                    for g in adj_p2gt[x]:
                        if g not in visited_g:
                            visited_g.add(g)
                            q.append(("g", g))

            yield comp_g, comp_p

    def _connected_components_postmatching(self, adj_gt_to_preds, adj_pred_to_gts):
        """
        Return connected components of a bipartite graph:
        - left nodes: GT labels
        - right nodes: Pred labels

        adj_gt_to_preds[g] = set of pred IDs connected to g
        adj_pred_to_gts[p] = set of gt IDs connected to p

        Returns a list of tuples: (set_of_gt_nodes, set_of_pred_nodes)
        """
        visited_gt = set()
        visited_pred = set()
        components = []

        # Iterate over GT nodes
        for g in adj_gt_to_preds.keys():
            if g in visited_gt:
                continue

            stack_gt = [g]
            comp_gt = set()
            comp_pred = set()

            while stack_gt:
                cg = stack_gt.pop()
                if cg in visited_gt:
                    continue
                visited_gt.add(cg)
                comp_gt.add(cg)

                # Visit preds connected to this gt
                for p in adj_gt_to_preds.get(cg, []):
                    if p not in visited_pred:
                        # mark pred
                        visited_pred.add(p)
                        comp_pred.add(p)
                        # explore gt neighbors from that pred
                        for ng in adj_pred_to_gts.get(p, []):
                            if ng not in visited_gt:
                                stack_gt.append(ng)

            if comp_gt or comp_pred:
                components.append((comp_gt, comp_pred))

        # Also handle not yet reached preds
        for p in adj_pred_to_gts.keys():
            if p in visited_pred:
                continue

            stack_pred = [p]
            comp_gt = set()
            comp_pred = set()

            while stack_pred:
                cp = stack_pred.pop()
                if cp in visited_pred:
                    continue
                visited_pred.add(cp)
                comp_pred.add(cp)

                for g in adj_pred_to_gts.get(cp, []):
                    if g not in visited_gt:
                        visited_gt.add(g)
                        comp_gt.add(g)
                        for np in adj_gt_to_preds.get(g, []):
                            if np not in visited_pred:
                                stack_pred.append(np)

            if comp_gt or comp_pred:
                components.append((comp_gt, comp_pred))

        return components

    #
    # Evaluation
    #
    def evaluate(self):
        adj_gt2p, adj_p2gt = self._adjacency_graph()

        if self.pair_df.empty:
            return self._empty_result()

        # Lookup dicts
        iou_lookup = {
            (int(r.gt), int(r.pred)): float(r.iou)
            for r in self.pair_df.itertuples(index=False)
        }
        dice_lookup = {
            (int(r.gt), int(r.pred)): float(r.dice)
            for r in self.pair_df.itertuples(index=False)
        }
        # moc_lookup = { (int(r.gt), int(r.pred)): float(r.moc) for r in self.pair_df.itertuples(index=False) } # not to be used as a metric, just for matching as it allows for smaller objects to match to larger ones

        matched_pairs = []
        matched_gt = set()
        matched_pred = set()
        iou_list = []
        dice_list = []

        # Linear sum assignment per connected component - not across all possible matches
        for comp_g, comp_p in self._connected_components_prematching(
            adj_gt2p, adj_p2gt
        ):
            gl = sorted(comp_g)
            pl = sorted(comp_p)
            ng, npred = len(gl), len(pl)

            # build local IoU matrix
            M = np.zeros((ng, npred), float)
            D = np.zeros((ng, npred), float)
            for i, g in enumerate(gl):
                for j, p in enumerate(pl):
                    M[i, j] = iou_lookup.get((g, p), 0.0)
                    D[i, j] = dice_lookup.get((g, p), 0.0)

            nloc = ng + npred
            C = np.ones((nloc, nloc), float)
            C[:ng, :npred] = 1.0 - M
            C[ng:, npred:] = (1.0 - M).T
            C[:ng, npred:] = self.unmatched_cost * np.eye(ng) + (1.0 - np.eye(ng))
            C[ng:, :npred] = self.unmatched_cost * np.eye(npred) + (1.0 - np.eye(npred))

            ridx, cidx = linear_sum_assignment(C)

            for r, c in zip(ridx, cidx):
                if r < ng and c < npred:
                    g = gl[r]
                    p = pl[c]
                    iou_val = M[r, c]
                    if iou_val >= self.iou_threshold:
                        matched_pairs.append((g, p))
                        matched_gt.add(g)
                        matched_pred.add(p)
                        iou_list.append(iou_val)
                        dice_list.append(D[r, c])

        # Get connected components for unmatched cells to construct segmentation error cases
        adj_gt_to_preds_unmatched = defaultdict(set)
        adj_pred_to_gts_unmatched = defaultdict(set)
        for row in self.pair_df.itertuples(index=False):
            if row.iou <= self.graph_iou_threshold:
                continue
            g, p = int(row.gt), int(row.pred)
            if g in matched_gt or p in matched_pred:
                continue
            adj_gt_to_preds_unmatched[g].add(p)
            adj_pred_to_gts_unmatched[p].add(g)

        splits, merges, catastrophes = 0, 0, 0
        split_details, merge_details, catastrophe_details = [], [], []

        for comp_gt, comp_pred in self._connected_components_postmatching(
            adj_gt_to_preds_unmatched, adj_pred_to_gts_unmatched
        ):
            ngc, npc = len(comp_gt), len(comp_pred)
            if ngc == 1 and npc > 1:
                splits += 1
                split_details.append(
                    {"gt": sorted(list(comp_gt))[0], "preds": sorted(list(comp_pred))}
                )
            elif ngc > 1 and npc == 1:
                merges += 1
                merge_details.append(
                    {"pred": sorted(list(comp_pred))[0], "gts": sorted(list(comp_gt))}
                )
            elif ngc > 1 and npc > 1:
                catastrophes += 1
                catastrophe_details.append(
                    {"gts": sorted(list(comp_gt)), "preds": sorted(list(comp_pred))}
                )

        all_gt_set = set(map(int, self.gt_ids))
        all_pred_set = set(map(int, self.pred_ids))
        fp_preds = list(all_pred_set - matched_pred)
        fn_gts = list(all_gt_set - matched_gt)

        tp = len(matched_pairs)
        fp = len(fp_preds)
        fn = len(fn_gts)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = hmean([precision, recall]) if precision > 0 and recall > 0 else 0.0

        # final metrics dict matching your original names
        self.metrics = {
            "iou_mean": float(np.mean(iou_list)) if iou_list else 0.0,
            "iou_list": iou_list,
            "dice_mean": float(np.mean(dice_list)) if dice_list else 0.0,
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
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "FP_list": fp_preds,
            "FN_list": fn_gts,
            "TTP_gt": list(matched_gt),
            "TTP_preds": list(matched_pred),
            "total_cells": int(len(self.gt_ids)),
            "total_pred_cells": int(len(self.pred_ids)),
        }

        return self.metrics

    def _empty_result(self):
        """Return a full metric dictionary when there are no overlaps between GT and predictions."""
        n_gt = len(self.gt_ids)
        n_pred = len(self.pred_ids)

        # True positives = 0 because no overlaps
        tp = 0
        fp = n_pred
        fn = n_gt

        precision = 0.0
        recall = 0.0
        f1 = 0.0

        self.metrics = {
            "iou_mean": 0.0,
            "iou_list": [],
            "dice_mean": 0.0 if self.extra_metric else None,
            "dice_list": [] if self.extra_metric else None,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "splits": 0,
            "merges": 0,
            "catastrophes": 0,
            "split_details": [],
            "merge_details": [],
            "catastrophe_details": [],
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "FP_list": list(self.pred_ids),
            "FN_list": list(self.gt_ids),
            "TTP_gt": [],
            "TTP_preds": [],
            "total_cells": n_gt,
            "total_pred_cells": n_pred,
        }
        return self.metrics


class SegmentationEvaluationBatch:
    """Run evaluation for a dataframe of segmentation pairs."""

    def __init__(
        self,
        df,
        iou_threshold=0.5,
        graph_iou_threshold=0.1,
        unmatched_cost=0.4,
        # cost_matrix_metric="iou",
    ):
        self.df = df.copy()
        self.iou_threshold = iou_threshold
        self.graph_iou_threshold = graph_iou_threshold
        self.unmatched_cost = unmatched_cost
        # self.cost_matrix_metric = cost_matrix_metric

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
