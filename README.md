# OBI-Segval - Object-Based Identification for Segmentation evaluation
![Python Version](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

OBI-Segval is a minimal, lightweight package for segmentation mask evaluation against a reference (ground-truth) mask. It performs cell matching
and computes metrics like the intersection-over-union (IoU), Dice, and classifies errors as splits, merges, and catastrophes, based on the descriptions in [Greenwald *et al.* 2022](https://doi.org/10.1038/s41587-021-01094-0) and [Schwartz *et al.* 2024](https://doi.org/10.1101/803205), with the added functionality of cost matrix selection, or matching approach.
Designed for cell segmentation evaluation, it can handle large batches of samples efficiently.

## Installation

Option 1. Install directory from the repository (recommended for development)
If you plan to develop or modify OBI-Segval, install it in editable mode:
```bash
# Clone the repository
git clone https://github.com/kbestak/obi-segval.git
cd obi-segval

# (Optional) create the conda environment
conda env create -f environment.yml
conda activate obi-segval

# Install in editable/development mode
pip install -e .
```
> The -e flag (editable mode) means the changes to the source code are immediately reflected without reinstalling.

Option 2. Install directly from GitHub
Once the repository is made public, users can install it directly via URL:
```bash
pip install git+https://github.com/kbestak/obi-segval.git
```
> This will work before uploading to PyPI, as soon as the repo is public because the `pyproject.toml` is set up

Verify installation with
```bash
python -m obi_segval --help
```
or simply:
```bash
obi-segval --help
```
to see the CLI help message.

## CLI

```bash
obi-segval \
    --input samples.csv \
    --output_dir results \
    --basename testing \
    --iou_threshold 0.5 \
    --graph_iou_threshold 0.1 \
    --unmatched_cost 0.4 \
    --save_plots
```

| Argument | Long form | Description |
|---|---|---|
| -i | --input_csv | File path to CSV with columns: sampleID, ref_mask, eval_mask, category |
| -o | --output_dir | Directory to save output metrics and plots |
| -b | --basename | Unique basename used when saving metrics and plots |
|  | --iou_threshold | IoU threshold for cell matching (0-1, default: 0.5). Match is true if pair is selected with linear_sum_assignment and IoU above this threshold. |
|  | --graph_iou_threshold | Graph IoU threshold for error detection (0-1, default: 0.1). Minimal IoU for cells to be considered 'connected'. |
|  | --unmatched_cost | Cost for unmatched objects in the cost matrix (0-1, default: 0.4) |
|  | --save_plots | Boolean specifying whether plots (barplot grouped by category and row-specific error overview) are saved |
|  | --version | Prints tool version. |

### Input format

Example of input CSV with potential usecase, comparing two methods across two samples (e.g. same ROI).

| sampleID | ref_mask                 | eval_mask                 | category |
|----------|--------------------------|---------------------------|----------|
| sample1  | path/to/groundtruth1.tif | path/to/prediction1_1.tif | method1  |
| sample1  | path/to/groundtruth1.tif | path/to/prediction1_2.tif | method2  |
| sample2  | path/to/groundtruth2.tif | path/to/prediction2_1.tif | method1  |
| sample2  | path/to/groundtruth2.tif | path/to/prediction2_2.tif | method2  |

### Outputs
Expected outputs given the CLI example and input CSV:
```
results/
├── testing_metrics.csv                          # Full per-sample segmentation evaluation results
├── testing_summary.csv                          # Aggregated metrics summarized by category
├── plots/
│   ├── testing_metrics_barplot.png              # Optional barplot of summary metrics (if --save_plots)
│   ├── testing_sample1_method1_error_plot.png   # Optional per-sample error visualizations (if --save_plots)
│   ├── testing_sample2_method1_error_plot.svg
│   ├── testing_sample1_method2_error_plot.svg
│   └── testing_sample2_method2_error_plot.png
```

Captured metrics in the `summary.csv` across all inputs grouped per category:
* IoU: mean and std
* Dice: mean and std
* Precision: mean and std
* Recall: mean and std
* F1 score: mean and std
* Splits: counts
* Merges: counts
* Catastrophes: counts
  
Captured metrics in the `metrics.csv` for each input CSV `row`:
* IoU: mean and list of all values
* Dice: mean and list of all values
* Precision
* Recall
* F1 score
* Splits: counts and dictionary of matched predictions to GTs
* Merges: counts and dictionary of matched GTs to predictions
* Catastrophes: counts and dictionary of groups of GTs and predictions involved
* IoU graph: constructed graph of cell overlaps
* True postives: counts
* False positives: counts
* False negatives: counts
* FP_list: list of false positive labels
* FN_list: list of false negative labels
* TTP_gt: list of GT labels of true positives
* TTP_preds: list of prediction labels of true positives
* total_cells: count of GT labels
* total_pred_cells: count of prediction labels

## Detailed description of what the tool does

### Overview

* **Start:** Label-images of ground truth and prediction cells

<img src="docs/images/GT_Prediction.png.png" alt="Description" width="40%"/>

* **Goals:**
  * Segmentation error type counts: True Positives (TP), False Positives (FP), False Negatives (FN), Merges, Splits, Catastrophes
  * Segmentation evaluation metrics: mean IoU, mean Dice, precision, recall, F1 score
  
### Cell matching

* Calculate the IoU between all GT (N) and PRED (M) pairs

<img src="docs/images/IoU_intro.png" alt="Description" width="40%"/>

* Construct the cost matrix:
  * `1 - IoU` (default)
  * `1 - Dice` (optional)
  * `1 - MOC` - Mean overlap coefficient defined as: `(Intersection / GT + Intersection / Prediction) / 2` (optional)
* Pad the NxM matrix to (N+M)x(M+N) to allow for an unassigned cost penalty (default 0.5)

<img src="docs/images/cost_matrix.png" alt="Description" width="40%"/>

* Apply the Hungarian method (Linear sum assignment problem) to find top matches (lowest cost)

<img src="docs/images/linear_sum_output.png.png" alt="Description" width="40%"/>

### Counting segmentation errors

* If the detected pair has an IoU above a predefined threshold (`iou_threshold`, e.g. 0.5), it is counted as a match - **True Positive**.
  * GT2 ←→ Pred1: IoU = 0.9 → match!
  * GT3 ←→ Pred2: IoU = 0.8 → match!
* Remaining GT and Prediction cells are used to build a graph with an edge if their IoU is above a threshold (`graph_iou_threshold`, e.g. 0.1).
* Ground truth cells without matches are labeled as **False Negatives** (FNs)
  * GT1 from previous example
* Prediction cells without matches are labeled as **False Positives** (FPs)
* Multiple GT cells connected to a prediction is a **Merge**
* Multiple Prediction cells connected to a GT cell is a **Split**
* Multiple GT cells connected to multiple Prediction cells constitutes a **Catastrophe**

> **Note:** If more than 2 Predictions match a GT cell or vice versa, they are counted as either a Merge or Split, not a Catastrophe
> 
> See Supplementary Figure 2. of [Schwartz **et al.** 2024](https://doi.org/10.1101/803205) for a visual example.

### Segmentation evaluation metrics

* These are calculated per image, not across all cells
* Calculated only across True Positive pairs
  * Mean IoU = Mean of Intersection / Union values
  * Mean Dice = Mean of 2 * Intersection / (Area sum of GT and Prediction) values 
  * Precision = TP / (TP + FP)
  * Recall = TO / (TP + FN)
  * F1 = 2*Precision*Recall / (Precision + Recall)

### Open questions
* What is the best approach for cell matching
  1. cost matrix choice for linear sum assignment
     * 1 - IoU
     * 1 - Dice
     * 1 - MOC
  2. Graph construction already only based on overlap thresholds (directional)
* Counting segmentation errors - should matched cells be excluded?
  * excluding nodes from the graph can lead to missed segmentation errors (merges, splits, catastrophes)
  * including the nodes would place "correct matches" within an error context
* Should merges and splits be treated as single objects for IoU and Dice measurements to capture shape of cells, while acknowledging over/undersegmentation?
* Excluding cells on edges needs to be a feature
* Are events involving 4 nodes always catastrophes? Here it is not implemented, and wording in original papers was imprecise.
* Parameter choice and impact on results

## Contributing
Contributions, issues, and feature requests are welcome!  
Feel free to open a pull request or submit an issue on [GitHub Issues](https://github.com/kbestak/obi-segval/issues).

Before submitting a PR:
- Run tests (not yet applicable)
- Follow existing code style and documentation patterns

## Citing

If you use **OBI-Segval** in your work, please cite:
>Bestak, K. OBI-Segval: Object-Based Identificatoin for Segmentation Evaluation.
>Available at: [https://github.com/kbestak/obi-segval](https://github.com/kbestak/obi-segval)

Note that for referencing the segmentation errors as used here, [Greenwald *et al.* 2022](https://doi.org/10.1038/s41587-021-01094-0) needs to be cited.