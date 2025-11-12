# SEGOBE - Object-Based Evaluation of Segmentation Results
![Python Version](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Segobe is a minimal, lightweight package for segmentation mask evaluation against a reference (ground-truth) mask. It performs cell matching
and computes metrics like the intersection-over-union (IoU), Dice, and classifies errors as splits, merges, and catastrophes, based on the descriptions in [Greenwald *et al.* 2022](https://doi.org/10.1038/s41587-021-01094-0) and [Schwartz *et al.* 2024](https://doi.org/10.1101/803205), with the added functionality of cost matrix selection, or matching approach.
Designed for cell segmentation evaluation, it can handle large batches of samples efficiently.

## Installation

Option 1. Install directory from the repository (recommended for development)
If you plan to develop or modify Segobe, install it in editable mode:
```bash
# Clone the repository
git clone https://github.com/schapirolabor/segobe.git
cd segobe

# (Optional) create the conda environment
conda env create -f environment.yml
conda activate segobe_env

# Install in editable/development mode
pip install -e .
```
> The -e flag (editable mode) means the changes to the source code are immediately reflected without reinstalling.

Option 2. Install directly from GitHub
Once the repository is made public, users can install it directly via URL:
```bash
pip install git+https://github.com/schapirolabor/segobe.git
```

Verify installation with
```bash
python -m segobe --help
```
or simply:
```bash
segobe --help
```
to see the CLI help message.

## CLI

```bash
segobe \
    --input samples.csv \
    --output_dir results \
    --basename testing \
    --iou_threshold 0.5 \
    --graph_iou_threshold 0.1 \
    --unmatched_cost 0.4 \
    --cost_matrix_metric 'iou' \
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
|  | --cost_matrix_metric | Specify which metric should be used for cost matrix construction (default: 'iou', other options 'dice', 'moc' - see details [here](docs/detailed_overview.md)) |
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

## Detailed description

Check the [Detailed overview](docs/detailed_overview.md) for a deeper clarification of tool functionality.

## Contributing
Contributions, issues, and feature requests are welcome!  
Feel free to open a pull request or submit an issue on [GitHub Issues](https://github.com/schapirolabor/segobe/issues).

Before submitting a PR:
- Run tests (not yet applicable)
- Follow existing code style and documentation patterns

## Citing

If you use **Segobe** in your work, please cite:

>Bestak, K. Segobe: Object-Based Evaluation of Segmentation Results.
>Available at: [https://github.com/schapirolabor/segobe](https://github.com/schapirolabor/segobe)

Note that for referencing the segmentation errors as used here, [Greenwald *et al.* 2022](https://doi.org/10.1038/s41587-021-01094-0) needs to be cited.