# SchapiroLabor/Segobe: Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## v0.1.1 - [2025.11.15]

Efficiency update - only cells with overlaps are taken into consideration when calculating IoU matrices. Mask comparisons not calculated through a direct comparison, but optimized through lookup tables.

### `Added`
- `plot_target_size` parameter for plotting - inputs will be "downscaled" to match the target size for plotting efficiency

### `Fixed`
- memory issues for large input masks with many cells

### `Removed`
- graph construction with networkx, replaced with functions
- cost matrix metric currently doesn't affect the execution - only IoU is used. To be updated soon.

## v0.1.0 - [2025.11.12]

First release of Segobe - a tool for object matching, segmentation error counting and metric evaluation.

### `Added`
- Everything

### `Fixed`
- 

### `Removed`
- 