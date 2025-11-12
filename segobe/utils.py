#!/usr/bin/env python3
import numpy as np


def filter_mask_by_ids(mask, ids):
    filtered_mask = np.zeros_like(mask)
    for id in ids:
        if isinstance(id, list):
            for instance in id:
                filtered_mask[mask == instance] = instance
        else:
            filtered_mask[mask == id] = id
    return filtered_mask.astype(np.uint8)
