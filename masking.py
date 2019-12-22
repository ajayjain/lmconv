# Masking utilities
import math
import os

from hilbertcurve.hilbertcurve import HilbertCurve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch 

from gilbert2d import gilbert2d_idx


####################
# Generation orders
####################

def raster_scan_idx(rows, cols):
    """Return indices of a raster scan"""
    idx = []
    for r in range(rows):
        for c in range(cols):
            idx.append((r, c))
    return np.array(idx)

def s_curve_idx(rows, cols):
    """Generate S shape curve"""
    idx = []
    for r in range(rows):
        col_idx = range(cols) if r % 2 == 0 else range(cols-1, -1, -1)
        for c in col_idx:
            idx.append((r, c))
    return np.array(idx)

def hilbert_idx(rows, cols):
    assert rows == cols, "Image must be square for Hilbert curve"
    assert (rows > 0 and (rows & (rows - 1)) == 0), "Must have power-of-two sized image"
    order = int(np.log2(rows))
    curve = HilbertCurve(order, 2)
    idx = np.zeros((rows * cols, 2), dtype=np.int)
    for i in range(rows * cols):
        coords = curve.coordinates_from_distance(i) # cols, then rows
        idx[i, 0] = coords[1]
        idx[i, 1] = coords[0]
    return idx

def get_generation_order_idx(order: str, rows: int, cols: int):
    """Get (rows*cols) x 2 np array given order that pixels are generated"""
    assert order in ["raster_scan", "s_curve", "hilbert", "gilbert2d"]
    return eval(f"{order}_idx")(rows, cols)

def reflect_rows(generation_idx, obs):
    return list(map(lambda loc: [obs[1] - loc[0] - 1, loc[1]], generation_idx))

def reflect_cols(generation_idx, obs):
    return list(map(lambda loc: [loc[0], obs[2] - loc[1] - 1], generation_idx))

def reflect_all(generation_idx, obs):
    return list(map(lambda loc: [obs[1] - loc[0] - 1, obs[2] - loc[1] - 1], generation_idx))

def transpose(generation_idx):
    return list(map(lambda loc: [loc[1], loc[0]], generation_idx))

def augment_orders(generation_idx, obs):
    return [
        generation_idx,
        reflect_rows(generation_idx, obs),
        reflect_cols(generation_idx, obs),
        reflect_all(generation_idx, obs),
        transpose(generation_idx),
        reflect_rows(transpose(generation_idx), obs),
        reflect_cols(transpose(generation_idx), obs),
        reflect_all(transpose(generation_idx), obs)
    ]

def plot_order(generation_idx, obs, out_path=None):
    """Plot generation coordinate list. A star on the curve
    denotes the pixel generated last. obs is a three-tuple of input image dimensions,
    (input-channels-unused, num_rows, num_cols)"""
    plt.figure(figsize=(3, 3))
    plt.hlines(np.arange(-1, obs[1])+0.5, xmin=-0.5, xmax=obs[2]-0.5, alpha=0.5)
    plt.vlines(np.arange(-1, obs[2])+0.5, ymin=-0.5, ymax=obs[1]-0.5, alpha=0.5)
    rows, cols = zip(*generation_idx)
    plt.plot(cols, rows, color="r")
    plt.scatter([cols[-1]], [rows[-1]], marker="*", s=100, c="k")
    plt.xticks(np.arange(obs[1]))
    plt.axis("equal")
    plt.gca().invert_yaxis()
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()

def plot_orders(generation_idx_list, obs, size=5, plot_rows=4, out_path=None):
    """Plot multiple generation coordinate lists in a single figure. A star on the curve
    denotes the pixel generated last. obs is a three-tuple of input image dimensions,
    (input-channels-unused, num_rows, num_cols)"""
    num = len(generation_idx_list)
    plot_cols = int(math.ceil(num / 4))
    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(size * plot_cols, size * plot_rows))
    pr, pc = 0, 0
    for generation_idx in generation_idx_list:
        axes[pr, pc].hlines(np.arange(-1, obs[1])+0.5, xmin=-0.5, xmax=obs[2]-0.5, alpha=0.5)
        axes[pr, pc].vlines(np.arange(-1, obs[2])+0.5, ymin=-0.5, ymax=obs[1]-0.5, alpha=0.5)
        rows, cols = zip(*generation_idx)
        axes[pr, pc].plot(cols, rows, color="r")
        axes[pr, pc].scatter([cols[-1]], [rows[-1]], marker="*", s=100, c="k")
        axes[pr, pc].axis("equal")
        axes[pr, pc].invert_yaxis()
        pc = (pc + 1) % plot_cols
        if pc == 0:
            pr += 1
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()


########
# Masks
########

def kernel_masks(generation_order_idx: np.ndarray, nrows, ncols, k=3,
                 dilation=1, mask_type='B', set_padding=0) -> np.ndarray:
    """Generate kernel masks given a pixel generation order."""
    assert k % 2 == 1, "Only odd sized kernels are implemented"
    half_k = int(k / 2)
    masks = np.zeros((len(generation_order_idx), k, k))
    locs_generated = set()
    for i, (r, c) in enumerate(generation_order_idx):
        row_major_index = r * ncols + c
        for dr in range(-half_k, half_k+1):
            for dc in range(-half_k, half_k+1):
                loc = (r + dr * dilation, c + dc * dilation)
                if loc in locs_generated:
                    # The desired location has been generated,
                    # so we can condition on it
                    masks[row_major_index, half_k + dr, half_k + dc] = 1
                elif not (0 <= loc[0] < nrows and 0 <= loc[1] < ncols):
                    # Kernel location overlaps with padding
                    masks[row_major_index, half_k + dr, half_k + dc] = set_padding
        locs_generated.add((r, c))

    if mask_type == 'B':
        masks[:, half_k, half_k] = 1
    else:
        assert np.all(masks[:, half_k, half_k] == 0)

    return masks

def get_unfolded_masks(generation_order_idx, nrows, ncols, k=3, dilation=1, mask_type='B'):
    assert mask_type in ['A', 'B']
    masks = kernel_masks(generation_order_idx, nrows, ncols, k, dilation, mask_type, set_padding=0)
    masks = torch.tensor(masks, dtype=torch.float)
    masks_unf = masks.view(1, nrows * ncols, -1).transpose(1, 2)
    return masks_unf

def get_masks(generation_idx, nrows: int, ncols: int, k: int=3, max_dilation: int=1, out_dir: str="/tmp", plot_suffix="", plot=True):
    """Get and plot three masks: mask type A for first layer, mask type B for later layers, and mask type B with dilation.
    Masks are copied to GPU and repeated along the batch dimension torch.cuda.device_count() times for DataParallel support."""
    mask_init = get_unfolded_masks(generation_idx, nrows, ncols, k=k, dilation=1, mask_type='A')
    mask_undilated = get_unfolded_masks(generation_idx, nrows, ncols, k=k, dilation=1, mask_type='B')
    if plot:
        plot_unfolded_masks(nrows, ncols, generation_idx, mask_init, k=k, out_path=os.path.join(out_dir, f"mask_init_{plot_suffix}.png"))
        plot_unfolded_masks(nrows, ncols, generation_idx, mask_undilated, k=k, out_path=os.path.join(out_dir, f"mask_undilated_{plot_suffix}.png"))
    mask_init = mask_init.cuda(non_blocking=True).repeat(torch.cuda.device_count(), 1, 1)
    mask_undilated = mask_undilated.cuda(non_blocking=True).repeat(torch.cuda.device_count(), 1, 1)

    if max_dilation == 1:
        mask_dilated = mask_undilated
    else:
        mask_dilated = get_unfolded_masks(generation_idx, nrows, ncols, k=k, dilation=max_dilation, mask_type='B')
        if plot:
            plot_unfolded_masks(nrows, ncols, generation_idx, mask_dilated, k=k, out_path=os.path.join(out_dir, f"mask_dilated_d{max_dilation}_{plot_suffix}.png"))
        mask_dilated = mask_dilated.cuda(non_blocking=True).repeat(torch.cuda.device_count(), 1, 1)

    return mask_init, mask_undilated, mask_dilated

def plot_masks(nrows, ncols, generation_order, masks, k=3, out_path=None):
    import time
    fig, axes = plt.subplots(nrows, ncols)
    plt.suptitle(f"Kernel masks")
    t1 = time.time()
    for row_major_index, ((r, c), mask) in enumerate(zip(generation_order, masks)):
        axes[row_major_index // ncols, row_major_index % ncols].imshow(mask, vmin=0, vmax=1)
    t2 = time.time()
    plt.setp(axes, xticks=[], yticks=[])
    t3 = time.time()
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()
    t4 = time.time()
    print(f"Times: {t2 - t1}, {t3 - t2}, {t4 - t3}")

def plot_unfolded_masks(nrows, ncols, generation_order, unfolded_masks, k=3, out_path=None):
    masks = unfolded_masks.view(k, k, -1).permute(2, 0, 1)
    print(f"Plotting kernel masks and saving to {out_path}...")
    plot_masks(nrows, ncols, generation_order, masks, k=3, out_path=out_path)
