# Masking utilities

from hilbertcurve.hilbertcurve import HilbertCurve
import numpy as np
import torch 

from gilbert2d import gilbert2d_idx


def get_input_mask_raster_scan(kernel_size, num_patches, mask_type='B'):
    """Generate mask for unfolded input image according to raster scan (row-major)
    autoregressive generation ordering.

    Args:
        kernel_size (int or Tuple[int, int]): kernel size of corresponding convolution.
                                              Influences rows in mask (ie patch size)
        num_patches (int): width of unfolded input, corresponding to the number of
                           locations a convolution will be applied. Generally equal
                           to H * W if appropriate padding is applied to the input
                           image before unfolding.
        mask_type (str): A or B, determining whether the center pixel can be conditioned
                         upon.

    Returns: Mask of shape 1 x (kernel_size^2) x num_patches,
             type torch.tensor(dtype=float)
    """
    if hasattr(kernel_size, "__iter__"):
        assert kernel_size[0] == kernel_size[1]
        kernel_size = kernel_size[0]

    assert kernel_size == 3, "Only kernel size 3 supported"
    assert mask_type in ["A", "B"]
    
    if mask_type == "A":
        col_mask_chunk = torch.tensor([1, 1, 1,
                                       1, 0, 0,
                                       0, 0, 0], dtype=torch.uint8)
    else:
        col_mask_chunk = torch.tensor([1, 1, 1,
                                       1, 1, 0,
                                       0, 0, 0], dtype=torch.uint8)

    inp_mask = col_mask_chunk.unsqueeze(1).repeat(1, num_patches)
    return inp_mask.unsqueeze(0)

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

def plot_kernels(nrows, ncols, generation_order, masks, k=3):
    fig, axes = plt.subplots(nrows, ncols)
    plt.suptitle(f"Kernel masks")
    for row_major_index, ((r, c), mask) in enumerate(zip(generation_order, masks)):
        axes[row_major_index // ncols, row_major_index % ncols].imshow(mask, vmin=0, vmax=1)
    plt.show()
