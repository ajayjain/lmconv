# from IPython import embed
import argparse
# import itertools
# from operator import itemgetter
# import os
# import re
# import time
import glob

# from PIL import Image
# from tensorboardX import SummaryWriter
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim import lr_scheduler
# from torchvision import datasets, transforms, utils
import numpy as np
import tqdm
# import wandb

# from masking import *
# from model import *
# from ours import *
# from utils import *


D = 1024  # For CIFAR10 images, 32x32


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob_query", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # Get PGMs from many files
    files = glob.glob(args.glob_query)
    all_idx = []
    all_edge_counts = []
    for f in tqdm.tqdm(files):
        conditional_index_dict = np.load(f, allow_pickle=True).item()
        num_pgm_edges = conditional_index_dict["e"]
        num_pgm_vertices = conditional_index_dict["v"]
        conditional_idx = conditional_index_dict["idx"]

        assert num_pgm_vertices == D
        assert len(conditional_idx.shape) == 1
        assert conditional_idx.shape[0] == D

        if conditional_idx is not None:
            all_edge_counts.append(num_pgm_edges)
            all_idx.append(conditional_idx)

    all_edge_counts = np.array(all_edge_counts, dtype=np.int)
    all_idx = np.stack(all_idx, axis=0)  # num_pgms x D

    print("Loaded files")
    print("Saving dictionary of grouped conditional indices to", args.output)
    np.save(args.output, {
        "all_idx": all_idx,
        "all_edge_counts": all_edge_counts
    })
