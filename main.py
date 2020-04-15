from IPython import embed
import argparse
import itertools
from operator import itemgetter
import os
import re
import time

from PIL import Image
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
import tqdm
import wandb

from masking import *
from model import *
from ours import *
from utils import *

parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='models',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--dataset', type=str,
                    default='cifar', help='Can be either cifar|mnist|celeba')
parser.add_argument('--binarize', action='store_true')
parser.add_argument('-p', '--print_every', type=int, default=20,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=20,
                    help='Every how many epochs to write checkpoint?')
parser.add_argument('-ts', '--sample_interval', type=int, default=4,
                    help='Every how many epochs to write samples?')
parser.add_argument('-tt', '--test_interval', type=int, default=1,
                    help='Every how many epochs to test model?')
parser.add_argument('-r', '--load_params', type=str, default=None,
                    help='Restore training from previous model checkpoint?')
parser.add_argument('--load_last_params', action="store_true",
                    help='Restore training from the last model checkpoint in the run dir?')
parser.add_argument('--do_not_load_optimizer', action="store_true")
parser.add_argument('-rd', '--run_dir', type=str, default=None,
                    help="Optionally specify run directory. One will be generated otherwise."
                         "Use to save log files in a particular place")
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('-ID', '--exp_id', type=int, default=0)
parser.add_argument('--ours', action='store_true')
# only for CelebAHQ
parser.add_argument('--max_celeba_train_batches', type=int, default=-1)
parser.add_argument('--max_celeba_test_batches', type=int, default=-1)
parser.add_argument('--celeba_size', type=int, default=256)
# parser.add_argument('--max_test_batches', type=int, default=-1)
parser.add_argument('--n_bits', type=int, default=8)
# pixelcnn++ and our model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-l', '--lr', type=float,
                    default=0.0002, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-wd', '--weight_decay', type=float,
                    default=0, help='Weight decay during optimization')
parser.add_argument('-c', '--clip', type=float, default=-1, help='Gradient norms clipped to this value')
parser.add_argument('-b', '--batch_size', type=int, default=64,
                    help='Batch size during training per GPU')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
# our model
parser.add_argument('-k', '--kernel_size', type=int, default=5,
                    help='Size of conv kernels')
parser.add_argument('-md', '--max_dilation', type=int, default=2,
                    help='Dilation in downsize stream')
parser.add_argument('-dp', '--dropout_prob', type=float, default=0.5,
                    help='Dropout prob used with nn.Dropout2d in gated resnet layers. '
                         'Argument only used if --ours is provided. Set to 0 to disable '
                         'dropout entirely.')
parser.add_argument('-nm', '--normalization', type=str, default='weight_norm',
                    choices=["none", "weight_norm", "instance_norm", "instance_norm_affine",
                             "order_rescale", "pono"])
parser.add_argument('-af', '--accum_freq', type=int, default=1,
                    help='Batches per optimization step. Used for gradient accumulation')
parser.add_argument('--two_stream', action="store_true", help="Enable two stream model")
parser.add_argument('--order', type=str, nargs="+",
                    choices=["raster_scan", "s_curve", "hilbert", "gilbert2d", "s_curve_center_quarter_last"],
                    help="Autoregressive generation order")
parser.add_argument('--randomize_order', action="store_true", help="Randomize between 8 variants of the "
                    "pixel generation order.")
parser.add_argument('--mode', type=str, choices=["train", "sample", "test", "count_params"],
                    default="train")
# configure sampling
parser.add_argument('--sample_region', type=str, choices=["full", "center", "random_near_center", "top", "custom"], default="full")
parser.add_argument('--sample_size_h', type=int, default=16, help="Only used for --sample_region center, top or random. =H of inpainting region.")
parser.add_argument('--sample_size_w', type=int, default=16, help="Only used for --sample_region center, top or random. =W of inpainting region.")
parser.add_argument('--sample_offset1', type=int, default=None, help="Manually specify box offset for --sample_region custom")
parser.add_argument('--sample_offset2', type=int, default=None, help="Manually specify box offset for --sample_region custom")
parser.add_argument('--sample_batch_size', type=int, default=25, help="Number of images to sample")
parser.add_argument('--sample_mixture_temperature', type=float, default=1.0)
parser.add_argument('--sample_logistic_temperature', type=float, default=1.0)
parser.add_argument('--sample_quantize', action="store_true", help="Quantize images during sampling to avoid train-sample distribution shift")
parser.add_argument('--save_nrow', type=int, default=4)
parser.add_argument('--save_padding', type=int, default=2)
# configure testing
parser.add_argument('--test_region', type=str, choices=["full", "custom"], default="full")
parser.add_argument('--test_minh', type=int, default=0, help="Specify conditional likelihood testing region. Only used with --test_region custom")
parser.add_argument('--test_maxh', type=int, default=32, help="Specify conditional likelihood testing region. Only used with --test_region custom")
parser.add_argument('--test_minw', type=int, default=0, help="Specify conditional likelihood testing region. Only used with --test_region custom")
parser.add_argument('--test_maxw', type=int, default=32, help="Specify conditional likelihood testing region. Only used with --test_region custom")
parser.add_argument('--order_variants', nargs="*", type=int)
# our model
parser.add_argument('--no_bias', action="store_true", help="Disable learnable bias for all convolutions")
parser.add_argument('--learn_weight_for_masks', action="store_true", help="Condition each masked conv on the mask itself, with a learned weight")
parser.add_argument('--minimize_bpd', action="store_true", help="Minimize bpd, scaling loss down by number of dimension")
parser.add_argument('--resize_sizes', type=int, nargs="*")
parser.add_argument('--resize_probs', type=float, nargs="*")
parser.add_argument('--base_order_reflect_rows', action="store_true")
parser.add_argument('--base_order_reflect_cols', action="store_true")
parser.add_argument('--base_order_transpose', action="store_true")
# memory and precision
parser.add_argument('--rematerialize', action="store_true", help="Recompute some activations during backwards to save memory")
parser.add_argument('--amp_opt_level', type=str, default=None)
# plotting
parser.add_argument('--plot_masks', action="store_true")

args = parser.parse_args()
# assert args.normalization != "weight_norm", "Weight normalization manually disabled in layers.py"


# Set seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Create run directory
if args.run_dir:
    run_dir = args.run_dir
else:
    dataset_name = args.dataset if not args.binarize else f"binary_{args.dataset}"
    _name = "{:05d}_{}_lr{:.5f}_bs{}_gc{}_k{}_md{}".format(
        args.exp_id, dataset_name, args.lr, args.batch_size, args.clip, args.kernel_size, args.max_dilation)
    if args.normalization != "none":
        _name = f"{_name}_{args.normalization}"
    if args.exp_name:
        _name = f"{_name}+{args.exp_name}"
    run_dir = os.path.join("runs", _name)
    if args.mode == "train":
        os.makedirs(run_dir, exist_ok=args.load_last_params)
assert os.path.exists(run_dir), "Did not find run directory, check --run_dir argument"

wandb.init(project="autoreg_orders", id=f"{args.exp_id}_{args.mode}", name=f"{run_dir}_{args.mode}", job_type=args.mode)

# Log arguments
wandb.config.update(args)
timestamp = time.strftime("%Y%m%d-%H%M%S")
if args.mode == "test" and args.test_region == "custom":
    logfile = f"{args.mode}_{args.test_minh}:{args.test_maxh}_{args.test_minw}:{args.test_maxw}_{timestamp}.log"
else:
    logfile = f"{args.mode}_{timestamp}.log"
logger = configure_logger(os.path.join(run_dir, logfile))
logger.info("Run directory: %s", run_dir)
logger.info("Arguments: %s", args)
for k, v in vars(args).items():
    logger.info(f"  {k}: {v}")


# Create data loaders
sample_batch_size = args.sample_batch_size
dataset_obs = {
    'mnist': (1, 28, 28),
    'cifar': (3, 32, 32),
    'celebahq': (3, args.celeba_size, args.celeba_size)
}[args.dataset]
input_channels = dataset_obs[0]
data_loader_kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True, 'batch_size':args.batch_size}
if args.resize_sizes:
    if not args.resize_probs:
        args.resize_probs = [1. / len(args.resize_sizes)] * len(args.resize_sizes)
    assert len(args.resize_probs) == len(args.resize_sizes)
    assert sum(args.resize_probs) == 1
    resized_obses = [(input_channels, s, s) for s in args.resize_sizes]
else:
    args.resize_sizes = [dataset_obs[1]]
    args.resize_probs = [1.]
    resized_obses = [dataset_obs]

def obs2str(obs):
    return 'x'.join(map(str, obs))

def random_resized_obs():
    idx = np.arange(len(resized_obses))
    obs_i = np.random.choice(idx, p=args.resize_probs)
    return resized_obses[int(obs_i)]

def get_resize_collate_fn(obs, default_collate=torch.utils.data.dataloader.default_collate):
    if obs == dataset_obs:
        return default_collate

    def resize_collate_fn(batch):
        X, y = default_collate(batch)
        X = torch.nn.functional.interpolate(X, size=obs[1:], mode="bilinear")
        return [X, y]
    return resize_collate_fn

def random_resize_collate(batch, default_collate=torch.utils.data.dataloader.default_collate):
    X, y = default_collate(batch)
    obs = random_resized_obs()
    if obs != dataset_obs:
        X = torch.nn.functional.interpolate(X, size=obs[1:], mode="bilinear")
    return [X, y]

# Create data loaders
if 'mnist' in args.dataset :
    assert args.n_bits == 8
    if args.binarize:
        rescaling = lambda x : (binarize_torch(x) - .5) * 2.  # binarze and rescale [0, 1] images into [-1, 1] range
    else:
        rescaling = lambda x : (x - .5) * 2.  # rescale [0, 1] images into [-1, 1] range
    rescaling_inv = lambda x : .5 * x + .5
    ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

    train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True,
        train=True, transform=ds_transforms), shuffle=True, collate_fn=random_resize_collate, **data_loader_kwargs)
    test_loader_by_obs = {
        obs: torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False,
            transform=ds_transforms), collate_fn=get_resize_collate_fn(obs), **data_loader_kwargs)
        for obs in resized_obses
    }

    # Default upper bounds for progress bars
    train_total = None
    test_total = None
elif 'cifar' in args.dataset :
    assert args.n_bits == 8
    rescaling = lambda x : (x - .5) * 2.  # rescale [0, 1] images into [-1, 1] range
    rescaling_inv = lambda x : .5 * x + .5
    ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms), shuffle=True, collate_fn=random_resize_collate, **data_loader_kwargs)
    test_loader_by_obs = {
        obs: torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False,
            transform=ds_transforms), collate_fn=get_resize_collate_fn(obs), **data_loader_kwargs)
        for obs in resized_obses
    }

    # Default upper bounds for progress bars
    train_total = None
    test_total = None
elif 'celebahq' in args.dataset :
    if args.n_bits == 8:
        rescaling = lambda x : (2. / 255) * x - 1.  # rescale uint8 images into [-1, 1] range
        rescaling_inv = lambda x : .5 * x + .5  # rescale [-1, 1] range to [0, 1] range
        #rescaling_inv = lambda x : (255. / 2) * (x + 1.)  # rescale [-1, 1] range to [0, 255] range
    else:
        assert 0 < args.n_bits < 8
        n_bins = 2. ** args.n_bits
        depth_divisor = (2. ** (8 - args.n_bits))
        def rescaling(x):
            # reduce bit depth, from [0, 255] to [0, n_bins-1] range
            x = torch.floor(x / depth_divisor)
            # rescale images from [0, n_bins-1] into [-1, 1] range
            x = (2. / (n_bins - 1)) * x - 1.
            return x
        rescaling_inv = lambda x : .5 * x + .5  # rescale [-1, 1] range to [0, 1] range
        #def rescaling_inv(x):
        #    # rescale images from [-1, 1] to [0, n_bins-1]
        #    x = ((n_bins - 1) / 2) * (x + 1.)
        #    # increase bit depth to [0, 255] range
        #    x = x * depth_divisor
        #    return x

    # NOTE: Random resizing of images during training is not supported for CelebA-HQ. Will use 256x256 resolution.
    def get_celeba_dataloaders():
        from celeba_data import get_celeba_dataloader
        kwargs = dict(data_loader_kwargs)
        kwargs["num_workers"] = 0
        train_loader = get_celeba_dataloader(args.data_dir, "train",
                                            collate_fn=itemgetter(0), # lambda batch: random_resize_collate(batch, itemgetter(0)),
                                            batch_transform=rescaling,
                                            max_batches=args.max_celeba_train_batches,
                                            size=args.celeba_size,
                                            **kwargs)
        test_loader_by_obs = {
            obs: get_celeba_dataloader(args.data_dir, "validation",
                                    collate_fn=get_resize_collate_fn(obs, itemgetter(0)),
                                    batch_transform=rescaling,
                                    max_batches=args.max_celeba_test_batches,
                                    size=args.celeba_size,
                                    **kwargs)
            for obs in resized_obses
        }
        return train_loader, test_loader_by_obs

    train_loader, test_loader_by_obs = get_celeba_dataloaders()

    # Manually specify upper bounds for progress bars
    train_total = 27000 // args.batch_size if args.max_celeba_train_batches <= 0 else args.max_celeba_train_batches
    test_total = 3000 // args.batch_size if args.max_celeba_test_batches <= 0 else args.max_celeba_test_batches
else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))


def quantize(x):
    # Quantize [-1, 1] images to uint8 range, then put back in [-1, 1]
    # Can be used during sampling with --sample_quantize argument
    assert args.n_bits == 8
    continuous_x = rescaling_inv(x) * 255  # Scale to [0, 255] range
    discrete_x = continuous_x.long().float()  # Round down
    quantized_x = discrete_x / 255.
    return rescaling(quantized_x)


# Select loss functions
if 'mnist' in args.dataset :
    # Losses for 1-channel images
    assert args.n_bits == 8
    if args.binarize:
        loss_op = binarized_loss
        loss_op_averaged = binarized_loss_averaged
        sample_op = sample_from_binary_logits
    else:
        loss_op = discretized_mix_logistic_loss_1d
        loss_op_averaged = discretized_mix_logistic_loss_1d_averaged
        sample_op = lambda x, i, j: sample_from_discretized_mix_logistic_1d(x, i, j, args.nr_logistic_mix)
else:
    # Losses for 3-channel images
    loss_op = lambda x, l : discretized_mix_logistic_loss(x, l, n_bits=args.n_bits)
    loss_op_averaged = lambda x, ls : discretized_mix_logistic_loss_averaged(x, ls, n_bits=args.n_bits)
    sample_op = lambda x, i, j: sample_from_discretized_mix_logistic(x, i, j, args.nr_logistic_mix,
                                                                              args.sample_mixture_temperature,
                                                                              args.sample_logistic_temperature)


# Construct model
if args.ours:
    logger.info("Constructing our model")

    if args.normalization == "instance_norm":
        raise NotImplementedError("Causal instance norm not implemented")
        # norm_op = lambda num_channels: nn.InstanceNorm2d(num_channels)
    elif args.normalization == "instance_norm_affine":
        raise NotImplementedError("Causal instance norm not implemented")
        # norm_op = lambda num_channels: nn.InstanceNorm2d(num_channels, affine=True)
    elif args.normalization == "order_rescale":
        norm_op = lambda num_channels: OrderRescale()
    elif args.normalization == "pono":
        norm_op = lambda num_channels: PONO()
    else:
        norm_op = None

    assert not args.two_stream, "--two_stream cannot be used with --ours"
    model = OurPixelCNN(
                nr_resnet=args.nr_resnet,
                nr_filters=args.nr_filters, 
                input_channels=input_channels,
                nr_logistic_mix=args.nr_logistic_mix,
                kernel_size=(args.kernel_size, args.kernel_size),
                max_dilation=args.max_dilation,
                weight_norm=(args.normalization == "weight_norm"),
                feature_norm_op=norm_op,
                dropout_prob=args.dropout_prob,
                conv_bias=(not args.no_bias),
                conv_mask_weight=args.learn_weight_for_masks,
                rematerialize=args.rematerialize,
                binarize=args.binarize)

    all_generation_idx_by_obs = {}
    all_masks_by_obs = {}
    for obs in resized_obses:
        # Get generation orders
        all_generation_idx = []
        for order in args.order:
            base_generation_idx = get_generation_order_idx(order, obs[1], obs[2])

            # Suggested orders
            #   BOTTOM HALF INPAINTING: (default)
            #   TOP HALF INPAINTING: --base_order_reflect_rows
            #   RIGHT HALF INPAINTING: --base_order_transpose
            #   LEFT HALF INPAINTING: --base_order_transpose --base_order_reflect_cols
            if args.base_order_transpose:
                base_generation_idx = transpose(base_generation_idx)
            if args.base_order_reflect_rows:
                base_generation_idx = reflect_rows(base_generation_idx, obs)
            if args.base_order_reflect_cols:
                base_generation_idx = reflect_cols(base_generation_idx, obs)

            if args.randomize_order:
                all_generation_idx.extend(augment_orders(base_generation_idx, obs))
            else:
                all_generation_idx.append(base_generation_idx)
        if args.order_variants:
            print("Selecting order variants", args.order_variants)
            all_generation_idx = [all_generation_idx[i] for i in args.order_variants]

        # Generate center square last for inpainting
        observed_idx = None
        #if args.mode == "test_center_quarter":
            #logger.info("Moving center coord generation to end for each variant of the order")
            #center_idx = center_quarter_coords(obs[1], obs[2])
            #all_generation_idx = [move_to_end(idx, center_idx) for idx in all_generation_idx]

            ## Full context
            #observed_idx = []
            #for r in range(obs[1]):
            #    for c in range(obs[2]):
            #        if (r, c) not in center_idx:
            #            observed_idx.append((r, c))

        # Plot orders
        if args.mode == "sample":
            plot_orders_out_path = os.path.join(run_dir, f"{args.mode}_{args.sample_region}_{args.sample_size_h}x{args.sample_size_w}_o1{args.sample_offset1}_o2{args.sample_offset2}_orderings_obs{obs2str(obs)}.png")
        elif args.mode == "test":
            plot_orders_out_path = os.path.join(run_dir, f"{args.mode}_{args.test_region}_{args.test_minh}:{args.test_maxh}_{args.test_minw}:{args.test_maxw}_orderings_obs{obs2str(obs)}.png")
        else:
            plot_orders_out_path = os.path.join(run_dir, f"{args.mode}_orderings_obs{obs2str(obs)}.png")

        try:
            plot_orders(all_generation_idx, obs, size=5, plot_rows=min(len(all_generation_idx), 4),
                        out_path=plot_orders_out_path)
            wandb.log({plot_orders_out_path: wandb.Image(plot_orders_out_path)})
        except IndexError as e:
            logger.error("Failed to plot orders: %s", e)

        all_generation_idx_by_obs[obs] = all_generation_idx

        # Make masks and plot
        all_masks = []
        for i, generation_idx in enumerate(all_generation_idx):
            masks = get_masks(generation_idx, obs[1], obs[2], args.kernel_size, args.max_dilation,
                              observed_idx=observed_idx,
                              out_dir=run_dir,
                              plot_suffix=f"obs{obs2str(obs)}_order{i}",
                              plot=args.plot_masks)
            logger.info(f"Mask shapes: {masks[0].shape}, {masks[1].shape}, {masks[2].shape}")
            all_masks.append(masks)
        all_masks_by_obs[obs] = all_masks
else:
    logger.info("Constructing original PixelCNN++")
    model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix,
                rematerialize=args.rematerialize)

    assert not args.randomize_order
    all_generation_idx_by_obs = {}
    all_masks_by_obs = {}
    for obs in resized_obses:
        all_generation_idx_by_obs[obs] = [get_generation_order_idx("raster_scan", obs[1], obs[2])]
        all_masks_by_obs[obs] = [(None, None, None)]
model = model.cuda()

# Create optimizer
# NOTE: PixelCNN++ TF repo uses betas=(0.95, 0.9995), different than PyTorch defaults
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

if args.amp_opt_level:
    # Enable mixed precision training
    from apex import amp
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)

model = nn.DataParallel(model)
wandb.watch(model)




# Load model parameters from checkpoint
if args.load_params:
    if os.path.exists(args.load_params):
        load_params = args.load_params
    else:
        load_params = os.path.join(run_dir, args.load_params)
    # Load params
    checkpoint_epochs, checkpoint_step = load_part_of_model(load_params,
                                           model=model.module,
                                           optimizer=None if args.do_not_load_optimizer else optimizer)
    logger.info(f"Model parameters loaded from {load_params}, from after {checkpoint_epochs} training epochs")
elif args.load_last_params:
    # Find the most recent checkpoint (highest epoch number).
    checkpoint_re = f"{args.exp_id}_ep([0-9]+)\\.pth"
    checkpoint_files = []
    checkpoint_epochs = []
    for f in os.listdir(run_dir):
        match = re.match(checkpoint_re, f)
        if match:
            checkpoint_files.append(f)
            ep = int(match.group(1))
            checkpoint_epochs.append(ep)
            logger.info(f"Found checkpoint {f} with {ep} epochs of training")
    if checkpoint_files:
        last_checkpoint_name = checkpoint_files[int(np.argmax(checkpoint_epochs))]
        load_params = os.path.join(run_dir, last_checkpoint_name)
        logger.info(f"Most recent checkpoint: {last_checkpoint_name}")
        # Load params
        checkpoint_epochs, checkpoint_step = load_part_of_model(load_params,
                                            model=model.module,
                                            optimizer=None if not args.do_not_load_optimizer else optimizer)
        logger.info(f"Model parameters loaded from {load_params}, from after {checkpoint_epochs} training epochs")
    else:
        logger.info("No checkpoints found")
        checkpoint_epochs = -1
        checkpoint_step = -1
else:
    checkpoint_epochs = -1
    checkpoint_step = -1


def test(model, all_masks, test_loader, epoch="N/A", progress_bar=True,
         slice_op=None, sliced_obs=obs):
    logger.info(f"Testing with ensemble of {len(all_masks)} orderings")
    test_loss = 0.
    pbar = tqdm.tqdm(test_loader,
                     desc=f"Test after epoch {epoch}",
                     disable=(not progress_bar),
                     total=test_total)
    num_images = 0
    for batch_idx, (input,_) in enumerate(pbar):
        num_images += input.shape[0]

        input = input.cuda(non_blocking=True)
        input_var = Variable(input)

        #mask_init, mask_undilated, mask_dilated = all_masks[0]
        #output = model(input_var, mask_init=mask_init, mask_undilated=mask_undilated, mask_dilated=mask_dilated)
        #loss = loss_op(input_var, output)

        # Average likelihoods over multiple orderings
        outputs = []
        for mask_init, mask_undilated, mask_dilated in all_masks:
            output = model(input_var, mask_init=mask_init, mask_undilated=mask_undilated, mask_dilated=mask_dilated)
            output = slice_op(output) if slice_op is not None else output
            outputs.append(output)

        # assert slice_op is not None  # FIXME: temporary check for whole image testing
        order_prefix = "_".join(args.order)
        np.save(f"{args.dataset}_{order_prefix}_all_generation_idx", all_generation_idx)

        input_var_for_loss = slice_op(input_var) if slice_op is not None else input_var
        loss = loss_op_averaged(input_var_for_loss, outputs)

        test_loss += loss.item()
        del loss, output

        deno = num_images * np.prod(sliced_obs) * np.log(2.)
        pbar.set_description(f"Test after epoch {epoch} {test_loss / deno}")

    deno = num_images * np.prod(sliced_obs) * np.log(2.)
    assert deno > 0, embed()
    test_bpd = test_loss / deno
    return test_bpd


def get_sampling_images(loader):
    # Get batch of images to complete for inpainting, or None for --sample_region=full
    if args.sample_region == "full":
        return None

    logger.info('getting batch of images to complete...')
    # Get sample_batch_size images from test set
    batches_to_complete = []
    sample_iter = iter(loader)
    for _ in range(sample_batch_size // args.batch_size + 1):
        batches_to_complete.append(next(sample_iter)[0])  # ignore labels
    del sample_iter

    batch_to_complete = torch.cat(batches_to_complete, dim=0)[:sample_batch_size]
    logger.info('got %d images to complete with shape %s', len(batch_to_complete), batch_to_complete.shape)

    return batch_to_complete


def sample(model, generation_idx, mask_init, mask_undilated, mask_dilated, batch_to_complete, obs):
    model.eval()
    if args.sample_region == "full":
        data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
        data = data.cuda()
        sample_idx = generation_idx
        context = None
        batch_to_complete = None
    else:
        if args.sample_region == "center":
            offset1 = -args.sample_size_h // 2
            offset2 = -args.sample_size_w // 2
        elif args.sample_region == "random_near_center":
            offset1 = int(np.random.randint(-obs[1] // 4, obs[1] // 4))
            offset2 = int(np.random.randint(-obs[2] // 4, obs[2] // 4)) 
        elif args.sample_region == "top":
            # should use with --sample_size_h = H/2, --sample_size_w = W. samples from top left corner
            offset1 = -(obs[1] // 2)
            offset2 = -(obs[2] // 2)
        elif args.sample_region == "custom":
            assert args.sample_offset1 is not None and args.sample_offset2 is not None
            offset1 = args.sample_offset1
            offset2 = args.sample_offset2
        else:
            raise NotImplementedError(f"Unknown sampling region {args.sample_region}")

        # Get indices of sampling region
        sample_region = set()
        for i in range(obs[1] // 2 + offset1,
                       obs[1] // 2 + offset1 + args.sample_size_h):
            for j in range(obs[2] // 2 + offset2,
                           obs[2] // 2 + offset2 + args.sample_size_w):
                sample_region.add((i, j))

        # Sort according to generation_idx
        sample_idx = []
        num_added = 0
        for i, j in generation_idx:
            if (i, j) in sample_region:
                sample_idx.append([i, j])
                num_added += 1 
        sample_idx = np.array(sample_idx, dtype=np.int)

        logger.info(f"Sample idx {sample_idx}")

        # Mask out region in network input image
        data = batch_to_complete.clone().cuda()
        print("batch_to_complete", type(batch_to_complete), batch_to_complete.shape, "data", type(data), data.shape)
        data[:, :, sample_idx[:, 0], sample_idx[:, 1]] = 0

        context = rescaling_inv(data).cpu()
        batch_to_complete = rescaling_inv(batch_to_complete).cpu()

        logger.info(f"Example context: {context.numpy()}")

    logger.info(f"Before sampling, data has range {data.min().item()}-{data.max().item()} (mean {data.mean().item()}), dtype={data.dtype} {type(data)}")
    for n_pix, (i, j) in enumerate(tqdm.tqdm(sample_idx, desc="Sampling pixels")):
        data_v = Variable(data)
        t1 = time.time()
        out = model(data_v, sample=True, mask_init=mask_init, mask_undilated=mask_undilated, mask_dilated=mask_dilated)
        t2 = time.time()
        out_sample = sample_op(out, i, j)
        if args.sample_quantize:
            out_sample = quantize(out_sample)
        logger.info("%d %d,%d Time to infer logits=%f s, sample=%f s", n_pix, i, j, t2-t1, time.time()-t2)
        data[:, :, i, j] = out_sample
        logger.info(f"Sampled pixel {i},{j}, with batchwise range {out_sample.min().item()}-{out_sample.max().item()} (mean {out_sample.mean().item()}), dtype={out_sample.dtype} {type(out_sample)}")

        if (n_pix <= 256 and n_pix % 32 == 0) or n_pix % 256 == 0:
            sample_save_path = os.path.join(run_dir, f'{args.mode}_{args.sample_region}_{args.sample_size_h}x{args.sample_size_w}_o1{args.sample_offset1}_o2{args.sample_offset2}_obs{obs2str(obs)}_ep{checkpoint_epochs}_order{sample_order_i}_{n_pix}of{len(sample_idx)}pix{"_quantize" if args.sample_quantize else ""}.png')
            utils.save_image(rescaling_inv(data), sample_save_path, nrow=4, padding=5, pad_value=1, scale_each=False)
            wandb.log({sample_save_path: wandb.Image(sample_save_path)}, step=n_pix)
    data = rescaling_inv(data).cpu()

    if batch_to_complete is not None and context is not None:
        # Interleave along batch dimension to visualize GT images
        difference = torch.abs(data - batch_to_complete)
        logger.info(f"Context range {context.min()}-{context.max()}. Data range {data.min()}-{data.max()}. batch_to_complete range {batch_to_complete.min()}-{batch_to_complete.max()}")
        data = torch.stack([context, data, batch_to_complete, difference], dim=1).view(-1, *data.shape[1:])

    return data


if args.mode == "train":
    logger.info("starting training")
    writer = SummaryWriter(log_dir=run_dir)
    global_step = checkpoint_step + 1
    min_train_bpd = 1e12
    min_test_bpd_by_obs = {obs: 1e12 for obs in resized_obses}
    last_saved_epoch = -1
    for epoch in range(checkpoint_epochs + 1, args.max_epochs):
        train_loss = 0.
        time_ = time.time()
        model.train()
        for batch_idx, (input,_) in enumerate(tqdm.tqdm(train_loader, desc=f"Train epoch {epoch}", total=train_total)):
            input = input.cuda(non_blocking=True)  # [-1, 1] range images

            obs = input.shape[1:]
            all_masks = all_masks_by_obs[obs]
            order_i = np.random.randint(len(all_masks))
            mask_init, mask_undilated, mask_dilated = all_masks[order_i]
            output = model(input, mask_init=mask_init, mask_undilated=mask_undilated, mask_dilated=mask_dilated)

            loss = loss_op(input, output)
            deno = args.batch_size * np.prod(obs) * np.log(2.)
            assert deno > 0, embed()
            train_bpd = loss / deno
            if args.minimize_bpd:
                loss = train_bpd

            if batch_idx % args.accum_freq == 0:
                optimizer.zero_grad()
            if args.amp_opt_level:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if (batch_idx + 1) % args.accum_freq == 0:
                if args.clip > 0:
                    # Compute and rescale gradient norm
                    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    # if gradient_norm > args.clip:
                        # logger.warning(f"Clipped gradients to norm {args.clip}")
                else:
                    # Just compute the gradient norm
                    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
                    gradient_norm = 0
                    for p in parameters:
                        param_norm = p.grad.data.norm(2)
                        gradient_norm += param_norm.item() ** 2
                    gradient_norm = gradient_norm ** (1. / 2)
                writer.add_scalar('train/gradient_norm', gradient_norm, global_step)
                wandb.log({"train/gradient_norm": gradient_norm, "epoch": epoch}, step=global_step)
                optimizer.step()
            train_loss += loss.item()

            writer.add_scalar('train/bpd', train_bpd.item(), global_step)
            min_train_bpd = min(min_train_bpd, train_bpd.item())
            writer.add_scalar('train/min_bpd', min_train_bpd, global_step)
            wandb.log({"train/bpd": train_bpd.item(),
                       "train/min_bpd": min_train_bpd,
                       "epoch": epoch}, step=global_step)

            if batch_idx >= 100 and train_bpd.item() >= 10:
                logger.warning("WARNING: main.py: large batch loss {} bpd".format(train_bpd.item()))

            if (batch_idx + 1) % args.print_every == 0: 
                deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
                average_bpd = train_loss / args.print_every if args.minimize_bpd else train_loss / deno
                logger.info('train bpd : {:.4f}, train loss : {:.1f}, time : {:.4f}, global step: {}'.format(
                    average_bpd,
                    train_loss / args.print_every,
                    (time.time() - time_),
                    global_step))
                train_loss = 0.
                time_ = time.time()

            if (batch_idx + 1) % args.accum_freq == 0:
                global_step += 1

        # decrease learning rate
        scheduler.step()

        model.eval()
        with torch.no_grad():
            save_dict = {}

            if (epoch + 1) % args.test_interval == 0:
                for obs in resized_obses:
                    logger.info(f"testing with obs {obs2str(obs)}...")
                    test_bpd = test(model,
                                    all_masks_by_obs[obs],
                                    test_loader_by_obs[obs],
                                    epoch,
                                    progress_bar=True)
                    writer.add_scalar(f'test/bpd_{obs2str(obs)}', test_bpd, global_step)
                    wandb.log({f'test/bpd_{obs2str(obs)}': test_bpd, "epoch": epoch}, step=global_step)
                    logger.info(f"test loss for obs {obs2str(obs)}: %s bpd" % test_bpd)
                    save_dict[f"test_loss_{obs2str(obs)}"] = test_bpd

                    # Log min test bpd for smoothness
                    min_test_bpd_by_obs[obs] = min(min_test_bpd_by_obs[obs], test_bpd)
                    writer.add_scalar(f'test/min_bpd_{obs2str(obs)}', min_test_bpd_by_obs[obs], global_step)
                    wandb.log({f'test/min_bpd_{obs2str(obs)}': min_test_bpd_by_obs[obs], "epoch": epoch}, step=global_step)
                    if obs == dataset_obs:
                        writer.add_scalar(f'test/bpd', test_bpd, global_step)
                        writer.add_scalar(f'test/min_bpd', min_test_bpd_by_obs[obs], global_step)
                        wandb.log({'test/bpd': test_bpd, 'epoch': epoch}, step=global_step)
                        wandb.log({'test/min_bpd': min_test_bpd_by_obs[obs], 'epoch': epoch}, step=global_step)

            # Save checkpoint so we have checkpoints every save_interval epochs, as well as a rolling most recent checkpoint
            save_path = os.path.join(run_dir, f"{args.exp_id}_ep{epoch}.pth")
            logger.info('saving model to %s...', save_path)
            save_dict["epoch"] = epoch
            save_dict["global_step"] = global_step
            save_dict["args"] = vars(args)
            try:
                save_dict["model_state_dict"] = model.module.state_dict()
                save_dict["optimizer_state_dict"] = optimizer.state_dict()
                torch.save(save_dict, save_path)
                if (epoch + 1) % args.save_interval != 0: 
                    # Remove last off-cycle checkpoint
                    remove_path = os.path.join(run_dir, f"{args.exp_id}_ep{last_saved_epoch}.pth")
                    if os.path.exists(os.path.join(run_dir, f"{args.exp_id}_ep{last_saved_epoch}.pth")):
                        logger.info('deleting checkpoint at %s', remove_path)
                        os.remove(remove_path)
                    last_saved_epoch = epoch
            except Exception as e:
                logger.error("Failed to save checkpoint! Error: %s", e)

            if (epoch + 1) % args.sample_interval == 0: 
                for obs in resized_obses:
                    try:
                        all_masks = all_masks_by_obs[obs]
                        all_generation_idx = all_generation_idx_by_obs[obs]
                        sample_order_i = np.random.randint(len(all_masks))
 
                        batch_to_complete = get_sampling_images(test_loader_by_obs[obs])

                        logger.info('sampling images with observation %s, ordering variant %d...', obs2str(obs), sample_order_i)
                        sample_t = sample(model,
                                          all_generation_idx[sample_order_i],
                                          *all_masks[sample_order_i],
                                          batch_to_complete,
                                          obs)
                        sample_save_path = os.path.join(run_dir, f"tsample_obs{obs2str(obs)}_{epoch}_order{sample_order_i}.png")
                        utils.save_image(sample_t, sample_save_path, nrow=4, padding=5, pad_value=1, scale_each=False)
                        wandb.log({sample_save_path: wandb.Image(sample_save_path), "epoch": epoch}, step=global_step)
                    except Exception as e:
                        logger.error("Failed to sample images! Error: %s", e)
        
        if "celeba" in args.dataset:
            # Need to manually re-create loaders to reset
            train_loader, test_loader_by_obs = get_celeba_dataloaders()
elif args.mode == "sample":
    assert dataset_obs in resized_obses
    batch_to_complete = get_sampling_images(test_loader_by_obs[dataset_obs])

    model.eval()
    with torch.no_grad():
        for obs in resized_obses:
            all_masks = all_masks_by_obs[obs]
            all_generation_idx = all_generation_idx_by_obs[obs]
            sample_order_i = np.random.randint(len(all_masks))
            logger.info('sampling images with observation %s, ordering variant %d...', obs2str(obs), sample_order_i)
            sample_t = sample(model, all_generation_idx[sample_order_i], *all_masks[sample_order_i], batch_to_complete, obs)
            sample_save_path = os.path.join(run_dir, f'{args.mode}_{args.sample_region}_ep{checkpoint_epochs}_{args.sample_size_h}x{args.sample_size_w}_o1{args.sample_offset1}_o2{args.sample_offset2}_obs{obs2str(obs)}_order{sample_order_i}_ltemp{args.sample_logistic_temperature}_mtemp{args.sample_mixture_temperature}{"_quantize" if args.sample_quantize else ""}.png')
            utils.save_image(sample_t, sample_save_path,
                             nrow=args.save_nrow, padding=args.save_padding, pad_value=1, scale_each=False)
            wandb.log({sample_save_path: wandb.Image(sample_save_path), "epoch": checkpoint_epochs})
elif args.mode == "test":
    if args.test_region == "full":
        slice_op = lambda x: x
        sliced_obs = obs
        region_str = f"full"
    else:
        def slice_op(x):
            # Take section of logits
            H, W = x.shape[2], x.shape[3]
            y = x[:, :, args.test_minh:args.test_maxh, args.test_minw:args.test_maxw]
            assert y.shape[2] == args.test_maxh - args.test_minh
            assert y.shape[3] == args.test_maxw - args.test_minw
            return y
        sliced_obs = (obs[0], args.test_maxh - args.test_minh, args.test_maxw - args.test_minw)
        region_str = f"{args.test_minh}:{args.test_maxh}_{args.test_minw}:{args.test_maxw}"

    model.eval()
    with torch.no_grad():
        for obs in resized_obses:
            logger.info(f"testing with obs {obs2str(obs)}...")
            test_bpd = test(model,
                            all_masks_by_obs[obs],
                            test_loader_by_obs[obs],
                            checkpoint_epochs,
                            progress_bar=True,
                            slice_op=slice_op,
                            sliced_obs=sliced_obs)
            test_nats = test_bpd * np.log(2) * np.prod(sliced_obs)
            logger.info(f"!!test loss with mode {args.mode}, randomize {args.randomize_order} for obs {obs2str(obs)}, sliced obs {obs2str(sliced_obs)}, region {region_str}: %s bpd = %s nats" % (test_bpd, test_nats))
elif args.mode == "count_params":
    model.train()

    print("Counting total number of parameters in model..")
    num_params = 0
    num_trainable_params = 0
    for param in model.parameters():
        num_params += np.prod(param.size())
        if param.requires_grad:
            num_trainable_params += np.prod(param.size())
    print("  Total number of parameters in model:", num_params)
    print("  Total number of trainable parameters in model:", num_trainable_params)

