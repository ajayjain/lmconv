from IPython import embed
import argparse
import os
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
                    default='cifar', help='Can be either cifar|mnist')
parser.add_argument('-p', '--print_every', type=int, default=20,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=20,
                    help='Every how many epochs to write checkpoint?')
parser.add_argument('-ts', '--sample_interval', type=int, default=4,
                    help='Every how many epochs to write samples?')
parser.add_argument('-r', '--load_params', type=str, default=None,
                    help='Restore training from previous model checkpoint?')
parser.add_argument('-rd', '--run_dir', type=str, default=None,
                    help="Optionally specify run directory. One will be generated otherwise."
                         "Use to save log files in a particular place")
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('-ID', '--exp_id', type=int, default=0)
parser.add_argument('--ours', action='store_true')
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
                    choices=["none", "weight_norm", "instance_norm", "instance_norm_affine"])
parser.add_argument('-af', '--accum_freq', type=int, default=1,
                    help='Batches per optimization step. Used for gradient accumulation')
parser.add_argument('--two_stream', action="store_true", help="Enable two stream model")
parser.add_argument('--order', type=str, choices=["raster_scan", "s_curve", "hilbert", "gilbert2d"],
                    help="Autoregressive generation order")
parser.add_argument('--randomize_order', action="store_true", help="Randomize between 8 variants of the "
                    "pixel generation order.")
parser.add_argument('--mode', type=str, choices=["train", "sample", "test"],
                    default="train")
parser.add_argument('--no_bias', action="store_true", help="Disable learnable bias for all convolutions")

args = parser.parse_args()
assert args.normalization != "weight_norm", "Weight normalization manually disabled in layers.py"


# Set seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)


# Create run directory
if args.run_dir:
    run_dir = args.run_dir
else:
    _name = "{:05d}_{}_lr{:.5f}_bs{}_gc{}_k{}_md{}".format(
        args.exp_id, args.dataset, args.lr, args.batch_size, args.clip, args.kernel_size, args.max_dilation)
    if args.normalization != "none":
        _name = f"{_name}_{args.normalization}"
    if args.exp_name:
        _name = f"{_name}+{args.exp_name}"
    run_dir = os.path.join("runs", _name)
    if args.mode == "train":
        os.makedirs(run_dir, exist_ok=False)
assert os.path.exists(run_dir), "Did not find run directory, check --run_dir argument"


# Log arguments
logger = configure_logger(os.path.join(run_dir, f"{args.mode}.log"))
logger.info("Run directory: %s", run_dir)
logger.info("Arguments: %s", args)
for k, v in vars(args).items():
    logger.info(f"  {k}: {v}")


# Create data loaders
sample_batch_size = 25
obs = (1, 28, 28) if 'mnist' in args.dataset else (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':True}
ds_transforms = transforms.Compose([transforms.ToTensor(), rescaling])

if 'mnist' in args.dataset :
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True, 
                        train=True, transform=ds_transforms), batch_size=args.batch_size, 
                            shuffle=True, **kwargs)

    test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

    logger.info("Using unaveraged loss discretized_mix_logistic_loss_1d, averaged loss discretized_mix_logistic_loss_1d_averaged")
    loss_op = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    loss_op_overaged = lambda real, fakes : discretized_mix_logistic_loss_1d_averaged(real, fakes)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

elif 'cifar' in args.dataset :
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

    logger.info("Using loss discretized_mix_logistic_loss")
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    # TODO: implement loss_op_averaged
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)
else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))


# Construct model
if args.ours:
    logger.info("Constructing our model")

    if args.normalization == "instance_norm":
        norm_op = lambda num_channels: nn.InstanceNorm2d(num_channels)
    elif args.normalization == "instance_norm_affine":
        norm_op = lambda num_channels: nn.InstanceNorm2d(num_channels, affine=True)
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
                conv_bias=(not args.no_bias))

    # Get generation orders
    base_generation_idx = get_generation_order_idx(args.order, obs[1], obs[2])
    if args.randomize_order:
        all_generation_idx = augment_orders(base_generation_idx, obs)
    else:
        all_generation_idx = [base_generation_idx]
    if args.mode == "train":
        plot_orders(all_generation_idx, obs, size=5, plot_rows=min(len(all_generation_idx), 4),
                    out_path=os.path.join(run_dir, "orderings.png"))

    # Make masks and plot
    all_masks = []
    for i, generation_idx in enumerate(all_generation_idx):
        masks = get_masks(generation_idx, obs[1], obs[2], args.kernel_size, args.max_dilation,
                          run_dir, plot_suffix=f"order{i}", plot=False)#(args.mode == "train"))
        logger.info(f"Mask shapes: {masks[0].shape}, {masks[1].shape}, {masks[2].shape}")
        all_masks.append(masks)
else:
    logger.info("Constructing original PixelCNN++")
    model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)

    assert not args.randomize_order
    all_generation_idx = [get_generation_order_idx("raster_scan", obs[1], obs[2])]
    all_masks = [(None, None, None)]
model = nn.DataParallel(model)
model = model.cuda()


# Load model parameters from checkpoint
if args.load_params:
    # TODO: Restore optimizer
    checkpoint_epochs = load_part_of_model(args.load_params, model=model.module, optimizer=None)
    logger.info(f"Model parameters loaded, from after {checkpoint_epochs} training epochs")
else:
    checkpoint_epochs = -1


# Create optimizer
# NOTE: PixelCNN++ TF repo uses betas=(0.95, 0.9995), different than PyTorch defaults
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)


def test(model, all_masks, test_loader, epoch="N/A"):
    logger.info(f"Testing with ensemble of {len(all_masks)} orderings")
    test_loss = 0.
    for batch_idx, (input,_) in enumerate(tqdm.tqdm(test_loader, desc=f"Test after epoch {epoch}")):
        input = input.cuda(non_blocking=True)
        input_var = Variable(input)

        #mask_init, mask_undilated, mask_dilated = all_masks[0]
        #output = model(input_var, mask_init=mask_init, mask_undilated=mask_undilated, mask_dilated=mask_dilated)
        #loss = loss_op(input_var, output)

        # Average likelihoods over multiple orderings
        outputs = []
        for mask_init, mask_undilated, mask_dilated in all_masks:
            output = model(input_var, mask_init=mask_init, mask_undilated=mask_undilated, mask_dilated=mask_dilated)
            outputs.append(output)
        loss = loss_op_overaged(input_var, outputs)

        test_loss += loss.item()
        del loss, output

    # FIXME: for final evaluation, don't use batch_idx * args.batch_size -- this slightly overestimates
    # the number of dims (10016 * prod(obs) * log(2) for mnist) since the last iteration might have fewer than
    # args.batch_size images. Leaving this code the same for now to allow comparison between training runs.
    deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
    assert deno > 0, embed()
    test_bpd = test_loss / deno
    return test_bpd


def sample(model, generation_idx, mask_init, mask_undilated, mask_dilated):
    model.eval()
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.cuda()
    for num_px_sampled, (i, j) in enumerate(tqdm.tqdm(generation_idx, desc="Sampling pixels")):
        data_v = Variable(data)
        out = model(data_v, sample=True, mask_init=mask_init, mask_undilated=mask_undilated, mask_dilated=mask_dilated)
        out_sample = sample_op(out)
        data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data


if args.mode == "train":
    logger.info("starting training")
    writer = SummaryWriter(log_dir=run_dir)
    global_step = 0
    min_train_bpd = 1e12
    min_test_bpd = 1e12
    for epoch in range(checkpoint_epochs + 1, args.max_epochs):
        train_loss = 0.
        time_ = time.time()
        model.train()
        for batch_idx, (input,_) in enumerate(tqdm.tqdm(train_loader, desc=f"Train epoch {epoch}")):
            input = input.cuda(non_blocking=True)

            order_i = np.random.randint(len(all_masks))
            output = model(input, mask_init=all_masks[order_i][0], mask_undilated=all_masks[order_i][1], mask_dilated=all_masks[order_i][2])

            loss = loss_op(input, output)

            if batch_idx % args.accum_freq == 0:
                optimizer.zero_grad()
            loss.backward()
            if (batch_idx + 1) % args.accum_freq == 0:
                if args.clip > 0:
                    # Compute and rescale gradient norm
                    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    if gradient_norm > args.clip:
                        logger.warning(f"Clipped gradients to norm {args.clip}")
                else:
                    # Just compute the gradient norm
                    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
                    gradient_norm = 0
                    for p in parameters:
                        param_norm = p.grad.data.norm(2)
                        gradient_norm += param_norm.item() ** 2
                    gradient_norm = gradient_norm ** (1. / 2)
                writer.add_scalar('train/gradient_norm', gradient_norm, global_step)
                optimizer.step()
            train_loss += loss.item()

            deno = args.batch_size * np.prod(obs) * np.log(2.)
            assert deno > 0, embed()
            train_bpd = loss.item() / deno
            writer.add_scalar('train/bpd', train_bpd, global_step)
            min_train_bpd = min(min_train_bpd, train_bpd)
            writer.add_scalar('train/min_bpd', min_train_bpd, global_step)

            if batch_idx >= 100 and (loss.item() / deno) >= 10:
                logger.warning("WARNING: main.py: large batch loss {} bpd".format(loss.item() / deno))

            if (batch_idx + 1) % args.print_every == 0: 
                deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
                logger.info('train bpd : {:.4f}, train loss : {:.1f}, time : {:.4f}, global step: {}'.format(
                    (train_loss / deno),
                    train_loss,
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
            test_bpd = test(model, all_masks, test_loader, epoch)
            writer.add_scalar('test/bpd', test_bpd, global_step)
            logger.info('test loss : %s' % test_bpd)

            # Log min test bpd for smoothness
            min_test_bpd = min(min_test_bpd, test_bpd)
            writer.add_scalar('test/min_bpd', min_test_bpd, global_step)

            if (epoch + 1) % args.save_interval == 0: 
                logger.info('saving model...')
                try:
                    torch.save({
                        "epoch": epoch,
                        "test_loss": test_bpd,
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "args": vars(args),
                    }, os.path.join(run_dir, f"{args.exp_id}_ep{epoch}.pth"))
                except Exception as e:
                    logger.error("Failed to save checkpoint! Error: %s", e)

            if (epoch + 1) % args.sample_interval == 0: 
                try:
                    sample_order_i = np.random.randint(len(all_masks))
                    logger.info('sampling images with ordering variant %d...', sample_order_i)
                    sample_t = sample(model, all_generation_idx[sample_order_i], *all_masks[sample_order_i])
                    sample_t = rescaling_inv(sample_t)
                    utils.save_image(sample_t, os.path.join(run_dir, f'tsample_{epoch}.png'), 
                                    nrow=5, padding=0)
                except Exception as e:
                    logger.error("Failed to sample images! Error: %s", e)
elif args.mode == "sample":
    model.eval()
    with torch.no_grad():
        sample_order_i = np.random.randint(len(all_masks))
        logger.info('sampling images with ordering variant %d...', sample_order_i)
        sample_t = sample(model, all_generation_idx[sample_order_i], *all_masks[sample_order_i])
        sample_t = rescaling_inv(sample_t)
        utils.save_image(sample_t, os.path.join(run_dir, f'sample_{checkpoint_epochs}.png'),
                         nrow=5, padding=0)
elif args.mode == "test":
    model.eval()
    with torch.no_grad():
        logger.info('testing...')
        test_bpd = test(model, all_masks, test_loader, checkpoint_epochs)
        logger.info('test loss : %s bpd' % test_bpd)
