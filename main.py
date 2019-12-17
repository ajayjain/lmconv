import time
import os
import argparse
from IPython import embed

from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, utils
from tensorboardX import SummaryWriter

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
parser.add_argument('-p', '--print_every', type=int, default=50,
                    help='how many iterations between print statements')
parser.add_argument('-t', '--save_interval', type=int, default=10,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=str, default=None,
                    help='Restore training from previous model checkpoint?')
parser.add_argument('--exp_name', type=str, default=None)
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
                    default=5e-4, help='Weight decay during optimization')
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
parser.add_argument('--normalization', type=str, default='weight_norm')
parser.add_argument('-af', '--accum_freq', type=int, default=1,
                    help='Batches per optimization step. Used for gradient accumulation')
parser.add_argument('--two_stream', action="store_true", help="Enable two stream model")
parser.add_argument('--order', type=str, choices=["raster_scan", "s_curve", "hilbert"],
                    help="Autoregressive generation order")

args = parser.parse_args()

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model_name = 'pcnn_{}_lr{:.5f}_wd{}_gc{}_nr-resnet{}_nr-filters{}_k{}_md{}_bs{}'.format(args.dataset, args.lr, args.weight_decay, args.clip, args.nr_resnet, args.nr_filters, args.kernel_size, args.max_dilation, args.batch_size)
if args.exp_name:
    model_name = f'{model_name}_{args.exp_name}'
assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)
writer = SummaryWriter(log_dir=os.path.join('runs', model_name))

print("Model name:", model_name)
print("Args:")
print(args)

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

    print("discretized_mix_logistic_loss_1d")
    loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

elif 'cifar' in args.dataset :
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

    print("discretized_mix_logistic_loss")
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)
else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))

assert args.normalization in ['weight_norm', 'none']

if args.ours:
    ks = (args.kernel_size, args.kernel_size)
    model = OurPixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix,
                kernel_size=ks, max_dilation=args.max_dilation,
                weight_norm=(args.normalization == "weight_norm"),
                two_stream=args.two_stream)

    # Make masks, copy to GPU and repeat along batch for DataParallel
    generation_idx = get_generation_order_idx(args.order, obs[1], obs[2])
    mask_init = get_unfolded_masks(generation_idx, obs[1], obs[2], k=args.kernel_size, dilation=1, mask_type='A')
    mask_init = mask_init.cuda(non_blocking=True).repeat(torch.cuda.device_count(), 1, 1)

    mask_undilated = get_unfolded_masks(generation_idx, obs[1], obs[2], k=args.kernel_size, dilation=1, mask_type='B')
    mask_undilated = mask_undilated.cuda(non_blocking=True).repeat(torch.cuda.device_count(), 1, 1)

    if args.max_dilation == 1:
        mask_dilated = mask_undilated
    else:
        mask_dilated = get_unfolded_masks(generation_idx, obs[1], obs[2], k=args.kernel_size, dilation=args.max_dilation, mask_type='B')
        mask_dilated = mask_dilated.cuda(non_blocking=True).repeat(torch.cuda.device_count(), 1, 1)

    print(f"Mask shapes: {mask_init.shape}, {mask_undilated.shape}, {mask_dilated.shape}")
else:
    assert False
    model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
    mask_init, mask_undilated, mask_dilated = None, None, None
model = nn.DataParallel(model)
model = model.cuda()

if args.load_params:
    load_part_of_model(model, args.load_params)
    # model.load_state_dict(torch.load(args.load_params))
    print('model parameters loaded')

# NOTE: PixelCNN++ TF repo uses betas=(0.95, 0.9995), different than PyTorch defaults
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

def sample(model):
    model.eval()
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.cuda()
    for i in range(obs[1]):
        for j in range(obs[2]):
            data_v = Variable(data)
            out = model(data_v, sample=True, mask_init=mask_init, mask_undilated=mask_undilated, mask_dilated=mask_dilated)
            out_sample = sample_op(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data

print('starting training')
global_step = 0
min_train_bpd = 1e12
min_test_bpd = 1e12
for epoch in range(args.max_epochs):
    train_loss = 0.
    time_ = time.time()
    model.train()
    for batch_idx, (input,_) in enumerate(tqdm(train_loader, desc=f"Train epoch {epoch}")):
        input = input.cuda(non_blocking=True)

        if not torch.isfinite(input).all().cpu().item():
            print("ERROR: main.py: NaN or Inf in input, embedding")
            embed()

        output = model(input, mask_init=mask_init, mask_undilated=mask_undilated, mask_dilated=mask_dilated)

        if not torch.isfinite(output).all().cpu().item():
            print("ERROR: main.py: NaN or Inf in returned tensor, embedding")
            embed()

        loss = loss_op(input, output)

        if not torch.isfinite(loss).all().cpu().item():
            print("ERROR: main.py: NaN or Inf in loss, embedding")
            embed()

        if batch_idx % args.accum_freq == 0:
            optimizer.zero_grad()
        loss.backward()
        if (batch_idx + 1) % args.accum_freq == 0:
            if args.clip > 0:
                # Compute and rescale gradient norm
                gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                if gradient_norm > args.clip:
                    print("WARN: Clipped gradients")
                else:
                    print("No need to clip gradients")
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
            print("WARNING: main.py: large batch loss {} bpd".format(loss.item() / deno))
            # pdb.set_trace()

        if (batch_idx + 1) % args.print_every == 0: 
            deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
            print('train bpd : {:.4f}, train loss : {:.1f}, time : {:.4f}, global step: {}'.format(
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
        test_loss = 0.
        for batch_idx, (input,_) in enumerate(tqdm(test_loader, desc=f"Test epoch {epoch}")):
            input = input.cuda(non_blocking=True)
            input_var = Variable(input)
            output = model(input_var, mask_init=mask_init, mask_undilated=mask_undilated, mask_dilated=mask_dilated)
            loss = loss_op(input_var, output)
    
            if not torch.isfinite(loss).all().cpu().item():
                print("ERROR: main.py: NaN or Inf in test loss in test loop, embedding")
                embed()

            test_loss += loss.item()
            del loss, output

        deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
        assert deno > 0, embed()
        test_bpd = test_loss / deno
        writer.add_scalar('test/bpd', test_bpd, global_step)
        min_test_bpd = min(min_test_bpd, test_bpd)
        writer.add_scalar('test/min_bpd', min_test_bpd, global_step)
        print('test loss : %s' % test_bpd)

        if (epoch + 1) % args.save_interval == 0: 
            torch.save({
                "epoch": epoch,
                "test_loss": test_bpd,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "mask_init": mask_init,
                "mask": mask
            }, 'models/{}_{}.pth'.format(model_name, epoch))

            print('sampling...')
            sample_t = sample(model)
            sample_t = rescaling_inv(sample_t)
            utils.save_image(sample_t,'images/{}_{}.png'.format(model_name, epoch), 
                    nrow=5, padding=0)
