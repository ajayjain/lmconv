import time
import os
import argparse

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
parser.add_argument('-c', '--clip', default=100, help='Gradient norms clipped to this value')
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

args = parser.parse_args()

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

model_name = 'pcnn_{}_lr{:.5f}_wd{}_gc{}_nr-resnet{}_nr-filters{}_k{}_md{}_bs{}'.format(args.dataset, args.lr, args.weight_decay, args.clip, args.nr_resnet, args.nr_filters, args.kernel_size, args.max_dilation, args.batch_size)
if args.exp_name:
    model_name = f'{model_name}_{args.exp_name}'
assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)
writer = SummaryWriter(log_dir=os.path.join('runs', model_name))

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

    loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

elif 'cifar' in args.dataset :
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)
else :
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))

if args.ours:
    model = OurPixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix,
                kernel_size=(args.kernel_size, args.kernel_size), max_dilation=args.max_dilation)
else:
    model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
                input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
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
            out   = model(data_v, sample=True)
            out_sample = sample_op(out)
            data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data

print('starting training')
global_step = 0
for epoch in range(args.max_epochs):
    train_loss = 0.
    time_ = time.time()
    model.train()
    for batch_idx, (input,_) in enumerate(tqdm(train_loader, desc=f"Train epoch {epoch}")):
        input = input.cuda(non_blocking=True)
        output = model(input)
        loss = loss_op(input, output)
        optimizer.zero_grad()
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        writer.add_scalar('train/gradient_norm', gradient_norm, global_step)
        optimizer.step()
        train_loss += loss.item()

        deno = args.batch_size * np.prod(obs) * np.log(2.)
        writer.add_scalar('train/bpd', (loss.item() / deno), global_step)

        if (batch_idx + 1) % args.print_every == 0: 
            deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
            print('loss : {:.4f}, time : {:.4f}, global step: {}'.format(
                (train_loss / deno), 
                (time.time() - time_),
                global_step))
            train_loss = 0.
            time_ = time.time()

        global_step += 1

    # decrease learning rate
    scheduler.step()

    model.eval()
    with torch.no_grad():
        test_loss = 0.
        for batch_idx, (input,_) in enumerate(tqdm(test_loader, desc=f"Test epoch {epoch}")):
            input = input.cuda(non_blocking=True)
            input_var = Variable(input)
            output = model(input_var)
            loss = loss_op(input_var, output)
            test_loss += loss.item()
            del loss, output

        deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
        test_loss = test_loss / deno
        writer.add_scalar('test/bpd', test_loss, global_step)
        print('test loss : %s' % test_loss)

        if (epoch + 1) % args.save_interval == 0: 
            torch.save({
                "epoch": epoch,
                "test_loss": test_loss,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, 'models/{}_{}.pth'.format(model_name, epoch))

            print('sampling...')
            sample_t = sample(model)
            sample_t = rescaling_inv(sample_t)
            utils.save_image(sample_t,'images/{}_{}.png'.format(model_name, epoch), 
                    nrow=5, padding=0)
