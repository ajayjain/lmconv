## Locally Masked Convolution
Code for the UAI 2020 paper "Locally Masked Convolution for Autoregressive Models", implemented with PyTorch. The Locally Masked Convolution layer allows PixelCNN-style autoregressive models to use a custom pixel generation order, rather than a raster scan. Training and evaluation code are available in `main.py`. To use locally masked convolutions in your project, see `locally_masked_convolution.py` for a memory-efficient implementation that depends only on torch. The layer uses masks that are generated in `masking.py`.

Paper: [https://arxiv.org/abs/2006.12486](https://arxiv.org/abs/2006.12486)

Website: [https://ajayjain.github.io/lmconv/](https://ajayjain.github.io/lmconv/)

<img src="https://ajayjain.github.io/lmconv/resources/lmconv_overview.png" width="600px" alt="Image completions and samples using a Locally Masked Convolutions">

### Citation
If you find our paper or code relevant to your research, please cite our UAI 2020 paper:
```
@inproceedings{jain2020lmconv,
    title={Locally Masked Convolution for Autoregressive Models},
    author={Ajay Jain and Pieter Abbeel and Deepak Pathak},
    year={2020},
    booktitle={Conference on Uncertainty in Artificial Intelligence (UAI)},
}
```

### Setup
Create a Python 3.7 environment with PyTorch installed following [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/). For example, with Miniconda/Anaconda, you can run:
```
conda create -n gen_py37 python=3.7
conda activate gen_py37
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

Install other dependencies:
```
pip install -r requirements.txt
```
NOTE: this installs the CPU-only version of Tensorflow, which is used to load CelebA-HQ data. We use the CPU version to prevent Tensorflow from using GPU memory.

CIFAR10 and MNIST images will be automatically downloaded by the code, but CelebA-HQ needs to be downloaded manually:
```
mkdir data
cd data
wget https://storage.googleapis.com/glow-demo/data/celeba-tfr.tar
tar -xvzf celeba-tfr.tar
```

### Evaluating pretrained models

Pretrained model checkpoints are available [here](https://drive.google.com/drive/folders/1ESeIKS3itwaFO_XigF4g-SAXFofwfEjF?usp=sharing).

Evaluate CIFAR10 whole-image likelihoods with 8 S-curve orders (2.887 bpd). Remove the `--randomize_order` flag to test with a single order (2.909 bpd):
```
mkdir runs/cifar_run  # for storing logs and images
python main.py -d cifar -b 32 --ours -k 3 --normalization pono --order s_curve --randomize_order -dp 0 --mode test --disable_wandb --run_dir runs/cifar_run --load_params cifar_s8.pth
```

Evaluate MNIST whole-image likelihoods with 8 S-curve orders (77.56 nats).
```
mkdir runs/mnist_run  # for storing logs and images
python main.py -d mnist -b 32 --ours -k 3 --normalization pono --order s_curve --randomize_order -dp 0 --binarize --mode test --disable_wandb --run_dir runs/mnist_run --load_params binary_mnist_ep159_s8.pth
```

Evaluate CIFAR10 conditional (inpainting) likelihoods for the top half of image with 2 orders (2.762 bpd). We add flags specifying the hidden region and generation order ([guide to `--test-mask` numbering](https://drive.google.com/open?id=1ETrntyAKvzYNMpntMFfj8WM5OgQMf2j_&authuser=ajayj%40berkeley.edu&usp=drive_fs)):
```
python main.py -d cifar -b 32 --ours -k 3 --normalization pono --order s_curve --randomize_order -dp 0 --mode test --disable_wandb --run_dir runs/cifar_run --load_params cifar_s8.pth --test_region custom --test_maxh 16 --test_maxw 32 --test_masks 1 3
```

Complete top region of CIFAR10 images:
```
python main.py -d cifar -b 32 --ours -k 3 --normalization pono --order s_curve -dp 0 --mode sample --disable_wandb --run_dir runs/cifar_run --load_params cifar_s8.pth --sample_region custom --sample_offset1 -16 --sample_offset2 -16 --sample_size_h 12 --sample_size_w 32 --sample_batch_size 48 --base_order_reflect_rows
```

Sample MNIST digits with hilbert curve order:
```
python main.py -d mnist --ours -k 3 --normalization pono --order gilbert2d -dp 0 --sample_region full --load_params grayscale_mnist_ep299_8h.pth --mode sample --sample_batch_size 16 --disable_wandb
```

### Train model
Train model on CIFAR10 (`-t` configures checkpoint save frequency, `-ID` allows runs to be numbered):
```
python main.py -d cifar -b 32 -t 10 --ours -c 2e6 -k 3 --normalization pono --order s_curve --randomize_order -dp 0 --exp_name s_rand_dp0_pono -ID 10000 --test_interval 4
```

Average checkpoints (optional, helps likelihoods slightly. need to train with `-t 1` to save checkpoints every epoch):
```
python average_checkpoints.py --run_id 10000 --inputs runs/<RUN_DIR> --output averaged.pth --num-epoch-checkpoints <NUM_CHECKPOINTS>
```

### Credits
This code was originally based on a [PyTorch implementation](https://github.com/pclucas14/pixel-cnn-pp) of [PixelCNN++](https://arxiv.org/pdf/1701.05517.pdf) by Lucas Caccia. `average_checkpoints.py` is sourced from the [fairseq](https://github.com/pytorch/fairseq/blob/master/scripts/average_checkpoints.py) project.

