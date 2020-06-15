## Locally Masked Convolution
Code for the UAI 2020 paper "Locally Masked Convolution for Autoregressive Models".

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

### Get CelebAHQ data
```
cd data
wget https://storage.googleapis.com/glow-demo/data/celeba-tfr.tar
tar -xvzf celeba-tfr.tar
```

### Running the code
```
python main.py
```

### Credits
This code is based on a [PyTorch implementation](https://github.com/pclucas14/pixel-cnn-pp) of [PixelCNN++.](https://arxiv.org/pdf/1701.05517.pdf) by Lucas Caccia.
