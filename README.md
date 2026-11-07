# AlexNet from Scratch in PyTorch

A PyTorch implementation of **AlexNet**, built by reading the original paper:
[*ImageNet Classification with Deep Convolutional Neural Networks*](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) (Krizhevsky, Sutskever, Hinton, 2012).

The model is trained and tested on **CIFAR-10**.

## Architecture

| Layer | Details |
|-------|---------|
| Conv1 | 3 → 96, 11×11, stride 4, ReLU, MaxPool 3×3 (stride 2) |
| Conv2 | 96 → 256, 5×5, padding 2, ReLU, MaxPool 3×3 (stride 2) |
| Conv3 | 256 → 384, 3×3, padding 1, ReLU |
| Conv4 | 384 → 384, 3×3, padding 1, ReLU |
| Conv5 | 384 → 256, 3×3, padding 1, ReLU, MaxPool 3×3 (stride 2) |
| FC1 | 9216 → 4096, ReLU |
| FC2 | 4096 → 4096, ReLU |
| FC3 | 4096 → 10 (CIFAR-10 classes) |

Input size is 227×227×3.

## Training Setup

- **Dataset:** CIFAR-10 (50,000 train / 10,000 test), images resized from 32×32 to 227×227
- **Augmentation:** random horizontal flip
- **Loss:** Cross-entropy
- **Optimizer:** Adam, learning rate 1e-4
- **Batch size:** 64
- **Epochs:** 10
- **Hardware:** Google Colab, T4 GPU

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | ~83% |

Predictions on test images are visualized with Matplotlib (green = correct, red = wrong). The images look blurry because CIFAR-10 images are only 32×32 and are upscaled to 227×227. This doesn't affect training.

## Getting Started

```bash
git clone https://github.com/krishtyagi0109-pixel/AlexNet.git
cd AlexNet
pip install torch torchvision matplotlib
```

Open the notebook in Jupyter or Google Colab and run all cells. CIFAR-10 downloads automatically. A GPU is strongly recommended, since training on CPU is very slow.

## What's Inside

- AlexNet model definition
- Training loop with per-epoch loss tracking
- Test accuracy evaluation
- Matplotlib visualization of predictions

## Possible Improvements

- Add Dropout and Local Response Normalization, as in the original paper
- Add a learning rate scheduler and weight decay
- Train for more epochs
- Compare against `torchvision.models.alexnet`

## Reference

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS.
