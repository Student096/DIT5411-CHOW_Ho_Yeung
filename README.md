# DIT5411 Machine Learning Assignment — Traditional Chinese Handwritten Character Recognition

## Table of Contents

- [Project Overview](#1-project-overview)
- [Dataset, Train/Test Split, and Data Augmentation](#2-dataset-traintest-split-and-data-augmentation)
- [Model Architectures and Training Configuration](#3-model-architectures-and-training-configuration)
- [Experimental Results](#4-experimental-results)
- [Model Performance Analysis](#5-model-performance-analysis)
- [Software and Hardware Requirements](#6-software-and-hardware-requirements)
- [Environment Setup](#7-environment-setup)
- [GPU Verification](#8-gpu-verification)
- [Running Smoke Tests and Full Training](#9-running-smoke-tests-and-full-training)
- [References](#10-references)

## 1. Project Overview

This project builds, trains, and evaluates three neural network architectures for Traditional Chinese handwritten character recognition:

- `Model_1_MLP`
- `Model_2_Basic_CNN`
- `Model_3_Deep_CNN_BN`

The models are trained on the [`chenkenanalytic/handwritting_data_all`](https://huggingface.co/datasets/chenkenanalytic/handwritting_data_all) dataset, which contains **13,065 Traditional Chinese character classes**. In accordance with the assignment specification, all models use a fixed output layer of **13,065 neurons** with `softmax` activation.

## 2. Dataset, Train/Test Split, and Data Augmentation

### Dataset and Preprocessing

- **Dataset source:** `chenkenanalytic/handwritting_data_all`
- **Number of character classes:** 13,065
- **Input image format:** Grayscale images resized to `64 × 64 × 1`
- **Pixel normalization:** `[0.0, 1.0]`

### Train/Test Split

For each character class:

- The first 40 images are assigned to the training set.
- The remaining images are assigned to the testing set.
- The testing set contains **162,550 images in total**.

### Data Augmentation

`CharDataSequence` performs on-the-fly affine transformations during training using OpenCV's `cv2.warpAffine`. The augmentation pipeline is designed to ensure that every character class has at least 200 training images:

- **Training samples:** 2,613,000 across 13,065 classes
- **Rotation:** Random angle in `[-15°, +15°]`
- **Scaling:** Random scale factor in `[0.85, 1.15]`
- **Shearing:** Random shear factors in `[-0.15, +0.15]`

## 3. Model Architectures and Training Configuration

All three models use the following training configuration:

- **Optimizer:** Adam with `learning_rate=1e-3`
- **Loss function:** `sparse_categorical_crossentropy`
- **Batch size:** `128` (`BATCH_SIZE = 128`)
- **Epochs:** `9` (`EPOCHS = 9`)
- **Output layer:** `Dense(13065, activation="softmax")`

### Model 1 — MLP

`Model_1_MLP` flattens each `64 × 64` grayscale image into a one-dimensional vector before applying fully connected layers:

```text
Flatten
Dense(1024, activation="relu")
Dropout(0.3)
Dense(512, activation="relu")
Dense(13065, activation="softmax")
```

### Model 2 — Basic CNN

`Model_2_Basic_CNN` uses two convolutional and pooling blocks, followed by fully connected classification layers:

```text
Conv2D(32, (3, 3), activation="relu", padding="same")
MaxPooling2D((2, 2))
Conv2D(64, (3, 3), activation="relu", padding="same")
MaxPooling2D((2, 2))
Flatten
Dropout(0.4)
Dense(512, activation="relu")
Dense(13065, activation="softmax")
```

### Model 3 — Deep CNN with Batch Normalization

`Model_3_Deep_CNN_BN` uses three convolutional blocks with Batch Normalization and ReLU activation. The channel sizes increase from 32 to 128:

```text
Conv2D(32, ...)
BatchNormalization
ReLU
MaxPooling2D((2, 2))

Conv2D(64, ...)
BatchNormalization
ReLU
MaxPooling2D((2, 2))

Conv2D(128, ...)
BatchNormalization
ReLU
MaxPooling2D((2, 2))

GlobalAveragePooling2D
Dense(512, activation="relu")
Dropout(0.4)
Dense(13065, activation="softmax")
```

In summary, this model uses `32 → 64 → 128` filters with `MaxPooling2D((2, 2))`, followed by `GlobalAveragePooling2D`, `Dense(512, activation="relu")`, `Dropout(0.4)`, and the final 13,065-class softmax layer.

## 4. Experimental Results

All models were trained for 9 epochs using 13,065 output classes.

| Rank | Model | Output Neurons | Epochs | Training Samples | Test Samples | Test Loss | Test Accuracy | Training Time (s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **1** | **`Model_2_Basic_CNN`** | 13,065 | 9 | 2,613,000 | 162,550 | **0.5927** | **0.8566** | 12,739.2 |
| **2** | **`Model_3_Deep_CNN_BN`** | 13,065 | 9 | 2,613,000 | 162,550 | 9.5467 | **0.0315** | 12,721.6 |
| **3** | **`Model_1_MLP`** | 13,065 | 9 | 2,613,000 | 162,550 | 9.4784 | **0.0001** | 13,036.8 |

> `model_results.csv` also records `Best_Train_Accuracy` and `Final_Train_Accuracy`. Evaluation across all 162,550 test samples is used as the primary comparison metric.

## 5. Model Performance Analysis

### Top Performer — `Model_2_Basic_CNN`

- **Test accuracy:** `0.8566`, or **85.66%**
- **Test loss:** `0.5927`

The Basic CNN achieves the best performance across all 13,065 classes. Its local `3 × 3` convolutions capture stroke patterns effectively, while `Flatten` preserves the spatial feature layout before the dense classification layers. This helps the model distinguish visually similar Traditional Chinese characters.

### `Model_3_Deep_CNN_BN`

- **Test accuracy:** `0.0315`, or **3.15%**
- **Test loss:** `9.5467`

Although this model uses Batch Normalization and deeper channel counts (`32 → 64 → 128`), it replaces `Flatten` with `GlobalAveragePooling2D()`. Spatial averaging may reduce sensitivity to fine stroke positions and radical arrangements, which is particularly harmful when distinguishing 13,065 fine-grained classes. The relatively short nine-epoch training schedule may also limit the benefits of the deeper architecture.

### `Model_1_MLP`

- **Test accuracy:** `0.0001`, or **0.01%**
- **Test loss:** `9.4784`

The MLP flattens the `64 × 64` pixels at the input, discarding two-dimensional spatial structure from the beginning. Across 13,065 fine-grained classes, uniform random guessing would achieve approximately:

```text
1 / 13,065 ≈ 0.000077
```

The pure dense architecture therefore has limited ability to generalize character shapes within nine epochs.

### Runtime Comparison

All three models completed training in approximately **3.5–3.6 hours** (`12,721.6–13,036.8` seconds). The similar runtimes reflect the shared cost of image decoding and on-the-fly affine augmentation across 2,613,000 training samples per epoch.

## 6. Software and Hardware Requirements

### Software Requirements

The project can be run on Linux/Ubuntu, Windows with WSL2, or a comparable environment. Use a dedicated virtual environment rather than installing packages into externally managed system Python such as `/usr/bin/python3`.

Recommended software:

- **Operating system:** Ubuntu/Linux or Windows 10/11 with WSL2
- **Python:** Python 3.10–3.12
- **Editor:** Visual Studio Code with the **Remote - WSL** extension when using WSL2
- **Virtual environment:** `~/tf_gpu` or a project-local `.venv`
- **GPU support:** Compatible NVIDIA drivers and a CUDA/cuDNN runtime supported by the installed TensorFlow version

### Required Python Libraries

- `tensorflow` or `tensorflow[and-cuda]` for a GPU-enabled installation
- `opencv-python` (`cv2`)
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

### GPU Acceleration

GPU acceleration is optional for subset or smoke testing but strongly recommended for the full 13,065-class, 9-epoch training run. A compatible NVIDIA GPU driver must be installed, and CUDA/cuDNN support must be accessible to TensorFlow.

Check NVIDIA driver visibility with:

```bash
nvidia-smi
```

### Hardware Requirements and Recommendations

#### Minimum Requirements for Subset or Smoke Testing

- Multi-core CPU, such as an Intel Core i5/i7 or equivalent
- `8 GB` RAM
- `10 GB+` free disk space
- A reduced test configuration such as `MAX_CLASSES = 100`

#### Recommended Requirements for the Full 13,065-Class Run

- **GPU:** Dedicated NVIDIA GPU with `6–8 GB+` VRAM, such as an NVIDIA GeForce RTX 3060 or equivalent
- **RAM:** `16 GB` or more for smooth data loading and `Sequence` batch generation
- **Storage:** SSD with `30–50 GB+` free space for extracted character folders, checkpoints, and logs
- **Power and stability:** Keep laptops connected to AC power. Run long sessions in the background with `nohup` or another persistent process manager so sleep or terminal disconnection does not interrupt training.

## 7. Environment Setup

### Option A: Windows, WSL2, and VS Code

#### Step 1: Install Required Windows Components

Run the following commands in **Windows PowerShell**:

```powershell
winget install -e --id Microsoft.VCRedist.2015+.x64 --source winget --accept-source-agreements --accept-package-agreements
wsl --install -d Ubuntu
code --install-extension ms-vscode-remote.remote-wsl
```

Restart Windows after installation completes. After rebooting, open Ubuntu once to complete the initial username and password setup.

#### Step 2: Verify WSL2 Status

Run the following command in **Windows PowerShell** and verify that Ubuntu shows `VERSION 2`:

```powershell
wsl -l -v
```

### Option B: Ubuntu/Linux Setup

Open an Ubuntu/Linux terminal and install the basic tools:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip unzip git
```

### Step 3: Create a Project Directory

Replace `your-name` with the desired project directory name:

```bash
mkdir -p ~/DIT5411-your-name
cd ~/DIT5411-your-name
```

The original project directory name may also be used:

```bash
mkdir -p ~/DIT5411-your-name
cd ~/DIT5411-your-name
```

### Step 4: Create and Activate a Virtual Environment

A project-local `.venv` is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Alternatively, create the virtual environment at `~/tf_gpu`:

```bash
python3 -m venv ~/tf_gpu
source ~/tf_gpu/bin/activate
python -m pip install --upgrade pip
```

### Step 5: Install Python Packages

For a standard installation:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
```

For the TensorFlow CUDA-enabled installation used in the original WSL2 setup:

```bash
pip install "tensorflow[and-cuda]" opencv-python numpy pandas matplotlib scikit-learn
```

Do not install packages directly into the externally managed system Python environment.

### Step 6: Verify the Python Installation

```bash
python -c "import tensorflow as tf, cv2, numpy as np, pandas as pd; print('TF:', tf.__version__, 'CV2:', cv2.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))"
```

## 8. GPU Verification

Create `01_check_gpu.py` inside the project directory:

```bash
cat << 'EOF' > 01_check_gpu.py
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("Detected GPUs:", gpus)

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    with tf.device("/GPU:0"):
        a = tf.ones((256, 256))
        b = tf.ones((256, 256))
        c = tf.matmul(a, b)

    print("GPU matrix multiplication succeeded; output shape:", c.shape)
else:
    print("No GPU detected. TensorFlow will run on the CPU.")
EOF
```

Run the verification script:

```bash
python3 01_check_gpu.py
```

You can also run it using an absolute path if the script is stored in `~/DIT5411-your-name`:

```bash
python3 ~/DIT5411-your-name/01_check_gpu.py
```

## 9. Running Smoke Tests and Full Training

Before launching the full experiment, run a reduced smoke test if the training code supports a class limit such as `MAX_CLASSES`:

```python
MAX_CLASSES = 100
```

A smoke test helps verify dataset loading, image preprocessing, augmentation, model construction, checkpoint writing, and evaluation without committing to the full multi-hour run.

After the smoke test succeeds, restore the full configuration:

```python
MAX_CLASSES = 13065
BATCH_SIZE = 128
EPOCHS = 9
```

For long-running training sessions, keep the virtual environment activated and use a persistent command such as:

```bash
nohup python train.py > training.log 2>&1 &
```

The exact training script name may differ depending on the project implementation. Monitor progress with:

```bash
tail -f training.log
```

## 10. References

1. [chenkenanalytic/handwritting_data_all — Traditional Chinese Handwritten Character Dataset](https://huggingface.co/datasets/chenkenanalytic/handwritting_data_all)
2. Abadi, M., et al. (2015). [TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](https://www.tensorflow.org/)
3. [Keras Documentation](https://keras.io/)
4. Bradski, G. (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools.
