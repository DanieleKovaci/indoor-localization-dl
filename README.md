# indoor-localization-dl

PyTorch MLP and 1D CNN for building and floor classification on UJIIndoorLoc Wi-Fi RSSI data.

## Background

My bachelor thesis (Kovaci, 2026) used SPSS linear regression and ANOVA on a 100-sample subset of UJIIndoorLoc to identify which signal and structural factors drive indoor positioning error. The best model explained 9.4% of the error variance (R² = 0.094), weak enough to conclude that linear RSSI models aren't sufficient on their own. This project trains an MLP and a 1D CNN on the full 21,048-sample dataset for building and floor classification.

## Dataset

**UJIIndoorLoc** ([UCI ML Repository #310](https://archive.ics.uci.edu/dataset/310/ujiindoorloc))
— 21,048 samples (19,937 train / 1,111 validation), 520 Wi-Fi access point RSSI features (WAP001–WAP520).
Raw value 100 denotes a non-detected access point; it is replaced with −105 dBm during preprocessing.
Features are scaled to [0, 1] with a MinMaxScaler fitted on training data only.

Two classification tasks: building (3 classes: 0, 1, 2) and floor (5 classes: 0–4).

## Models

**MLP** — Three fully-connected hidden layers (512 → 256 → 128 units), each followed by BatchNorm1d, ReLU, and Dropout(0.3). Final linear layer outputs class logits. Trained with Adam and StepLR decay (step=10, γ=0.5).

**CNN-1D** — The 520-dimensional input is treated as a 1D signal. Three convolutional blocks (1→32→64→128 channels, kernel sizes 5/5/3) with BatchNorm and ReLU, two MaxPool(2) layers reducing the sequence to length 130, then a two-layer classifier head (Linear 16640→256→num_classes) with Dropout(0.3).

## Results

| Task     | MLP    | CNN-1D |
|----------|--------|--------|
| Building | 83.80% | 91.72% |
| Floor    | 61.03% | 65.89% |

Evaluated on the official UJIIndoorLoc validation split (1,111 samples). The train and validation sets were collected in separate campaigns with different class distributions, which holds the numbers below cross-validated results in the literature (building accuracy in cross-validated settings typically exceeds 99%).

The thesis measured positioning error in metres, not classification accuracy, so the numbers aren't directly comparable. What they share: the thesis found linear RSSI models explain almost nothing (R² = 0.094); the CNN here gets to 91.7% building accuracy using the full 520-WAP fingerprint.

## Setup and usage

```bash
pip install -r requirements.txt
```

Train all four combinations:

```bash
# Building classification
python train.py --model mlp --task building --epochs 30 --lr 0.001  --batch_size 256 --device cuda
python train.py --model cnn --task building --epochs 30 --lr 0.001  --batch_size 256 --device cuda

# Floor classification
python train.py --model mlp --task floor    --epochs 50 --lr 0.001  --batch_size 256 --device cuda
python train.py --model cnn --task floor    --epochs 50 --lr 0.0005 --batch_size 128 --device cuda
```

Evaluate and print comparison table:

```bash
python evaluate.py
```

Smoke-test model architectures without training:

```bash
python models.py
```

## File structure

```
indoor-localization-dl/
├── data.py          # Dataset download and preprocessing
├── models.py        # MLP and CNN-1D definitions + smoke test
├── train.py         # Training script (CLI)
├── evaluate.py      # Evaluation and comparison table
├── requirements.txt
├── README.md
└── .gitignore
```
