# DenseNet-121 Image Classification

This project provides a basic framework for image classification using the DenseNet-121 model.

## Directory Structure

```
DenseNet/
├── data/
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your training, validation, and test images in the corresponding directories under `data/`. Each subdirectory in `train`, `validation`, and `test` should represent a class.

### 3. Train the Model

```bash
python train.py --epochs 50 -lr 0.001 --pretrained --fine-tune --save-model
```

- `--epochs`: Number of epochs to train for.
- `-lr`: Learning rate.
- `--pretrained`: Use a pretrained model.
- `--fine-tune`: Fine-tune all layers of the model.
- `--save-model`: Save the trained model.
- '--out-dir': Directory to save outputs (default: `../outputs`).

### 4. Evaluate the Model

```bash
python evaluate.py --model ../outputs/model.pth --data ../data/test
```

- `--model`: Path to the trained model.
- `--data`: Path to the test data.

### 5. Make Predictions

```bash
python predict.py --model ../outputs/model.pth --input path/to/your/image.jpg
```

- `--model`: Path to the trained model.
- `--input`: Path to the input image.
