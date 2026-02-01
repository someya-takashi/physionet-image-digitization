# PhysioNet ECG Image Digitization

A deep learning pipeline for digitizing ECG (electrocardiogram) images back into time-series signals using segmentation models.

## Environment Setup

### Docker

This project uses Docker for environment management. Pull the pre-configured PyTorch image and start a container with the following command:

```bash
docker pull vastai/pytorch:cuda-13.0.2-auto
```

### Installing Dependencies

Once inside the container, install the required Python libraries:

```bash
pip install -r requirements.txt
```

## Training

This project provides two types of segmentation models for ECG image digitization.

### Whole Model

The Whole Model processes the entire ECG image at once and segments all four lead series simultaneously. It uses a CoordConv-based decoder for position-aware decoding.

**Configuration:** `configs/whole_model.yaml`

```bash
# Train with default settings
python train.py --config configs/whole_model.yaml

# Train with custom experiment name
python train.py --config configs/whole_model.yaml wandb.exp_name=whole_exp1

# Train with different validation fold
python train.py --config configs/whole_model.yaml cv.val_fold=1 wandb.exp_name=whole_fold1
```

### Series Model

The Series Model crops and processes each of the four lead series individually, then applies Cross-Series Feature Fusion to learn relationships between leads.

**Configuration:** `configs/series_model.yaml`

```bash
# Train with default settings
python train.py --config configs/series_model.yaml

# Train with custom batch size and experiment name
python train.py --config configs/series_model.yaml training.batch_size=8 wandb.exp_name=series_exp1
```

### Configuration Options

You can override any configuration parameter from the command line using dot notation:

```bash
python train.py --config configs/whole_model.yaml \
    training.batch_size=4 \
    optimizer.lr=5e-4 \
    cv.val_fold=2 \
    wandb.exp_name=custom_experiment
```

Model checkpoints and training logs are saved to the directory specified by `output_dir` in the configuration (default: `outputs/{wandb.exp_name}`).

## Prediction

Use `predict.py` to run inference with a trained model and evaluate the Signal-to-Noise Ratio (SNR) on the validation set.

### Basic Usage

```bash
python predict.py --weight <path_to_checkpoint> --fold <fold_number>
```

### Examples

```bash
# Run prediction using the best checkpoint from fold 0
python predict.py --weight outputs/whole_exp1/best.pth --fold 0

# Run prediction with a specific batch size
python predict.py --weight outputs/series_exp1/best.pth --fold 1 --batch_size 8

# Run prediction on CPU
python predict.py --weight outputs/whole_exp1/best.pth --fold 0 --device cpu
```

### Output

The prediction script outputs:
- **Loss:** Average pixel-wise loss on the validation set
- **SNR:** Signal-to-Noise Ratio in decibels (dB), measuring the quality of signal reconstruction
- **Samples evaluated:** Number of samples used for evaluation

Example output:
```
============================================================
Results:
  Loss: 0.0083
  SNR: 10.38 dB
  Samples evaluated: 1764
============================================================
```
