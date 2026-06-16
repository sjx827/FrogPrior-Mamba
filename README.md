# FrogPrior-Mamba

FrogPrior-Mamba is a time-frequency speech enhancement model for tree frog calls under airport-noise interference. The model follows a magnitude/phase enhancement pipeline with Mamba-based time-frequency modeling and adds frog-call frequency priors in the training objective.

This repository is adapted from the SEMamba enhancement framework. The public code is organized for frog-call enhancement experiments and does not include datasets, checkpoints, logs, or generated enhanced audio.

## Main Features

- Time-frequency Mamba generator for magnitude and phase enhancement.
- SI-SDR based discriminator objective for bioacoustic enhancement.
- Frog-call frequency weighted magnitude and complex losses.
- Scripts for dataset JSON preparation, training, inference, and frequency-band analysis.

## Environment

The experiments were developed with:

- Python 3.9
- PyTorch 2.2.2
- CUDA 11.8
- mamba-ssm

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install `mamba-ssm` according to your CUDA and PyTorch versions.

## Data Format

Training uses paired JSON files:

```text
data/train_clean.json
data/train_noisy.json
data/valid_clean.json
data/valid_noisy.json
```

Each JSON maps a wav filename to an absolute or relative wav path.

If your data is organized as:

```text
data/processed/train/clean/*.wav
data/processed/train/noisy/*.wav
data/processed/valid/clean/*.wav
data/processed/valid/noisy/*.wav
data/processed/test/clean/*.wav
data/processed/test/noisy/*.wav
```

prepare JSON files with:

```bash
bash make_dataset.sh data/processed
```

## Training

```bash
python train.py \
  --exp_name FrogPriorMamba \
  --config recipes/FrogPriorMamba/FrogPriorMamba.yaml
```

Checkpoints and TensorBoard logs are written under `exp/<exp_name>/`.

## Inference

```bash
python inference.py \
  --config recipes/FrogPriorMamba/FrogPriorMamba.yaml \
  --checkpoint_file exp/FrogPriorMamba/g_00000000.pth \
  --input_folder data/processed/valid/noisy \
  --output_folder results/FrogPriorMamba_valid
```

## Synthetic Data Generation

To synthesize noisy frog-call mixtures:

```bash
python generate_data.py \
  --clean_dir /path/to/frog_calls \
  --noise_dir /path/to/airport_noise \
  --output_dir data/processed
```

## Frequency-Band Analysis

```bash
python analyze_frog_bands.py \
  --input /path/to/frog_calls \
  --output_csv frog_band_energy.csv
```

## Repository Notes

Large or generated files are intentionally ignored by `.gitignore`, including:

- datasets and wav files
- checkpoints
- experiment logs
- inference outputs
- TensorBoard event files

Before publishing results, keep model checkpoints and datasets outside the GitHub repository.
