# SPARD: Single-step Inference with Adaptive Sampling in Residual Diffusion for Human Motion Prediction

*AAAI 2026*

This repository contains the official implementation of SPARD, a novel approach for human motion prediction using adaptive sampling in residual diffusion that enables single-step inference.

## 📁 Project Structure

```
.
├── models/                    # Model architectures
│   ├── model.py              # SPARD main model
│   ├── noise_predictor.py    # Noise predictor module
│   └── train.py              # Training classes
├── data_loader/              # Data handling
│   ├── dataset_amass.py      # AMASS dataset loader
│   └── dataset_h36m.py       # Human3.6M dataset loader
├── utils/                    # Utilities
│   ├── log.py               # Logging utilities
│   └── scripts.py           # Helper scripts
├── config.py                # Configuration management
├── h36m_main.py             # Human3.6M training script
├── amass_main.py            # AMASS training script
├── results/                     # Training logs and checkpoints (generated)
│   ├── h36m/
│   │   ├── train/
│   │   ├── distill/
│   │   └── np/
│   └── amass/
│       ├── train/
│       ├── distill/
│       └── np/

```

### Dataset Preparation

#### Human3.6M
Follow [GSPS](https://github.com/wei-mao-2019/gsps) for Human3.6M dataset preparation. Download the data and place it in the `data` folder.

#### AMASS  
Follow [BeLFusion](https://github.com/BarqueroGerman/BeLFusion) for AMASS dataset preparation. Due to distribution policies, we cannot provide the data directly.

## 📈 Three-Stage Training Strategy

1. **Base Training**: Full diffusion model with iterative DDIM sampling
2. **Distillation**: Knowledge distillation for one-step sampling
3. **Noise Prediction**: Specialized module for fast residual prediction

## 📦 Pre-trained Models

We provide pre-trained models in the [Google Drive](https://drive.google.com/drive/folders/1X27WXq_g-7eYJqITGdxEJ1idPtYDrRuw?usp=sharing):

- `results/h36m/`: Human3.6M trained models
- `results/amass/`: AMASS trained models

Each contains:
- `train/ckpt.pt`: Base diffusion model
- `distill/distill_ckpt.pt`: Distilled one-step model  
- `np/np_ckpt.pt`: Noise predictor for fast sampling

## 🏃‍♂️ Training Pipeline

### Human3.6M Dataset

#### Stage 1: Base Model Training
```bash
python h36m_main.py --mode train --cfg h36m --device cuda:0 --batch_size 64 --num_epoch 1000 --lr 1.e-4 
```

#### Stage 2: Distillation Training
```bash
python h36m_main.py --mode distill --cfg h36m --device cuda:0 --batch_size 64 --num_epoch 1000 --lr 1.e-4 --ckpt "results/h36m/train/ckpt.pt" 
```

#### Stage 3: Noise Predictor Training
```bash
python h36m_main.py --mode np --cfg h36m --device cuda:0 --batch_size 64 --num_epoch 100 --lr 1.e-4 --distill_ckpt "results/h36m/distill/distill_ckpt.pt"
```

### AMASS Dataset

#### Stage 1: Base Model Training
```bash
python amass_main.py --mode train --cfg amass --device cuda:0 --batch_size 64 --num_epoch 1000 --lr 1.e-4
```

#### Stage 2: Distillation Training
```bash
python amass_main.py --mode distill --cfg amass --device cuda:0 --batch_size 64 --num_epoch 1000 --lr 1.e-4 --ckpt "results/amass/train/ckpt.pt"
```

#### Stage 3: Noise Predictor Training
```bash
python amass_main.py --mode np --cfg amass --device cuda:0 --batch_size 64 --num_epoch 100 --lr 1.e-4 --sample np --distill_ckpt "results/amass/distill/distill_ckpt.pt"
```

## 📊 Evaluation

### Human3.6M Evaluation

#### Base Model (DDIM Sampling)
```bash
python h36m_main.py --mode eval --cfg h36m --device cuda:0 --batch_size 64 --sample ddim --ckpt "results/h36m/train/ckpt.pt"
```

#### Distilled Model (One-step Sampling)
```bash
python h36m_main.py --mode distill_eval --cfg h36m --device cuda:0 --batch_size 64 --sample onestep --distill_ckpt "results/h36m/distill/distill_ckpt.pt"
```

#### Noise Predictor (Fast Sampling)
```bash
python h36m_main.py --mode np_eval --cfg h36m --device cuda:0 --batch_size 64 --sample np --distill_ckpt "results/h36m/distill/distill_ckpt.pt" --np_ckpt "results/h36m/np/np_ckpt.pt"
```

### AMASS Evaluation

#### Base Model (DDIM Sampling)
```bash
python amass_main.py --mode eval --cfg amass --device cuda:0 --batch_size 64 --sample ddim --ckpt "results/amass/train/ckpt.pt"
```

#### Distilled Model (One-step Sampling)
```bash
python amass_main.py --mode distill_eval --cfg amass --device cuda:0 --batch_size 64 --sample onestep --distill_ckpt "results/amass/distill/distill_ckpt.pt"
```

#### Noise Predictor (Fast Sampling)
```bash
python amass_main.py --mode np_eval --cfg amass --device cuda:0 --batch_size 64 --sample np --distill_ckpt "results/amass/distill/distill_ckpt.pt" --np_ckpt "results/amass/np/np_ckpt.pt"
```

## ⚙️ Key Parameters

- `--mode`: Operation mode (`train`, `distill`, `np`, `eval`, `distill_eval`, `np_eval`)
- `--device`: Training device (`cuda:0`, `cpu`)
- `--batch_size`: Batch size (default: 64)
- `--num_epoch`: Training epochs (default: 1000)
- `--lr`: Learning rate (default: 1e-4)
- `--mode_test`: Classifier-Free Guidance Coefficient (default: 0.3)
- `--div_k`: Relaxation Parameter h (default: 10)
- `--sample`: Sampling strategy (`ddim`, `onestep`, `np`)
- `--model_ckpt/--ckpt`: Model checkpoint path
- `--distill_ckpt`: Distillation checkpoint path  
- `--np_ckpt`: Noise predictor checkpoint path

## 🤝 Acknowledgement

Our code builds upon previous work :[HumanMAC](https://github.com/LinghaoChan/HumanMAC) [ResShift](https://github.com/zsyOAOA/ResShift) [SinSR](https://github.com/wyf0912/SinSR). Thanks for the help from the authors.
