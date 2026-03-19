# Controllable Diffusion Policy (CDP)
### Controllable Imitation Learning via Classifier-Free Guidance for Data-Efficient Robot Manipulation
***

## Overview

This repository contains the full implementation of **Controllable Diffusion Policy (CDP)**, a score-based diffusion policy augmented with **Classifier-Free Guidance (CFG)** for fine-grained intra-task control in robotic manipulation.

The core algorithm is implemented from scratch inside the **robomimic** training framework, extending it with:
- A custom **score-based diffusion policy** using a causal transformer backbone
- **Karras EDM-style preconditioning** for stable training across all noise levels
- **CFG-guided DDIM sampling** that enables generalization to unseen execution parameters at inference time
- A custom **box-push environment** built on top of robosuite for controlled distance-conditioned data collection using Phantom Omni

***

## Codebase Structure

```
.
├── robosuite/                           # Simulation environment & data collection
│   ├── inject_goal.py                   # Injects push_distance labels into HDF5 demos
│   ├── scripts/
│   │   └── box_push_cdp_tele.py         # Teleoperation script for demo collection
│   └── robosuite/
│       ├── environments/
│       │   └── box_push_cdp.py          # Box-push environment with distance conditioning
│       └── src/
│           ├── utils/
│           │   ├── dataset_utils.py     # HDF5 dataset utilities for CDP
│           │   └── teleop_node_utils.py # Teleoperation ROS utilities
│           └── device/
│               └── phantom.py           # Phantom Omni haptic device driver
│
└── robomimic/                           # Training, evaluation & algorithm implementation
    └── robomimic/
        ├── algo/
        │   └── cdp.py                   # Core CDP algorithm: training loop, CFG inference, EMA
        ├── models/
        │   └── cdp_nets.py              # Nets used for CDP
        ├── config/
        │   └── cdp_config.py            # Hyperparameter config for CDP
        ├── exps/templates/
        │   ├── cdp.json                 # Full experiment JSON config
        │   └── cdp_baseline             # Baseline experiment JSON config
        └── scripts/
            └── cdp_run_trained_agent.py # CFG rollout & evaluation script
```

***

## Installation

```bash
cd robomimic
conda env create -f environment.yml
conda activate ece176_final
cd ../robosuite
pip install -e .
cd ../robomimic
pip install -e .
```

***

## Evaluating a Trained Model

```bash
python3 robomimic/scripts/cdp_run_trained_agent.py \
    --ckpt ./cdp_trained_models/cdp_box_push/models/model_epoch_500.pth \
    --target_dist 0.10 \
    --cond_lambda 2.0 \
    --n_rollouts 20 \
    --video_path ./videos/rollout_10_20.mp4
```

| Argument | Description |
|---|---|
| `--ckpt` | Path to trained model checkpoint (`.pth`) |
| `--target_dist` | Target push distance in meters (e.g., `0.10`, `0.15`, `0.25`) |
| `--cond_lambda` | CFG guidance strength λ — use `1.0` for naive conditional, `2.0` for CDP (full CFG) |
| `--n_rollouts` | Number of evaluation episodes |
| `--video_path` | Output path for rollout video |

