# LingBot-VA for UR5

A single-arm UR5 workflow built on [LingBot-VA](https://github.com/robbyant/lingbot-va) and [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin). It collects demonstrations in simulation, converts them to LeRobot datasets, prepares training inputs, and connects policies to simulation or a guarded ROS 2 interface.

The simulation uses a UR5 with a WSG gripper. The public ALOHA checkpoint is used for its original embodiment; it is not a trained UR5 policy.

## Installation

Use Linux with an NVIDIA GPU, a working driver and Vulkan support, Git, and Conda. Setup creates two Python 3.10 environments: `robotwin` for simulation and `lingbot` for the model. Their PyTorch versions differ, so keep them separate. Run commands from the repository root; the shell scripts select the environment automatically.

```bash
bash scripts/00_system_check.sh
bash scripts/clone_sources.sh
bash scripts/01_create_envs.sh
bash scripts/02_install_robotwin.sh
bash scripts/download_assets.sh
bash scripts/03_test_robotwin.sh
bash scripts/05_install_lingbot.sh
bash scripts/06_download_checkpoints.sh
```

Source and model revisions are pinned in [configs/sources.lock.json](configs/sources.lock.json). Downloads resume when interrupted. The two model checkpoints occupy about 49 GB; simulation assets, extracted files, and environments require additional space. Setup downloads upstream code into `third_party/` and prepares patched copies in `outputs/runtime/`.

Tested with Ubuntu 24.04, Python 3.10.16, PyTorch 2.4.1/cu121 for RoboTwin and 2.9.0/cu126 for LingBot. Dependency lists are in [envs/](envs/).

## Simulation and data

Run one expert demonstration:

```bash
bash scripts/04_test_ur5_sim.sh
# Add --headless when no display is available.
```

Each run writes video and HDF5 files to a new directory under `outputs/ur5_sim_test/`. The console prints its location.

Collect one training episode, convert it, and extract the VAE and text latents:

```bash
bash scripts/08_generate_ur5_dataset.sh --task move_can_pot --episodes 1
bash scripts/09_validate_dataset.sh --dataset datasets/ur5_move_can_pot
bash scripts/extract_latents.sh --dataset datasets/ur5_move_can_pot
bash scripts/09_validate_dataset.sh --dataset datasets/ur5_move_can_pot --require-latents
```

Existing successful datasets are protected against overwrite. To collect another dataset, pass `--output-dir datasets/my_run` to the collection script and use that path with `--dataset` afterward. Simulation and camera settings live in [configs/sim/ur5.json](configs/sim/ur5.json).

## Training and evaluation

Check the dataset and training configuration without starting training:

```bash
bash scripts/11_train_ur5.sh --dataset datasets/ur5_move_can_pot --dry-run
```

On a machine with sufficient memory, training and evaluation use:

```bash
bash scripts/11_train_ur5.sh --dataset datasets/ur5_move_can_pot --steps 3000 --lr 1e-5
bash scripts/12_eval_ur5_sim.sh \
  --checkpoint checkpoints/lingbot_ur5/checkpoints/checkpoint_step_3000 \
  --episodes 10
```

The training guard requires at least 48 GiB GPU memory and 64 GiB host RAM; this is a conservative entry check, not a guarantee that every configuration fits. Inference settings are in [configs/inference/runtime.json](configs/inference/runtime.json). `bash scripts/07_test_pretrained.sh` checks and, when resources permit, evaluates the original ALOHA checkpoint. On a 16 GB GPU it prints `INFERENCE SKIPPED: EXPECTED VRAM LIMIT` without starting the model. A successful preflight exit alone does not mean inference ran.

## Real robot

Test action conversion and safety checks without connecting to hardware:

```bash
bash scripts/13_real_dry_run.sh
```

The result must report `physical_commands_sent: 0`. Real execution requires ROS 2 Jazzy, the UR driver, calibrated joint/TCP/world frames, a camera and gripper implementation, and a completed [deployment configuration](configs/real/deployment.json). `scripts/prepare_ros_env.sh` creates a separate system-Python ROS environment after ROS is installed. Physical execution is gated by configuration validation and operator approval; the supplied configuration is incomplete and disabled.

## Validation status

Expert simulation, one LeRobot episode, VAE/text latent extraction, dataset loading, CPU checkpoint round-trips, and mock action execution have been tested. Full model training, learned-policy evaluation, and physical UR5 execution have not been validated. One demonstration checks the data pipeline; it does not establish policy performance.

## Development

```text
src/lingbot_ur5/    Dataset, training, policy, and robot/camera adapters
configs/           Simulation, model, training, and deployment settings
scripts/           Setup and workflow entry points
envs/              Dependency lists for the separate environments
patches/           Changes applied to upstream runtime copies
tests/             CPU and mock regression tests
```

Run tests in the model environment:

```bash
conda run -n lingbot python -m pytest
```

For direct Python commands, set `PYTHONPATH="$PWD/src:$PWD"` and `LINGBOT_UR5_ROOT="$PWD"`. Shell entry points set these automatically. Keep the source checkout: scripts, configuration, and runtime assets are part of the workflow.

`local/`, downloaded dependencies, datasets, checkpoints, logs, and generated outputs are excluded from Git. To create a clean source archive without these files:

```bash
python3 scripts/export_source.py
```

The archive is written under `dist/`. Upstream code, weights, and assets retain their own licenses and are downloaded separately.
