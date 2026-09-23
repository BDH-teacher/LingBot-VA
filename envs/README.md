# Python environments

| Environment | Python | PyTorch | Purpose |
| --- | --- | --- | --- |
| robotwin | 3.10.16 | 2.4.1 / cu121 | SAPIEN, CuRobo, demonstration collection |
| lingbot | 3.10.16 | 2.9.0 / cu126 | Model, LeRobot data, latent extraction |
| .venv-ros | System Python 3.12 | None required | ROS 2 Jazzy transport |

Use the setup scripts in the root README. They install the matching PyTorch wheels separately from the requirements here. LeRobot 0.3.3 is installed with `--no-deps` because the upstream model requires a different PyTorch version; optional LeRobot device/UI features are not part of this setup.

Shell entry points use `conda run`, so manual activation is optional. Set `CONDA_EXE` if Conda is not on PATH or installed in the usual location.

`bash scripts/export_versions.sh` saves local Conda history, pip package lists, and source revision checks in `local/environment/`. These describe the installed machine and are excluded from Git; installation scripts and requirements remain the reproducible setup inputs.
