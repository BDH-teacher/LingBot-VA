# Conda environments

| Environment | Definition | Purpose |
| --- | --- | --- |
| robotwin | [robotwin.yml](robotwin.yml) | Python 3.10.16; SAPIEN and CuRobo; PyTorch 2.4.1/cu121 |
| lingbot | [lingbot.yml](lingbot.yml) | Python 3.10.16; LingBot and LeRobot; PyTorch 2.9.0/cu126 |
| ur5-ros (optional) | [ur5-ros.yml](ur5-ros.yml) | Python 3.12; RoboStack Jazzy transport/message dependencies |

From the repository root:

```bash
conda run --no-capture-output -n base python scripts/manage.py create-envs
conda run --no-capture-output -n base python scripts/manage.py install-sim
conda run --no-capture-output -n base python scripts/manage.py install-model
```

`create-envs` preserves existing environments. The installer tasks install their matching PyTorch wheels separately from the requirement lists. LeRobot 0.3.3 uses `--no-deps` to retain the model's PyTorch version; optional LeRobot device/UI features are outside this setup.

Run tasks through `scripts/manage.py` using `conda run`, or activate Conda and run `python scripts/manage.py`. The launcher sets paths only for child processes. It never writes `.bashrc`, `.profile`, or global Conda configuration. No system Python or system pip is used by workflow entry points.

The optional ROS environment uses the channels in its manifest, without mixing `/opt/ros` or system Python packages. `prepare-ros` creates the environment and checks imports; no ROS nodes or drivers are started. Installation and physical transport remain unverified on this machine.

```bash
conda run --no-capture-output -n base python scripts/manage.py export-versions
```

This records installed versions in ignored `local/environment/`. The YAML files in this directory are portable setup inputs, not machine exports. CUDA driver and Vulkan support are host prerequisites rather than Conda packages.
