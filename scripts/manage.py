"""Conda-only workflow entry point. Does not change shell startup files."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    'server': ('lingbot', 'model_server.py', []),
    'convert': ('lingbot', 'convert_dataset.py', []),
    'clone': ('base', 'clone_sources.py', []),
    'download-assets': ('robotwin', 'download.py', ['assets']),
    'sim': ('robotwin', 'run_sim.py', ['--episodes', '1']),
    'download-models': ('lingbot', 'download.py', ['models']),
    'pretrained': ('lingbot', 'run_evaluation.py', ['--official', '--checkpoint', str(ROOT/'checkpoints/pretrained/lingbot-va-posttrain-robotwin'), '--episodes', '1']),
    'validate': ('lingbot', 'validate_dataset.py', []),
    'latents': ('lingbot', 'extract_latents.py', []),
    'smoke': ('lingbot', 'module:lingbot_ur5.train', ['--smoke']),
    'train': ('lingbot', 'module:lingbot_ur5.train', []),
    'evaluate': ('lingbot', 'run_evaluation.py', []),
    'mock': ('lingbot', 'real_run.py', ['--dry-run']),
    'real': ('ur5-ros', 'real_run.py', []),
    'test': ('lingbot', 'module:pytest', []),
    'export-source': ('base', 'export_source.py', []),
}
SPECIAL = ['system-check', 'create-envs', 'install-sim', 'install-model', 'check-sim',
           'collect', 'python', 'export-versions', 'prepare-ros', 'check-ros']


def child_environment(root=ROOT):
    env = os.environ.copy()
    env.update(LINGBOT_UR5_ROOT=str(root), PROJECT_ROOT=str(root), PYTHONUNBUFFERED='1')
    # Scope source paths to child processes; never change the user's shell.
    env['PYTHONPATH'] = os.pathsep.join((str(root/'src'), str(root)))
    return env


class Runner:
    def __init__(self):
        if not (Path(sys.prefix)/'conda-meta').is_dir():
            raise RuntimeError('Start with: conda run -n base python scripts/manage.py <task>')
        self.conda = os.environ.get('CONDA_EXE') or shutil.which('conda')
        if not self.conda:
            raise RuntimeError('Conda executable not found; use conda run to start this command')
        self.env = child_environment(ROOT)
        self.env['CONDA_EXE'] = self.conda
        (ROOT/'logs').mkdir(exist_ok=True)
        (ROOT/'local/environment').mkdir(parents=True, exist_ok=True)

    def run(self, argv, capture=False, optional=False):
        argv = [str(x) for x in argv]
        print('+ ' + ' '.join(argv), flush=True)
        start = time.time()
        code = 127
        try:
            result = subprocess.run(argv, cwd=ROOT, env=self.env, text=True,
                                    stdout=subprocess.PIPE if capture else None,
                                    stderr=subprocess.STDOUT if capture else None)
            code = result.returncode
            if code and not optional:
                if capture:
                    print(result.stdout, flush=True)
                raise subprocess.CalledProcessError(code, argv)
            return result
        finally:
            with (ROOT/'logs/commands.jsonl').open('a') as log:
                log.write(json.dumps({'started_at': datetime.datetime.fromtimestamp(start, datetime.timezone.utc).isoformat(),
                                      'cwd': str(ROOT), 'argv': argv, 'exit_code': code,
                                      'elapsed_seconds': round(time.time()-start, 3)})+'\n')

    def conda_python(self, env, *args, **kwargs):
        return self.run([self.conda, 'run', '--no-capture-output', '-n', env, 'python', *args], **kwargs)

    def script(self, env, script, args=()):
        command = ['-m', script[7:]] if script.startswith('module:') else [str(ROOT/'scripts'/script)]
        return self.conda_python(env, *command, *args)

    def snapshot(self, name):
        history = self.run([self.conda, 'env', 'export', '-n', name, '--from-history'], capture=True).stdout
        history = '\n'.join(line for line in history.splitlines() if not line.startswith('prefix:'))+'\n'
        (ROOT/f'local/environment/{name}.yml').write_text(history)
        freeze = self.conda_python(name, '-m', 'pip', 'freeze', capture=True).stdout
        (ROOT/f'local/environment/{name}_pip_freeze.txt').write_text(freeze)

    def create(self, name):
        info = self.run([self.conda, 'env', 'list', '--json'], capture=True)
        paths = json.loads(info.stdout)['envs']
        if any(Path(p).name == name for p in paths):
            print(f'Keeping existing environment: {name}')
            return
        self.run([self.conda, 'env', 'create', '--file', ROOT/f'envs/{name}.yml'])


def collect(runner, args):
    parser = argparse.ArgumentParser(prog='manage.py collect')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--task', default='move_can_pot')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'datasets/ur5_move_can_pot')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--randomization', action='store_true')
    a = parser.parse_args(args)
    if a.episodes < 1:
        parser.error('episodes must be positive')
    output = a.output_dir.expanduser().resolve()
    raw = Path(str(output)+'_raw')
    sim = ['--headless', '--task', a.task, '--episodes', str(a.episodes), '--seed', str(a.seed), '--output-dir', str(raw)]
    if a.randomization:
        sim.append('--randomization')
    runner.script('robotwin', 'run_sim.py', sim)
    runner.script('lingbot', 'convert_dataset.py', ['--raw', str(raw), '--output', str(output)])
    runner.script('lingbot', 'validate_dataset.py', ['--dataset', str(output)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=sorted([*TASKS, *SPECIAL]))
    parser.add_argument('args', nargs=argparse.REMAINDER, help='Arguments forwarded to the selected task')
    a = parser.parse_args(argv)
    if a.task in SPECIAL and a.task not in ('collect', 'python') and a.args:
        parser.error(f'{a.task} takes no extra arguments')
    if a.task == 'real' and '--execute-real' not in a.args:
        parser.error('Use mock for dry-run; real requires explicit --execute-real and reviewed configuration')
    r = Runner()
    if a.task in TASKS:
        env, script, defaults = TASKS[a.task]
        r.script(env, script, defaults+a.args)
    elif a.task == 'python':
        if not a.args:
            parser.error('python requires Python arguments, for example -c or a script path')
        r.run([r.conda, 'run', '--no-capture-output', '-p', sys.prefix, 'python', *a.args])
    elif a.task == 'collect':
        collect(r, a.args)
    elif a.task == 'create-envs':
        for name in ('robotwin', 'lingbot'):
            r.create(name)
    elif a.task == 'install-sim':
        r.conda_python('robotwin', '-m', 'pip', 'install', 'torch==2.4.1', 'torchvision==0.19.1', '--index-url', 'https://download.pytorch.org/whl/cu121')
        r.conda_python('robotwin', '-m', 'pip', 'install', '-r', str(ROOT/'envs/robotwin-requirements.txt'))
        r.run([r.conda, 'install', '-y', '-n', 'robotwin', '--override-channels', '-c', 'conda-forge', '-c', 'nvidia/label/cuda-12.1.1', 'cuda-toolkit=12.1', 'gcc_linux-64=12', 'gxx_linux-64=12', 'cmake', 'ninja', 'ffmpeg', 'git-lfs'])
        r.script('robotwin', 'install_robotwin_extras.py')
        r.snapshot('robotwin')
    elif a.task == 'install-model':
        r.conda_python('lingbot', '-m', 'pip', 'install', 'torch==2.9.0', 'torchvision==0.24.0', 'torchaudio==2.9.0', '--index-url', 'https://download.pytorch.org/whl/cu126')
        r.conda_python('lingbot', '-m', 'pip', 'install', '-r', str(ROOT/'envs/lingbot-requirements.txt'))
        r.conda_python('lingbot', '-m', 'pip', 'install', 'lerobot==0.3.3', '--no-deps')
        r.run([r.conda, 'install', '-y', '-n', 'lingbot', '--override-channels', '-c', 'conda-forge', 'ffmpeg=7.1'])
        r.script('lingbot', 'prepare_lingbot_runtime.py')
        r.conda_python('lingbot', '-c', 'from lingbot_ur5.paths import model_path; model_path(); import torch, diffusers, transformers, lerobot; from wan_va.modules.model import WanTransformer3DModel; print("LINGBOT IMPORT OK", torch.__version__, torch.cuda.is_available())')
        r.snapshot('lingbot')
    elif a.task == 'check-sim':
        for script in ('audit_embodiment.py', 'prepare_robotwin_runtime.py', 'test_renderer.py'):
            r.script('robotwin', script)
    elif a.task == 'export-versions':
        for name in ('robotwin', 'lingbot'):
            r.snapshot(name)
        r.script('base', 'export_versions.py')
    elif a.task in ('prepare-ros', 'check-ros'):
        if a.task == 'prepare-ros':
            r.create('ur5-ros')
            r.conda_python('ur5-ros', '-m', 'pip', 'install', '-r', str(ROOT/'envs/ros-requirements.txt'))
        r.conda_python('ur5-ros', '-c', 'import rclpy, control_msgs, ur_dashboard_msgs, ur_msgs, moveit_msgs; print("ROS IMPORT OK; no node initialized")')
    elif a.task == 'system-check':
        outputs = []
        commands = [['uname', '-a'], ['lscpu'], ['nvidia-smi'], ['free', '-h'], ['df', '-h', str(ROOT)], [r.conda, 'info', '--envs'], ['git', '--version']]
        for command in commands:
            if shutil.which(command[0]):
                result = r.run(command, capture=True, optional=True)
                outputs.append('$ '+' '.join(command)+'\n'+result.stdout)
        report = '\n'.join(outputs)
        (ROOT/'logs/system_check.txt').write_text(report)
        print(report)


if __name__ == '__main__':
    main()
