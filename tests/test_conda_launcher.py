"""Verify environment isolation and workflow routing without installing packages."""
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import manage


def test_child_paths_do_not_modify_parent_environment(monkeypatch, tmp_path):
    monkeypatch.setenv('PYTHONPATH', '/unrelated/python')
    before = os.environ.copy()
    child = manage.child_environment(tmp_path)
    assert child['PYTHONPATH'] == os.pathsep.join((str(tmp_path/'src'), str(tmp_path)))
    assert child['LINGBOT_UR5_ROOT'] == str(tmp_path)
    assert dict(os.environ) == before


def test_system_python_cannot_launch_workflow(monkeypatch, tmp_path):
    monkeypatch.setattr(manage.sys, 'prefix', str(tmp_path))
    with pytest.raises(RuntimeError, match='conda run'):
        manage.Runner()


def test_conda_subprocess_is_argv_based_and_failure_is_logged(monkeypatch, tmp_path):
    (tmp_path/'conda-meta').mkdir()
    monkeypatch.setattr(manage, 'ROOT', tmp_path)
    monkeypatch.setattr(manage.sys, 'prefix', str(tmp_path))
    monkeypatch.setenv('CONDA_EXE', '/conda/bin/conda')
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=7, stdout='failed')
    monkeypatch.setattr(manage.subprocess, 'run', run)
    runner = manage.Runner()
    with pytest.raises(manage.subprocess.CalledProcessError):
        runner.conda_python('lingbot', '-c', 'print("literal;$HOME")')
    argv, options = calls[0]
    assert argv[:7] == ['/conda/bin/conda', 'run', '--no-capture-output', '-n', 'lingbot', 'python', '-c']
    assert not options.get('shell')
    assert '"exit_code": 7' in (tmp_path/'logs/commands.jsonl').read_text()


def test_collection_routes_sim_and_conversion_in_order(tmp_path):
    calls = []
    runner = SimpleNamespace(script=lambda *args: calls.append(args))
    manage.collect(runner, ['--output-dir', str(tmp_path/'space name'), '--episodes', '1'])
    assert [(c[0], c[1]) for c in calls] == [
        ('robotwin', 'run_sim.py'), ('lingbot', 'convert_dataset.py'), ('lingbot', 'validate_dataset.py')]
    assert calls[0][2][-1] == str(tmp_path/'space name_raw')
    assert calls[2][2][-1] == str(tmp_path/'space name')


def test_real_requires_explicit_flag_before_starting_runner(monkeypatch):
    monkeypatch.setattr(manage, 'Runner', lambda: pytest.fail('must not start a process'))
    with pytest.raises(SystemExit):
        manage.main(['real'])


def test_mock_keeps_dry_run_and_uses_model_environment(monkeypatch):
    calls = []
    monkeypatch.setattr(manage, 'Runner', lambda: SimpleNamespace(script=lambda *args: calls.append(args)))
    manage.main(['mock', '--max-actions', '2'])
    assert calls == [('lingbot', 'real_run.py', ['--dry-run', '--max-actions', '2'])]
