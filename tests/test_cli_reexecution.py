"""Rerun/path and preflight regressions; no simulator or large model is started."""
import json,pathlib,sys
from types import SimpleNamespace
import pytest
from scripts import run_sim,run_evaluation
from lingbot_ur5 import sim

def test_sim_cli_repeated_default_preserves_old_results(tmp_path,monkeypatch):
    parent=tmp_path/'outputs/ur5_sim_test';parent.mkdir(parents=True)
    original=parent/'episode_000000.hdf5';original.write_bytes(b'original')
    calls=[]
    monkeypatch.setattr(run_sim,'ROOT',tmp_path)
    monkeypatch.setattr(run_sim,'collect',lambda *args:calls.append(args))
    monkeypatch.setattr(sys,'argv',['run_sim.py'])
    run_sim.main();run_sim.main()
    destinations=[args[1] for args in calls]
    assert len(set(destinations))==2
    assert all(p.parent==parent and p.is_dir() for p in destinations)
    assert original.read_bytes()==b'original'

def test_explicit_sim_output_is_anchored_to_calling_directory(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);calls=[]
    monkeypatch.setattr(run_sim,'collect',lambda *args:calls.append(args))
    monkeypatch.setattr(sys,'argv',['run_sim.py','--output-dir','my_run'])
    run_sim.main()
    assert calls[0][1]==tmp_path/'my_run'

def test_collection_survives_robotwin_changing_cwd(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    runtime=tmp_path/'runtime';runtime.mkdir()
    task=SimpleNamespace(plan_success=True,play_once=lambda:{},check_success=lambda:True,close_env=lambda:None)
    def setup(*args):
        monkeypatch.chdir(runtime)
        return task,{}
    class FakeRecorder:
        def __init__(self,path):
            # Test-only placeholders, not demonstration data.
            path.write_bytes(b'test')
            path.with_suffix('.mp4').write_bytes(b'test')
        def close(self,*args):pass
    monkeypatch.setattr(sim,'make_task',setup);monkeypatch.setattr(sim,'EpisodeRecorder',FakeRecorder)
    result=sim.collect(1,'result')
    assert result==tmp_path/'result'
    assert (result/'episode_000000.hdf5').exists()
    assert not (runtime/'result').exists()

@pytest.mark.parametrize('status,ready',[('SKIPPED_EXPECTED_VRAM_LIMIT',False),('PREFLIGHT_PASSED',True)])
def test_preflight_result_controls_server_start(tmp_path,monkeypatch,status,ready):
    monkeypatch.setattr(run_evaluation,'ROOT',tmp_path)
    def preflight(args,**kwargs):
        pathlib.Path(args[args.index('--preflight-report')+1]).write_text(json.dumps({'status':status}))
        return SimpleNamespace(returncode=0,stdout='preflight output\n')
    monkeypatch.setattr(run_evaluation.subprocess,'run',preflight)
    assert run_evaluation.run_preflight(['mock-server']) is ready

def test_failed_preflight_cannot_reuse_stale_report(tmp_path,monkeypatch):
    old=tmp_path/'outputs/inference';old.mkdir(parents=True)
    (old/'preflight.json').write_text(json.dumps({'status':'SKIPPED_EXPECTED_VRAM_LIMIT'}))
    monkeypatch.setattr(run_evaluation,'ROOT',tmp_path)
    monkeypatch.setattr(run_evaluation.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=1,stdout='missing checkpoint\n'))
    with pytest.raises(SystemExit,match='PREFLIGHT FAILED'):run_evaluation.run_preflight(['mock-server'])

def test_vram_skip_never_spawns_server_or_simulator(monkeypatch):
    monkeypatch.setattr(run_evaluation,'run_preflight',lambda args:False)
    monkeypatch.setattr(sys,'argv',['run_evaluation.py'])
    def forbidden(*args,**kwargs):raise AssertionError('Must not launch after skip')
    monkeypatch.setattr(run_evaluation.subprocess,'Popen',forbidden)
    run_evaluation.main()
