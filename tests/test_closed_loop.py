import dataclasses,json,pathlib,time
import numpy as np
import pytest
from lingbot_ur5.adapters.action.real_executor import RealUR5Executor
from lingbot_ur5.adapters.action.safety import RobotState
from lingbot_ur5.adapters.action.lingbot_ur5_action_adapter import UR5ActionAdapter
from lingbot_ur5.adapters.camera.real_camera import MockCamera
from lingbot_ur5.policy import rollout
from scripts.real_run import MockPolicy

def test_mock_closed_loop_no_transport(tmp_path):
    root=pathlib.Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/real/safety_mock.json').read_text())
    q=np.asarray(config['home_joints']);eef=np.array([0.3,0,0.5,0,0,0,1.])
    state=RobotState(q,eef,np.zeros(6),time.monotonic());policy=MockPolicy(q)
    executor=RealUR5Executor(config,state,tmp_path/'actions.jsonl')
    count=rollout(policy,MockCamera(),executor,UR5ActionAdapter(eef),'mock instruction',32)
    assert count==32 and policy.feedback==4 and policy.requests>=4
    rows=[json.loads(l) for l in (tmp_path/'actions.jsonl').read_text().splitlines()]
    assert len(rows)==32 and all(r['mode']=='DRY_RUN' for r in rows)
    assert executor.transport is None

def test_real_mode_cannot_be_enabled_with_mock_config(tmp_path):
    root=pathlib.Path(__file__).resolve().parents[1];config=json.loads((root/'configs/real/safety_mock.json').read_text())
    with pytest.raises(ValueError,match='Calibrated'):
        RealUR5Executor(config,None,tmp_path/'log',execute_real=True)
    with pytest.raises(ValueError,match='Dry run'):
        RealUR5Executor(config,None,tmp_path/'log',transport=object())
