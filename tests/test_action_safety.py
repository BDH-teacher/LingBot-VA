import dataclasses,json,pathlib
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from lingbot_ur5.adapters.action.lingbot_ur5_action_adapter import *
from lingbot_ur5.adapters.action.safety import *

ORIGIN=np.r_[[0.3,0,0.5],Rotation.from_euler('xyz',[0.4,-0.2,0.9]).as_quat()]

def test_world_translation_and_orientation_roundtrip():
    target=np.r_[[0.32,-0.01,0.52],Rotation.from_euler('xyz',[-0.2,0.1,1]).as_quat()]
    rel=relative_pose(target,ORIGIN)
    np.testing.assert_allclose(rel[:3],target[:3]-ORIGIN[:3])
    np.testing.assert_allclose(absolute_pose(rel,ORIGIN),target,atol=1e-12)

def test_raw_quaternion_boundary():
    raw=[1,2,3,1,0,0,0]
    np.testing.assert_array_equal(wxyz_to_xyzw(raw),[1,2,3,0,0,0,1])
    np.testing.assert_allclose(xyzw_to_wxyz(wxyz_to_xyzw(raw)),raw)

def test_layout_padding_and_selected_roundtrip():
    adapter=UR5ActionAdapter(ORIGIN)
    q=np.arange(6)/10
    a=adapter.pack(ORIGIN,q,0.8)
    np.testing.assert_array_equal(a[14:20],q)
    np.testing.assert_array_equal(a[[*range(7,14),*range(20,28),29]],0)
    cmd=adapter.decode(a[list(UR5_CHANNELS)],selected=True,issued_at=1)
    np.testing.assert_allclose(cmd.eef,ORIGIN,atol=1e-12)
    assert cmd.gripper==0.8

def test_normalization_masks_unused_dimensions():
    adapter=UR5ActionAdapter(ORIGIN,{'q01':[-1]*30,'q99':[1]*30})
    a=adapter.pack(ORIGIN,np.zeros(6),0.5)
    b=adapter.normalize(a)
    assert not b[~adapter.mask].any()
    np.testing.assert_allclose(adapter.denormalize(b),a,atol=2e-6)

@pytest.fixture
def safety_case():
    p=pathlib.Path(__file__).resolve().parents[1]/'configs/real/safety_mock.json'
    c=json.loads(p.read_text());q=np.array(c['home_joints'])
    return SafetyFilter(c),UR5Command(q,ORIGIN,0.5,1),RobotState(q,ORIGIN,np.zeros(6),1)

def test_valid_and_replay(safety_case):
    f,c,s=safety_case; f.validate(c,s,1.01)
    with pytest.raises(ValueError,match='monotonic'): f.validate(c,s,1.02)

@pytest.mark.parametrize('change',[
    {'joints':np.ones(6)*np.nan},{'joints':np.ones(6)*np.inf},
    {'joints':np.ones(6)*10},{'eef':np.r_[[2,0,0.5],[0,0,0,1]]},
    {'eef':np.r_[ORIGIN[:3],[0,0,0,0]]},{'issued_at':0},
    {'issued_at':2},{'gripper':1.1},{'frame':'base'}])
def test_invalid_rejected(safety_case,change):
    f,c,s=safety_case
    with pytest.raises(ValueError):f.validate(dataclasses.replace(c,**change),s,1.01)

def test_velocity_acceleration_step_watchdog_and_estop(safety_case):
    f,c,s=safety_case
    for delta in (0.1,0.01):
        with pytest.raises(ValueError):f.validate(dataclasses.replace(c,joints=c.joints+delta),s,1.01)
    with pytest.raises(ValueError,match='EEF step'):f.validate(dataclasses.replace(c,eef=np.r_[c.eef[:3]+0.02,c.eef[3:]]),s,1.01)
    with pytest.raises(ValueError,match='Stale'):f.validate(c,dataclasses.replace(s,observed_at=0),1.01)
    with pytest.raises(ValueError,match='Emergency'):f.validate(c,dataclasses.replace(s,emergency_stop=True),1.01)
    with pytest.raises(ValueError,match='Emergency'):f.validate(c,s,1.01)
