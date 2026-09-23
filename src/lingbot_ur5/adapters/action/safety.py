"""Fail-closed command checks. This is not a certified robot safety system."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from .lingbot_ur5_action_adapter import finite,quaternion,UR5Command

@dataclass
class RobotState:
    joints: np.ndarray
    eef: np.ndarray
    velocity: np.ndarray
    observed_at: float
    emergency_stop: bool=False
    frame: str='world'

class SafetyFilter:
    def __init__(self, config):
        self.config=config; self.latched_stop=False;self.last_issue=None
        self.lo=finite(config['joint_lower'],(6,));self.hi=finite(config['joint_upper'],(6,))
        self.bounds=finite(config['workspace'],(2,3))
        if (self.hi<=self.lo).any() or (self.bounds[1]<=self.bounds[0]).any(): raise ValueError('Invalid safety bounds')
        for key in ('max_velocity','max_acceleration','max_eef_step','max_eef_rotation','watchdog_seconds','control_dt'):
            if not np.isfinite(config[key]) or config[key]<=0: raise ValueError('Invalid limit '+key)

    def stop(self): self.latched_stop=True

    def validate(self, command:UR5Command,state:RobotState,now:float):
        c=self.config;now=float(now)
        if self.latched_stop or state.emergency_stop:
            self.stop();raise ValueError('Emergency stop latched')
        for timestamp in (command.issued_at,state.observed_at,now):
            if not np.isfinite(timestamp): raise ValueError('Invalid timestamp')
        if not 0<=now-command.issued_at<=c['watchdog_seconds']: raise ValueError('Stale/future command')
        if not 0<=now-state.observed_at<=c['watchdog_seconds']: raise ValueError('Stale/future state')
        if self.last_issue is not None and command.issued_at<=self.last_issue: raise ValueError('Non-monotonic command')
        if command.frame!=state.frame or command.frame!=c['frame']: raise ValueError('Coordinate frame mismatch')
        q=finite(command.joints,(6,));current=finite(state.joints,(6,));v0=finite(state.velocity,(6,))
        p=finite(command.eef,(7,));curp=finite(state.eef,(7,))
        quaternion(p[3:]);quaternion(curp[3:])
        if (q<self.lo).any() or (q>self.hi).any(): raise ValueError('Joint limits')
        if (current<self.lo).any() or (current>self.hi).any(): raise ValueError('Measured joint limits')
        if (p[:3]<self.bounds[0]).any() or (p[:3]>self.bounds[1]).any(): raise ValueError('Workspace bounds')
        if np.linalg.norm(p[:3]-curp[:3])>c['max_eef_step']: raise ValueError('EEF step limit')
        rotation=Rotation.from_quat(curp[3:]).inv()*Rotation.from_quat(p[3:])
        if rotation.magnitude()>c['max_eef_rotation']: raise ValueError('EEF rotation limit')
        v=(q-current)/c['control_dt']
        if (np.abs(v)>c['max_velocity']).any(): raise ValueError('Velocity limits')
        if (np.abs(v-v0)/c['control_dt']>c['max_acceleration']).any(): raise ValueError('Acceleration limits')
        if not np.isfinite(command.gripper) or not 0<=command.gripper<=1: raise ValueError('Gripper range')
        self.last_issue=command.issued_at
        return command
