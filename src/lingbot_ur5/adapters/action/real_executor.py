"""Dry-run by default. Real transport is injected only after deployment validation."""
import dataclasses,json,pathlib,time
import numpy as np
from .safety import RobotState,SafetyFilter

class RealUR5Executor:
    def __init__(self,config,initial_state,log_path,transport=None,execute_real=False):
        if execute_real and (not config.get('hardware_validated') or transport is None):
            raise ValueError('Calibrated hardware config and explicit transport required')
        if transport is not None and not execute_real:raise ValueError('Dry run must not receive hardware transport')
        self.filter=SafetyFilter(config);self.config=config;self.state=initial_state
        self.transport=transport;self.execute_real=execute_real
        self.log_path=pathlib.Path(log_path);self.log_path.parent.mkdir(parents=True,exist_ok=True)
    def execute(self,command):
        now=time.monotonic()
        if self.execute_real:
            self.state=self.transport.read_state()
            # Transport must obtain FK in the configured frame for proposed joints.
            fk=self.transport.forward_kinematics(command.joints)
            from scipy.spatial.transform import Rotation
            if np.linalg.norm(fk[:3]-command.eef[:3])>self.config['fk_position_tolerance']:raise ValueError('Joint/EEF FK disagreement')
            if (Rotation.from_quat(fk[3:]).inv()*Rotation.from_quat(command.eef[3:])).magnitude()>self.config['fk_rotation_tolerance']:raise ValueError('Joint/EEF FK rotation disagreement')
            command=dataclasses.replace(command,eef=fk)
            now=time.monotonic()
        else:self.state.observed_at=now
        checked=self.filter.validate(command,self.state,now)
        if self.execute_real:self.transport.send(checked)
        record=dataclasses.asdict(checked);record['mode']='REAL' if self.execute_real else 'DRY_RUN'
        with self.log_path.open('a') as f:f.write(json.dumps(record,default=lambda v:v.tolist())+'\n')
        if not self.execute_real:
            velocity=(checked.joints-self.state.joints)/self.config['control_dt']
            self.state=RobotState(checked.joints.copy(),checked.eef.copy(),velocity,now,False,checked.frame)
    def watchdog(self):
        if time.monotonic()-self.state.observed_at>self.config['watchdog_seconds']:
            self.stop();raise TimeoutError('State watchdog expired')
    def stop(self):
        self.filter.stop()
        if self.execute_real:self.transport.cancel()
    def success(self):return False
