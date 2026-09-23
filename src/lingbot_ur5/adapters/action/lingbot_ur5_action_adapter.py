"""30D LingBot convention, with explicit UR5 frame and quaternion semantics.

Evidence: wan_va/configs/va_robotwin_cfg.py and dataset/lerobot_latent_dataset.py.
UR5 profile: left EEF xyzw relative to episode origin; absolute joints, gripper 0..1.
Translation difference is expressed in world axes (not rotated into initial EEF).
RoboTwin's raw endpose is wxyz; convert at the data/simulator boundary.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

UR5_CHANNELS = tuple(range(7)) + tuple(range(14, 20)) + (28,)
ROBOTWIN_CHANNELS = tuple(range(7)) + (28,) + tuple(range(7, 14)) + (29,)
JOINT_NAMES = ('shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
               'wrist_1_joint','wrist_2_joint','wrist_3_joint')

def finite(value, shape=None):
    x=np.asarray(value,dtype=np.float64)
    if shape is not None and x.shape!=shape: raise ValueError(f'Expected {shape}, got {x.shape}')
    if not np.isfinite(x).all(): raise ValueError('NaN/Inf rejected')
    return x

def quaternion(q):
    q=finite(q,(4,))
    if abs(np.linalg.norm(q)-1)>1e-3: raise ValueError('Invalid quaternion norm')
    return q/np.linalg.norm(q)

def wxyz_to_xyzw(pose):
    p=finite(pose,(7,)).copy()
    return np.r_[p[:3],p[4:7],p[3]]

def xyzw_to_wxyz(pose):
    p=finite(pose,(7,)).copy()
    quaternion(p[3:])
    return np.r_[p[:3],p[6],p[3:6]]

def relative_pose(pose, origin):
    pose=finite(pose,(7,)); origin=finite(origin,(7,))
    r=Rotation.from_quat(quaternion(origin[3:])).inv()*Rotation.from_quat(quaternion(pose[3:]))
    return np.r_[pose[:3]-origin[:3],r.as_quat()]

def absolute_pose(pose, origin):
    pose=finite(pose,(7,)); origin=finite(origin,(7,))
    r=Rotation.from_quat(quaternion(origin[3:]))*Rotation.from_quat(quaternion(pose[3:]))
    return np.r_[pose[:3]+origin[:3],r.as_quat()]

@dataclass(frozen=True)
class UR5Command:
    joints: np.ndarray
    eef: np.ndarray
    gripper: float
    issued_at: float
    frame: str='world'

class UR5ActionAdapter:
    channels=UR5_CHANNELS
    def __init__(self, origin, norm_stat=None, frame='world'):
        self.origin=finite(origin,(7,)).copy();quaternion(self.origin[3:])
        self.frame=frame
        self.mask=np.zeros(30,dtype=bool);self.mask[list(self.channels)]=True
        self.norm_stat=norm_stat

    def pack(self, eef_xyzw, joints, gripper, *, relative=True):
        eef=finite(eef_xyzw,(7,));quaternion(eef[3:])
        joints=finite(joints,(6,))
        if not np.isfinite(gripper) or not 0<=gripper<=1: raise ValueError('Gripper outside [0,1]')
        a=np.zeros(30)
        a[:7]=relative_pose(eef,self.origin) if relative else eef
        a[14:20]=joints;a[28]=gripper
        return a

    def normalize(self, action):
        a=finite(action)
        if a.shape[-1]!=30: raise ValueError('Expected 30 action channels')
        if self.norm_stat is None: raise ValueError('Normalization statistics required')
        lo=finite(self.norm_stat['q01'],(30,));hi=finite(self.norm_stat['q99'],(30,))
        if (hi<lo).any(): raise ValueError('Invalid quantiles')
        out=np.clip((a-lo)/(hi-lo+1e-6)*2-1,-1.5,1.5)
        return np.where(self.mask,out,0)

    def denormalize(self, action):
        if self.norm_stat is None: raise ValueError('Normalization statistics required')
        a=finite(action,(30,));lo=finite(self.norm_stat['q01'],(30,));hi=finite(self.norm_stat['q99'],(30,))
        # Match official postprocess including its epsilon on inverse.
        return np.where(self.mask,(a+1)/2*(hi-lo+1e-6)+lo,0)

    def decode(self, action, *, issued_at, normalized=False, selected=False):
        if selected:
            raw=finite(action,(len(self.channels),));a=np.zeros(30);a[list(self.channels)]=raw
        else: a=finite(action,(30,)).copy()
        if normalized: a=self.denormalize(a)
        if np.any(np.abs(a[~self.mask])>1e-6): raise ValueError('Inactive arm/padded joint must be zero')
        # Diffusion outputs may have slightly nonunit quaternions; normalize here,
        # but reject degenerate values. SafetyFilter validates the resulting pose.
        norm=np.linalg.norm(a[3:7])
        if not 0.5<=norm<=1.5: raise ValueError('Degenerate predicted quaternion')
        a[3:7]/=norm
        pose=absolute_pose(a[:7],self.origin)
        if not 0<=a[28]<=1: raise ValueError('Gripper outside [0,1]')
        return UR5Command(a[14:20].copy(),pose,float(a[28]),float(issued_at),self.frame)

def legacy_robotwin_restore(selected_action, raw_origin):
    """Exact official checkpoint convention, including wxyz-as-xyzw legacy quirk.

    Output is raw RoboTwin ordering; NEVER use this for a physical robot.
    """
    a=finite(selected_action,(16,));origin=finite(raw_origin,(16,));out=a.copy()
    for offset in (0,8):
        q=a[offset+3:offset+7]; n=np.linalg.norm(q)
        if n<1e-6: raise ValueError('Degenerate quaternion')
        p=np.r_[a[offset:offset+3],q/n]
        out[offset:offset+7]=absolute_pose(p,origin[offset:offset+7])
    return out
