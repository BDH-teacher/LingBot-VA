import time
import numpy as np
class SimCamera:
    def __init__(self,task,official=False):self.task=task;self.official=official
    def read(self):
        obs=self.task.get_obs();cam=obs['observation']
        return {'observation.images.cam_high':cam['head_camera']['rgb'],
                'observation.images.cam_left_wrist':cam['left_camera']['rgb'],
                'observation.images.cam_right_wrist':cam['right_camera']['rgb'] if self.official else np.zeros_like(cam['left_camera']['rgb']),
                'captured_at':time.monotonic()}
    def close(self):pass
