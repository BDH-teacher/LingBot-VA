"""Camera abstraction: replace this class with calibrated RealSense capture later."""
from typing import Protocol
import time
import numpy as np

class Camera(Protocol):
    def read(self)->dict:...
    def close(self)->None:...

class MockCamera:
    def read(self):
        return {'observation.images.cam_high':np.zeros((256,320,3),np.uint8),
                'observation.images.cam_left_wrist':np.zeros((128,160,3),np.uint8),
                'observation.images.cam_right_wrist':np.zeros((128,160,3),np.uint8),
                'captured_at':time.monotonic()}
    def close(self):pass

class OpenCVCamera:
    def __init__(self,head_device,wrist_device):
        import cv2
        self.cv2=cv2;self.head=cv2.VideoCapture(head_device);self.wrist=cv2.VideoCapture(wrist_device)
        if not self.head.isOpened() or not self.wrist.isOpened():self.close();raise RuntimeError('Both calibrated cameras are required')
    def read(self):
        cv2=self.cv2;frames=[]
        for cap,size in ((self.head,(320,256)),(self.wrist,(160,128))):
            ok,frame=cap.read()
            if not ok:raise RuntimeError('Camera read failed')
            frames.append(cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),size))
        return {'observation.images.cam_high':frames[0],'observation.images.cam_left_wrist':frames[1],
                'observation.images.cam_right_wrist':np.zeros_like(frames[1]),'captured_at':time.monotonic()}
    def close(self):
        for cap in (getattr(self,'head',None),getattr(self,'wrist',None)):
            if cap is not None:cap.release()
