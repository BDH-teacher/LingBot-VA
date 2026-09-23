"""The same model client and observation/action feedback loop for SIM and REAL."""
import time
import numpy as np
from .paths import model_path

class LingBotPolicy:
    def __init__(self,host='127.0.0.1',port=29056,timeout=600):
        model_path()
        from evaluation.robotwin.msgpack_numpy import Packer,unpackb
        from websockets.sync.client import connect
        self.ws=connect(f'ws://{host}:{port}',compression=None,max_size=None,ping_interval=None,open_timeout=30)
        self.packer=Packer();self.unpack=unpackb;self.timeout=timeout
        self.metadata=self.unpack(self.ws.recv(timeout=30));self.latencies=[]
    def infer(self,payload):
        start=time.monotonic();self.ws.send(self.packer.pack(payload));result=self.ws.recv(timeout=self.timeout)
        if isinstance(result,str):raise RuntimeError(result)
        self.latencies.append(time.monotonic()-start)
        return self.unpack(result)
    def close(self):self.ws.close()

def read_camera(camera):
    obs=camera.read()
    stamp=obs.get('captured_at')
    if stamp is None or not np.isfinite(stamp) or not 0<=time.monotonic()-stamp<=0.25:
        raise ValueError('Missing/stale camera timestamp')
    for key in ('observation.images.cam_high','observation.images.cam_left_wrist','observation.images.cam_right_wrist'):
        a=np.asarray(obs[key])
        if a.ndim!=3 or a.shape[2]!=3 or a.dtype!=np.uint8 or min(a.shape[:2])<1:
            raise ValueError('Camera must return RGB uint8 HWC')
    return obs

def rollout(policy,camera,executor,adapter,instruction,max_actions=500,record=None):
    policy.infer({'reset':True,'prompt':instruction,'save_visualization':False})
    first=True;count=0;first_obs=read_camera(camera)
    while count<max_actions:
        reply=policy.infer({'obs':first_obs if first else None,'prompt':instruction,'save_visualization':False})
        actions=np.asarray(reply['action']);frames=[]
        if actions.ndim!=3 or actions.shape[0]!=14 or actions.shape[2]%4:raise ValueError('UR5 action response shape')
        for i in range(1 if first else 0,actions.shape[1]):
            for j in range(actions.shape[2]):
                if count>=max_actions:return count
                command=adapter.decode(actions[:,i,j],selected=True,issued_at=time.monotonic())
                executor.execute(command);count+=1
                obs=read_camera(camera)
                if record:record(count,obs,command,reply)
                if (j+1)%(actions.shape[2]//4)==0:frames.append(obs)
                if executor.success():return count
        if not frames:raise RuntimeError('No fresh observations collected')
        policy.infer({'obs':frames,'compute_kv_cache':True,'imagine':False,'save_visualization':False,'state':actions})
        first=False
    return count
