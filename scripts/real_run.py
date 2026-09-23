import argparse,json,pathlib,time
import numpy as np
from lingbot_ur5.paths import ROOT
from lingbot_ur5.policy import LingBotPolicy,rollout
from lingbot_ur5.adapters.action.lingbot_ur5_action_adapter import UR5ActionAdapter
from lingbot_ur5.adapters.action.real_executor import RealUR5Executor
from lingbot_ur5.adapters.action.safety import RobotState
from lingbot_ur5.adapters.camera.real_camera import MockCamera

class MockPolicy:
    def __init__(self,q):self.q=q;self.feedback=0;self.requests=0
    def infer(self,payload):
        self.requests+=1
        if payload.get('reset'):return {}
        if payload.get('compute_kv_cache'):
            if not payload['obs']:raise ValueError('No closed-loop observations')
            self.feedback+=len(payload['obs']);return {}
        a=np.zeros((14,2,16),dtype='float32');a[6]=1;a[7:13]=self.q[:,None,None];a[13]=0.5
        return {'action':a,'metrics':{'mock':True}}
    def close(self):pass

def main():
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group();group.add_argument('--dry-run',action='store_true');group.add_argument('--execute-real',action='store_true')
    p.add_argument('--config',default=str(ROOT/'configs/real/deployment.json'));p.add_argument('--model-server',action='store_true');p.add_argument('--port',type=int,default=29056)
    p.add_argument('--max-actions',type=int,default=32);a=p.parse_args()
    transport=None;camera=None;policy=None;executor=None
    try:
        if a.execute_real:
            cfg=json.loads(pathlib.Path(a.config).read_text())
            required=['robot_ip','calibration_file','safety_config','camera_factory','gripper_factory','tcp_link','world_frame']
            if not cfg.get('hardware_validated') or not cfg.get('operator_approved') or any(not cfg.get(k) for k in required):
                raise RuntimeError('REAL EXECUTION BLOCKED: complete reviewed deployment/calibration/camera/gripper config first')
            if not a.model_server:raise ValueError('Real execution requires the trained model server; mock policy is forbidden')
            if not pathlib.Path(cfg['calibration_file']).is_file():raise FileNotFoundError('Robot-specific calibration file required')
            safety=json.loads(pathlib.Path(cfg['safety_config']).read_text())
            if not safety.get('hardware_validated') or safety['frame']!=cfg['world_frame']:raise ValueError('Reviewed safety frame required')
            from lingbot_ur5.adapters.action.safety import SafetyFilter
            SafetyFilter(safety)
            if abs(safety['control_dt']-0.04)>1e-9:raise ValueError('UR5 profile requires 25Hz control; validate transport timing first')
            for key in ('fk_position_tolerance','fk_rotation_tolerance'):
                if not np.isfinite(safety.get(key,float('nan'))) or safety[key]<=0:raise ValueError('Reviewed FK tolerance required: '+key)
            from lingbot_ur5.adapters.action.ros2_transport import ROS2Transport,factory
            transport=ROS2Transport(cfg,safety);state=transport.read_state();camera=factory(cfg['camera_factory'],cfg['camera_config'])
        else:
            safety=json.loads((ROOT/'configs/real/safety_mock.json').read_text());safety['control_dt']=0.04
            state=RobotState(np.asarray(safety['home_joints']),np.array([0.3,0,0.5,0,0,0,1.]),np.zeros(6),time.monotonic())
            camera=MockCamera()
        policy=LingBotPolicy(port=a.port) if a.model_server else MockPolicy(state.joints)
        if a.model_server and policy.metadata.get('profile')!='ur5-canonical-v1':raise ValueError('Real adapter requires UR5 model profile')
        adapter=UR5ActionAdapter(state.eef,frame=state.frame)
        logfile=ROOT/'outputs/real_dry_run/commands.jsonl' if not a.execute_real else ROOT/'outputs/real_execution/commands.jsonl'
        executor=RealUR5Executor(safety,state,logfile,transport,a.execute_real)
        count=rollout(policy,camera,executor,adapter,'Pick up the can and place it beside the pot.',a.max_actions)
        print(json.dumps({'mode':'REAL' if a.execute_real else 'DRY_RUN','actions_validated':count,'model':'LingBot' if a.model_server else 'MOCK',
                         'feedback_frames':getattr(policy,'feedback',None),'physical_commands_sent':count if a.execute_real else 0,'log':str(logfile)},indent=2))
    finally:
        if executor:executor.stop()
        if policy:policy.close()
        if camera:camera.close()
        if transport:transport.close()
if __name__=='__main__':main()
