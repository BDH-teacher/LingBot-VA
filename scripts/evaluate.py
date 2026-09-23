import argparse,json,pathlib,time
import numpy as np
from lingbot_ur5.paths import ROOT
from lingbot_ur5.sim import make_task
from lingbot_ur5.policy import LingBotPolicy,rollout
from lingbot_ur5.adapters.camera.sim_camera import SimCamera
from lingbot_ur5.adapters.action.sim_executor import SimExecutor
from lingbot_ur5.adapters.action.lingbot_ur5_action_adapter import UR5ActionAdapter,wxyz_to_xyzw,legacy_robotwin_restore

def main():
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,default=10);p.add_argument('--port',type=int,default=29056)
    p.add_argument('--output-dir',default=str(ROOT/'outputs/evaluation'));p.add_argument('--gui',action='store_true');p.add_argument('--official',action='store_true');p.add_argument('--seed',type=int,default=1000)
    a=p.parse_args();out=pathlib.Path(a.output_dir)/time.strftime('%Y%m%d-%H%M%S');out.mkdir(parents=True)
    import imageio.v2 as imageio
    results=[]
    for episode in range(a.episodes):
        task=None;policy=None;writer=None;errors=None;actions=0;metrics=[];started=time.monotonic()
        try:
            task,cfg=make_task(a.gui,a.seed+episode,official=a.official)
            camera=SimCamera(task,a.official);policy=LingBotPolicy(port=a.port)
            expected='official-robotwin-legacy' if a.official else 'ur5-canonical-v1'
            if policy.metadata.get('profile')!=expected:raise ValueError('Server action profile mismatch')
            writer=imageio.get_writer(str(out/f'episode_{episode:04d}.mp4'),fps=25,codec='libx264')
            def record(count,obs,cmd,reply):
                nonlocal actions
                actions=count
                writer.append_data(obs['observation.images.cam_high']);metrics.append(reply.get('metrics',{}))
            if not a.official:
                origin=wxyz_to_xyzw(task.get_obs()['endpose']['left_endpose']);adapter=UR5ActionAdapter(origin)
                limits=json.loads((ROOT/'configs/sim/ur5_asset_audit.json').read_text())['joint_limits']
                actions=rollout(policy,camera,SimExecutor(task,limits),adapter,cfg.get('instruction','Pick up the can and place it beside the pot.'),task.step_lim,record)
            else:
                prompt='Pick up the can and place it beside the pot.';obs=task.get_obs()
                origin=np.r_[obs['endpose']['left_endpose'],obs['endpose']['left_gripper'],obs['endpose']['right_endpose'],obs['endpose']['right_gripper']]
                policy.infer({'reset':True,'prompt':prompt,'save_visualization':False});first=True
                while actions<task.step_lim and not task.eval_success:
                    reply=policy.infer({'obs':camera.read() if first else None,'prompt':prompt,'save_visualization':False})
                    chunk=np.asarray(reply['action']);frames=[]
                    for i in range(1 if first else 0,chunk.shape[1]):
                        for j in range(chunk.shape[2]):
                            task.take_action(legacy_robotwin_restore(chunk[:,i,j],origin),action_type='ee');actions+=1
                            obs=camera.read();record(actions,obs,None,reply)
                            if (j+1)%(chunk.shape[2]//4)==0:frames.append(obs)
                            if task.eval_success or actions>=task.step_lim:break
                        if task.eval_success or actions>=task.step_lim:break
                    if task.eval_success or actions>=task.step_lim:break
                    policy.infer({'obs':frames,'compute_kv_cache':True,'imagine':False,'save_visualization':False,'state':chunk});first=False
        except Exception as exc:
            errors=repr(exc)
            import traceback;traceback.print_exc()
        finally:
            success=bool(errors is None and task and task.eval_success)
            result={'episode':episode,'seed':a.seed+episode,'success':success,'actions':actions,'elapsed_seconds':time.monotonic()-started,
                    'error':errors,'latency_seconds':policy.latencies if policy else [],'model_metrics':metrics,'closed_loop':True}
            results.append(result);(out/f'episode_{episode:04d}.json').write_text(json.dumps(result,indent=2))
            if writer:writer.close()
            if policy:policy.close()
            if task:
                task.close_env()
                if a.gui and hasattr(task,'viewer'):task.viewer.close()
    summary={'episodes':len(results),'success_rate':sum(r['success'] for r in results)/len(results),'errors':sum(r['error'] is not None for r in results),'profile':'official' if a.official else 'single-ur5','results_dir':str(out)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    if summary['errors']:raise SystemExit(1)
if __name__=='__main__':main()
