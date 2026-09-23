"""One UR5 articulation using the official WSG asset and RoboTwin expert.

The inactive right interface is an alias required by RoboTwin's bimanual API.
All right drive commands are no-ops; right Cartesian plans are rejected.
"""
import copy, json, pathlib, time
import numpy as np
import yaml
from .paths import ROOT,robotwin_path

def make_task(gui=False,seed=0,randomization=False,official=False):
    path=robotwin_path()
    from envs.move_can_pot import move_can_pot
    from envs.robot.robot import Robot
    from envs.robot.planner import CuroboPlanner,MplibPlanner
    from envs.utils.action import ArmTag

    class SingleRobot(Robot):
        def set_planner(self,scene=None):
            self.communication_flag=False
            self.left_planner=CuroboPlanner(self.left_entity_origion_pose,self.left_arm_joints_name,
                  [j.get_name() for j in self.left_entity.get_active_joints()],yml_path=self.left_curobo_yml_path)
            self.right_planner=self.left_planner
            self.left_mplib_planner=MplibPlanner(self.left_urdf_path,self.left_srdf_path,self.left_move_group,
                  self.left_entity_origion_pose,self.left_entity,self.left_planner_type,scene)
            self.right_mplib_planner=self.left_mplib_planner
        def move_to_homestate(self):
            for j,q in zip(self.left_arm_joints,self.left_homestate):j.set_drive_target(q)
        def set_arm_joints(self,position,velocity,arm_tag):
            if str(arm_tag)=='left':super().set_arm_joints(position,velocity,arm_tag)
        def set_gripper(self,value,arm_tag,gripper_eps=0.1):
            if str(arm_tag)=='left':super().set_gripper(value,arm_tag,gripper_eps)
            else:self.right_gripper_val=1.0
        def is_right_gripper_open(self):return True
        def right_plan_path(self,*args,**kwargs):raise ValueError('Single UR5: right-arm Cartesian plan forbidden')
        def right_plan_multi_path(self,*args,**kwargs):raise ValueError('Single UR5: right-arm Cartesian plan forbidden')

    class SingleTask(move_can_pot):
        recorder=None
        sim_steps=0
        def load_robot(self,**kwargs):
            self.robot=SingleRobot(self.scene,self.need_topp,**kwargs)
            self.robot.set_planner(self.scene);self.robot.init_joints()
            assert self.robot.left_entity is self.robot.right_entity
            for link in self.robot.left_entity.get_links():link.set_mass(1)
        def load_actors(self):
            super().load_actors()
            if self.arm_tag=='right':
                pose=self.can.get_pose();position=pose.p.copy();position[0]=-abs(position[0])
                pose.p=position;self.can.set_pose(pose)
            self.arm_tag=ArmTag('left')
            pose=self.pot.get_pose()
            import sapien
            self.target_pose=sapien.Pose([pose.p[0]-0.18,pose.p[1],0.741+self.table_z_bias],pose.q)
        def _after_sim_step(self):
            self.sim_steps+=1
            if self.recorder is not None and self.sim_steps%10==0:
                self.recorder.record(self,self.sim_steps/250)

    config=yaml.safe_load((path/'task_config/demo_clean.yml').read_text())
    name='aloha-agilex' if official else 'ur5-wsg'
    embodiment=path/'assets/embodiments'/name
    emb=yaml.safe_load((embodiment/'config.yml').read_text())
    # One shared articulation expects both API slots to refer to the same UR5 joints.
    if not official:
        for key in ('arm_joints_name','ee_joints','move_group','gripper_name','homestate'):
            emb[key]=[copy.deepcopy(emb[key][0]),copy.deepcopy(emb[key][0])]
        emb['robot_pose']=[emb['robot_pose'][0],emb['robot_pose'][0]]
        settings=json.loads((ROOT/'configs/sim/ur5.json').read_text())
        head=settings.get('head_camera')
        if head:
            forward=np.asarray(head['target'])-np.asarray(head['position']);forward/=np.linalg.norm(forward)
            left=np.cross([0.,0.,1.],forward);left/=np.linalg.norm(left)
            emb['static_camera_list']=[dict(name='head_camera',position=head['position'],forward=forward.tolist(),left=left.tolist())]
    config.update(task_name='move_can_pot',task_config='ur5',left_robot_file=str(embodiment),right_robot_file=str(embodiment),
                  left_embodiment_config=emb,right_embodiment_config=emb,dual_arm_embodied=True,
                  embodiment_name=name,render_freq=10 if gui else 0,save_freq=10,save_data=False,
                  need_plan=True,eval_mode=True,eval_video_save_dir=None,seed=seed,now_ep_num=0)
    if randomization:
        config['domain_randomization'].update(random_background=True,random_light=True,clean_background_rate=0.5)
    task=move_can_pot() if official else SingleTask()
    task.setup_demo(**config)
    task.set_instruction('Pick up the can and place it beside the pot.')
    return task,config

class EpisodeRecorder:
    def __init__(self,path):
        import h5py,imageio.v2 as imageio
        self.path=pathlib.Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file=h5py.File(self.path,'w');self.count=0;self.start=None
        self.video=imageio.get_writer(str(self.path.with_suffix('.mp4')),fps=25,codec='libx264')
    def record(self,task,sim_time):
        from lingbot_ur5.adapters.action.lingbot_ur5_action_adapter import wxyz_to_xyzw
        obs=task.get_obs()
        joints=task.robot.get_left_arm_real_jointState()
        state=np.zeros(30);state[:7]=wxyz_to_xyzw(obs['endpose']['left_endpose'])
        state[14:20]=joints[:6];state[28]=obs['endpose']['left_gripper']
        if self.start is None:self.start=sim_time
        row={'state':state.astype('float32'),'commanded_joints':np.asarray(task.robot.get_left_arm_jointState()[:6],dtype='float32'),
             'timestamp':np.float64(sim_time-self.start)}
        for name in ('head_camera','left_camera'):
            row[name]=obs['observation'][name]['rgb']
        row['right_camera']=np.zeros_like(row['left_camera'])
        for key,value in row.items():
            value=np.asarray(value)
            if key not in self.file:
                self.file.create_dataset(key,shape=(0,*value.shape),maxshape=(None,*value.shape),dtype=value.dtype,
                       compression='lzf' if value.ndim else None,chunks=True)
            ds=self.file[key];ds.resize(self.count+1,axis=0);ds[self.count]=value
        self.video.append_data(row['head_camera']);self.count+=1
    def close(self,success,seed):
        self.file.attrs.update(success=bool(success),seed=int(seed),fps=25,embodiment='single-ur5-wsg',quaternion='xyzw',frame='world')
        self.file.close();self.video.close()

def collect(episodes,output,gui=False,seed=0,randomization=False,max_attempts=20):
    # RoboTwin changes cwd during setup; anchor user-relative paths before that.
    output=pathlib.Path(output).expanduser().resolve();output.mkdir(parents=True,exist_ok=True)
    existing=list(output.glob('episode_*.hdf5'))
    if existing:raise FileExistsError(f'{output} already contains episodes; choose a new --output-dir')
    completed=0;attempts=[]
    for candidate in range(seed,seed+episodes*max_attempts):
        task=None;rec=None;started=time.monotonic()
        try:
            task,_=make_task(gui,candidate,randomization)
            rec=EpisodeRecorder(output/f'attempt_{candidate:06d}.hdf5');task.recorder=rec
            info=task.play_once();success=bool(task.plan_success and task.check_success())
            task.recorder=None
            rec.close(success,candidate);rec=None
            attempts.append({'seed':candidate,'success':success,'seconds':time.monotonic()-started,'info':info})
            if success:
                for suffix in ('.hdf5','.mp4'):
                    (output/f'attempt_{candidate:06d}{suffix}').rename(output/f'episode_{completed:06d}{suffix}')
                completed+=1
                print(f'UR5 EXPERT EPISODE SUCCESS {completed}/{episodes}',flush=True)
            if completed==episodes:break
        finally:
            if rec:rec.close(False,candidate)
            if task:
                task.recorder=None;task.close_env()
                if gui and hasattr(task,'viewer'):task.viewer.close()
            (output/'episodes.json').write_text(json.dumps(attempts,indent=2,default=str))
    if completed!=episodes:raise RuntimeError(f'Only {completed}/{episodes} expert successes; inspect {output}')
    return output
