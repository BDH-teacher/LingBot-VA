"""Real LeRobot v2.1 conversion and a 30D subclass of the official latent loader."""
import json,pathlib
import numpy as np
from lingbot_ur5.adapters.action.lingbot_ur5_action_adapter import UR5_CHANNELS,relative_pose
CAMERAS={'head_camera':'observation.images.cam_high','left_camera':'observation.images.cam_left_wrist','right_camera':'observation.images.cam_right_wrist'}
RESOLUTIONS={'head_camera':(320,256),'left_camera':(160,128),'right_camera':(160,128)}

def convert(raw_dir,destination,instruction,segment_frames=256):
    import cv2,h5py
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    raw_dir=pathlib.Path(raw_dir);destination=pathlib.Path(destination).resolve()
    files=sorted(raw_dir.glob('episode_*.hdf5'))
    if not files:raise FileNotFoundError(f'No successful expert episodes in {raw_dir}')
    if destination.exists():raise FileExistsError(f'Dataset already exists: {destination}')
    features={'action':{'dtype':'float32','shape':(30,),'names':None},
              'observation.state':{'dtype':'float32','shape':(30,),'names':None}}
    for raw,key in CAMERAS.items():
        w,h=RESOLUTIONS[raw];features[key]={'dtype':'video','shape':(h,w,3),'names':['height','width','channel']}
    ds=LeRobotDataset.create('local/'+destination.name,25,features,root=destination,
          robot_type='single-ur5-wsg',image_writer_threads=2,video_backend='pyav')
    for path in files:
        with h5py.File(path) as h:
            if not h.attrs['success']:raise ValueError(f'Failed expert episode {path}')
            states=h['state'][:].astype('float32');times=h['timestamp'][:];n=len(states)-1
            if n<32:raise ValueError('Episode too short for temporal VAE')
            np.testing.assert_allclose(np.diff(times),1/25,atol=1e-5)
            for i in range(n):
                frame={'observation.state':states[i].copy(),'action':states[i+1].copy()}
                for raw,key in CAMERAS.items():frame[key]=cv2.resize(h[raw][i],RESOLUTIONS[raw],interpolation=cv2.INTER_LINEAR)
                ds.add_frame(frame,task=instruction,timestamp=i/25)
            ds.save_episode()
    ds.stop_image_writer()
    p=destination/'meta/episodes.jsonl';rows=[json.loads(line) for line in p.read_text().splitlines()]
    relative_actions=[]
    import pyarrow.parquet as pq
    for row in rows:
        n=row['length']; segments=[]
        for start in range(0,n,segment_frames):
            end=min(start+segment_frames,n)
            if end-start<32:
                if segments:segments[-1]['end_frame']=end
                continue
            segments.append({'start_frame':start,'end_frame':end,'action_text':instruction})
        row['action_config']=segments
        table=pq.read_table(destination/ds.meta.get_data_file_path(row['episode_index'])).to_pydict()
        actions=np.asarray(table['action'])
        for seg in segments:
            a=actions[seg['start_frame']:seg['end_frame']].copy()
            # The first observation pose anchors both training and evaluation.
            anchor=np.asarray(table['observation.state'][seg['start_frame']])[:7]
            for i in range(len(a)):a[i,:7]=relative_pose(a[i,:7],anchor)
            relative_actions.append(a)
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    joined=np.concatenate(relative_actions)
    lo=np.quantile(joined,0.01,axis=0);hi=np.quantile(joined,0.99,axis=0)
    lo[3:7]=-1;hi[3:7]=1;lo[28]=0;hi[28]=1
    # Constant active channels retain a small finite normalization range.
    for c in UR5_CHANNELS:
        if hi[c]-lo[c]<1e-4:lo[c]-=1e-4;hi[c]+=1e-4
    profile={'format':'ur5-canonical-v1','active_channels':list(UR5_CHANNELS),'quaternion':'xyzw',
             'frame':'world','eef':'RoboTwin transformed EEF, 0.12m behind gripper center',
             'action':'next measured state; relative EEF, absolute joint angles after loader',
             'action_hz':25,'latent_frame_stride':4,'norm_stat':{'q01':lo.tolist(),'q99':hi.tolist()},
             'sources':[str(f.resolve()) for f in files],'instruction':instruction}
    (destination/'meta/ur5_profile.json').write_text(json.dumps(profile,indent=2))
    print('LEROBOT CONVERSION COMPLETE',destination)

def dataset_class():
    from .paths import model_path
    model_path()
    import torch
    from einops import rearrange
    from dataset.lerobot_latent_dataset import LatentLeRobotDataset
    class UR5LatentDataset(LatentLeRobotDataset):
        def _action_post_process(self,start,end,frame_ids,action):
            action=np.asarray(action,dtype=np.float64).copy()
            # Read the matching observation so the origin matches evaluation exactly.
            global_start=int(self.episode_data_index['from'][self._current_episode])+start
            origin=np.asarray(self.hf_dataset.with_format('numpy')[global_start]['observation.state'])[:7]
            for i in range(len(action)):action[i,:7]=relative_pose(action[i,:7],origin)
            stride=int(frame_ids[1]-frame_ids[0]);shift=int(frame_ids[0]-start)
            frames=(len(frame_ids)-1)//4+1
            action=np.pad(action[shift:],((stride*4,0),(0,0)))[:frames*stride*4]
            if len(action)!=frames*stride*4:raise ValueError('Insufficient actions for latent alignment')
            mask=np.zeros_like(action,dtype=bool);mask[:,list(UR5_CHANNELS)]=True
            action=np.clip((action-self.q01)/(self.q99-self.q01+1e-6)*2-1,-1.5,1.5)*mask
            return torch.tensor(rearrange(action,'(f n) c -> c f n 1',f=frames),dtype=torch.float32),torch.tensor(rearrange(mask,'(f n) c -> c f n 1',f=frames))
        def __getitem__(self,index):
            self._current_episode=self.new_metas[index%len(self.new_metas)]['episode_index']
            return super().__getitem__(index)
    return UR5LatentDataset

def load_latent_dataset(config):
    cls=dataset_class()
    return cls(str(pathlib.Path(config.dataset_path).resolve()),config=config)
