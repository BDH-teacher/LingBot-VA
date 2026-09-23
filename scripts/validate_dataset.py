import argparse,json,pathlib
import numpy as np
import pyarrow.parquet as pq
import av
from lingbot_ur5.adapters.action.lingbot_ur5_action_adapter import UR5_CHANNELS
from lingbot_ur5.dataset import CAMERAS,RESOLUTIONS

def validate(root,require_latents=False):
    root=pathlib.Path(root).resolve();meta=root/'meta'
    info=json.loads((meta/'info.json').read_text());profile=json.loads((meta/'ur5_profile.json').read_text())
    episodes=[json.loads(l) for l in (meta/'episodes.jsonl').read_text().splitlines()]
    assert len(episodes)>0 and len(episodes)==info['total_episodes'],'episode count'
    assert info['fps']==25==profile['action_hz'],'action frequency'
    mask=np.zeros(30,bool);mask[list(UR5_CHANNELS)]=True
    stat=profile['norm_stat'];lo=np.asarray(stat['q01']);hi=np.asarray(stat['q99'])
    assert lo.shape==hi.shape==(30,) and np.isfinite([lo,hi]).all(),'normalization'
    assert np.all(hi[mask]>lo[mask]) and np.all(lo[~mask]==0) and np.all(hi[~mask]==0),'quantile ranges'
    total=0
    for ep in episodes:
        idx=ep['episode_index'];n=ep['length'];chunk=idx//info.get('chunks_size',1000)
        data=pq.read_table(root/info['data_path'].format(episode_chunk=chunk,episode_index=idx)).to_pydict()
        assert len(data['timestamp'])==n,'length'
        timestamps=np.asarray(data['timestamp']);np.testing.assert_allclose(timestamps,np.arange(n)/25,atol=1e-4)
        for key in ('action','observation.state'):
            a=np.asarray(data[key]);assert a.shape==(n,30) and np.isfinite(a).all(),key+' shape/finite'
            np.testing.assert_allclose(np.linalg.norm(a[:,3:7],axis=1),1,atol=1e-3)
            assert not np.any(a[:,~mask]),'inactive arm/padding'
            assert np.all((a[:,28]>=0)&(a[:,28]<=1)),'gripper'
            # URDF numerical limits, captured by audit_embodiment.py when assets exist.
            audit=pathlib.Path(__file__).resolve().parents[1]/'configs/sim/ur5_asset_audit.json'
            if not audit.exists():raise FileNotFoundError('Run scripts/audit_embodiment.py to derive URDF joint limits')
            limits=json.loads(audit.read_text())['joint_limits']
            q=a[:,14:20];assert np.all(q>=np.asarray(limits['lower'])-1e-4) and np.all(q<=np.asarray(limits['upper'])+1e-4),'URDF joint limits'
        assert ep['tasks'] and all(t.strip() for t in ep['tasks']),'instruction'
        for raw,key in CAMERAS.items():
            video=root/info['video_path'].format(episode_chunk=chunk,video_key=key,episode_index=idx)
            with av.open(str(video)) as container:
                stream=container.streams.video[0];assert abs(float(stream.average_rate)-25)<1e-3,'video fps'
                frames=0;last_time=-1
                for frame in container.decode(stream):
                    assert (frame.width,frame.height)==RESOLUTIONS[raw],'resolution'
                    rgb=frame.to_ndarray(format='rgb24');assert rgb.dtype==np.uint8 and rgb.shape[-1]==3,'RGB'
                    assert frame.time is not None and frame.time>last_time,'video timestamps';last_time=frame.time;frames+=1
                assert frames==n,'video/action count'
        assert ep['action_config'],'action segmentation'
        for seg in ep['action_config']:
            assert 0<=seg['start_frame']<seg['end_frame']<=n and seg['action_text'].strip()
            if require_latents:
                import torch
                for key in CAMERAS.values():
                    p=root/f'latents/chunk-{chunk:03d}'/key/f'episode_{idx:06d}_{seg["start_frame"]}_{seg["end_frame"]}.pth'
                    obj=torch.load(p,weights_only=False,map_location='cpu');x=obj['latent']
                    assert x.shape==(obj['latent_num_frames']*obj['latent_height']*obj['latent_width'],48),'latent dimensions'
                    assert torch.isfinite(x).all() and torch.isfinite(obj['text_emb']).all(),'latent NaN/Inf'
                    assert len(obj['frame_ids'])%4==1 and np.all(np.diff(obj['frame_ids'])==4),'latent temporal stride'
        total+=n
    assert total==info['total_frames'],'total frame count'
    if require_latents:assert (root/'empty_emb.pt').exists(),'empty text embedding'
    report={'valid':True,'episodes':len(episodes),'frames':total,'fps':25,'action_dim':30,'state_dim':30,'latents_checked':require_latents}
    (root/'validation.json').write_text(json.dumps(report,indent=2));print('DATASET VALID',json.dumps(report))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='datasets/ur5_move_can_pot');p.add_argument('--require-latents',action='store_true')
    a=p.parse_args();validate(a.dataset,a.require_latents)
