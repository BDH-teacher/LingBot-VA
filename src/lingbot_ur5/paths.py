import os,pathlib,sys
ROOT=pathlib.Path(os.environ.get('LINGBOT_UR5_ROOT',pathlib.Path(__file__).resolve().parents[2])).resolve()
def resolve_inference_checkpoint(value):
    path=pathlib.Path(value).resolve()
    if (path/'transformer/config.json').is_file():return path
    candidates=[]
    for parent in (path,path/'checkpoints'):
        for item in parent.glob('checkpoint_step_*'):
            if (item/'resume_ready.json').is_file() and (item/'transformer/config.json').is_file():
                try:candidates.append((int(item.name.rsplit('_',1)[1]),item))
                except ValueError:continue
    if not candidates:raise FileNotFoundError(f'No complete inference checkpoint at {path}')
    return max(candidates,key=lambda item:item[0])[1]

def model_path():
    path=ROOT/'outputs/runtime/lingbot-va'
    if not path.exists(): raise FileNotFoundError('Run conda run -n base python scripts/manage.py install-model first')
    for p in (path,path/'wan_va'):
        if str(p) not in sys.path:sys.path.insert(0,str(p))
    return path
def robotwin_path():
    path=ROOT/'outputs/runtime/robotwin'
    if not path.exists(): raise FileNotFoundError('Run conda run -n base python scripts/manage.py check-sim first')
    sys.path.insert(0,str(path));os.chdir(path)
    return path
