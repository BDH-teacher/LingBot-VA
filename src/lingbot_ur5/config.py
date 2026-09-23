import copy,json,pathlib
from .paths import ROOT,model_path
from lingbot_ur5.adapters.action.lingbot_ur5_action_adapter import UR5_CHANNELS
def make_config(dataset=None,checkpoint=None,training=False,official=False):
    model_path()
    from configs import VA_CONFIGS
    cfg=copy.deepcopy(VA_CONFIGS['robotwin_train' if training else 'robotwin'])
    cfg.wan22_pretrained_model_name_or_path=str(pathlib.Path(checkpoint or ROOT/'checkpoints/pretrained/lingbot-va-base').resolve())
    checkpoint_root=pathlib.Path(cfg.wan22_pretrained_model_name_or_path)
    model_config=json.loads((checkpoint_root/'transformer/config.json').read_text())
    if model_config['action_dim']!=30 or model_config['in_channels']!=48:
        raise ValueError('Expected public Wan VA 30D/48-channel model')
    weights=list((checkpoint_root/'transformer').glob('*.safetensors'))
    if not weights or any(p.stat().st_size==0 for p in weights):raise FileNotFoundError('Transformer weights missing')
    cfg.enable_offload=True;cfg.infer_mode='server';cfg.host='127.0.0.1'
    cfg.save_root=str(ROOT/'outputs/inference');cfg.enable_wandb=False;cfg.load_worker=0
    if not training:
        runtime=json.loads((ROOT/'configs/inference/runtime.json').read_text())
        cfg.enable_offload=runtime['enable_offload'];cfg.num_inference_steps=runtime['video_steps']
        cfg.action_num_inference_steps=runtime['action_steps']
    if not official:
        dataset=pathlib.Path(dataset or ROOT/'datasets/ur5_move_can_pot').resolve()
        profile_path=dataset/'meta/ur5_profile.json' if training else pathlib.Path(cfg.wan22_pretrained_model_name_or_path)/'ur5_profile.json'
        if not profile_path.exists():raise FileNotFoundError('UR5 action profile missing: '+str(profile_path)+'; public RoboTwin weights require --official')
        profile=json.loads(profile_path.read_text())
        if profile['format']!='ur5-canonical-v1':raise ValueError('Wrong dataset/action profile')
        cfg.dataset_path=str(dataset);cfg.empty_emb_path=str(dataset/'empty_emb.pt')
        cfg.norm_stat=profile['norm_stat'];cfg.used_action_channel_ids=list(UR5_CHANNELS)
        if profile['active_channels']!=list(UR5_CHANNELS) or profile['quaternion']!='xyzw' or profile['frame']!='world' or profile['action_hz']!=25:
            raise ValueError('Incompatible UR5 action contract')
        inverse=[len(UR5_CHANNELS)]*30
        for i,c in enumerate(UR5_CHANNELS):inverse[c]=i
        cfg.inverse_used_action_channel_ids=inverse
    return cfg
