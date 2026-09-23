import argparse,json,pathlib,os,time
from lingbot_ur5.paths import ROOT,model_path,resolve_inference_checkpoint

def main():
    runtime=json.loads((ROOT/'configs/inference/runtime.json').read_text())
    if runtime['host']!='127.0.0.1' or runtime['attention']!='torch':raise ValueError('This launcher supports localhost + official torch attention')
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=str(ROOT/'checkpoints/pretrained/lingbot-va-posttrain-robotwin'))
    p.add_argument('--dataset',default=str(ROOT/'datasets/ur5_move_can_pot'));p.add_argument('--official',action='store_true')
    p.add_argument('--port',type=int,default=runtime['port']);p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--preflight-report',type=pathlib.Path,help='Additional report path for a launcher invocation')
    a=p.parse_args();model_path()
    import torch
    from lingbot_ur5.config import make_config
    checkpoint=resolve_inference_checkpoint(a.checkpoint)
    original_checkpoint=checkpoint
    if not (checkpoint/'transformer/config.json').exists():raise FileNotFoundError('Transformer checkpoint missing: '+str(checkpoint))
    # Training exports only transformer; compose an inference view using pinned base components.
    if not (checkpoint/'vae').exists():
        view=ROOT/'outputs/inference_models'/checkpoint.name;view.mkdir(parents=True,exist_ok=True)
        for name in ('vae','text_encoder','tokenizer','transformer'):
            source=checkpoint/name if name=='transformer' else ROOT/'checkpoints/pretrained/lingbot-va-base'/name
            if not source.exists():raise FileNotFoundError(source)
            link=view/name
            if link.is_symlink() and link.resolve()!=source.resolve():link.unlink()
            if not link.exists():link.symlink_to(source,target_is_directory=True)
        checkpoint=view
    cfg=make_config(a.dataset,original_checkpoint,official=a.official)
    cfg.wan22_pretrained_model_name_or_path=str(checkpoint)
    total=torch.cuda.get_device_properties(0).total_memory/1024**3 if torch.cuda.is_available() else 0
    report={'gpu_total_gib':total,'official_approx_vram_gb':24,'offload':cfg.enable_offload,'attention':'torch','checkpoint':str(checkpoint),'profile':'official-robotwin-legacy' if a.official else 'ur5-canonical-v1'}
    report.update(status='SKIPPED_EXPECTED_VRAM_LIMIT' if total<runtime['min_gpu_gib'] else 'PREFLIGHT_PASSED',model_loaded=False)
    output=ROOT/'outputs/inference';output.mkdir(parents=True,exist_ok=True)
    (output/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    if a.preflight_report:a.preflight_report.write_text(json.dumps(report,indent=2))
    if total<runtime['min_gpu_gib']:
        if a.preflight_only:
            print('INFERENCE SKIPPED: EXPECTED VRAM LIMIT. No model allocation/OOM attempted.');return
        raise SystemExit('INFERENCE SKIPPED: EXPECTED VRAM LIMIT. No model allocation/OOM attempted; use >=24GB GPU (simulation shares VRAM).')
    if a.preflight_only:return
    cfg.local_rank=0;cfg.rank=0;cfg.world_size=1;cfg.infer_mode='server'
    cfg.port=a.port;cfg.host='127.0.0.1'
    # Official server relies on initialized distributed context even for one GPU.
    os.environ.setdefault('MASTER_ADDR','127.0.0.1');os.environ.setdefault('MASTER_PORT',str(runtime['distributed_port']))
    from distributed.util import init_distributed
    from wan_va_server import VA_Server
    from utils.Simple_Remote_Infer.deploy.websocket_policy_server import WebsocketPolicyServer
    init_distributed(1,0,0);model=VA_Server(cfg)
    class MeteredPolicy:
        def infer(self,request):
            torch.cuda.reset_peak_memory_stats();started=time.monotonic();response=model.infer(request)
            if response is None:response={}
            response['metrics']={'seconds':time.monotonic()-started,'gpu_peak_mib':torch.cuda.max_memory_allocated()/1024**2}
            return response
    WebsocketPolicyServer(MeteredPolicy(),host='127.0.0.1',port=a.port,metadata=report).serve_forever()
if __name__=='__main__':main()
