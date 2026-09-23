import argparse,json,os,pathlib,subprocess,sys
from .paths import ROOT,model_path

def main():
    defaults=json.loads((ROOT/'configs/training/ur5.json').read_text())
    p=argparse.ArgumentParser()
    for key in ('dataset','checkpoint','output_dir'):p.add_argument('--'+key.replace('_','-'),default=str(ROOT/defaults[key]))
    for key in ('steps','batch_size','grad_accum','save_interval'):p.add_argument('--'+key.replace('_','-'),type=int,default=defaults[key])
    p.add_argument('--lr',type=float,default=defaults['lr']);p.add_argument('--resume');p.add_argument('--warmstart')
    p.add_argument('--gpus',type=int,default=1);p.add_argument('--dry-run',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if min(a.steps,a.batch_size,a.grad_accum,a.gpus,a.save_interval)<1 or a.lr<=0:p.error('Positive training settings required')
    if a.resume and a.warmstart:p.error('Use resume or warmstart, not both')
    model_path()
    import torch
    from .config import make_config
    from .dataset import load_latent_dataset
    cfg=make_config(a.dataset,a.checkpoint,True)
    cfg.update(num_steps=1 if a.smoke else a.steps,batch_size=a.batch_size,learning_rate=a.lr,gradient_accumulation_steps=a.grad_accum,
               save_root=str(pathlib.Path(a.output_dir).resolve()),save_interval=1 if a.smoke else a.save_interval)
    if a.warmstart:cfg.resume_from=str(pathlib.Path(a.warmstart).resolve())
    if a.resume and not (pathlib.Path(a.resume)/'resume_ready.json').exists():raise FileNotFoundError('Incomplete/no DCP resume checkpoint')
    from scripts.validate_dataset import validate
    validate(a.dataset,require_latents=True)
    ds=load_latent_dataset(cfg)
    if len(ds)==0:raise ValueError('No valid latent segments')
    sample=ds[0]
    for key,tensor in sample.items():
        if not torch.isfinite(tensor).all():raise ValueError('Nonfinite training tensor '+key)
    if sample['actions'].shape[0]!=30 or sample['latents'].shape[0]!=48:raise ValueError('Wrong training dimensions')
    print('CONFIG AND DATASET LOADER VALID', {k:tuple(v.shape) for k,v in sample.items()},flush=True)
    out=pathlib.Path(cfg.save_root);out.mkdir(parents=True,exist_ok=True)
    resolved={k:str(v) if isinstance(v,torch.dtype) else v for k,v in cfg.items()}
    (out/'resolved_config.json').write_text(json.dumps(resolved,indent=2,default=str))
    if a.dry_run:
        print('TRAINING PIPELINE READY (config + actual latent dataset loader; model not loaded)');return
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    mem=torch.cuda.get_device_properties(0).total_memory/1024**3
    # Conservative local guard, not an official measured training minimum.
    if mem<48 or os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')<60*1024**3:
        if a.smoke:
            print('TRAINING PIPELINE READY\nSMOKE TEST SKIPPED: EXPECTED VRAM LIMIT\nForward/loss/backward NOT executed.');return
        raise RuntimeError('Local resource guard: prepare >=48GiB GPU and >=64GiB host RAM; see README.md#training-and-evaluation')
    if not a.worker:
        argv=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node',str(a.gpus),'-m','lingbot_ur5.train',*sys.argv[1:],'--worker']
        subprocess.run(argv,check=True);return
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from .checkpointing import TrainingState
    from distributed.util import init_distributed
    import train as official_train
    cfg.rank=int(os.environ['RANK']);cfg.local_rank=int(os.environ['LOCAL_RANK']);cfg.world_size=int(os.environ['WORLD_SIZE'])
    init_distributed(cfg.world_size,cfg.local_rank,cfg.rank)
    # Swap only the robot dataset adapter, keeping the official model/loss/optimizer.
    official_train.MultiLatentLeRobotDataset=lambda config:load_latent_dataset(config)
    class UR5Trainer(official_train.Trainer):
        def save_checkpoint(self):
            super().save_checkpoint()
            dest=self.save_dir/f'checkpoint_step_{self.step}'
            if not (dest/'transformer/config.json').exists():raise RuntimeError('Inference checkpoint export failed')
            dcp.save({'training':TrainingState(self)},checkpoint_id=str(dest/'training_state'))
            if cfg.rank==0:
                import shutil
                shutil.copy2(pathlib.Path(a.dataset)/'meta/ur5_profile.json',dest/'ur5_profile.json')
                (dest/'resume_ready.json').write_text(json.dumps({'step':self.step,'dataset':str(pathlib.Path(a.dataset).resolve()),'base_checkpoint':cfg.wan22_pretrained_model_name_or_path,'torch':torch.__version__,'resume_semantics':'model/optimizer/scheduler/global step; data shuffle and RNG restart'}))
            dist.barrier()
    trainer=UR5Trainer(cfg)
    if a.resume:
        saved=json.loads((pathlib.Path(a.resume)/'resume_ready.json').read_text())
        if saved['dataset']!=str(pathlib.Path(a.dataset).resolve()):raise ValueError('Resume dataset differs')
        dcp.load({'training':TrainingState(trainer)},checkpoint_id=str(pathlib.Path(a.resume)/'training_state'))
    trainer.train()
    if trainer.step%cfg.save_interval:trainer.save_checkpoint()
    dist.destroy_process_group()
if __name__=='__main__':main()
