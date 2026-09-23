"""Tiny CPU checkpoint test; no LingBot weights or robot connections."""
from types import SimpleNamespace
import torch
import torch.distributed.checkpoint as dcp
from lingbot_ur5.checkpointing import TrainingState
from lingbot_ur5.paths import resolve_inference_checkpoint

def make_trainer():
    model=torch.nn.Linear(2,1)
    opt=torch.optim.AdamW(model.parameters(),lr=0.01)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=0.9)
    return SimpleNamespace(transformer=model,optimizer=opt,lr_scheduler=scheduler,step=0)

def test_cpu_adam_checkpoint_resume(tmp_path):
    original=make_trainer()
    original.transformer(torch.ones(1,2)).sum().backward()
    original.optimizer.step();original.lr_scheduler.step();original.step=7
    dcp.save({'training':TrainingState(original)},checkpoint_id=tmp_path)
    restored=make_trainer()
    dcp.load({'training':TrainingState(restored)},checkpoint_id=tmp_path)
    assert restored.step==7
    assert restored.lr_scheduler.state_dict()==original.lr_scheduler.state_dict()
    for before,after in zip(original.transformer.parameters(),restored.transformer.parameters()):
        torch.testing.assert_close(before,after)
        for key in ('step','exp_avg','exp_avg_sq'):
            torch.testing.assert_close(original.optimizer.state[before][key],restored.optimizer.state[after][key])

def test_checkpoint_root_selects_latest_complete_export(tmp_path):
    for step,complete in [(250,True),(500,True),(750,False)]:
        folder=tmp_path/'checkpoints'/f'checkpoint_step_{step}'
        (folder/'transformer').mkdir(parents=True)
        (folder/'transformer/config.json').write_text('{}')
        if complete:(folder/'resume_ready.json').write_text('{}')
    assert resolve_inference_checkpoint(tmp_path).name=='checkpoint_step_500'
