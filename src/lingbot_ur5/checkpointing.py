"""PyTorch DCP stateful wrapper; optimizer/step resume in addition to inference export."""
class TrainingState:
    def __init__(self,trainer):self.trainer=trainer
    def state_dict(self):
        from torch.distributed.checkpoint.state_dict import get_state_dict
        t=self.trainer;model,optimizer=get_state_dict(t.transformer,t.optimizer)
        return {'model':model,'optimizer':optimizer,'step':t.step,'scheduler':t.lr_scheduler.state_dict()}
    def load_state_dict(self,state):
        from torch.distributed.checkpoint.state_dict import set_state_dict
        t=self.trainer
        set_state_dict(t.transformer,t.optimizer,model_state_dict=state['model'],optim_state_dict=state['optimizer'])
        t.step=state['step'];t.lr_scheduler.load_state_dict(state['scheduler'])
