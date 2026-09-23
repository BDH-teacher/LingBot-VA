import numpy as np
class SimExecutor:
    def __init__(self,task,joint_limits):self.task=task;self.limits=joint_limits
    def execute(self,command):
        q=command.joints
        if not np.isfinite(q).all() or np.any(q<self.limits['lower']) or np.any(q>self.limits['upper']):raise ValueError('Sim joint limit violation')
        robot=self.task.robot
        robot.set_arm_joints(q,np.zeros(6),'left');robot.set_gripper(command.gripper,'left')
        for _ in range(10):self.task.scene.step()
        self.task.take_action_cnt+=1;self.task._update_render()
        if self.task.render_freq:self.task.viewer.render()
        self.task.eval_success=bool(self.task.check_success())
    def success(self):return self.task.eval_success
