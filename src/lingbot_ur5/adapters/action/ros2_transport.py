"""ROS 2 Jazzy UR driver transport. Import/init happens only in explicit real mode.

Run with system ROS Python, NOT a Conda Python. It requires calibrated MoveIt FK,
an independently running UR driver, and a user-supplied gripper implementation.
"""
import importlib,threading,time
import numpy as np
from .safety import RobotState,SafetyFilter
from .lingbot_ur5_action_adapter import JOINT_NAMES

def factory(spec,config):
    module,name=spec.split(':');return getattr(importlib.import_module(module),name)(**config)

class ROS2Transport:
    def __init__(self,config,safety):
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.executors import MultiThreadedExecutor
        from control_msgs.action import FollowJointTrajectory
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool
        from ur_msgs.msg import IOStates
        from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy,qos_profile_sensor_data
        from ur_dashboard_msgs.msg import SafetyMode,RobotMode
        from moveit_msgs.srv import GetPositionFK
        self.rclpy=rclpy;self.config=config;self.safety=safety;self.lock=threading.RLock()
        self.final_filter=SafetyFilter(safety)
        self.program_running=None;self.io_time=0
        self.state_msg=None;self.state_time=0;self.safety_mode=None;self.safety_time=0;self.robot_mode=None;self.robot_time=0
        self.goal=None;self.closed=False;self.stop_event=threading.Event();self.last_send=time.monotonic()
        rclpy.init();self.node=rclpy.create_node('lingbot_ur5_executor')
        latched=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        self.node.create_subscription(JointState,config['joint_state_topic'],self._state_cb,qos_profile_sensor_data)
        self.node.create_subscription(IOStates,config['io_state_topic'],self._io_cb,qos_profile_sensor_data)
        # UR publishes modes/program state on change, with transient-local durability.
        self.node.create_subscription(SafetyMode,config['safety_mode_topic'],self._safety_cb,latched)
        self.node.create_subscription(RobotMode,config['robot_mode_topic'],self._robot_cb,latched)
        self.node.create_subscription(Bool,config['program_running_topic'],self._program_cb,latched)
        self.client=ActionClient(self.node,FollowJointTrajectory,config['controller'])
        self.fk=self.node.create_client(GetPositionFK,config['fk_service'])
        self.pool=MultiThreadedExecutor(2);self.pool.add_node(self.node)
        self.thread=threading.Thread(target=self.pool.spin,daemon=True);self.thread.start()
        if not self.client.wait_for_server(timeout_sec=10) or not self.fk.wait_for_service(timeout_sec=10):raise RuntimeError('UR trajectory controller/MoveIt FK unavailable')
        self.gripper=factory(config['gripper_factory'],config['gripper_config'])
        # Read-only subscriptions/FK until first explicit send; no automatic home move.
        self.watchdog_thread=threading.Thread(target=self._watchdog,daemon=True);self.watchdog_thread.start()
    def _state_cb(self,msg):
        with self.lock:self.state_msg=msg;self.state_time=time.monotonic()
    def _io_cb(self,msg):
        with self.lock:self.io_time=time.monotonic()
    def _program_cb(self,msg):
        with self.lock:self.program_running=bool(msg.data)
        if not msg.data:self.cancel()
    def _safety_cb(self,msg):
        with self.lock:self.safety_mode=int(msg.mode);self.safety_time=time.monotonic()
        if int(msg.mode) not in (1,2):self.cancel()
    def _robot_cb(self,msg):
        with self.lock:self.robot_mode=int(msg.mode);self.robot_time=time.monotonic()
        if int(msg.mode)!=7:self.cancel()
    def _wait(self,future,timeout):
        event=threading.Event();future.add_done_callback(lambda _:event.set())
        if not event.wait(timeout):raise TimeoutError('ROS response deadline exceeded')
        return future.result()
    def forward_kinematics(self,joints):
        from moveit_msgs.srv import GetPositionFK
        req=GetPositionFK.Request();req.header.frame_id=self.config['world_frame'];req.fk_link_names=[self.config['tcp_link']]
        req.robot_state.joint_state.name=list(JOINT_NAMES);req.robot_state.joint_state.position=np.asarray(joints).tolist()
        res=self._wait(self.fk.call_async(req),self.safety['watchdog_seconds'])
        if res.error_code.val!=1 or len(res.pose_stamped)!=1:raise RuntimeError('Calibrated FK failed')
        p=res.pose_stamped[0].pose
        return np.array([p.position.x,p.position.y,p.position.z,p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w])
    def read_state(self):
        with self.lock:
            msg=self.state_msg;stamp=self.state_time;safety_mode=self.safety_mode;robot_mode=self.robot_mode
            io_time=self.io_time;program_running=self.program_running
        now=time.monotonic()
        if msg is None or min(stamp,io_time)<now-self.safety['watchdog_seconds']:raise RuntimeError('Missing/stale robot feedback')
        if safety_mode is None or robot_mode is None or program_running is None:raise RuntimeError('Missing latched UR status')
        order=[msg.name.index(name) for name in JOINT_NAMES]
        if len(msg.velocity)!=len(msg.name):raise RuntimeError('Measured joint velocities required')
        q=np.asarray(msg.position)[order];v=np.asarray(msg.velocity)[order]
        return RobotState(q,self.forward_kinematics(q),v,stamp,safety_mode not in (1,2) or robot_mode!=7 or not program_running,self.config['world_frame'])
    def send(self,command):
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint
        if self.stop_event.is_set():raise RuntimeError('Transport stop latched')
        # Recheck fresh robot state immediately before any command crosses ROS.
        state=self.read_state()
        if state.emergency_stop:raise RuntimeError('Robot not in safe RUNNING mode')
        self.final_filter.validate(command,state,time.monotonic())
        goal=FollowJointTrajectory.Goal();goal.trajectory.joint_names=list(JOINT_NAMES)
        point=JointTrajectoryPoint();point.positions=command.joints.tolist()
        ns=int(self.safety['control_dt']*1e9);point.time_from_start.sec=ns//1000000000;point.time_from_start.nanosec=ns%1000000000
        goal.trajectory.points=[point]
        with self.lock:
            if self.stop_event.is_set():raise RuntimeError('Transport stop latched')
            self.last_send=time.monotonic()
            pending=self.client.send_goal_async(goal)
            pending.add_done_callback(self._goal_received)
        try:
            self.goal=self._wait(pending,self.safety['watchdog_seconds'])
        except Exception:
            self.cancel();raise
        if not self.goal.accepted:raise RuntimeError('Trajectory rejected')
        with self.lock:
            if self.stop_event.is_set():
                self.goal.cancel_goal_async();raise RuntimeError('Stopped before goal acknowledgement')
            self.gripper.set_normalized(command.gripper)
        result=self._wait(self.goal.get_result_async(),max(1.0,self.safety['control_dt']*5))
        if result.result.error_code!=0:raise RuntimeError('Trajectory execution failed: '+result.result.error_string)
        self.goal=None
    def _goal_received(self,future):
        # A timed-out request can still be accepted later. Cancel that goal too.
        try:goal=future.result()
        except Exception:return
        with self.lock:
            self.goal=goal
            if self.stop_event.is_set() and goal.accepted:goal.cancel_goal_async()
    def _watchdog(self):
        while not self.closed:
            if self.goal is not None and time.monotonic()-self.last_send>max(self.safety['control_dt']*3,self.safety['watchdog_seconds']):self.cancel()
            time.sleep(0.02)
    def cancel(self):
        with self.lock:
            self.stop_event.set();self.final_filter.stop()
            if self.goal is not None and self.goal.accepted:self.goal.cancel_goal_async()
            if hasattr(self,'gripper'):self.gripper.stop()
    def close(self):
        self.cancel();self.closed=True;self.pool.shutdown();self.node.destroy_node();self.rclpy.shutdown()
