"""Exercise stop races without importing ROS or creating a transport/node."""
import threading
from concurrent.futures import Future
from types import SimpleNamespace
from lingbot_ur5.adapters.action.ros2_transport import ROS2Transport

def test_late_goal_acceptance_after_timeout_is_cancelled():
    obj=ROS2Transport.__new__(ROS2Transport)
    obj.lock=threading.RLock();obj.stop_event=threading.Event();obj.stop_event.set()
    calls=[]
    handle=SimpleNamespace(accepted=True,cancel_goal_async=lambda:calls.append('cancel'))
    future=Future();future.set_result(handle)
    obj._goal_received(future)
    assert calls==['cancel']
    assert obj.goal is handle
