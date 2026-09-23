"""Actual official WebSocket server + project client, backed by a CPU mock policy."""
import json,os,pathlib,socket,subprocess,sys,time,urllib.request
import numpy as np
import pytest
from lingbot_ur5.paths import ROOT
from lingbot_ur5.policy import LingBotPolicy,read_camera
from lingbot_ur5.adapters.camera.real_camera import MockCamera

def test_stale_camera_is_rejected():
    class Stale(MockCamera):
        def read(self):
            obs=super().read();obs['captured_at']-=1;return obs
    with pytest.raises(ValueError,match='stale camera'):read_camera(Stale())

def test_official_websocket_numpy_roundtrip():
    if not (ROOT/'outputs/runtime/lingbot-va').exists():pytest.skip('Run LingBot runtime preparation first')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    code='''import numpy as np
from lingbot_ur5.paths import model_path
model_path()
from scripts.real_run import MockPolicy
from utils.Simple_Remote_Infer.deploy.websocket_policy_server import WebsocketPolicyServer
WebsocketPolicyServer(MockPolicy(np.zeros(6)),host="127.0.0.1",port=PORT,metadata={"profile":"mock-protocol"}).serve_forever()
'''.replace('PORT',str(port))
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join((str(ROOT/'src'),str(ROOT)))
    server=subprocess.Popen([sys.executable,'-c',code],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    client=None
    try:
        deadline=time.monotonic()+15
        while True:
            if server.poll() is not None:raise AssertionError(server.stderr.read().decode())
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz',timeout=0.2) as response:assert response.status==200
                break
            except OSError:
                if time.monotonic()>deadline:raise TimeoutError('Test server startup')
                time.sleep(0.05)
        client=LingBotPolicy(port=port,timeout=2)
        assert client.metadata['profile']=='mock-protocol'
        client.infer({'reset':True})
        reply=client.infer({'obs':MockCamera().read()})
        assert reply['action'].shape==(14,2,16)
        assert reply['action'].dtype==np.float32
        ack=client.infer({'compute_kv_cache':True,'obs':[MockCamera().read()],'state':reply['action']})
        assert 'server_timing' in ack
    finally:
        if client:client.close()
        server.terminate();server.wait(timeout=5);server.stderr.close()
