"""Start model server, wait for readiness, run real simulator feedback, stop server."""
import argparse,json,os,pathlib,subprocess,sys,tempfile,time,urllib.request
from lingbot_ur5.paths import ROOT

def run_preflight(server_args):
    log_dir=ROOT/'logs';log_dir.mkdir(parents=True,exist_ok=True)
    # A unique response prevents a previous run's report from hiding a failure.
    with tempfile.TemporaryDirectory(prefix='eval_preflight_',dir=log_dir) as folder:
        report_path=pathlib.Path(folder)/'report.json'
        result=subprocess.run(server_args+['--preflight-only','--preflight-report',str(report_path)],
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (log_dir/'eval_preflight.log').write_text(result.stdout)
        print(result.stdout,end='',flush=True)
        if result.returncode:
            raise SystemExit(f'INFERENCE PREFLIGHT FAILED: see {log_dir / "eval_preflight.log"}')
        if not report_path.is_file():raise SystemExit('INFERENCE PREFLIGHT FAILED: report missing')
        report=json.loads(report_path.read_text())
    if report['status']=='SKIPPED_EXPECTED_VRAM_LIMIT':
        print('추론을 건너뜁니다: 현재 GPU 메모리가 공식 RoboTwin 요구량(약 24GB)보다 작습니다.\n'
              '모델과 평가용 시뮬레이터를 시작하지 않았습니다.\n'
              '이 PC에서는 README STEP 9(기존 데이터 검증)와 STEP 10(loader 검사)을 진행하세요.',flush=True)
        return False
    if report['status']!='PREFLIGHT_PASSED':raise SystemExit('INFERENCE PREFLIGHT FAILED: unexpected status')
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=str(ROOT/'checkpoints/lingbot_ur5'))
    p.add_argument('--dataset',default=str(ROOT/'datasets/ur5_move_can_pot'));p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--official',action='store_true');p.add_argument('--gui',action='store_true');p.add_argument('--port',type=int,default=29056)
    p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    if a.episodes<1:p.error('episodes must be positive')
    conda=os.environ.get('CONDA_EXE','conda')
    server_args=[conda,'run','--no-capture-output','-n','lingbot','python',str(ROOT/'scripts/model_server.py'),'--checkpoint',a.checkpoint,'--dataset',a.dataset,'--port',str(a.port)]
    if a.official:server_args+=['--official']
    if not run_preflight(server_args) or a.preflight_only:return
    log=(ROOT/'logs/eval_server.log').open('w');server=subprocess.Popen(server_args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        deadline=time.monotonic()+1200
        while time.monotonic()<deadline:
            if server.poll() is not None:raise RuntimeError(f'Model server exited {server.returncode}; see logs/eval_server.log')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/healthz',timeout=2) as r:
                    if r.status==200:break
            except OSError:time.sleep(2)
        else:raise TimeoutError('Model server readiness deadline exceeded')
        client=[conda,'run','--no-capture-output','-n','robotwin','python',str(ROOT/'scripts/evaluate.py'),'--episodes',str(a.episodes),'--port',str(a.port)]
        if a.official:client+=['--official']
        if a.gui:client+=['--gui']
        subprocess.run(client,check=True)
    finally:
        import signal
        if server.poll() is None:
            os.killpg(server.pid,signal.SIGTERM)
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
        log.close()
if __name__=='__main__':main()
