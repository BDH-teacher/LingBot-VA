import datetime,json,os,pathlib,subprocess
root=pathlib.Path(os.environ['LINGBOT_UR5_ROOT']);lock=json.loads((root/'configs/sources.lock.json').read_text())
def cmd(args):
    r=subprocess.run(args,capture_output=True,text=True);return (r.stdout+r.stderr).strip()
lines=['# 재현 버전 기록','',f'확인 시각: {datetime.datetime.now().astimezone().isoformat()}','',
       '설치 완료 여부는 local/user/STATUS.md 참조. 아래 표는 실제 checkout/다운로드 검증 파일을 기준으로 합니다.','','## 소스','',
       '|Directory|Remote|Actual HEAD|Lock matches|','|---|---|---|---|']
for name,spec in lock['repos'].items():
    actual=cmd(['git','-C',str(root/'third_party'/name),'rev-parse','HEAD'])
    lines.append(f'|{name}|{spec["url"]}|`{actual}`|{actual==spec["commit"]}|')
lines+=['','## Hugging Face','', '|Repo|Pinned revision|Download verified|','|---|---|---|']
for name,spec in lock['huggingface'].items():
    dest=root/('third_party/RoboTwin-lingbot/assets' if spec['type']=='dataset' else 'checkpoints/pretrained/'+name.split('/')[-1])
    marker=dest/'download_verified.json'
    verified=marker.exists() and all((dest/f['path']).exists() and (dest/f['path']).stat().st_size==f['size'] for f in spec['files'])
    lines.append(f'|{name}|`{spec["revision"]}`|{verified}|')
lines+=['','## GPU/driver','', '```text',cmd(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv']),'```','',
        'driver CUDA compatibility: 13.0 (initial system check). robotwin torch wheel: cu121; lingbot: cu126. Actual package lists: `local/environment/*_pip_freeze.txt`. Conda from-history: `local/environment/*.yml`.','',
        'RoboTwin current checkout is kept for reference; execution uses the LingBot-pinned checkout. CuRobo v0.7.6 is a project compatibility pin, not a SHA explicitly mandated by LingBot.','',
        'All modifications to upstream runtime are documented by `patches/`. This local project has no configured remote unless the user adds one.']
(root/'local/environment').mkdir(parents=True,exist_ok=True)
(root/'local/environment/VERSIONS.md').write_text('\n'.join(lines)+'\n');print('VERSIONS EXPORTED')
